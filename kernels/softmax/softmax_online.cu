// CUDALab Softmax — SFM-0002: online (m, l) 单遍累积 + 一次归一化写回。
//
// 算法（推导与 CPU 恒等门禁见 docs/softmax_algorithm.md）:
//   每行一个 block。遍 1: 每个线程沿 stride 下标在线累积局部
//   (m_t, l_t) —— 部分 (m,l) = (段内 max, Σexp(x−m)) 是行的充分统计量;
//   block 归约用 merge 恒等成对合并:
//       (m,l) ⊕ (m',l') = (M, l·exp(m−M) + l'·exp(m'−M)), M = max(m,m')
//   （warp shuffle 两两合并 + shared memory 跨 warp 合并）;
//   遍 2: 重读 x, 写 y_i = exp(x_i − M)/L。
//
// 内部流量: 2 读 1 写 = 3× 算法字节数（baseline/vec4 的 3 读 1 写 = 4×）,
// 且 exp 只在归约后计算一次（baseline 在遍 2/遍 3 各算一遍）。
//
// 访存宽度: 4 元素（fp16 8B / fp32 16B, 继承 SFM-0001 的向量化结论）;
// H % 4 != 0 或指针未按向量宽度对齐时回退到标量 online 内核
// （同一 merge 归约, 数值路径相同）。
//
// 数值: FP32 中间量。merge 因子 exp(m−M) ∈ (0,1] 永不放大（无溢出
// 路径）; 空线程局部 (−FLT_MAX, 0) 合并贡献为 0。归约为固定二叉树
// 顺序 ⇒ 确定性。

#include "softmax_common.h"
#include "softmax_scalar.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>

namespace {

struct __align__(8) Half4On {
    __half2 lo;
    __half2 hi;
};

template <typename T>
struct V4On {
    static constexpr size_t vec_align_bytes = 0;
};

template <>
struct V4On<__half> {
    static constexpr size_t vec_align_bytes = 8;
    static __device__ float4 load(const __half* p, int v) {
        const Half4On h = reinterpret_cast<const Half4On*>(p)[v];
        return make_float4(el_to_float(h.lo.x), el_to_float(h.lo.y),
                           el_to_float(h.hi.x), el_to_float(h.hi.y));
    }
    static __device__ void store(__half* p, int v, float4 f) {
        Half4On h;
        h.lo = __floats2half2_rn(f.x, f.y);
        h.hi = __floats2half2_rn(f.z, f.w);
        reinterpret_cast<Half4On*>(p)[v] = h;
    }
};

template <>
struct V4On<float> {
    static constexpr size_t vec_align_bytes = 16;
    static __device__ float4 load(const float* p, int v) {
        return reinterpret_cast<const float4*>(p)[v];
    }
    static __device__ void store(float* p, int v, float4 f) {
        reinterpret_cast<float4*>(p)[v] = f;
    }
};

// block 内 (m, l) 对的 merge 归约: warp shuffle 两两合并 +
// shared memory 跨 warp 顺序合并; 结果广播在 s_m[0] / s_l[0]。
// 注意: 必须成对合并完整 (m,l) 对（⊕ 运算）, 不能拆成
// max 归约 + sum 归约各走各的（重标度因子依赖两个操作数的 m）。
template <int NT>
__device__ void block_merge(float m, float l, float* s_m, float* s_l) {
    const int tid = threadIdx.x;
    const unsigned mask = 0xffffffffu;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float om = __shfl_down_sync(mask, m, offset);
        const float ol = __shfl_down_sync(mask, l, offset);
        const float M = fmaxf(m, om);
        l = l * expf(m - M) + ol * expf(om - M);
        m = M;
    }
    const int nwarp = NT / 32;
    if ((tid & 31) == 0) {
        s_m[tid >> 5] = m;
        s_l[tid >> 5] = l;
    }
    __syncthreads();
    if (tid == 0) {
        float rm = s_m[0];
        float rl = s_l[0];
        for (int i = 1; i < nwarp; i++) {
            const float M = fmaxf(rm, s_m[i]);
            rl = rl * expf(rm - M) + s_l[i] * expf(s_m[i] - M);
            rm = M;
        }
        s_m[0] = rm;
        s_l[0] = rl;
    }
    __syncthreads();
}

// ---- 向量化 online 内核（4 元素/线程）----
template <typename T>
__global__ void softmax_online_vec4_kernel(const T* __restrict__ x,
                                           T* __restrict__ y,
                                           int H) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const T* __restrict__ xrow = x + (size_t)row * H;
    T* __restrict__ yrow = y + (size_t)row * H;
    __shared__ float s_m[SB_BLOCK / 32];
    __shared__ float s_l[SB_BLOCK / 32];
    const int nvec = H / 4;

    // ---- 遍 1: 在线累积局部 (m, l) ----
    float m = -FLT_MAX;
    float l = 0.f;
    for (int v = tid; v < nvec; v += nthreads) {
        const float4 f = V4On<T>::load(xrow, v);
        const float m2 = fmaxf(m, fmaxf(fmaxf(f.x, f.y),
                                        fmaxf(f.z, f.w)));
        l = l * expf(m - m2) + expf(f.x - m2) + expf(f.y - m2)
          + expf(f.z - m2) + expf(f.w - m2);
        m = m2;
    }
    block_merge<SB_BLOCK>(m, l, s_m, s_l);
    const float M = s_m[0];
    const float L = s_l[0];

    // ---- 遍 2: 重读 x, 写 y = exp(x − M) / L ----
    const float inv_l = 1.0f / L;
    for (int v = tid; v < nvec; v += nthreads) {
        const float4 f = V4On<T>::load(xrow, v);
        V4On<T>::store(yrow, v, make_float4(
            expf(f.x - M) * inv_l, expf(f.y - M) * inv_l,
            expf(f.z - M) * inv_l, expf(f.w - M) * inv_l));
    }
}

// ---- 标量 online 内核（回退路径, 与向量化同一数值结构）----
template <typename T>
__global__ void softmax_online_scalar_kernel(const T* __restrict__ x,
                                             T* __restrict__ y,
                                             int H) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const T* __restrict__ xrow = x + (size_t)row * H;
    T* __restrict__ yrow = y + (size_t)row * H;
    __shared__ float s_m[SB_BLOCK / 32];
    __shared__ float s_l[SB_BLOCK / 32];

    float m = -FLT_MAX;
    float l = 0.f;
    for (int i = tid; i < H; i += nthreads) {
        const float v = el_to_float(xrow[i]);
        const float m2 = fmaxf(m, v);
        l = l * expf(m - m2) + expf(v - m2);
        m = m2;
    }
    block_merge<SB_BLOCK>(m, l, s_m, s_l);
    const float M = s_m[0];
    const float L = s_l[0];
    const float inv_l = 1.0f / L;
    for (int i = tid; i < H; i += nthreads)
        yrow[i] = el_from_float<T>(expf(el_to_float(xrow[i]) - M) * inv_l);
}

template <typename T>
void softmax_online_fwd(const at::Tensor& x, at::Tensor& out) {
    c10::cuda::CUDAGuard guard(x.device());
    const int M = x.size(0);
    const int H = x.size(1);
    const char* xp = reinterpret_cast<const char*>(x.data_ptr());
    char* yp = reinterpret_cast<char*>(out.data_ptr());
    const size_t align = V4On<T>::vec_align_bytes;
    const bool vec_ok = (H % 4 == 0)
                        && ((reinterpret_cast<uintptr_t>(xp) & (align - 1)) == 0)
                        && ((reinterpret_cast<uintptr_t>(yp) & (align - 1)) == 0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (vec_ok) {
        softmax_online_vec4_kernel<T><<<dim3(M), SB_BLOCK, 0, stream>>>(
            reinterpret_cast<const T*>(xp), reinterpret_cast<T*>(yp), H);
    } else {
        softmax_online_scalar_kernel<T><<<dim3(M), SB_BLOCK, 0, stream>>>(
            reinterpret_cast<const T*>(xp), reinterpret_cast<T*>(yp), H);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void softmax_online_fwd(const at::Tensor& x, at::Tensor& out) {
    switch (x.scalar_type()) {
        case at::kHalf:
            softmax_online_fwd<__half>(x, out);
            break;
        case at::kFloat:
            softmax_online_fwd<float>(x, out);
            break;
        default:
            TORCH_CHECK(false, "softmax online 不支持该 dtype");
    }
}

}  // namespace

static struct SoftmaxOnlineRegistrar {
    SoftmaxOnlineRegistrar() {
        register_softmax_variant("softmax_online", softmax_online_fwd);
    }
} softmax_online_registrar;
