// CUDALab Softmax — SFM-0003: vec4 结构 + 2 路 ILP 展开。
//
// 与 incumbent `softmax_vec4` 的唯一差异: 三遍的 stride 循环每轮处理
// **2 个独立**的 4 元素向量（下标 v 与 v+nthreads），两个 8B/16B
// 加载在消费第一个之前都已发射 —— 每线程在途加载数 ×2，用计算
// 隐藏内存往返延迟。
//
// 动机（SFM-0002 的 NCU 教训）: vec4 与 online 的 long_scoreboard
// 都停在 ~51%，DRAM 仅 ~31% 峰值 —— 瓶颈是全局内存**延迟**而非
// **带宽**；减少流量（SFM-0002）收益 ~4%，增加并发在途加载才是
// 直接对策。本实验保持 3 遍结构与 4× 流量不变，只加 ILP ——
// 单变量。
//
// 数值: 与 vec4 完全相同（同一 FP32 中间路径，同一归约骨架，
// 求和/比较顺序逐元素不变 —— 仅循环分组不同；max 满足交换律，
// 遍 2 的加法顺序仍是"线程内沿 stride 升序、每线程 4 个连续
// exp 相加"，与 vec4 逐位一致）。
//
// 回退: H % 4 != 0 或指针未按向量宽度对齐 → 与 vec4 相同的
// 标量内核（softmax_scalar.h）。

#include "softmax_common.h"
#include "softmax_scalar.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>

namespace {

struct __align__(8) Half4Ilp {
    __half2 lo;
    __half2 hi;
};

template <typename T>
struct V4Ilp {
    static constexpr size_t vec_align_bytes = 0;
};

template <>
struct V4Ilp<__half> {
    static constexpr size_t vec_align_bytes = 8;
    static __device__ float4 load(const __half* p, int v) {
        const Half4Ilp h = reinterpret_cast<const Half4Ilp*>(p)[v];
        return make_float4(el_to_float(h.lo.x), el_to_float(h.lo.y),
                           el_to_float(h.hi.x), el_to_float(h.hi.y));
    }
    static __device__ void store(__half* p, int v, float4 f) {
        Half4Ilp h;
        h.lo = __floats2half2_rn(f.x, f.y);
        h.hi = __floats2half2_rn(f.z, f.w);
        reinterpret_cast<Half4Ilp*>(p)[v] = h;
    }
};

template <>
struct V4Ilp<float> {
    static constexpr size_t vec_align_bytes = 16;
    static __device__ float4 load(const float* p, int v) {
        return reinterpret_cast<const float4*>(p)[v];
    }
    static __device__ void store(float* p, int v, float4 f) {
        reinterpret_cast<float4*>(p)[v] = f;
    }
};

__device__ __forceinline__ float v4_max4(float4 f) {
    return fmaxf(fmaxf(f.x, f.y), fmaxf(f.z, f.w));
}

template <typename T>
__global__ void softmax_vec4_ilp2_kernel(const T* __restrict__ x,
                                         T* __restrict__ y,
                                         int H) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const T* __restrict__ xrow = x + (size_t)row * H;
    T* __restrict__ yrow = y + (size_t)row * H;
    __shared__ float s_red[SB_BLOCK / 32];
    const int nvec = H / 4;

    // ---- 遍 1: 行内 max（FP32）—— 2 路展开 ----
    float m = -FLT_MAX;
    int v = tid;
    for (; v + nthreads < nvec; v += 2 * nthreads) {
        const float4 fa = V4Ilp<T>::load(xrow, v);
        const float4 fb = V4Ilp<T>::load(xrow, v + nthreads);
        m = fmaxf(m, v4_max4(fa));
        m = fmaxf(m, v4_max4(fb));
    }
    if (v < nvec) {
        const float4 fa = V4Ilp<T>::load(xrow, v);
        m = fmaxf(m, v4_max4(fa));
    }
    m = block_max<SB_BLOCK>(m, s_red);

    // ---- 遍 2: sum(exp(x - max))（FP32）—— 2 路展开 ----
    float l = 0.f;
    v = tid;
    for (; v + nthreads < nvec; v += 2 * nthreads) {
        const float4 fa = V4Ilp<T>::load(xrow, v);
        const float4 fb = V4Ilp<T>::load(xrow, v + nthreads);
        l += expf(fa.x - m) + expf(fa.y - m) + expf(fa.z - m) + expf(fa.w - m);
        l += expf(fb.x - m) + expf(fb.y - m) + expf(fb.z - m) + expf(fb.w - m);
    }
    if (v < nvec) {
        const float4 fa = V4Ilp<T>::load(xrow, v);
        l += expf(fa.x - m) + expf(fa.y - m) + expf(fa.z - m) + expf(fa.w - m);
    }
    l = block_sum<SB_BLOCK>(l, s_red);

    // ---- 遍 3: y = exp(x - max) / sum —— 2 路展开 ----
    const float inv_l = 1.0f / l;
    v = tid;
    for (; v + nthreads < nvec; v += 2 * nthreads) {
        const float4 fa = V4Ilp<T>::load(xrow, v);
        const float4 fb = V4Ilp<T>::load(xrow, v + nthreads);
        V4Ilp<T>::store(yrow, v, make_float4(
            expf(fa.x - m) * inv_l, expf(fa.y - m) * inv_l,
            expf(fa.z - m) * inv_l, expf(fa.w - m) * inv_l));
        V4Ilp<T>::store(yrow, v + nthreads, make_float4(
            expf(fb.x - m) * inv_l, expf(fb.y - m) * inv_l,
            expf(fb.z - m) * inv_l, expf(fb.w - m) * inv_l));
    }
    if (v < nvec) {
        const float4 fa = V4Ilp<T>::load(xrow, v);
        V4Ilp<T>::store(yrow, v, make_float4(
            expf(fa.x - m) * inv_l, expf(fa.y - m) * inv_l,
            expf(fa.z - m) * inv_l, expf(fa.w - m) * inv_l));
    }
}

template <typename T>
void softmax_vec4_ilp2_fwd(const at::Tensor& x, at::Tensor& out) {
    c10::cuda::CUDAGuard guard(x.device());
    const int M = x.size(0);
    const int H = x.size(1);
    const char* xp = reinterpret_cast<const char*>(x.data_ptr());
    char* yp = reinterpret_cast<char*>(out.data_ptr());
    const size_t align = V4Ilp<T>::vec_align_bytes;
    const bool vec_ok = (H % 4 == 0)
                        && ((reinterpret_cast<uintptr_t>(xp) & (align - 1)) == 0)
                        && ((reinterpret_cast<uintptr_t>(yp) & (align - 1)) == 0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (vec_ok) {
        softmax_vec4_ilp2_kernel<T><<<dim3(M), SB_BLOCK, 0, stream>>>(
            reinterpret_cast<const T*>(xp), reinterpret_cast<T*>(yp), H);
    } else {
        softmax_scalar_kernel<T><<<dim3(M), SB_BLOCK, 0, stream>>>(
            reinterpret_cast<const T*>(xp), reinterpret_cast<T*>(yp), H);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void softmax_vec4_ilp2_fwd(const at::Tensor& x, at::Tensor& out) {
    switch (x.scalar_type()) {
        case at::kHalf:
            softmax_vec4_ilp2_fwd<__half>(x, out);
            break;
        case at::kFloat:
            softmax_vec4_ilp2_fwd<float>(x, out);
            break;
        default:
            TORCH_CHECK(false, "softmax vec4_ilp2 不支持该 dtype");
    }
}

}  // namespace

static struct SoftmaxVec4Ilp2Registrar {
    SoftmaxVec4Ilp2Registrar() {
        register_softmax_variant("softmax_vec4_ilp2", softmax_vec4_ilp2_fwd);
    }
} softmax_vec4_ilp2_registrar;
