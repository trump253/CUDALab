// CUDALab Softmax — SFM-0001: 4 元素向量化访存的 3 遍 softmax。
//
// 与 baseline 的唯一算法级差异: 访存宽度 1 -> 4 元素
// （fp16: 8B = __half2 x2; fp32: 16B = float4）。
// 3 遍结构（max -> sum(exp) -> normalize）、FP32 中间量、遍 3 重算
// exp、block 归约骨架 —— 全部与 baseline 相同（共用 softmax_scalar.h）。
//
// 输入契约与 baseline 一致: 任意 H、任意合法对齐的连续 2D 张量。
// H % 4 != 0 或 x/out 基址未按向量宽度对齐时，回退到与 baseline
// 相同的标量内核（softmax_scalar_kernel），不拒绝输入。
//
// 剖析动机见 experiments/softmax/SFM-0001.md:
// baseline long_scoreboard 60.6% stall、DRAM 仅 ~18-22% 峰值，
// 瓶颈是标量小事务的内存指令数 / 往返延迟，而非 DRAM 带宽本身。

#include "softmax_common.h"
#include "softmax_scalar.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>

namespace {

struct __align__(8) Half4 {
    __half2 lo;
    __half2 hi;
};

// 4 元素加载/存储 traits（T -> float4 中间表示）。
template <typename T>
struct V4Traits {
    static constexpr size_t vec_align_bytes = 0;
};

template <>
struct V4Traits<__half> {
    static constexpr size_t vec_align_bytes = 8;
    static __device__ float4 load(const __half* p, int v) {
        const Half4 h = reinterpret_cast<const Half4*>(p)[v];
        return make_float4(el_to_float(h.lo.x), el_to_float(h.lo.y),
                           el_to_float(h.hi.x), el_to_float(h.hi.y));
    }
    static __device__ void store(__half* p, int v, float4 f) {
        Half4 h;
        h.lo = __floats2half2_rn(f.x, f.y);
        h.hi = __floats2half2_rn(f.z, f.w);
        reinterpret_cast<Half4*>(p)[v] = h;
    }
};

template <>
struct V4Traits<float> {
    static constexpr size_t vec_align_bytes = 16;
    static __device__ float4 load(const float* p, int v) {
        return reinterpret_cast<const float4*>(p)[v];
    }
    static __device__ void store(float* p, int v, float4 f) {
        reinterpret_cast<float4*>(p)[v] = f;
    }
};

template <typename T>
__global__ void softmax_vec4_kernel(const T* __restrict__ x,
                                    T* __restrict__ y,
                                    int H) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const T* __restrict__ xrow = x + (size_t)row * H;
    T* __restrict__ yrow = y + (size_t)row * H;
    __shared__ float s_red[SB_BLOCK / 32];
    const int nvec = H / 4;

    // ---- 遍 1: 行内 max（FP32）----
    float m = -FLT_MAX;
    for (int v = tid; v < nvec; v += nthreads) {
        const float4 f = V4Traits<T>::load(xrow, v);
        m = fmaxf(m, fmaxf(fmaxf(f.x, f.y), fmaxf(f.z, f.w)));
    }
    m = block_max<SB_BLOCK>(m, s_red);

    // ---- 遍 2: sum(exp(x - max))（FP32）----
    float l = 0.f;
    for (int v = tid; v < nvec; v += nthreads) {
        const float4 f = V4Traits<T>::load(xrow, v);
        l += expf(f.x - m) + expf(f.y - m) + expf(f.z - m) + expf(f.w - m);
    }
    l = block_sum<SB_BLOCK>(l, s_red);

    // ---- 遍 3: y = exp(x - max) / sum（重读 x、重算 exp）----
    const float inv_l = 1.0f / l;
    for (int v = tid; v < nvec; v += nthreads) {
        const float4 f = V4Traits<T>::load(xrow, v);
        V4Traits<T>::store(yrow, v, make_float4(
            expf(f.x - m) * inv_l, expf(f.y - m) * inv_l,
            expf(f.z - m) * inv_l, expf(f.w - m) * inv_l));
    }
}

template <typename T>
void softmax_vec4_fwd(const at::Tensor& x, at::Tensor& out) {
    c10::cuda::CUDAGuard guard(x.device());
    const int M = x.size(0);
    const int H = x.size(1);
    const char* xp = reinterpret_cast<const char*>(x.data_ptr());
    char* yp = reinterpret_cast<char*>(out.data_ptr());
    const size_t align = V4Traits<T>::vec_align_bytes;
    const bool vec_ok = (H % 4 == 0)
                        && ((reinterpret_cast<uintptr_t>(xp) & (align - 1)) == 0)
                        && ((reinterpret_cast<uintptr_t>(yp) & (align - 1)) == 0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (vec_ok) {
        softmax_vec4_kernel<T><<<dim3(M), SB_BLOCK, 0, stream>>>(
            reinterpret_cast<const T*>(xp), reinterpret_cast<T*>(yp), H);
    } else {
        // 回退: 与 baseline 同一标量内核（数值行为逐位一致）。
        softmax_scalar_kernel<T><<<dim3(M), SB_BLOCK, 0, stream>>>(
            reinterpret_cast<const T*>(xp), reinterpret_cast<T*>(yp), H);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void softmax_vec4_fwd(const at::Tensor& x, at::Tensor& out) {
    switch (x.scalar_type()) {
        case at::kHalf:
            softmax_vec4_fwd<__half>(x, out);
            break;
        case at::kFloat:
            softmax_vec4_fwd<float>(x, out);
            break;
        default:
            TORCH_CHECK(false, "softmax vec4 不支持该 dtype");
    }
}

}  // namespace

static struct SoftmaxVec4Registrar {
    SoftmaxVec4Registrar() {
        register_softmax_variant("softmax_vec4", softmax_vec4_fwd);
    }
} softmax_vec4_registrar;
