// CUDALab RoPE — baseline 变体: 一个 thread -> 一个 RoPE pair。
//
// grid 覆盖 M * (D/2) 个线程（1D grid, block 128）:
//   t  -> row m = t / (D/2), pair i = t % (D/2)
// 每线程: 载入 x 的一对 (a, b) + cos/sin 各一个值（FP32 提升），
// 旋转后写回一对。无复杂融合、无向量化 —— 作为 v0.4 RoPE 的
// 基准（incumbent），实验候选（ROPE-0001+）相对它做 paired 对比。
//
// 输入契约（launch 前由 bindings.cpp 的 validate_rope_inputs 保证，
// 内核内不含设备端断言）:
//   x (M,D) 连续, D 偶数; positions (M,) int64 连续, 0 <= p < L;
//   cos_t / sin_t (L, D/2) 连续, dtype 与 x 相同。

#include "rope_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

namespace {

static constexpr int kBlock = 128;

template <typename T>
__global__ void rope_baseline_kernel(const T* __restrict__ x,
                                     const int64_t* __restrict__ positions,
                                     const T* __restrict__ cos_t,
                                     const T* __restrict__ sin_t,
                                     T* __restrict__ out,
                                     int64_t n_pairs, int64_t d2) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= n_pairs) return;
    int64_t m = t / d2;          // row
    int64_t i = t - m * d2;      // pair index within row
    int64_t pos = positions[m];

    const T* xp = x + m * (2 * d2) + 2 * i;
    float a = el_to_float(xp[0]);
    float b = el_to_float(xp[1]);
    float c = el_to_float(cos_t[pos * d2 + i]);
    float s = el_to_float(sin_t[pos * d2 + i]);

    T* yp = out + m * (2 * d2) + 2 * i;
    yp[0] = el_from_float<T>(a * c - b * s);
    yp[1] = el_from_float<T>(a * s + b * c);
}

template <typename T>
void launch_baseline(const at::Tensor& x, const at::Tensor& positions,
                     const at::Tensor& cos_t, const at::Tensor& sin_t,
                     at::Tensor& out) {
    const int64_t M = x.size(0);
    const int64_t d2 = x.size(1) / 2;
    const int64_t n_pairs = M * d2;
    const int grid = static_cast<int>((n_pairs + kBlock - 1) / kBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rope_baseline_kernel<T><<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<const T*>(x.const_data_ptr()),
        positions.const_data_ptr<int64_t>(),
        reinterpret_cast<const T*>(cos_t.const_data_ptr()),
        reinterpret_cast<const T*>(sin_t.const_data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()),
        n_pairs, d2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rope_baseline_fwd(const at::Tensor& x, const at::Tensor& positions,
                       const at::Tensor& cos_t, const at::Tensor& sin_t,
                       at::Tensor& out) {
    // dtype 已在 host 侧验证（仅 half / float）
    if (x.scalar_type() == at::kHalf) {
        launch_baseline<__half>(x, positions, cos_t, sin_t, out);
    } else {
        launch_baseline<float>(x, positions, cos_t, sin_t, out);
    }
}

static struct Registrar {
    Registrar() { register_rope_variant("rope_baseline", rope_baseline_fwd); }
} registrar;

}  // namespace
