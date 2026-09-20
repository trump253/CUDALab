// CUDALab RoPE — ROPE-0001 候选: 一个 thread -> 2 个 RoPE pair。
//
// 相对 rope_baseline 的唯一改动: 每线程处理相邻的 2 个 pair
// （j*P, j*P+1），grid 减半为 M * (D/4)。8 次 global load 全部
// hoist 到 8 次 FP32 乘法之前（显式提高每线程在途字节数 /
// memory-level parallelism），store 同样推迟到计算之后。
//
// 依据（baseline NCU, M=1024 D=128 fp16, cc=all clkbase）:
//   long_scoreboard = 69.3% 的 stall（内存延迟等待）,
//   dram 23.25% / sm 11.61% —— 延迟受限而非带宽/计算受限;
//   baseline 每线程 4 个标量 load 在途（a, b, c, s; fp16 共 8B）。
// 假设: 在途字节 x2（fp16: 8B -> 16B load）将把 long_scoreboard
// 占比与 kernel 时长同时压下来, 且 32768 线程（0.535 wave）的
// 占用损失不抵消收益。
//
// 无向量化访存（标量 hoist）: 本实验只测 MLP 这一个杠杆,
// 避免与访存宽度变化混淆归因。无对齐契约（标量访问）。

#include "rope_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

namespace {

static constexpr int kBlock = 128;
static constexpr int kPairs = 2;

template <typename T>
__global__ void rope_v1_2pair_kernel(const T* __restrict__ x,
                                     const int64_t* __restrict__ positions,
                                     const T* __restrict__ cos_t,
                                     const T* __restrict__ sin_t,
                                     T* __restrict__ out,
                                     int64_t n_chunks, int64_t d2) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= n_chunks) return;
    const int64_t per_row = d2 / kPairs;   // 每行的 chunk 数
    const int64_t m = t / per_row;
    const int64_t j = t - m * per_row;    // chunk 在行内索引: pair [j*2, j*2+1)
    const int64_t pos = positions[m];

    const T* xp = x + m * (2 * d2) + 2 * j * kPairs;
    const T* cp = cos_t + pos * d2 + j * kPairs;
    const T* sp = sin_t + pos * d2 + j * kPairs;

    // ---- 全部 load hoist（MLP）----
    float a[kPairs], b[kPairs], c[kPairs], s[kPairs];
#pragma unroll
    for (int k = 0; k < kPairs; ++k) {
        a[k] = el_to_float(xp[2 * k]);
        b[k] = el_to_float(xp[2 * k + 1]);
        c[k] = el_to_float(cp[k]);
        s[k] = el_to_float(sp[k]);
    }
    // ---- 计算 ----
    T* yp = out + m * (2 * d2) + 2 * j * kPairs;
#pragma unroll
    for (int k = 0; k < kPairs; ++k) {
        yp[2 * k] = el_from_float<T>(a[k] * c[k] - b[k] * s[k]);
        yp[2 * k + 1] = el_from_float<T>(a[k] * s[k] + b[k] * c[k]);
    }
}

template <typename T>
void launch_v1_2pair(const at::Tensor& x, const at::Tensor& positions,
                     const at::Tensor& cos_t, const at::Tensor& sin_t,
                     at::Tensor& out) {
    const int64_t M = x.size(0);
    const int64_t d2 = x.size(1) / 2;
    TORCH_CHECK(d2 % kPairs == 0,
                "rope_v1_2pair 要求 D/2 可被 ", kPairs,
                " 整除（D%4==0），实际 D=", x.size(1));
    const int64_t n_chunks = M * (d2 / kPairs);
    const int grid = static_cast<int>((n_chunks + kBlock - 1) / kBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rope_v1_2pair_kernel<T><<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<const T*>(x.const_data_ptr()),
        positions.const_data_ptr<int64_t>(),
        reinterpret_cast<const T*>(cos_t.const_data_ptr()),
        reinterpret_cast<const T*>(sin_t.const_data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()),
        n_chunks, d2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rope_v1_2pair_fwd(const at::Tensor& x, const at::Tensor& positions,
                       const at::Tensor& cos_t, const at::Tensor& sin_t,
                       at::Tensor& out) {
    if (x.scalar_type() == at::kHalf) {
        launch_v1_2pair<__half>(x, positions, cos_t, sin_t, out);
    } else {
        launch_v1_2pair<float>(x, positions, cos_t, sin_t, out);
    }
}

static struct Registrar {
    Registrar() { register_rope_variant("rope_v1_2pair", rope_v1_2pair_fwd); }
} registrar;

}  // namespace
