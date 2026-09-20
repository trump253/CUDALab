// CUDALab RoPE — ROPE-0004 候选: 一个 thread -> 8 个 RoPE pair。
//
// 相对 rope_baseline 的唯一改动: 每线程处理相邻的 8 个 pair，
// grid 缩为 M * (D/16)。32 次 global load（16 x + 8 cos + 8 sin）
// 全部 hoist 到计算之前（fp16 下每线程在途 load 字节 8B -> 64B），
// store 推迟到最后。（v0.4 review 更正: 原注释误写 24 次/48B,
// 按源码 kPairs=8 × 4 loads 实为 32 次/64B。）
//
// 依据（baseline NCU, M=1024 D=128 fp16, cc=all clkbase）:
//   long_scoreboard = 69.3% 的 stall —— 延迟受限; DRAM 仅 23%。
//   ROPE-0001/0002 已把 MLP 提到 2x/4x; 本实验把 MLP 推到 8x,
//   测试 (a) 收益是否继续增长, (b) 8192 线程（0.133 wave,
//   每 SM ~273 线程 / 2048 容量 = 13% 占用）的 wave 塌缩何时
//   开始抵消 MLP 收益 —— 即找 pairs/thread 的甜点。
//
// 要求 D%16==0（D=64/128 均满足; launcher launch 前检查）。
// 标量 hoist 访存, 无对齐契约, 无向量化 —— 只测 MLP 杠杆的上限。
// 寄存器压力显著上升（24 个活跃 float + 地址）, 由 NCU 记录。

#include "rope_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

namespace {

static constexpr int kBlock = 128;
static constexpr int kPairs = 8;

template <typename T>
__global__ void rope_v4_8pair_kernel(const T* __restrict__ x,
                                     const int64_t* __restrict__ positions,
                                     const T* __restrict__ cos_t,
                                     const T* __restrict__ sin_t,
                                     T* __restrict__ out,
                                     int64_t n_chunks, int64_t d2) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= n_chunks) return;
    const int64_t per_row = d2 / kPairs;
    const int64_t m = t / per_row;
    const int64_t j = t - m * per_row;    // pair [j*8, j*8+8)
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
void launch_v4_8pair(const at::Tensor& x, const at::Tensor& positions,
                     const at::Tensor& cos_t, const at::Tensor& sin_t,
                     at::Tensor& out) {
    const int64_t M = x.size(0);
    const int64_t d2 = x.size(1) / 2;
    TORCH_CHECK(d2 % kPairs == 0,
                "rope_v4_8pair 要求 D/2 可被 ", kPairs,
                " 整除（D%16==0），实际 D=", x.size(1));
    const int64_t n_chunks = M * (d2 / kPairs);
    const int grid = static_cast<int>((n_chunks + kBlock - 1) / kBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rope_v4_8pair_kernel<T><<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<const T*>(x.const_data_ptr()),
        positions.const_data_ptr<int64_t>(),
        reinterpret_cast<const T*>(cos_t.const_data_ptr()),
        reinterpret_cast<const T*>(sin_t.const_data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()),
        n_chunks, d2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rope_v4_8pair_fwd(const at::Tensor& x, const at::Tensor& positions,
                       const at::Tensor& cos_t, const at::Tensor& sin_t,
                       at::Tensor& out) {
    if (x.scalar_type() == at::kHalf) {
        launch_v4_8pair<__half>(x, positions, cos_t, sin_t, out);
    } else {
        launch_v4_8pair<float>(x, positions, cos_t, sin_t, out);
    }
}

static struct Registrar {
    Registrar() { register_rope_variant("rope_v4_8pair", rope_v4_8pair_fwd); }
} registrar;

}  // namespace
