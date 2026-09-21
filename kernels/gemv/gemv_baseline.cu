// CUDALab GEMV — GEMV-0000 基线: 每输出行一个 block, FP32 归约, FP16 输出。
//
// 这是 v0.5 的 **parent / incumbent 起点**（用户指定的基线形态, 不是
// 从 cuBLAS / llama.cpp 成熟 GEMV kernel 抄来的优化实现）:
//
//   grid  = N 个 block（一个 block 负责一个输出行 y[n]）
//   block = 256 线程固定
//   每线程: 以 blockDim.x 步长 strided 访问该行
//              acc += W[n, k] * x[k]     （el_to_float 提升到 FP32,
//                                          FP32 累加, FMA 由 nvcc 收缩）
//   归约:  warp shuffle（5 步）+ shared 跨 warp（8 个 warp sum）
//              + 二次 warp shuffle（8 值, 3 步）→ 线程 0 执行
//              el_from_float（fp16 路径为逐值 RN 舍入）并写 out[n]
//
// 访存: 纯标量（fp16 路径每次 2B load）, 行内相邻线程访问相邻元素
// （coalesced）; x 被每行重新读取（K 个元素, L2 缓存, 8KB @ K=4096
// fp16）——不预取、不共享内存 staging、不向量化。这些是后续实验
// （GEMV-0001..）的候选杠杆, 基线刻意不引入。
//
// 无对齐契约: 标量 2B/4B load 在连续契约下天然满足, 任何连续合法
// 输入（含 storage offset 视图）都必须成功（negative 套件
// valid_offset_view_control 钉死）。
//
// 数值约定: 累加顺序 = 每线程步长部分和（k = tid, tid+B, ...）→
// warp 树 → shared 树。该顺序是 baseline 自己的, 候选变体（warp-per-
// row / vector / split-K）顺序不同, 正确性合同（gemv_correctness.py
// 的固定 arith 界）对任何合法 FP32 累加顺序都成立, 变体间不要求
// 逐位一致。

#include "gemv_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

namespace {

static constexpr int kBlock = 256;
static constexpr int kWarps = kBlock / 32;  // 8, 2 的幂（二次 shuffle 用）

template <typename T>
__global__ void gemv_baseline_kernel(const T* __restrict__ W,
                                     const T* __restrict__ x,
                                     T* __restrict__ out,
                                     int64_t N, int64_t K) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const T* __restrict__ wrow = W + row * K;

    // 步长部分和（FP32 累加; K 不要求整除 blockDim.x）
    float acc = 0.f;
    for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
        acc += el_to_float(wrow[k]) * el_to_float(x[k]);
    }

    // 阶段 1: warp 内 shuffle 归约
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    __shared__ float warp_sums[kWarps];
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();

    // 阶段 2: warp 0 归约 8 个 warp sum（kWarps 为 2 的幂, 越界 lane 补 0）
    if (wid == 0) {
        acc = (lane < kWarps) ? warp_sums[lane] : 0.f;
#pragma unroll
        for (int off = kWarps / 2; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) out[row] = el_from_float<T>(acc);
    }
}

template <typename T>
void launch_gemv_baseline(const at::Tensor& W, const at::Tensor& x,
                          at::Tensor& out) {
    const int64_t N = W.size(0);
    const int64_t K = W.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gemv_baseline_kernel<T><<<static_cast<int>(N), kBlock, 0, stream>>>(
        reinterpret_cast<const T*>(W.const_data_ptr()),
        reinterpret_cast<const T*>(x.const_data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()),
        N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemv_baseline_fwd(const at::Tensor& W, const at::Tensor& x,
                       at::Tensor& out) {
    if (W.scalar_type() == at::kHalf) {
        launch_gemv_baseline<__half>(W, x, out);
    } else {
        launch_gemv_baseline<float>(W, x, out);
    }
}

static struct Registrar {
    Registrar() { register_gemv_variant("gemv_baseline", gemv_baseline_fwd); }
} registrar;

}  // namespace
