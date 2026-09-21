// CUDALab GEMV — GEMV-0004: 4 路 split-K（partial + combine 两 kernel）。
//
// 实验假设（GEMV-0004）: 把归约维 K 切成 4 段, 每段由独立的
// (行, 段) block 归约并写 FP32 部分和, 第二个 kernel 求和 + cast。
// 动机: 主目标 N=K=4096 时 baseline 已有 4096 个 block（并行度充足,
// 预期收益有限）, 但 **小 N / 小 K 形状**（1024×4096: 1024 blocks;
// 4096×1024: 每 block 只做 2KB 行, 归约开销占比大）并行度/每 block
// 工作量都偏小 —— split-K×4 把 block 数 ×4、每 block 工作量 ÷4,
// 测试该结构杠杆对 DRAM 延迟受限 regime 的普遍性。本实验按用户
// 要求"可以考虑但不要机械执行": 主目标上预期收益有限, 若 paired
// 证据不支持, 记录为失败实验并保留。
//
// 结构:
//   kernel 1  gemv_splitk_partial_kernel
//     grid = (N, 4), block = 256 线程（标量 load —— 本实验只测
//     split-K 结构, 不混入向量化; 与 GEMV-0001/0002 隔离）
//     每 block 归约 W[row, c·K/4 : (c+1)·K/4] · x 的同段, 写
//     partials[row·4 + c]（FP32, [N][4] 布局）
//   kernel 2  gemv_splitk_combine_kernel
//     每线程 1 行: out[row] = cast(Σ_c partials[row·4 + c])
//   两次 launch 在同一 stream 上顺序执行, 对 harness 是一次算子调用。
//
// workspace: partials 缓冲（N·4·4B, 主目标 64KB）在**首次调用**时
// 分配, 之后按 N 增长单调复用（static 缓存）—— 计时区的 warmup
// （≥150 launches）覆盖首次分配, 稳态计时区域内无 malloc。
//
// 契约: K % 4 == 0（段长整数）; 不满足 → 回退 gemv_scalar_kernel
// （逐位一致于 baseline）; 合法输入不得被拒。标量访存, **无指针
// 对齐契约**（任意连续合法输入在 K%4==0 时走 split-K 路径）。
//
// 数值: 部分和顺序 = 每段内步长部分和 → 段内归约树 → FP32 段和
// 相加 → 最终 cast。合法 FP32 累加顺序（split-K 的额外 FP32 段间
// 加法在界内: 每段部分和自身满足 2·(K/4)·2^-24·S_seg 界, 4 段相加
// 3 次额外舍入被 TOL_K 余量覆盖 —— 固定 arith 界按 2K 全量系数计,
// 对任意合法顺序成立）; 与 baseline 不要求逐位一致。

#include "gemv_common.h"
#include <ATen/ATen.h>

namespace {

static constexpr int kSplit = 4;

template <typename T>
__global__ void gemv_splitk_partial_kernel(const T* __restrict__ W,
                                           const T* __restrict__ x,
                                           float* __restrict__ partials,
                                           int64_t N, int64_t K) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int64_t c = static_cast<int64_t>(blockIdx.y);
    const int64_t seg = K / kSplit;
    const T* __restrict__ wrow = W + row * K + c * seg;
    const T* __restrict__ xs = x + c * seg;

    float acc = 0.f;
    for (int64_t k = threadIdx.x; k < seg; k += blockDim.x) {
        acc += el_to_float(wrow[k]) * el_to_float(xs[k]);
    }

    // 两级归约（同 baseline 结构）
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    __shared__ float warp_sums[kGemvScalarWarps];
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = (lane < kGemvScalarWarps) ? warp_sums[lane] : 0.f;
#pragma unroll
        for (int off = kGemvScalarWarps / 2; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) partials[row * kSplit + c] = acc;
    }
}

template <typename T>
__global__ void gemv_splitk_combine_kernel(const float* __restrict__ partials,
                                           T* __restrict__ out, int64_t N) {
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const float s = (partials[row * kSplit + 0] + partials[row * kSplit + 1]) +
                    (partials[row * kSplit + 2] + partials[row * kSplit + 3]);
    out[row] = el_from_float<T>(s);
}

// workspace 缓存: 首次调用分配, N 增长时重分配（单调）。
// 单线程 harness 使用; 计时 warmup 覆盖首次分配。
static at::Tensor g_splitk_partials;

template <typename T>
void launch_gemv_splitk4(const at::Tensor& W, const at::Tensor& x,
                         at::Tensor& out) {
    const int64_t N = W.size(0);
    const int64_t K = W.size(1);
    if (g_splitk_partials.numel() < N * kSplit) {
        g_splitk_partials = at::empty({N * kSplit},
                                      W.options().dtype(at::kFloat));
    }
    float* partials = reinterpret_cast<float*>(g_splitk_partials.data_ptr());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid1(static_cast<unsigned>(N), kSplit);
    gemv_splitk_partial_kernel<T><<<grid1, kGemvScalarBlock, 0, stream>>>(
        reinterpret_cast<const T*>(W.const_data_ptr()),
        reinterpret_cast<const T*>(x.const_data_ptr()),
        partials, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int grid2 = static_cast<int>((N + kGemvScalarBlock - 1) /
                                       kGemvScalarBlock);
    gemv_splitk_combine_kernel<T><<<grid2, kGemvScalarBlock, 0, stream>>>(
        partials, reinterpret_cast<T*>(out.data_ptr()), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemv_splitk4_fwd(const at::Tensor& W, const at::Tensor& x,
                      at::Tensor& out) {
    if (W.size(1) % kSplit != 0) {
        launch_gemv_scalar_dispatch(W, x, out);  // 回退: 不得拒绝
        return;
    }
    if (W.scalar_type() == at::kHalf) {
        launch_gemv_splitk4<__half>(W, x, out);
    } else {
        launch_gemv_splitk4<float>(W, x, out);
    }
}

static struct Registrar {
    Registrar() { register_gemv_variant("gemv_splitk4", gemv_splitk4_fwd); }
} registrar;

}  // namespace
