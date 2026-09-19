// CUDALab Softmax — baseline: 每行一个 block，三遍（max → sum → normalize）。
//
// 目标: 正确、可读、可剖析的参照实现。不是为了快，而是为了给
// profiler 一个明确的瓶颈画像，让后续优化实验从剖析证据出发。
//
// 结构:
//   - grid = (M,)，block = 256 线程，线程沿行内下标 stride-loop。
//   - 遍 1: 行内 max（FP32），warp shuffle + shared memory 归约。
//   - 遍 2: sum(exp(x - max))（FP32），同样的归约。
//   - 遍 3: y = exp(x - max) / sum —— 重新读取 x 并重新计算 exp
//     （"exp reuse / 更少重读"是后续实验的方向，baseline 不优化）。
//
// 数值: 中间量全部 FP32（max / exp / sum / 归一化），输出转回原 dtype。
// 对任意 H（无向量化，无 H 对齐要求）与任意 M 成立。
//
// 显式不做的事（保持 baseline 身份）:
//   - 不向量化访存（标量 T 加载）。
//   - 不复用 exp 结果（遍 3 重算）。
//   - 不在线/单遍化（online softmax 需先过 docs/softmax_algorithm.md
//     推导 + CPU merge 恒等测试，之后才允许进 CUDA）。
//
// 设备代码（归约 + 标量内核）自 SFM-0001 起放在 softmax_scalar.h，
// 与向量化变体的 fallback 共用；本文件的 dispatch / registrar 不变，
// 生成的二进制行为与原版逐指令一致。

#include "softmax_common.h"
#include "softmax_scalar.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>

namespace {

template <typename T>
void launch(const at::Tensor& x, at::Tensor& out) {
    const int M = x.size(0);
    const int H = x.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    softmax_scalar_kernel<T><<<dim3(M), SB_BLOCK, 0, stream>>>(
        reinterpret_cast<const T*>(x.data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()), H);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void softmax_baseline_fwd(const at::Tensor& x, at::Tensor& out) {
    c10::cuda::CUDAGuard guard(x.device());
    switch (x.scalar_type()) {
        case at::kHalf:
            launch<__half>(x, out);
            break;
        case at::kFloat:
            launch<float>(x, out);
            break;
        default:
            TORCH_CHECK(false, "softmax baseline 不支持该 dtype");
    }
}

}  // namespace

static struct SoftmaxBaselineRegistrar {
    SoftmaxBaselineRegistrar() {
        register_softmax_variant("softmax_baseline", softmax_baseline_fwd);
    }
} softmax_baseline_registrar;
