// CUDALab GEMV — GEMV-0001: 16B 向量化 load, 结构不变（每行一个 block）。
//
// 实验假设（GEMV-0001）: baseline 的瓶颈是 **单线程在途 load 数太少**
// （NCU: long_scoreboard 占 79.3% warp stall, DRAM 仅 49.3% ——
// DRAM 延迟受限而非吞吐受限）。把 2B 标量 load 换成 16B 向量 load
// （8×fp16 / 4×fp32）, 每线程每次迭代取 8 个元素: 相同线程数下
// 在途字节 ×8, load 指令数 ÷8, 无需改变 block/行 结构 —— 与
// GEMV-0002 的 warp-per-row 结构变化**隔离**向量化的单独收益。
//
// 结构（与 baseline 相同）:
//   grid  = N 个 block; block = 256 线程
//   每线程: i = tid, tid+256, ... 的 16B 向量（fp16 K=4096: 每线程 2 个）
//   归约: 与 baseline 完全相同（warp shuffle + shared + warp0）
//
// 对齐契约（v0.5 总则; 见 gemv_common.h gemv_vec_contract_ok）:
//   W 基指针 16B 对齐 ∧ x 基指针 16B 对齐 ∧ K % epv == 0
//   （epv = 8 for fp16, 4 for fp32）。
//   不满足 → host 侧回退 gemv_scalar_kernel（与 baseline 同一来源,
//   输出逐位一致）; 合法输入（奇数 offset 视图 / K 非 8 倍数）不得
//   被拒。negative 套件 per-variant 回退用例钉死该行为。
//
// 数值: 累加顺序 = 每线程步长 16B 向量部分和（每向量内 8 元素按
// 低→高序 FMA）→ 同一归约树。合法 FP32 累加顺序, 固定 arith 界适用;
// 与 baseline 不要求逐位一致（对齐输入下）。

#include "gemv_common.h"

namespace {

template <typename T>
__global__ void gemv_vec4_row_kernel(const T* __restrict__ W,
                                     const T* __restrict__ x,
                                     T* __restrict__ out,
                                     int64_t N, int64_t K) {
    constexpr int64_t epv = 16 / static_cast<int64_t>(sizeof(T));
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const U16* __restrict__ wrow = reinterpret_cast<const U16*>(W + row * K);
    const U16* __restrict__ xv = reinterpret_cast<const U16*>(x);
    const int64_t nvec = K / epv;  // 每行 16B 向量数

    // 步长 16B 向量部分和（FP32 累加, 逐元素 FMA）
    float acc = 0.f;
#pragma unroll 4
    for (int64_t i = threadIdx.x; i < nvec; i += blockDim.x) {
        U16 w = wrow[i];
        U16 xv_ = xv[i];
        vec_acc<T>(w, xv_, acc);
    }

    // 归约: 与 baseline 相同
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
        if (lane == 0) out[row] = el_from_float<T>(acc);
    }
}

template <typename T>
void launch_gemv_vec4_row(const at::Tensor& W, const at::Tensor& x,
                          at::Tensor& out) {
    const int64_t N = W.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gemv_vec4_row_kernel<T><<<static_cast<int>(N), kGemvScalarBlock, 0,
                              stream>>>(
        reinterpret_cast<const T*>(W.const_data_ptr()),
        reinterpret_cast<const T*>(x.const_data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()),
        W.size(0), W.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemv_vec4_row_fwd(const at::Tensor& W, const at::Tensor& x,
                       at::Tensor& out) {
    if (W.scalar_type() == at::kHalf) {
        if (gemv_vec_contract_ok<__half>(W, x)) {
            launch_gemv_vec4_row<__half>(W, x, out);
        } else {
            launch_gemv_scalar_dispatch(W, x, out);  // 回退: 不得拒绝
        }
    } else {
        if (gemv_vec_contract_ok<float>(W, x)) {
            launch_gemv_vec4_row<float>(W, x, out);
        } else {
            launch_gemv_scalar_dispatch(W, x, out);  // 回退: 不得拒绝
        }
    }
}

static struct Registrar {
    Registrar() { register_gemv_variant("gemv_vec4_row", gemv_vec4_row_fwd); }
} registrar;

}  // namespace
