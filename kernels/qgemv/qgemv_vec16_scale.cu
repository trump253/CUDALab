// CUDALab QGEMV — QGEMV-0002: scale 提升（数学恒等变形, 同向量化结构）。
//
// 实验假设（QGEMV-0002, 由 QGEMV-0001 的 NCU 证据驱动）: vec16_row 已把
// DRAM 利用率从 27.2% 提到 86.0%（dram_throughput_pct）, 但
// long_scoreboard 仍是第一大 stall（51.4% of stall cycles, 7.76
// cyc/iss）, sm_throughput 39.9% —— 剩余 ~14% 头部空间来自两个可
// 隔离的杠杆:（a）每 term 计算指令（baseline 语义每 term 3 条:
// CVT(q→fp32) + MUL(×scale) + FFMA(×x+acc））,（b）归约 barrier
// （7.3%）。本变体只动（a）: **scale 提升** ——
//
//   y[n] = s_n · Σ_k q[n,k]·x[k]     （scale 对整行是常数, 数学恒等）
//
// 每 term 只剩 CVT + FFMA 2 条（q·x 在 FP32 中**精确**: 7 位整数 ×
// 11 位 fp16 尾数 ≤ 18 位 ≤ 24 位, 乘积无舍入; 加法 1 次舍入/term,
// 行末 1 次 ×scale）—— 每 W 向量 16×3=48 条计算 → 16×2=32 条 +
// 1 条行末 MUL, 指令流 ÷~1.5, 在不改变访存模式（同样的 16B W 向量
// + 两个 16B x 向量）下提高在途字节/指令的比值, 目标是把 DRAM
// 利用率从 86% 进一步推向 DRAM ceiling（v0.5 FP16 vec4_row 91%
// regime）。
//
// 结构（与 vec16_row 相同）:
//   grid = N blocks; block = 256 threads
//   每线程: i = tid, tid+256, ... 的 16B W 向量 + 两个连续 16B x 向量
//   归约: 与 baseline 完全相同（warp shuffle + shared + warp0）
//   行末: warp0 lane0 执行 out[row] = __float2half_rn(acc * s)
//
// 对齐契约: 与 vec16_row 相同 —— qgemv_vec_contract_ok
// （W_q 基址 16B ∧ x 基址 16B ∧ K%16==0）, 不满足 → 回退
// qgemv_scalar_kernel（bit-identical）。
//
// 数值: 累加顺序 = 每线程步长 16B 向量部分和（每向量内 16 元素
// 低→高序, 精确乘 + FP32 加）→ 同一归约树 → 行末 ×scale。合法 FP32
// 累加顺序, 每 term 舍入次数（≤2）不高于 baseline 语义（3）, 固定
// arith 界（系数 3K·2^-24, TOL_K=2）原样适用, 不放宽; 与 baseline /
// vec16_row 不要求逐位一致。scale=0 行: acc*0 = 0（精确, 无 NaN）。

#include "qgemv_common.h"

namespace {

__global__ void qgemv_vec16_scale_kernel(const int8_t* __restrict__ Wq,
                                         const float* __restrict__ scale,
                                         const __half* __restrict__ x,
                                         __half* __restrict__ out,
                                         int64_t N, int64_t K) {
    constexpr int64_t epv = 16;
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const U16Q* __restrict__ wrow =
        reinterpret_cast<const U16Q*>(Wq + row * K);
    const U16Q* __restrict__ xv = reinterpret_cast<const U16Q*>(x);
    const int64_t nvec = K / epv;

    // 步长 16B W 向量部分和（FP32 累加, q·x 精确乘 + 单次加）
    float acc = 0.f;
#pragma unroll 4
    for (int64_t i = threadIdx.x; i < nvec; i += blockDim.x) {
        U16Q w = wrow[i];
        U16Q xv0 = xv[2 * i];
        U16Q xv1 = xv[2 * i + 1];
        qgemv_vec_acc_hoist(w, xv0, xv1, acc);
    }

    // 归约: 与 baseline 相同
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    __shared__ float warp_sums[kQgemvScalarWarps];
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = (lane < kQgemvScalarWarps) ? warp_sums[lane] : 0.f;
#pragma unroll
        for (int off = kQgemvScalarWarps / 2; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) {
            // 行末 ×scale（每行 1 次, 恒等变形; scale=0 → 精确 0）
            out[row] = __float2half_rn(acc * scale[row]);
        }
    }
}

void qgemv_vec16_scale_fwd(const at::Tensor& Wq, const at::Tensor& scale,
                           const at::Tensor& x, at::Tensor& out) {
    if (qgemv_vec_contract_ok(Wq, x)) {
        const int64_t N = Wq.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        qgemv_vec16_scale_kernel<<<static_cast<int>(N), kQgemvScalarBlock, 0,
                                   stream>>>(
            static_cast<const int8_t*>(Wq.const_data_ptr()),
            static_cast<const float*>(scale.const_data_ptr()),
            reinterpret_cast<const __half*>(x.const_data_ptr()),
            reinterpret_cast<__half*>(out.data_ptr()),
            Wq.size(0), Wq.size(1));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        launch_qgemv_scalar(Wq, scale, x, out);  // 回退: 不得拒绝
    }
}

static struct Registrar {
    Registrar() { register_qgemv_variant("qgemv_vec16_scale",
                                         qgemv_vec16_scale_fwd); }
} registrar;

}  // namespace
