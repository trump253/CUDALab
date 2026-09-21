// CUDALab QGEMV — QGEMV-0004: warp-per-row + ILP=4（4 独立累加器）,
// 16B 向量 load, 无 shared memory / 无 barrier。
//
// 实验假设（QGEMV-0004, 由 QGEMV-0003 的 NCU 证据驱动）:
//   warp_vec16 (QGEMV-0003, NEUTRAL 1.0107x): dram 87.24%（各变体
//   最高）, occ 90.43%, regs 41, **barrier stall 完全消失**, 但
//   long_scoreboard = 84.1% of stall cycles（18.53 cyc/iss）—— 几乎
//   所有 stall 都是 warp 在等 DRAM。这是"贴着 DRAM 墙"的特征
//   （v0.5 FP16 vec4_row 达 91.4% logical BW 后同样是 latency-wait
//   主导）, 剩余 ~4-13% 头部的唯一杠杆是**提高每 lane 在途 load 数**:
//   QGEMV-0003 的 ILP=2（每 lane 2 个 W 向量在途, K=4096 下 4 个
//   迭代）加深到 ILP=4（4 个 W 向量在途, 2 个迭代）—— 每 warp
//   在途 16B load 字节 ×2（4 W + 8 x 向量）, 赌占用率不掉档
//   （QGEMV-0002 的教训: 占用率掉档会让 latency 暴露更久, 适得其
//   反; 本变体寄存器压力上升是**预期内的风险**, NCU 的 occ/dram
//   两项共同裁决）。
//
// 结构:
//   grid  = ceil(N/8) blocks; block = 256 threads = 8 warps
//   每 warp 1 行: row = blockIdx.x*8 + warp_id（row >= N 退出）
//   每 lane: for (i = lane; i < nvec; i += 128), 每迭代 4 个相位
//       i, i+32, i+64, i+96（4 个独立累加器, 4 路独立在途）
//   归约: 4 累加器相加 → 纯 warp shuffle（5 步）, lane0 写 out[row]
//
// 覆盖性（每向量恰好一次）: 相位 p ∈ {0,1,2,3} 覆盖区间
//   [128t + 32p, 128t + 32p + 32),  t = 0,1,... —— 四个相位分区
//   完整覆盖 [0, nvec), 两两不相交。
//
// 对齐契约: 与前三个变体相同 —— qgemv_vec_contract_ok（W_q 基址
// 16B ∧ x 基址 16B ∧ K%16==0）, 不满足 → 回退 qgemv_scalar_kernel
// （bit-identical）。
//
// 数值: 累加顺序 = 每 lane 4 路步长部分和（每向量内 16 元素低→高
// 序, 逐元素 q→fp32 ×scale ×x, 与 vec16_row 相同语义）→ 4 路相加 →
// warp 树。合法 FP32 累加顺序, 固定 arith 界（3K·2^-24, TOL_K=2）
// 原样适用, 不放宽; 与其余变体不要求逐位一致。

#include "qgemv_common.h"

namespace {

constexpr int kQgemvIlp4WarpsPerBlock = kQgemvScalarBlock / 32;  // 8

__global__ void qgemv_warp_vec16_ilp4_kernel(const int8_t* __restrict__ Wq,
                                             const float* __restrict__ scale,
                                             const __half* __restrict__ x,
                                             __half* __restrict__ out,
                                             int64_t N, int64_t K) {
    constexpr int64_t epv = 16;
    const int64_t row = (static_cast<int64_t>(blockIdx.x)
                         * kQgemvIlp4WarpsPerBlock)
                        + (threadIdx.x >> 5);
    if (row >= N) return;  // N 非 8 倍数时的尾部 warp
    const int lane = threadIdx.x & 31;
    const U16Q* __restrict__ wrow =
        reinterpret_cast<const U16Q*>(Wq + row * K);
    const U16Q* __restrict__ xv = reinterpret_cast<const U16Q*>(x);
    const int64_t nvec = K / epv;
    const float s = scale[row];

    // ILP=4: 4 个独立累加器, 4 路在途（步长 128 向量）
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int64_t i = lane; i < nvec; i += 128) {
        {
            const int64_t p0 = i;
            U16Q w = wrow[p0];
            U16Q x0 = xv[2 * p0];
            U16Q x1 = xv[2 * p0 + 1];
            qgemv_vec_acc_dequant(w, x0, x1, s, a0);
        }
        if (i + 32 < nvec) {
            const int64_t p1 = i + 32;
            U16Q w = wrow[p1];
            U16Q x0 = xv[2 * p1];
            U16Q x1 = xv[2 * p1 + 1];
            qgemv_vec_acc_dequant(w, x0, x1, s, a1);
        }
        if (i + 64 < nvec) {
            const int64_t p2 = i + 64;
            U16Q w = wrow[p2];
            U16Q x0 = xv[2 * p2];
            U16Q x1 = xv[2 * p2 + 1];
            qgemv_vec_acc_dequant(w, x0, x1, s, a2);
        }
        if (i + 96 < nvec) {
            const int64_t p3 = i + 96;
            U16Q w = wrow[p3];
            U16Q x0 = xv[2 * p3];
            U16Q x1 = xv[2 * p3 + 1];
            qgemv_vec_acc_dequant(w, x0, x1, s, a3);
        }
    }
    // 4 路合并 → 纯 warp 归约（无 shared, 无 barrier）
    float acc0 = (a0 + a1) + (a2 + a3);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, off);
    }
    if (lane == 0) out[row] = __float2half_rn(acc0);
}

void qgemv_warp_vec16_ilp4_fwd(const at::Tensor& Wq,
                               const at::Tensor& scale,
                               const at::Tensor& x, at::Tensor& out) {
    if (qgemv_vec_contract_ok(Wq, x)) {
        const int64_t N = Wq.size(0);
        const int64_t grid = (N + kQgemvIlp4WarpsPerBlock - 1)
                             / kQgemvIlp4WarpsPerBlock;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        qgemv_warp_vec16_ilp4_kernel<<<static_cast<int>(grid),
                                       kQgemvScalarBlock, 0, stream>>>(
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
    Registrar() { register_qgemv_variant("qgemv_warp_vec16_ilp4",
                                         qgemv_warp_vec16_ilp4_fwd); }
} registrar;

}  // namespace
