// CUDALab QGEMV — QGEMV-0003: warp-per-row（每 warp 一个输出行）,
// 16B 向量 load, ILP=2 双累加器, 无 shared memory / 无 barrier。
//
// 实验假设（QGEMV-0003, 由 QGEMV-0001/0002 的 NCU 证据驱动）:
//   vec16_row:  dram 86.0%, long_scoreboard 51.4% of stalls (7.76
//               cyc/iss), barrier 7.3%, occ 82.6%, regs 40, sm 39.9%
//   vec16_scale: dram 82.7%（更差）, occ 68.4%（掉档）, long_scoreboard
//               49.2% —— **计算指令削减不是杠杆**（×scale 乘法被访存
//               延迟完全隐藏, 削减后占用率反而掉档, 延迟暴露更久）。
// 结论: 剩余瓶颈是**每线程在途字节太少**（K=4096, 256 threads/block:
// 每线程恰好 1 个 W 向量, 在途 load = 3 条 16B）+ 归约 barrier
// （每行 2 次 __syncthreads）。本变体做**结构变化**:
//
//   grid  = ceil(N/8) blocks; block = 256 threads = 8 warps
//   每 warp 负责 1 行: row = blockIdx.x*8 + warp_id（row >= N 时退出）
//   每 lane: 步长 32 个 W 向量, ILP=2 双累加器（i 与 i+32 两路
//       独立在途, 每 lane 在 K=4096 下 4+4=8 个 W 向量 in flight
//       链, 在途 load 字节 ×~8 vs vec16_row 的每线程 1 向量）
//   归约: 纯 warp shuffle（5 步）, lane0 写 out[row]
//          —— **没有 shared memory, 没有 __syncthreads**（barrier
//          stall 8.6% 整个消除）
//
// 访存合并性: 同一 warp 的 32 lane 每迭代覆盖连续 32×16B = 512B
// （与 vec16_row 的 warp 级合并模式相同）; x 被 8 个 warp 重读
// （8KB @ K=4096, L1/L2 缓存, 算法流量口径不变）。
//
// 对齐契约: 与 vec16_row 相同 —— qgemv_vec_contract_ok（W_q 基址
// 16B ∧ x 基址 16B ∧ K%16==0）, 不满足 → 回退 qgemv_scalar_kernel
// （bit-identical）。
//
// 数值: 累加顺序 = 每 lane 双路步长部分和（i 路 + i+32 路, 每向量
// 内 16 元素低→高序, 逐元素 q→fp32 ×scale ×x, 与 vec16_row 相同
// 的每 term 语义）→ acc0+acc1 → warp 树。合法 FP32 累加顺序, 固定
// arith 界（3K·2^-24, TOL_K=2）原样适用, 不放宽; 与 baseline /
// vec16_row 不要求逐位一致。
//
// 尾部: K=11008 → nvec=688 非 64 倍数, i+32 路在尾部按 lane 分歧
// （每向量恰好被处理一次, 见文件尾推导; 仅尾部 1 次迭代分歧, 成本
// 可忽略）。

#include "qgemv_common.h"

namespace {

constexpr int kQgemvWarpWarpsPerBlock = kQgemvScalarBlock / 32;  // 8

__global__ void qgemv_warp_vec16_kernel(const int8_t* __restrict__ Wq,
                                        const float* __restrict__ scale,
                                        const __half* __restrict__ x,
                                        __half* __restrict__ out,
                                        int64_t N, int64_t K) {
    constexpr int64_t epv = 16;
    const int64_t row = (static_cast<int64_t>(blockIdx.x)
                         * kQgemvWarpWarpsPerBlock)
                        + (threadIdx.x >> 5);
    if (row >= N) return;  // N 非 8 倍数时的尾部 warp
    const int lane = threadIdx.x & 31;
    const U16Q* __restrict__ wrow =
        reinterpret_cast<const U16Q*>(Wq + row * K);
    const U16Q* __restrict__ xv = reinterpret_cast<const U16Q*>(x);
    const int64_t nvec = K / epv;
    const float s = scale[row];

    // ILP=2 双累加器: 两路独立在途（i 路 / i+32 路, 步长 64 向量）
    float acc0 = 0.f;
    float acc1 = 0.f;
    for (int64_t i = lane; i < nvec; i += 64) {
        U16Q w0 = wrow[i];
        U16Q x00 = xv[2 * i];
        U16Q x01 = xv[2 * i + 1];
        qgemv_vec_acc_dequant(w0, x00, x01, s, acc0);
        const int64_t j = i + 32;
        if (j < nvec) {  // 均匀分支（j 只依赖 i/nvec, 除尾部外 warp 一致）
            U16Q w1 = wrow[j];
            U16Q x10 = xv[2 * j];
            U16Q x11 = xv[2 * j + 1];
            qgemv_vec_acc_dequant(w1, x10, x11, s, acc1);
        }
    }
    // 纯 warp 归约（无 shared, 无 barrier）
    acc0 += acc1;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, off);
    }
    if (lane == 0) out[row] = __float2half_rn(acc0);
}

void qgemv_warp_vec16_fwd(const at::Tensor& Wq, const at::Tensor& scale,
                          const at::Tensor& x, at::Tensor& out) {
    if (qgemv_vec_contract_ok(Wq, x)) {
        const int64_t N = Wq.size(0);
        const int64_t grid = (N + kQgemvWarpWarpsPerBlock - 1)
                             / kQgemvWarpWarpsPerBlock;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        qgemv_warp_vec16_kernel<<<static_cast<int>(grid),
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
    Registrar() { register_qgemv_variant("qgemv_warp_vec16",
                                         qgemv_warp_vec16_fwd); }
} registrar;

}  // namespace
