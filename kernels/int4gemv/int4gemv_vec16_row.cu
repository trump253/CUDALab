// CUDALab INT4GEMV — INT4GEMV-0001 候选: 16B 向量化 packed load
// (uint4 = 16B = 32 个 INT4) + 每 16B W 向量单次 group scale lookup +
// x 侧 4×16B 向量 load。
//
// 假设（来自 int4gemv_baseline 的 NCU @4096², clkbase）:
//   long_scoreboard = 64.2% 停顿（9.8 cycles/issue）, SM 吞吐 46.2%,
//   DRAM 仅 25.4% —— 瓶颈是标量 1B packed load（LDG.1）+ 2B x load
//   （LDG.2×2）+ 每 byte 一次 2B scale load（LDG.2, 大多 L1 命中）
//   的内存指令延迟, 不是 DRAM 带宽。把 W 侧 16 条 LDG.1 换成 1 条
//   LDG.128（16B = 32 个 INT4）, x 侧 32 条 LDG.2 换成 4 条 LDG.128,
//   scale 侧 16 次 lookup 换成 1 次（见下）, 应大幅压缩 long_scoreboard
//   并把 SM 推向更高指令吞吐。
//
// 结构:
//   grid  = N 个 block（一行一个 block, 同 baseline）
//   block = 128 线程固定
//   每 thread（对 K=4096 主目标恰好 1 个 uint4/row: K/32 = 128）:
//       strided uint4 循环 v = tid, tid+128, ...
//         1× LDG.128 W_packed 向量（16B = 32 个 INT4）
//         4× LDG.128 x 向量（32 个 fp16 = 4×8）
//         1× group scale（见"单 group 引理"）
//         32 个 nibble unpack → 32 个 (q·s)·x（MUL + FFMA）
//   归约: 4 warp（128 线程）warp shuffle（5 步）+ shared + 二次
//         warp shuffle（2 步）→ 线程 0 __float2half_rn 写 out[row]
//
// **单 group 引理**（scale lookup 32→1 的根据, 纯对齐算术）:
//   16B W 向量 v 覆盖 byte [16v, 16v+15], 即元素 k ∈ [32v, 32v+31]
//   （32 个连续 k）。group 边界每 128 个元素 = 每 64 个 byte 一条;
//   16B 对齐（契约）⇒ 16v % 64 ∈ {0,16,32,48}; 最大偏移 48,
//   48+15 = 63 < 64 ⇒ **每个 16B 向量完整落在单一 group 内**,
//   g = (16v) >> 6 = v >> 2。因此每向量 1 次 scale load 即覆盖全部
//   32 个元素（baseline 是每 byte 1 次）。
//
// 对齐契约（用户 §6, int4gemv_common.h 总则）: 16B 向量 load 要求
//   W_packed 基指针 16B 对齐 ∧ x 基指针 16B 对齐 ∧ K%32==0
// （K%128==0 已保证, 检查保留作自文档化）——
// `int4gemv_vec_contract_ok` host 侧检查; 不满足时**回退**
// `launch_int4gemv_scalar`（int4gemv_baseline 的同一代码源, 同输入
// 输出 bit-identical, 由 negative 套件 per-variant 用例
// fallback_Wp_misaligned / fallback_x_misaligned 钉死）, 合法输入
// （含 1 字节 storage offset 的连续视图）不得被拒绝。
//
// 数值约定: 每 term (q·s) 乘 1 次 + (·x) 乘 1 次 + 加 1 次（FFMA
// 收缩后更少）—— 每 term ≤ 3 次 FP32 舍入, 固定 arith 界
// （int4gemv_correctness.py, 3K·2^-24, TOL_K=2）覆盖本累加顺序
// （每 thread 步长向量部分和, 向量内元素顺序 k 升序, lo 先 hi 后
// 与 baseline 同序）。

#include "int4gemv_common.h"

namespace {

static constexpr int kVec16RowBlock = 128;
static constexpr int kVec16RowWarps = kVec16RowBlock / 32;  // 4

static __global__ void int4gemv_vec16_row_kernel(
        const uint4* __restrict__ Wp,      // (N, K/32) uint4 视图
        const __half* __restrict__ scale,  // (N, K/128)
        const uint4* __restrict__ x,       // (K/8,) uint4 视图
        __half* __restrict__ out,
        int64_t N, int64_t K) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int64_t nvec = K / 32;                       // uint4/row
    const uint4* __restrict__ wrow = Wp + row * nvec;
    const __half* __restrict__ srow = scale + row * (K / 128);
    float acc = 0.f;
    for (int64_t v = threadIdx.x; v < nvec; v += blockDim.x) {
        U32I4 w;
        w.v = wrow[v];                                  // 1× LDG.128
        // 32 个连续 x 值 = 4 个连续 uint4（k = 32v..32v+31）
        const uint4* xv = x + 4 * v;
        U32I4 xa; xa.v = xv[0];
        U32I4 xb; xb.v = xv[1];
        U32I4 xc; xc.v = xv[2];
        U32I4 xd; xd.v = xv[3];                         // 4× LDG.128
        // 单 group 引理: g = v >> 2（16B 向量恒在单一 group 内）
        const float s = __half2float(srow[v >> 2]);     // 1× 2B load
        int q[32];
        int4gemv_vec_acc_unpack(w, q);                  // 32 nibble
        // 每元素: (q·s)·x —— 1 MUL + 1 FFMA（nvcc 收缩 q*s*x 的
        // 第二乘加）; x 按 __half2 对转 float2（8 次 h2f2/向量）
        const __half2* ha = reinterpret_cast<const __half2*>(&xa);
        const __half2* hb = reinterpret_cast<const __half2*>(&xb);
        const __half2* hc = reinterpret_cast<const __half2*>(&xc);
        const __half2* hd = reinterpret_cast<const __half2*>(&xd);
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const float2 fa = (p < 4) ? __half22float2(ha[p])
                                      : __half22float2(hb[p - 4]);
            const float2 fb = (p < 4) ? __half22float2(hc[p])
                                      : __half22float2(hd[p - 4]);
            acc += (static_cast<float>(q[2 * p]) * s) * fa.x;
            acc += (static_cast<float>(q[2 * p + 1]) * s) * fa.y;
            // fb 覆盖 x = 16+2p, 16+2p+1（p<4: xc[p]; p>=4: xd[p-4]）
            // → q 索引 16+2p, 16+2p+1
            acc += (static_cast<float>(q[2 * p + 16]) * s) * fb.x;
            acc += (static_cast<float>(q[2 * p + 17]) * s) * fb.y;
        }
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    __shared__ float warp_sums[kVec16RowWarps];
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        acc = (lane < kVec16RowWarps) ? warp_sums[lane] : 0.f;
#pragma unroll
        for (int off = kVec16RowWarps / 2; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) out[row] = __float2half_rn(acc);
    }
}

void int4gemv_vec16_row_fwd(const at::Tensor& Wp, const at::Tensor& scale,
                            const at::Tensor& x, at::Tensor& out) {
    // 对齐契约不满足 → scalar 回退（bit-identical, 不得拒绝）
    if (!int4gemv_vec_contract_ok(Wp, x)) {
        launch_int4gemv_scalar(Wp, scale, x, out);
        return;
    }
    const int64_t N = Wp.size(0);
    const int64_t K = 2 * Wp.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int4gemv_vec16_row_kernel<<<static_cast<int>(N), kVec16RowBlock, 0,
                                stream>>>(
        reinterpret_cast<const uint4*>(Wp.const_data_ptr()),
        reinterpret_cast<const __half*>(scale.const_data_ptr()),
        reinterpret_cast<const uint4*>(x.const_data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static struct Registrar {
    Registrar() { register_int4gemv_variant("int4gemv_vec16_row",
                                            int4gemv_vec16_row_fwd); }
} registrar;

}  // namespace
