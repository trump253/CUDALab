// CUDALab INT4GEMV — INT4GEMV-0002 候选: x-stationary R=4 row tile
// （每 block 4 行, x 片段在寄存器里跨 4 行复用）。
//
// 假设（来自 int4gemv_vec16_row 的 NCU @4096², clkbase）:
//   lg_throttle = 43.9% 停顿（6.555 cpi, LSU 发射队列满）取代
//   long_scoreboard（18.4%）成为第一瓶颈; DRAM 利用率 48.98%, SM 42.3%。
//   vec16_row 的内存指令/行（K=4096, 128 线程, 1 uint4/线程/行）:
//       128×(1 W LDG.128 + 4 x LDG.128 + 1 scale LDG.16) = 768 条/行,
//   其中 x 侧 512 条 LDG.128/行 —— x (K,) 对所有行共享, 每行被完整
//   重发一次（8KB @K=4096, L2 常驻, DRAM 影响小, 但 LSU 发射压力全
//   额计入）。
//
// 结构（R=4）:
//   grid  = ceil(N/4) 个 block, block = 128 线程
//   每 thread（对 K=4096 恰好 1 个 uint4 片段: K/32 = 128）:
//       strided uint4 循环 v = tid, tid+128, ...（同 vec16_row）:
//         4× LDG.128 x 片段（k = 32v..32v+31）加载**一次**进寄存器,
//         转 float2[32] 一次;
//         内层 unroll R=4: row = 4*blockIdx.x + i（guard row < N）:
//             1× LDG.128 W 向量 + 1× scale（单 group 引理 g = v>>2,
//             与 vec16_row 相同）+ 32 nibble unpack + 32×(MUL+FFMA)
//             进该行的独立累加器 acc[i];
//   归约: 4 个 acc[i] 各做 4-warp 归约（shfl 5 步 + shared[4][4] +
//         shfl 2 步）, warp 0 lane 0 写 out[row]。
//
// 内存指令/行（K=4096）: (128×(4 W + 4 x + 4 scale))/4 行 = 384 条
//   （vec16_row 768 条, 2.0× 压缩; 主要是 x 512→128 条/行）。
// 数值: 每 thread 每 acc[i] 的 term 序列与 vec16_row 逐位同序
//   （v 升序, 向量内 k 升序, lo 先 hi 后）, 归约顺序相同 ——
//   预期与 vec16_row bit-identical（层 A 硬门仍独立核验, 不做
//   交叉等价假设）。
//
// 寄存器/占用率权衡: x 片段 float2[32]（32 寄存器, 跨 R=4 复用）+
//   acc[4] + 瞬态 q[32]（逐 r 复用）≈ 80-90 寄存器 → 占用率预期
//   从 vec16_row 的 ~85% 降到 ~45-50%, 换取 (1) 内存指令 2× 压缩
//   (2) 4 条独立 FMA 链（ILP 4, 掩盖 long_scoreboard 残余 18.4%）。
//   若 NCU 显示占用率损失吃掉收益, 这是 0003 的直接证据。
//
// 对齐契约（用户 §6, 同 vec16_row）: `int4gemv_vec_contract_ok`
//   host 侧检查; 不满足 → 回退 `launch_int4gemv_scalar`（与
//   int4gemv_baseline 同一代码源, bit-identical, negative 套件
//   per-variant 回退用例钉死）, 合法输入（含 1 字节 storage offset
//   的连续视图）不得被拒绝。
//
// 边界: N 不整除 4（如 N=1, N=3 的 edge 形状）时末 block 的部分
//   row 被 guard row<N 跳过（不读 W/scale, 不写 out）; K=128 时
//   nvec=4 < 128 线程, 只有 4 个线程持有片段, 其余线程 acc≡0 参与
//   归约 —— 两条路径都不需要额外分支。

#include "int4gemv_common.h"

namespace {

static constexpr int kRowTile4Block = 128;
static constexpr int kRowTile4Warps = kRowTile4Block / 32;  // 4
static constexpr int kRowTile4Rows = 4;

static __global__ void int4gemv_rowtile4_kernel(
        const uint4* __restrict__ Wp,       // (N, K/32) uint4 视图
        const __half* __restrict__ scale,   // (N, K/128)
        const uint4* __restrict__ x,        // (K/8,) uint4 视图
        __half* __restrict__ out,
        int64_t N, int64_t K) {
    const int64_t nvec = K / 32;            // uint4/row
    const int64_t ngroup = K / 128;
    float acc[kRowTile4Rows];
#pragma unroll
    for (int i = 0; i < kRowTile4Rows; ++i) acc[i] = 0.f;

    for (int64_t v = threadIdx.x; v < nvec; v += blockDim.x) {
        // x 片段加载一次（k = 32v..32v+31 = 4 个连续 16B 向量）,
        // 转 float2 一次, 跨 R=4 行复用 —— 本变体唯一的结构变化。
        const uint4* xv = x + 4 * v;
        U32I4 xa; xa.v = xv[0];
        U32I4 xb; xb.v = xv[1];
        U32I4 xc; xc.v = xv[2];
        U32I4 xd; xd.v = xv[3];                       // 4× LDG.128
        const __half2* ha = reinterpret_cast<const __half2*>(&xa);
        const __half2* hb = reinterpret_cast<const __half2*>(&xb);
        const __half2* hc = reinterpret_cast<const __half2*>(&xc);
        const __half2* hd = reinterpret_cast<const __half2*>(&xd);
        // xw[i] = x[32v + i]: ha/hb 覆盖 k = 0..15, hc/hd 覆盖 k = 16..31
        float xw[32];
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const float2 fa = (p < 4) ? __half22float2(ha[p])
                                      : __half22float2(hb[p - 4]);
            const float2 fb = (p < 4) ? __half22float2(hc[p])
                                      : __half22float2(hd[p - 4]);
            xw[2 * p] = fa.x;
            xw[2 * p + 1] = fa.y;
            xw[16 + 2 * p] = fb.x;
            xw[16 + 2 * p + 1] = fb.y;
        }
#pragma unroll
        for (int r = 0; r < kRowTile4Rows; ++r) {
            const int64_t row =
                    static_cast<int64_t>(kRowTile4Rows * blockIdx.x + r);
            if (row >= N) continue;                   // 末 block 守卫
            const uint4* __restrict__ wrow = Wp + row * nvec;
            const __half* __restrict__ srow = scale + row * ngroup;
            U32I4 w;
            w.v = wrow[v];                            // 1× LDG.128
            // 单 group 引理（同 vec16_row）: g = v >> 2
            const float s = __half2float(srow[v >> 2]);   // 1× 2B load
            int q[32];
            int4gemv_vec_acc_unpack(w, q);            // 32 nibble
#pragma unroll
            for (int p = 0; p < 8; ++p) {
                acc[r] += (static_cast<float>(q[2 * p]) * s) * xw[2 * p];
                acc[r] += (static_cast<float>(q[2 * p + 1]) * s) *
                          xw[2 * p + 1];
                acc[r] += (static_cast<float>(q[2 * p + 16]) * s) *
                          xw[16 + 2 * p];
                acc[r] += (static_cast<float>(q[2 * p + 17]) * s) *
                          xw[16 + 2 * p + 1];
            }
        }
    }

#pragma unroll
    for (int r = 0; r < kRowTile4Rows; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
        }
    }

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    __shared__ float warp_sums[kRowTile4Rows][kRowTile4Warps];
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < kRowTile4Rows; ++r) warp_sums[r][wid] = acc[r];
    }
    __syncthreads();

    if (wid == 0) {
#pragma unroll
        for (int r = 0; r < kRowTile4Rows; ++r) {
            const int64_t row =
                    static_cast<int64_t>(kRowTile4Rows * blockIdx.x + r);
            if (row >= N) continue;
            float t = (lane < kRowTile4Warps) ? warp_sums[r][lane] : 0.f;
#pragma unroll
            for (int off = kRowTile4Warps / 2; off > 0; off >>= 1) {
                t += __shfl_down_sync(0xffffffffu, t, off);
            }
            if (lane == 0) out[row] = __float2half_rn(t);
        }
    }
}

void int4gemv_rowtile4_fwd(const at::Tensor& Wp, const at::Tensor& scale,
                           const at::Tensor& x, at::Tensor& out) {
    // 对齐契约不满足 → scalar 回退（bit-identical, 不得拒绝）
    if (!int4gemv_vec_contract_ok(Wp, x)) {
        launch_int4gemv_scalar(Wp, scale, x, out);
        return;
    }
    const int64_t N = Wp.size(0);
    const int64_t K = 2 * Wp.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int grid = static_cast<int>((N + kRowTile4Rows - 1) / kRowTile4Rows);
    int4gemv_rowtile4_kernel<<<grid, kRowTile4Block, 0, stream>>>(
        reinterpret_cast<const uint4*>(Wp.const_data_ptr()),
        reinterpret_cast<const __half*>(scale.const_data_ptr()),
        reinterpret_cast<const uint4*>(x.const_data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static struct Registrar {
    Registrar() { register_int4gemv_variant("int4gemv_rowtile4",
                                            int4gemv_rowtile4_fwd); }
} registrar;

}  // namespace
