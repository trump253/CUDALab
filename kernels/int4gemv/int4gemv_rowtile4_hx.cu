// CUDALab INT4GEMV — INT4GEMV-0004 候选: rowtile4 + half 驻留 x 片段
// （x 片段从 float[32] 降到 __half2[16], 逐行转换）。
//
// 假设（来自 rowtile4 / rowtile8 的 NCU 对照 @4096², clkbase）:
//   rowtile4: DRAM 62.34%, issue 45.80%, long_scoreboard 44.3% (4.31
//   cpi), 80 寄存器, 占用率 62.83%。rowtile8 (MLP 4->8): DRAM 62.55%
//   几乎不动, 但 long_scoreboard cpi 4.31->3.34 (MLP 有效) 且占用率
//   62.83%->43.09% 塌方 (117 寄存器) —— 「更多 MLP per thread」与
//   「更多 resident thread」对 DRAM 的边际贡献互相抵消, R 维度
//   (4->8) 已到局部最优。
//   剩余未测试点: **保持 R=4 结构 (MLP 4, 384 内存指令/行) 但把
//   x 片段驻留寄存器减半**, 让占用率从 63% 回升到 ~78%, 在飞 W
//   字节 +24% (resident thread +24%), 测试「占用率」是否是比
//   「MLP」更强的 DRAM 杠杆。
//
// 结构（R=4, 与 rowtile4 唯一差异在 x 驻留表示）:
//   grid = ceil(N/4), block = 128 线程, strided v 循环（同 rowtile4）。
//   每 thread 每 v:
//       4× LDG.128 x 片段加载一次, **保留原始 __half2[16]**
//       （16 寄存器, rowtile4 是 float xw[32] = 32 寄存器）;
//       内层 unroll R=4: row = 4*blockIdx.x + r（guard row < N）:
//           逐行 16× __half22float2 转换（rowtile4 每 v 只做 16 次,
//           本变体每 v 做 64 次, +48 次 float2 转换 / thread / v）
//           + 1× LDG.128 W + 1× scale（单 group 引理 g = v>>2）
//           + 32 nibble unpack + 32×(MUL+FFMA) 进独立累加器 acc[r]。
//   归约: 与 rowtile4 逐位相同（shfl 5 步 + shared[4][4] + shfl 2 步）。
//
// 数值: 每 thread 每 acc[r] 的 term 序列与 rowtile4 逐位同序（v 升序,
//   向量内 k 升序, lo 先 hi 后; __half22float2 是精确的半精度->单精度
//   提升, 与 rowtile4 的转换值相同）, 归约顺序相同 —— 预期与
//   rowtile4 bit-identical（层 A 硬门独立核验, 不做交叉等价假设）。
//
// 寄存器/占用率权衡（本实验的赌注）: x 驻留 32 -> 16 寄存器;
//   预期 80 -> ~64 → 占用率 62.83% -> ~78%（占用率与寄存器数实测
//   严格成反比, rowtile4/rowtile8 两点验证: 80/117 = 0.684 vs
//   43.09/62.83 = 0.686）。代价: 每 v 多 48 次 half2->float2 转换
//   （ALU/convert 管线上移; rowtile4 实测 ALU pipe 36.58%, 有余量）。
//   若 ptxas 选择把转换 hoist 出 r 循环（float[32] 常驻）, 寄存器
//   回到 ~80, 则占用率杠杆不可用 —— 该结果本身即结论（编译期已
//   最优, 结构上无 64 寄存器驻留路径）。
//
// 对齐契约（用户 §6, 同 rowtile4）: `int4gemv_vec_contract_ok`
//   host 侧检查; 不满足 → 回退 `launch_int4gemv_scalar`（与
//   int4gemv_baseline 同一代码源, bit-identical, negative 套件
//   per-variant 回退用例钉死）, 合法输入不得被拒绝。
//
// 边界: N 不整除 4 时末 block 部分 row 被 guard 跳过; K=128 时
//   nvec=4 < 128 线程, 只有 4 个线程持有片段, 其余线程 acc≡0 参与
//   归约 —— 与 rowtile4 相同, 无额外分支。

#include "int4gemv_common.h"

namespace {

static constexpr int kRowTile4HxBlock = 128;
static constexpr int kRowTile4HxWarps = kRowTile4HxBlock / 32;  // 4
static constexpr int kRowTile4HxRows = 4;

static __global__ void int4gemv_rowtile4_hx_kernel(
        const uint4* __restrict__ Wp,       // (N, K/32) uint4 视图
        const __half* __restrict__ scale,   // (N, K/128)
        const uint4* __restrict__ x,        // (K/8,) uint4 视图
        __half* __restrict__ out,
        int64_t N, int64_t K) {
    const int64_t nvec = K / 32;            // uint4/row
    const int64_t ngroup = K / 128;
    float acc[kRowTile4HxRows];
#pragma unroll
    for (int i = 0; i < kRowTile4HxRows; ++i) acc[i] = 0.f;

    for (int64_t v = threadIdx.x; v < nvec; v += blockDim.x) {
        // x 片段加载一次, 保留 __half2 原始表示（16 寄存器）,
        // 跨 R=4 行复用 —— 本变体与 rowtile4 的唯一结构差异。
        // xh[j] = x[2j..2j+1]: xa/xb -> j=0..7, xc/xd -> j=8..15
        const uint4* xv = x + 4 * v;
        U32I4 xa; xa.v = xv[0];
        U32I4 xb; xb.v = xv[1];
        U32I4 xc; xc.v = xv[2];
        U32I4 xd; xd.v = xv[3];                       // 4× LDG.128
        const __half2* ha = reinterpret_cast<const __half2*>(&xa);
        const __half2* hb = reinterpret_cast<const __half2*>(&xb);
        const __half2* hc = reinterpret_cast<const __half2*>(&xc);
        const __half2* hd = reinterpret_cast<const __half2*>(&xd);
        __half2 xh[16];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            xh[p] = ha[p];
            xh[4 + p] = hb[p];
            xh[8 + p] = hc[p];
            xh[12 + p] = hd[p];
        }
#pragma unroll
        for (int r = 0; r < kRowTile4HxRows; ++r) {
            const int64_t row =
                    static_cast<int64_t>(kRowTile4HxRows * blockIdx.x + r);
            if (row >= N) continue;                   // 末 block 守卫
            const uint4* __restrict__ wrow = Wp + row * nvec;
            const __half* __restrict__ srow = scale + row * ngroup;
            U32I4 w;
            w.v = wrow[v];                            // 1× LDG.128
            // 单 group 引理（同 rowtile4/vec16_row）: g = v >> 2
            const float s = __half2float(srow[v >> 2]);   // 1× 2B load
            int q[32];
            int4gemv_vec_acc_unpack(w, q);            // 32 nibble
            // 逐行转换: xh[p] -> k = 2p, 2p+1; xh[8+p] -> k = 16+2p, ..+1
#pragma unroll
            for (int p = 0; p < 8; ++p) {
                const float2 fa = __half22float2(xh[p]);
                const float2 fb = __half22float2(xh[8 + p]);
                acc[r] += (static_cast<float>(q[2 * p]) * s) * fa.x;
                acc[r] += (static_cast<float>(q[2 * p + 1]) * s) * fa.y;
                acc[r] += (static_cast<float>(q[2 * p + 16]) * s) * fb.x;
                acc[r] += (static_cast<float>(q[2 * p + 17]) * s) * fb.y;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < kRowTile4HxRows; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
        }
    }

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    __shared__ float warp_sums[kRowTile4HxRows][kRowTile4HxWarps];
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < kRowTile4HxRows; ++r) warp_sums[r][wid] = acc[r];
    }
    __syncthreads();

    if (wid == 0) {
#pragma unroll
        for (int r = 0; r < kRowTile4HxRows; ++r) {
            const int64_t row =
                    static_cast<int64_t>(kRowTile4HxRows * blockIdx.x + r);
            if (row >= N) continue;
            float t = (lane < kRowTile4HxWarps) ? warp_sums[r][lane] : 0.f;
#pragma unroll
            for (int off = kRowTile4HxWarps / 2; off > 0; off >>= 1) {
                t += __shfl_down_sync(0xffffffffu, t, off);
            }
            if (lane == 0) out[row] = __float2half_rn(t);
        }
    }
}

void int4gemv_rowtile4_hx_fwd(const at::Tensor& Wp, const at::Tensor& scale,
                              const at::Tensor& x, at::Tensor& out) {
    // 对齐契约不满足 → scalar 回退（bit-identical, 不得拒绝）
    if (!int4gemv_vec_contract_ok(Wp, x)) {
        launch_int4gemv_scalar(Wp, scale, x, out);
        return;
    }
    const int64_t N = Wp.size(0);
    const int64_t K = 2 * Wp.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int grid =
            static_cast<int>((N + kRowTile4HxRows - 1) / kRowTile4HxRows);
    int4gemv_rowtile4_hx_kernel<<<grid, kRowTile4HxBlock, 0, stream>>>(
        reinterpret_cast<const uint4*>(Wp.const_data_ptr()),
        reinterpret_cast<const __half*>(scale.const_data_ptr()),
        reinterpret_cast<const uint4*>(x.const_data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static struct Registrar {
    Registrar() { register_int4gemv_variant("int4gemv_rowtile4_hx",
                                            int4gemv_rowtile4_hx_fwd); }
} registrar;

}  // namespace
