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
//
// 实现注记（GEMV-0001.. 落地时重构）: 标量计算本体移入
// gemv_common.h 的 gemv_scalar_kernel（**单一来源**）—— 本文件与
// 所有向量化变体的标量回退都调用它, 保证回退输出与 baseline 在同一
// 输入上逐位一致（negative 套件 per-variant 回退回归用例的参照）。
// 计算代码本体未改动, 只换了位置。

#include "gemv_common.h"

namespace {

void gemv_baseline_fwd(const at::Tensor& W, const at::Tensor& x,
                       at::Tensor& out) {
    launch_gemv_scalar_dispatch(W, x, out);
}

static struct Registrar {
    Registrar() { register_gemv_variant("gemv_baseline", gemv_baseline_fwd); }
} registrar;

}  // namespace
