// CUDALab QGEMV — QGEMV-0000 基线: 每输出行一个 block, 标量 int8 load,
// FP32 归约, FP16 输出。
//
// 这是 v0.6 的 **parent / incumbent 起点**（用户 §4 指定的基线形态,
// 不是从 cuBLAS / llama.cpp 成熟 INT8 GEMV kernel 抄来的优化实现）:
//
//   grid  = N 个 block（一个 block 负责一个输出行 y[n]）
//   block = 256 线程固定
//   每线程: 以 blockDim.x 步长 strided 访问该行
//              int8 标量 load → FP32 → ×scale[n] → ×x[k]（fp16→fp32）
//              FP32 累加（FMA 由 nvcc 收缩）
//   归约:  warp shuffle（5 步）+ shared 跨 warp（8 个 warp sum）
//              + 二次 warp shuffle（8 值, 3 步）→ 线程 0 执行
//              __float2half_rn 并写 out[n]
//
// 访存: 纯标量（int8 路径每次 1B W_q load + 2B x load）, 行内相邻
// 线程访问相邻元素（coalesced）; x 被每行重新读取（K 个元素, L2
// 缓存, 8KB @ K=4096 fp16）—— 不预取、不共享内存 staging、不向量化
// （向量化需要 16B 对齐契约, 是 QGEMV-0001 起的实验杠杆, 基线刻意
// 不引入）。scale 每行 1 次 4B load（寄存器内复用 K 次）。
//
// 预期瓶颈（基线假设, NCU 实证在 Phase 4）: 与 v0.5 GEMV 基线同型
// —— 每 32B W_q 流量对应 ~6 条 warp load 指令（1B 标量 int8 load
// 32 个/32B, 每 warp 1 条 4B… 实际每 lane 1B 不可合并为宽 load,
// 指令/字节比是 FP16 2B load 的 2 倍）, 很可能是 **instruction
// bound**（DRAM 利用率明显低于 v0.5 FP16 基线的 49%）。
//
// 无对齐契约: 标量 1B/2B load 在连续契约下天然满足, 任何连续合法
// 输入（含 1 字节 storage offset 视图）都必须成功（negative 套件
// valid_offset_view_Wq_control 钉死）。
//
// 数值约定: 累加顺序 = 每线程步长部分和（k = tid, tid+B, ...）→
// warp 树 → shared 树。每 term 最多 3 次 FP32 舍入（q·s 乘、t1·x
// 乘、加）—— 固定 arith 界（qgemv_correctness.py, 系数 3K·2^-24,
// TOL_K=2）对任何合法 FP32 累加顺序成立; 候选变体（向量 / warp-
// per-row / scale 提升）顺序不同, 变体间不要求逐位一致。
//
// 实现注记: 标量计算本体在 qgemv_common.h 的 qgemv_scalar_kernel
// （**单一来源**）—— 本文件与所有向量化变体的标量回退都调用它,
// 保证回退输出与 baseline 在同一输入上逐位一致（negative 套件
// per-variant 回退回归用例的参照）。

#include "qgemv_common.h"

namespace {

void qgemv_baseline_fwd(const at::Tensor& Wq, const at::Tensor& scale,
                        const at::Tensor& x, at::Tensor& out) {
    launch_qgemv_scalar(Wq, scale, x, out);
}

static struct Registrar {
    Registrar() { register_qgemv_variant("qgemv_baseline",
                                         qgemv_baseline_fwd); }
} registrar;

}  // namespace
