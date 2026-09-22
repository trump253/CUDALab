// CUDALab INT4GEMV — INT4GEMV-0000 基线: 每输出行一个 block, 标量
// packed byte load, nibble unpack, group scale lookup, FP32 归约,
// FP16 输出。
//
// 这是 v0.7 的 **parent / incumbent 起点**（用户 §4 指定的基线形态,
// 不是从 cuBLAS / llama.cpp / Marlin 成熟 INT4 GEMV kernel 抄来的
// 优化实现）:
//
//   grid  = N 个 block（一个 block 负责一个输出行 y[n]）
//   block = 256 线程固定
//   每线程: 以 blockDim.x 步长 strided 访问该行 packed byte
//              1B packed load → unpack 两个 signed INT4（low = k=2b,
//              high = k=2b+1, two's complement 符号扩展）
//              → 按 k/128 读 group scale（g = b/64, 1 byte 恒属同一
//              group）→ fp16 scale → fp32
//              → 每元素 (q·s)·x（fp16 x → fp32）, FP32 累加
//   归约:  warp shuffle（5 步）+ shared 跨 warp（8 个 warp sum）
//              + 二次 warp shuffle（8 值, 3 步）→ 线程 0 执行
//              __float2half_rn 并写 out[n]
//
// 访存: 纯标量（packed 路径每次 1B uint8 load + 2B×2 fp16 x load +
// 每 byte 1 次 2B scale load, L1 缓存 —— group 内 64 byte 共享同一
// scale）, 行内相邻线程访问相邻元素（coalesced）; x 被每行重新读取
// （K 个元素, 8KB @ K=4096 fp16, L2 缓存）—— 不预取、不共享内存
// staging、不向量化（向量化需要 16B 对齐契约, 是 INT4GEMV-0001 起的
// 实验杠杆, 基线刻意不引入）。
//
// 预期瓶颈（基线假设, NCU 实证在后续 Phase）: 每 32B W_packed 流量
// 对应 32 条 LDG.1（1B 标量 uint8 load, 每 warp 32B）+ 64 条 x 的
// LDG.2（2B/元素, 每 32B W 对应 64 个 x 元素）+ 每 byte 的 unpack
// 指令（2 次 AND + 2 次符号扩展 + 2 次 half2float + 4 次 CVT/MUL/FFMA
// 量级）—— INT4 每 32B 权重流量做 32 个元素的 unpack+dequant, 指令/
// 字节比是 v0.6 QGEMV 标量 int8 基线的 ~2 倍, 很可能是 **instruction
// bound**（DRAM 利用率低于 v0.6 INT8 基线的水平）。
//
// 无对齐契约: 标量 1B/2B load 在连续契约下天然满足, 任何连续合法
// 输入（含 1 字节 storage offset 视图）都必须成功（negative 套件
// valid_offset_view_*_control 钉死）。
//
// 数值约定: 累加顺序 = 每线程步长 byte 部分和（b = tid, tid+B, ...;
// 每 byte 内 lo 先 hi 后）→ warp 树 → shared 树。每 term 最多 3 次
// FP32 舍入（q·s 乘、t1·x 乘、加）—— 固定 arith 界
// （int4gemv_correctness.py, 系数 3K·2^-24, TOL_K=2）对任何合法 FP32
// 累加顺序成立; 候选变体（向量 / warp-per-row / scale 提升）顺序
// 不同, 变体间不要求逐位一致。
//
// 实现注记: 标量计算本体在 int4gemv_common.h 的 int4gemv_scalar_kernel
// （**单一来源**）—— 本文件与所有向量化变体的标量回退都调用它,
// 保证回退输出与 baseline 在同一输入上逐位一致（negative 套件
// per-variant 回退回归用例的参照）。

#include "int4gemv_common.h"

namespace {

void int4gemv_baseline_fwd(const at::Tensor& Wp, const at::Tensor& scale,
                           const at::Tensor& x, at::Tensor& out) {
    launch_int4gemv_scalar(Wp, scale, x, out);
}

static struct Registrar {
    Registrar() { register_int4gemv_variant("int4gemv_baseline",
                                            int4gemv_baseline_fwd); }
} registrar;

}  // namespace
