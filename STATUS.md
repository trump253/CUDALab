# CUDALab — 状态

**日期：** 2026-07-16
**阶段：** 8 已完成 — v0.1 收尾
**状态：** 全部停止条件满足（见下方清单）。

## 当前状态

| 项目 | 取值 |
|---|---|
| 当前最佳内核 | `v4_vec_reg`（向量化 + 寄存器驻留，每行一个 block） |
| 主目标（M=128, H=4096, fp16） | baseline 7.33 µs → **5.27 µs（1.39×）**，复验 76/76 PASS |
| 实验 | 7 个（EXP-0001…EXP-0007）：3 个有效 KEEP、1 个 NEUTRAL、1 个 REJECT、2 个已作废 |
| 正确性 | 5/5 变体通过固定容差的 76 例套件（fp16 + fp32，含边界用例） |
| 基准框架 | `cuda-event-batched-v1`（32 连发批量样本，5×100，中位数主指标） |
| 剖析器 | ncu 2022.3.0 **工作正常**（权限 OK）；结构化 JSON + 原始输出 |
| git | 本地仓库，`main` 分支，repo-local 身份，无 remote；工作树干净 |

## 停止条件清单

- [x] 干净的 git 仓库（本地 `main`，repo-local 身份，无 remote）
- [x] 参考实现可用
- [x] 正确性框架通过（固定容差，76 例 × 5 变体，从不放宽）
- [x] 基线内核 + 扩展构建可用（sm_75，CUDA 11.8，torch 2.4.1+cu118）
- [x] 带结构化 JSON/CSV 结果的基准框架（完整矩阵、GPU 状态、预热、批量计时）
- [x] 真实 GPU 剖析尝试：**ncu 成功** —— 真实剖析 JSON + 真实原始输出已保存
- [x] 可用的实验追踪（EXP-*.json + 判定规则 + best.json）
- [x] ≥3 个真实数据驱动的优化实验，带真实 KEEP/REJECT/NEUTRAL（v1 KEEP、v2 NEUTRAL、v3 REJECT、v4 KEEP）
- [x] 确定最佳内核（v4_vec_reg）并完成复验（完整正确性 76/76 + 完整基准矩阵）
- [x] 只含真实数字的 README.md
- [x] STATUS.md（本文件）
- [x] 带阶段进度的 PROJECT_PLAN.md
- [x] 最终本地 git 提交，工作树干净

## 已知局限（README 中同样记载）

- 容器内无法锁定 GPU 时钟 → ~5 µs 尺度 ±10% 逐次运行方差。
- warm-L2 批量框架 vs 冷 L2 ncu 对接近的变体可能给出不同排序。
- `effective_bw_gbps` 是逻辑流量 ÷ 时间（小工作集下存在 L2 效应）。
- v2/v4 仅限 H ≤ 8192（H/256 ∈ {2,4,8,16,32}）。

## 值得注意的事故（已记录，未隐藏）

- EXP-0002 误判 REJECT：单发事件计时（约 6 µs 启动噪声）与 ncu 结论
  相反（v1 实际快 2.35×）。修复框架 → `cuda-event-batched-v1`；
  EXP-0001/EXP-0002 标记 `superseded`；所有变体重测。
