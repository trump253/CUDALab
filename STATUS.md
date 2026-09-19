# CUDALab — 状态

**日期：** 2026-09-19
**阶段：** v0.2 已完成 — Evaluator Hardening & Revalidation
**状态：** 全部停止条件满足（见 v0.2 清单）。

## v0.2 完成摘要（2026-09-19）

分支 `v0.2-evaluator-hardening`，基线 `main` @ 94179b6（v0.1 完成状态），
10 个阶段全部完成，10 个增量提交。v0.1 数据（EXP-0001…0007、`benchmarks/` 顶层、
`profiles/rmsnorm/`、`correctness/` 顶层）原样保留；`best_v0.1.json` 存档了
v0.1 的 best，`best.json` 为 v0.2 当前最佳。

| 项目 | v0.2 结果 |
|---|---|
| 正确性 | 5/5 变体 76/76 PASS（`correctness/v0.2/`）+ 负例套件 27/28 符合预期（1 多 GPU 用例单卡环境跳过） |
| 基准 harness | `paired-streaming-v2`：配对 A/B 交替、DVFS guard（>5% 拒轮）、hot/streaming 双模式、预分配 16-buffer 池（streaming 工作集 33.5 MB > 5.5 MB L2） |
| 全矩阵 | 7 形状 × {fp16,fp32} × {hot,streaming} × 5 变体 = 28 run，**369/369 rounds valid**（0 DVFS invalid，全程 1350 MHz） |
| 主形状 fp16 (128,4096) | v4_vec_reg 1.56×（hot）/ 1.87×（streaming）vs baseline；v4 vs v2 统计平局（NEUTRAL）→ v4 保留 incumbent；v4 vs v1：hot REJECT v1（median v4/v1 0.9327，0/9 轮 v1 更快）、streaming NEUTRAL（v1 反微快 1.1%） |
| fp32 | **v2_reg 为最佳**（vs v4：1.37× streaming / 1.62× hot，KEEP） |
| (128,8192) 两 dtype | **v2_reg 为最佳**（fp16 paired 1.38× KEEP；v4 在 H=8192 退化） |
| EXP-0007 1.231× | **REVISED**：同频复验 v4 vs v1 = 1.011×（streaming）/ v4 快约 7%（hot，median 0.9327）；原值系 1350 vs ~1905 MHz 混频膨胀 |
| NCU 方法学 | v0.1 "cold L2" 说法**推翻**（`--cache-control` 默认 all = 不失效缓存）；v0.2 双缓存模式剖析（`profiles/rmsnorm/v0.2/`）；小 kernel 下 cc=all/none 差异可忽略 → 缓存杠杆在 bench 层 |
| 统计 | round-level paired speedup + bootstrap CI95（固定种子 20260919）+ KEEP/REJECT/NEUTRAL/UNSTABLE；18 个 CPU 单元测试全过 |
| 分派 | `cudalab/dispatch.py`：28 单元格实测分发表 + 保守 fallback（5 个 CPU 测试 + e2e 验证） |
| API 加固 | Finding A–D 修复（非法 H 显式报错、对齐契约、forward_into 验证、launch 检查）；28 例负例套件 |
| PyTorch 参照 | F.rms_norm 66.2 µs（主形状 fp16，非融合路径，仅记录不决策） |
| 审计 | `docs/benchmark_audit_v0.2.md`（独立 subagent 审计，Lead 复核） |
| 未做 | 无新内核/变体（v0.2 约束）；未 push（等待指示） |

## v0.2 阶段清单

- [x] Phase 1 — API 正确性加固（bindings 统一验证 + 5 变体 TORCH_CHECK + launch 检查）
- [x] Phase 2 — 非法输入负例套件（28 例）
- [x] Phase 3 — 基准重构：paired A/B + 预分配缓冲池 + 顺序去偏
- [x] Phase 4 — DVFS guard（pair/matrix 双校验、重试、UNSTABLE）
- [x] Phase 5 — hot/streaming 双缓存模式（streaming 工作集 > L2）
- [x] Phase 6 — round-level 统计 + bootstrap CI + 四态决策 + CPU 单元测试
- [x] Phase 7 — NCU 方法学审计（cache-control 语义）+ 双模式剖析 + L1/L2 命中率
- [x] Phase 8 — 完整复验（正确性 ×5、负例、全矩阵、13 组配对、shape winners、EXP-0008、best 归档/更新）
- [x] Phase 9 — 形状/dtype 分发表（基于实测显著证据）
- [x] Phase 10 — 文档（README 重组、STATUS、PLAN、独立审计）

## 当前状态（v0.1，历史保留）

| 项目 | 取值 |
|---|---|
| 当前最佳内核（v0.1） | `v4_vec_reg`（向量化 + 寄存器驻留，每行一个 block） |
| 主目标（M=128, H=4096, fp16） | baseline 7.33 µs → 5.27 µs（1.39×）——**v0.2 判定该对比有效，但 EXP-0007 的 1.231× vs v1 为 DVFS 膨胀（REVISED）** |
| 实验 | 7 个（EXP-0001…EXP-0007）+ EXP-0008（v0.2 复验） |
| 正确性 | 5/5 变体通过 76 例套件（v0.1 与 v0.2 复验均通过） |
| 基准框架 | `cuda-event-batched-v1`（v0.1，保留作历史对照）；v0.2 = `paired-streaming-v2` |
| git | `main`（v0.1，含 remote）+ `v0.2-evaluator-hardening`（v0.2，未 push） |

## 已知局限（v0.2 更新，README 同步）

- DVFS guard 只能检测并拒绝失配轮，不能预防；nvidia-smi 轮询是区间外代理采样。
- NCU `--clock-control base` 在容器内无锁频正面证据。
- 矩阵模式 round-level ratio 对离群干扰轮敏感（winner 仅指示性，判定以 paired 为准）。
- streaming 33.5 MB 工作集未达 DRAM 带宽饱和；M=1 区域 launch-bound。
- compute-sanitizer 不可用（未做越界/竞态检查）。
- 分发表覆盖 14 个实测 (M,H) 组合，其余保守回退。
- v0.1/v0.2 数字跨版本不可直接比较（harness/时钟/缓存策略均不同）。

## 值得注意的事故（已记录，未隐藏）

- **EXP-0002**（v0.1）：单发事件计时（约 6 µs 启动噪声）把 v1 误判 REJECT，
  与 ncu 结论相反。修复 → `cuda-event-batched-v1`；EXP-0001/0002 标记 superseded。
- **EXP-0007**（v0.1 → v0.2 复核）：batched 框架下变体间非配对测量遭遇 DVFS
  混频（v1@~1350 MHz vs v4@~1905 MHz），1.231× 加速比虚高。v0.2 paired
  harness + DVFS guard 复验后 REVISED 为 1.011×（streaming）/ v4 快约 7%（hot，median 0.9327）。
  两条教训共同支撑客观层设计：独立于智能体、完整保留作废记录、结论可复现可审计。
