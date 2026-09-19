# CUDALab — 状态

**日期：** 2026-09-19
**阶段：** v0.2 已完成 → v0.2.1 Review Fix（4 项 review findings 全部修复）
**状态：** v0.2.1 修复与验证完成，已 push 至 `v0.2-evaluator-hardening`，等待 merge review。

## v0.2 完成摘要（2026-09-19）

分支 `v0.2-evaluator-hardening`，基线 `main` @ 94179b6（v0.1 完成状态），
10 个阶段全部完成，10 个增量提交。v0.1 数据（EXP-0001…0007、`benchmarks/` 顶层、
`profiles/rmsnorm/`、`correctness/` 顶层）原样保留；`best_v0.1.json` 存档了
v0.1 的 best，`best.json` 为 v0.2 当前最佳。

| 项目 | v0.2 结果 |
|---|---|
| 正确性 | 5/5 变体 76/76 PASS（`correctness/v0.2/`）+ 负例套件 29/30 符合预期（1 多 GPU 用例单卡环境跳过） |
| 基准 harness | `paired-streaming-v2`：配对 A/B 交替、DVFS guard（>5% 拒轮）、hot/streaming 双模式、预分配 16-buffer 池（streaming 工作集 33.5 MB > 5.5 MB L2） |
| 全矩阵 | 7 形状 × {fp16,fp32} × {hot,streaming} × 5 变体 = 28 run，**369/369 rounds valid**（0 DVFS invalid，全程 1350 MHz） |
| 主形状 fp16 (128,4096) | **无统计唯一胜出者（NO_UNIQUE_WINNER，v0.2.1 语义修正）**：streaming（primary）v1/v2/v4 两两 NEUTRAL；hot（secondary）v4 vs v1 REJECT v1（median v4/v1 0.9327，0/9 轮 v1 更快）、v4 vs v2 NEUTRAL；v4_vec_reg 保留为 v0.1 incumbent（非统计确认唯一最佳）；vs baseline 1.56×（hot）/ 1.87×（streaming） |
| fp32 | **v2_reg 为最佳**（vs v4：1.37× streaming / 1.62× hot，KEEP） |
| (128,8192) 两 dtype | **v2_reg 为最佳**（fp16 paired 1.38× KEEP；v4 在 H=8192 退化） |
| EXP-0007 1.231× | **REVISED**：同频复验 v4 vs v1 = 1.011×（streaming）/ v4 快约 7%（hot，median 0.9327）；原值系 1350 vs ~1905 MHz 混频膨胀 |
| NCU 方法学 | `--cache-control` 语义修正（v0.2.1，此前写反）：`all`（默认）= cache flush/reset（每 replay 前失效缓存，确定性 flushed 状态）、`none` = no-flush（不失效，状态不受控）；v0.1 走默认 all（= 失效），其 "cold L2" 说法与默认配置一致（v0.2 曾误判"推翻"，已更正）；v0.2 双缓存模式剖析（`profiles/rmsnorm/v0.2/`）；小 kernel 下 cc=all/none 差异可忽略 → 缓存杠杆在 bench 层 |
| 统计 | round-level paired speedup + bootstrap CI95（固定种子 20260919）+ KEEP/REJECT/NEUTRAL/UNSTABLE；18 个 CPU 单元测试全过 |
| 分派 | `cudalab/dispatch.py`（v0.2.1 证据政策）：仅 2 个 paired 确认格（(128,4096) fp32、(128,8192) fp16 → v2_reg）+ 1 个显式 incumbent 格（(128,4096) fp16 → v4）路由优化变体；矩阵-only/模式冲突/未实测一律 baseline，`dispatch_info` 四类 evidence_source（6 个 CPU 测试） |
| API 加固 | Finding A–D 修复（非法 H 显式报错、对齐契约、forward_into 验证、launch 检查）；30 例负例套件（v0.2 28 例 + v0.2.1 增补 2 例 v4 FP32 H=1024 对齐回归） |
| PyTorch 参照 | F.rms_norm 66.2 µs（主形状 fp16，非融合路径，仅记录不决策） |
| 审计 | `docs/benchmark_audit_v0.2.md`（独立 subagent 审计，Lead 复核） |
| 未做 | 无新内核/变体（v0.2 约束）；未 push（等待指示） |

## v0.2.1 Review Fix（2026-09-19）

v0.2 review 提出的 4 项 findings 全部修复（无新内核、无 v0.2 数据改动、
无 NCU 重跑）：

| # | Finding | 修复 | Commit |
|---|---|---|---|
| 1 | NCU `--cache-control` 语义写反 | 本机 ncu 2022.3 `--help` + raw 输出 + NVIDIA 文档核实：`all`（默认）= cache flush/reset、`none` = no-flush；profiler/profile_v2/README/STATUS/PLAN/审计/EXP-0008 标注修正；未重跑、未删数据；v0.1 "cold L2" 与默认配置一致（撤回"推翻"说法） | `e052c2e` |
| 2 | v4 FP32 PER=4 对齐 bug | `v4_precheck` dtype 分路径（fp32 恒 16B float4，fp16 依 PER 16B/4B）；负例套件 +2 例（4B offset 必须拒 / 16B offset 必须 PASS）→ 30 例；CUDA 重建后负例 29/30+1 跳过、5 变体 76/76 无回归 | `94609b8` |
| 3 | best.json 结论过强 | 主形状 fp16 → NO_UNIQUE_WINNER（streaming 无唯一胜出者，hot 为 secondary 记录）；v4 保留 v0.1 incumbent、v1/v2 竞争性变体；best.json 重构 v0.2.1 schema（primary_streaming/secondary_hot 分开）；EXP-0008 措辞修正（paired 原始数据未动） | `694a5a6` |
| 4 | Dispatcher 外推/硬编码 | evidence > coverage：仅 2 个 paired-evidence 格（(128,4096) fp32、(128,8192) fp16 → v2_reg）+ 1 个 incumbent-fallback 格（(128,4096) fp16 → v4）路由优化变体；matrix-only/冲突/未实测一律 baseline；`dispatch_info` 四类 evidence_source；(16,4096) fp16 冲突格不再声称 v4 稳定 | `8ee7030` |

## v0.2 阶段清单

- [x] Phase 1 — API 正确性加固（bindings 统一验证 + 5 变体 TORCH_CHECK + launch 检查）
- [x] Phase 2 — 非法输入负例套件（28 例；v0.2.1 增补 2 例对齐回归后为 30 例）
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
- 分发表（v0.2.1）仅在 3 个实测格路由优化变体（2 个 paired-evidence + 1 个
  incumbent-fallback），其余实测/未实测组合一律 baseline（evidence > coverage）。
- v0.1/v0.2 数字跨版本不可直接比较（harness/时钟/缓存策略均不同）。

## 值得注意的事故（已记录，未隐藏）

- **EXP-0002**（v0.1）：单发事件计时（约 6 µs 启动噪声）把 v1 误判 REJECT，
  与 ncu 结论相反。修复 → `cuda-event-batched-v1`；EXP-0001/0002 标记 superseded。
- **EXP-0007**（v0.1 → v0.2 复核）：batched 框架下变体间非配对测量遭遇 DVFS
  混频（v1@~1350 MHz vs v4@~1905 MHz），1.231× 加速比虚高。v0.2 paired
  harness + DVFS guard 复验后 REVISED 为 1.011×（streaming）/ v4 快约 7%（hot，median 0.9327）。
  两条教训共同支撑客观层设计：独立于智能体、完整保留作废记录、结论可复现可审计。
