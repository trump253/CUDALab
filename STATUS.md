# CUDALab — 状态

**日期：** 2026-09-19
**阶段：** v0.3 + v0.3.1 合并修复（Merge Fix，4 项 review finding，无新内核）
**状态：** 分支 `v0.3-softmax`（基线 main = v0.2.1 = dfe9e9b），**不 merge 回 main、不开始 v0.4**；v0.3.1 完成后 STOP 等最终 merge review。

## v0.3.1 合并修复摘要（2026-09-19）

针对 v0.3 分支 review 的 4 项 finding，只修 finding、不加内核、不重跑全矩阵：

| # | Finding | 处置 |
|---|---|---|
| 1 | `softmax_hsplit2` 依赖 CUDA 调度模型不保证的跨 block 并发驻留假设（spin-wait → liveness 风险）+ HsGlobal 进程级 scratch race 风险 | **隔离**：UNSAFE_HISTORICAL_EXPERIMENT / REJECTED / NOT_FOR_NORMAL_DISPATCH；默认 `ext.variants()` 移除（bindings.cpp quarantine 集），CLI/引擎显式请求明确拒绝；内核源码与全部 SFM-0004 数据保留。SFM-0004.md §6 |
| 2 | "CI 排除 1.0 但 <5%" 被误称为"统计平局"；"结构最优 / 设计空间闭合"过度外推 | 区分 `statistical_relation`（FASTER/SLOWER/UNRESOLVED）与 `policy_decision`（KEEP/REJECT/NEUTRAL/UNSTABLE）；README/报告/best.json 措辞更正为"`softmax_vec4` 是当前 acceptance policy 下的 incumbent；后续候选尚未达到 ≥5% 的替换门槛" |
| 3 | 2080 Ti 峰值误写 550 GB/s；由 `algorithmic_bw_gbps`（逻辑流量）推出"DRAM 饱和 / 带宽墙"不成立 | 更正为 **616 GB/s**；(1024,4096) fp16 ≈ 485 GB/s 逻辑吞吐 = 78.7% 卡规格，**是否真正达到 DRAM 饱和需要对应 NCU 验证（该形状无 NCU DRAM 证据）**；相关"饱和/带宽墙"结论撤回或弱化 |
| 4 | v2.2 spike / cross-block guard 不对称（只拒绝异常慢状态）→ 潜在选择偏差 | 在 `docs/evaluator_hardening_v0.3.md`、`docs/benchmark_audit_v0.3.md`、最终报告登记 KNOWN LIMITATION + Evaluator v2.3 TODO；注明 SFM-0001 primary streaming 记录 invalid_spikes=0 / invalid_crossblock=0（1.68× 不依赖这些过滤）；不重跑 v0.3 数据 |

顺手清理：无效轮计数字段更名 `invalid_environment_rounds`（旧名 `invalid_dvfs_*`
保留为 legacy alias）；README streaming 工作集表述改为"是否 > L2 取决于 shape，
以 `working_set_gt_l2` 为准"（主目标 (128,4096) fp16 为 33.5 MB）。

## v0.3 完成摘要（2026-09-19）

核心问题：**v0.2 的闭环（正确性 → 配对 bench → 统计 → 决策 → 剖析 →
实验史）能否原样迁移到第二个算子？** 算子：row-wise Softmax
（FP32 内部，输出原 dtype；FP16 主 + FP32，禁 BF16；sm_75 / CUDA 11.8）。
最终报告：`docs/report_v0.3_result.md`。

| 项目 | v0.3 结果 |
|---|---|
| Evaluator 通用化 | `cudalab/evaluator/`（bench v2.2 + stats/decision/profiler/negative/experiment，operator-agnostic）+ 算子 adapter `cudalab/operators/{rmsnorm,softmax}.py`；stats.py/decision.py 与 v0.2.1 逐字节相同（独立审计确认） |
| harness 升级 | v2.2（`docs/evaluator_hardening_v0.3.md`）：移除 round 内 nvidia-smi 采样 → 时间基准 burn（≥150 launches 且 ≥300ms）+ 逐样本 spike guard（1.5×）+ 跨块一致性 guard（1.15×）；v2.1 DVFS guard 偏离已作为"基于证据的机器态适配"记录并在全部分支文档与最终报告中声明 |
| RMSNorm 回归硬门 | **PASS**（eae07bb；最终复跑 85faeca）：CPU tests + negative 29/30+1 skip + 正确性 v4_vec_reg/baseline 76/76 + (128,4096) fp16 paired 全兼容 v0.2 结论 |
| 正确性 / 负例 | 5 个 softmax 变体全部 72/72（容差逐变体相同，fp16 atol 2e-3/rtol 5e-3）；negative 14/14+1 skip（launch 前 TORCH_CHECK） |
| 基准矩阵 | 36 格 × 5 变体 full5 + base/inc 各 36 格，全部 9/9 valid；主目标 (128,4096) fp16 |
| 自主优化实验 | **4/4**（profiler→hypothesis 驱动）：SFM-0001 `softmax_vec4` **KEEP**（streaming 1.6772→1.6890 稳健复现；hot 记录值 1.2916 存在机器态漂移，final_reval 0.9865 NEUTRAL，已披露）→ **incumbent = `softmax_vec4`**；SFM-0002 online NEUTRAL（瓶颈是延迟不是带宽）；SFM-0003 vec4_ilp2 NEUTRAL（每线程 ILP 不是杠杆）；SFM-0004 hsplit2 **REJECT**（occupancy 44%→86% 但 barrier stall 5.6%→31-35%，不 occupancy-bound），v0.3.1 起 **隔离**（UNSAFE_HISTORICAL_EXPERIMENT / NOT_FOR_NORMAL_DISPATCH，见 §6）。**失败内核全部保留（历史证据）**；四个正交维度测完 ≠ 设计空间穷尽（v0.3.1 措辞更正） |
| best.json | `experiments/softmax/best.json`（classify_cell，v0.2.1 语义）：36 格全部 **NO_UNIQUE_WINNER**（win/runner-up 比值 0.998–1.048 < 1.05）；17 格 INCUMBENT 标签（vec4 在 top-2）/ 19 NO_UNIQUE_WINNER |
| NCU | baseline vs 4 候选（双 cache-control，v0.2.1 语义）+ incumbent 复验（<1% 漂移）；per-launch µs 与 NCU dur 的时钟/L2 语义差异已在报告说明 |
| 独立审计 | `docs/benchmark_audit_v0.3.md`：**PASS WITH CAVEATS**（数据逐项可复现、决策与规则一致、重构忠实；caveat #1 SFM-0001 hot 漂移已在 SFM-0001.md §6 + 报告披露，caveat #3b 日期笔误已修正） |
| 机器态限定 | (128,4096) fp16 **hot** 模式跨运行配对结果漂移（vec4 hot 1.29→1.37→0.99 三次运行；streaming 1.68 稳定）；RMSNorm 回归 hot 1.0144 NEUTRAL（v0.2 为 0.9327 REJECT）/ streaming 0.9419 REJECT（v0.2 为 0.9592 NEUTRAL）——点估计漂移、机制不变，定位为机器态而非 evaluator 缺陷 |
| 未做 | dispatcher 默认不做（无 paired 确认的 per-shape 路由证据）；不 merge main；不开始 v0.4 |

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
| 4 | Dispatcher 外推/硬编码 | evidence > coverage：仅 2 个 paired-evidence 格（(128,4096) fp32、(128,8192) fp16 → v2_reg）+ 1 个 incumbent-fallback 格（(128,4096) fp16 → v4）路由优化变体；matrix-only/冲突/未实测一律 baseline；`dispatch_info` 四类 evidence_source；(16,4096) fp16 冲突格不再声称 v4 稳定 | `774b1f7` |

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
