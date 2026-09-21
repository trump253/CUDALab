# Evaluator v2.3（paired-streaming-v2.3）——对称 guard、raw/filtered 双轨、filter-sensitivity

日期: 2026-09-19（v0.4-rope 分支）
前序: v2.2（环境稳定化, 见 `evaluator_hardening_v0.3.md` 第 8 条）
状态: 已实现，CPU 单测 36/36 通过（v2.3 初始 28 + v0.4 review 约定
钉死 1 + v0.4.1 gate 收紧 2 + v0.4.1 statistical_relation/policy_decision
形式分离 5），GPU 回归门（Phase 5）**PASS**
（见 §6 与 `benchmarks/v2.3_regression/gate_summary.json`）

## 1. 动机：v2.2 的不对称 guard

v2.2 的两道 guard 都只拒绝**异常慢**的状态：

- per-sample spike guard: 样本 > 运行中 accepted 基线中位数 × 1.5 → 剔除；
- cross-block guard: block 中位数 > 该 variant 运行中位数 × 1.15 → round 无效。

后果（v0.3.1 已登记为 KNOWN LIMITATION，见 `bench.py` 模块 docstring 第 11 条
与 `evaluator_hardening_v0.3.md`）:

1. **选择偏差风险**: 只拒慢、不拒快，理论上偏好性剔除慢 excursion，
   可能系统性偏向某个 variant（例如快 variant 的快 outlier 保留、慢
   variant 的慢 outlier 剔除）。
2. **不可审计**: v2.2 记录只保存 guard 后的中位数（`parent_us` /
   `candidate_us` 与 `n_clean`/`n_spike` 计数），**无法回答"guard 是否
   改变了结论"**——若 guard 不对称地剔除了某 variant 的样本，事后无从
   检验。

v2.3 同时修复这两点：guard 对称化 + raw/filtered 双轨记录 +
filter-sensitivity 判定。

## 2. 对称 guard（明确、可解释、可单测、固定规则、parent/candidate 完全同一）

判据统一为 **log 空间对称偏差**：`|log(t / ref)| > log(F)`。
等价于 ratio 空间 `t/ref > F` 或 `t/ref < 1/F`，快/慢双向拒绝。
阈值 F 为固定常数，写入每条记录（`environment_guard` 块）与本文档，
由单元测试钉死。

### 2.1 per-sample spike guard（`stats.apply_spike_guard`）

- `ref` = 最近 `SPIKE_WINDOW = 50` 个 **accepted** 样本的中位数；
- `t > ref × 1.5` → 拒绝，reason = `"slow"`（慢 spike）；
- `t < ref / 1.5` → 拒绝，reason = `"fast"`（快 spike，**v2.3 新增**）；
- 否则接受；边界（恰好 = 阈值）接受（严格不等式）；
- block 首个样本（无 accepted 基线）恒接受；
- parent 与 candidate 使用**完全相同**的规则、窗口与阈值。

### 2.2 cross-block guard（`stats.crossblock_flag`）

- `ref` = 该 variant 本 run 内此前各 block **filtered 中位数**的运行
  中位数（调用方持有 hist；判定后把当前 med 追加，warmup block 同样
  追加——与 v2.2 行为一致，仅判据对称化）；
- warmup：每 variant 前 `CROSSBLOCK_WARMUP = 3` 个 block 不判（吸收
  run 前 nvidia-smi 空隙 / first-touch 态）；
- warmup 后：`ratio = med / ref`；
  - `ratio > 1.15` → flagged, direction = `"slow"`（均匀慢块）；
  - `ratio < 1 / 1.15` → flagged, direction = `"fast"`（均匀快块，
    **v2.3 新增**）；
  - 其余不判；
- med 为 None（无 accepted 样本）时不判、不入 hist。

### 2.3 为什么是 log 空间对称偏差

- **对称**: 快 excursion 与慢 excursion 用同一 F，方向无关；
  parent/candidate 同规则 → 无方向性偏好（修复 limitation 的核心）。
- **可解释**: "与运行基线偏差超过因子 F" 是一句话能讲清的固定规则，
  没有自适应、没有方向特定阈值。
- **稳健**: 中位数基线 + log 空间对称阈值，对单个极端 outlier 不敏感；
  等价 ratio 判据在工程上就是 v2.2 的慢侧判据加上对称的快侧。
- **可单测**: 三个 guard 逻辑全部抽成**纯 CPU 函数**
  （`stats.apply_spike_guard` / `stats.block_stats` /
  `stats.crossblock_flag`，无 torch 依赖），阈值即函数默认参数，
  由 `tests/test_evaluator_v23_cpu.py` 确定性单测钉死。

## 3. raw / filtered 双轨记录

每个 block（variant × round）同时记录：

| 字段 | 含义 |
| --- | --- |
| `raw_median_us` | guard **前**全部样本中位数（raw 轨） |
| `n_raw` | 全部样本数 |
| `median_us` | guard **后** accepted 样本中位数（filtered 轨，进入 round 统计） |
| `n_accepted` | accepted 样本数 |
| `n_rejected_fast` / `n_rejected_slow` | 快/慢拒绝计数（对称） |
| `n_clean` / `n_spike` | legacy alias（= n_accepted / fast+slow，v2.1/v2.2 兼容） |
| `raw_samples_us` / `accepted_samples_us` | 全量样本（3 位小数，审计用） |

round 级（pair 记录）: `parent_us`/`candidate_us`（filtered）+
`parent_raw_us`/`candidate_raw_us`/`raw_speedup`（raw 轨，同口径
paired）。

pair 级记录新增（schema 要求）：

- `raw_speedup` / `filtered_speedup`：各自 = valid round 的 per-round
  paired ratio 的中位数（`filtered_speedup` = 原 `median_speedup` 同值）；
- `raw{parent_median_us, candidate_median_us, speedup, bootstrap_ci_95}`、
  `filtered{...}`：双轨的跨轮中位数 + CI95（bootstrap seed 20260919,
  n=10000，确定性）；
- `rejected_samples{fast, slow}`：全部 round（含无效重试）两 variant 的
  快/慢拒绝合计；
- `environment_guard{method, symmetric, spike_factor, spike_window,
  crossblock_factor, crossblock_warmup, min_accepted_samples,
  filter_sensitive_log_delta, raw_and_filtered_recorded}`：自描述 guard
  参数块。

矩阵记录：每 round `us`（filtered）+ `us_raw`（raw）；per-variant
`median_us` + `raw_median_us`；记录级 `winner_raw` /
`filter_sensitive` / `rejected_samples`。

**兼容性**：旧字段（`invalid_environment_rounds` 及 legacy
`invalid_dvfs_*` 等）原样保留；历史 v2/v2.1/v2.2 JSON **永不被修改**
（按其各自 harness 版本解释）。旧记录没有 raw 字段时，
filter-sensitivity 判定为"未评估"（不假装可信，也不强行敏感）。

## 4. filter-sensitivity（raw vs filtered 方向一致性）

`stats.filter_sensitive(raw_speedup, filtered_speedup, log_delta)`，
纯 CPU、确定性：

- 任一缺失/非法 → `(False, "未评估")`；
- **方向翻转**（raw<1<filtered 或 filtered<1<raw，严格跨 1.0）→ True；
- 否则 `|log(filtered/raw)| > FILTER_LOG_DELTA = log(1.10)`（≈10%
  相对差）→ True；
- 否则 False。

**pair 记录**：raw/filtered speedup 都定义为 parent/candidate（固定
语义），直接套用上述判据。

**矩阵记录**（winner 上下文）：矩阵中"runner/winner 比值"在各自排序内
恒 >1，方向翻转表现为 **winner 不同**，故矩阵判据为：
(1) raw 排序的 winner ≠ filtered 排序的 winner → 敏感（winner flip，
"guard 改变了结论"）；(2) 否则用共同 top-2（filtered winner W /
runner R）的 raw 比值 vs filtered 比值套 `filter_sensitive` 的 10%
判据。

### 4.1 决策层 gate（`decision.apply_filter_gate`）

在 `decide_v2` 之后应用（**v0.4.1 语义收紧**）：记录
`filter_sensitive=true` 时，**无论原决策是 KEEP / REJECT 还是
NEUTRAL，最终 policy_decision 一律 UNSTABLE**——raw/filtered 已分歧
的记录连"中性"都不可信，不强行给出任何 policy 判定；detail 记录
`original_decision` 与敏感原因。仅原决策已是 UNSTABLE 时保持不变（只
记录标记）。v0.4 的旧语义只降级 KEEP/REJECT、NEUTRAL 留标记；v0.4
实际记录 filter_sensitive 全部为 false，故该收紧不改变任何历史判定。
`cmd_optimize` 的实验记录 `decision` 块即 gate 之后的结果（v0.4.1 起
同时记录 `statistical_relation`——只基于 CI95、与 5% 阈值无关——与
`policy_decision` 两个字段：统计陈述与 acceptance policy 形式分离，
见 decision.statistical_relation 与 experiment.classify_cell）。

## 5. 测试（`tests/test_evaluator_v23_cpu.py`）

确定性构造，无 GPU：

1. **对称 spike**（10µs 基线）: 15.1µs → 拒（slow）；6.5µs → 拒
   （fast）；14.9µs / 6.7µs → 接受（阈值内）；边界 15.0 / 6.666… 按
   严格不等式处理；快慢使用同一 1.5 因子。
2. **对称 block**（运行中位数 10）: 11.6 → flagged slow；8.6 →
   flagged fast（ratio 0.86 < 1/1.15≈0.8696）；11.4 不判。
3. **无偏 swap**（对称性核心）: parent 带快 outlier、candidate 带慢
   outlier 的 block_stats 与两者互换后的结果对比——guard 行为对
   parent/candidate 完全对称（同数据同规则，拒绝计数与 filtered
   中位数一致，只是 reason 标签随方向）。
4. **filter-sensitive 构造**（方向翻转必须 UNSTABLE）:
   parent = 50×10.0 + 50×6.5（6.5 全部 fast 拒；raw 中位数 8.25,
   filtered 10.0）；candidate = 50×9.0 + 50×15.1（15.1 全部 slow 拒；
   raw 中位数 12.05, filtered 9.0）。raw speedup = 8.25/12.05 ≈
   0.685（<1，raw 看 candidate 更慢），filtered speedup = 10/9 ≈
   1.111（>1，filtered 看 candidate 更快）→ 翻转 →
   `filter_sensitive=True`；filtered 轨本可判 KEEP，经
   `apply_filter_gate` 后必须 **UNSTABLE**。
5. **filter_sensitive 纯函数**: 翻转 / 10% 内 / 缺值 三类。
6. **apply_filter_gate**: KEEP→UNSTABLE（敏感）、REJECT→UNSTABLE
   （敏感）、非敏感 KEEP 不变、非敏感 NEUTRAL 不变、detail 含
   original_decision；**v0.4.1 收紧**：NEUTRAL+敏感 → UNSTABLE，
   双向必测（raw 明显更快 + filtered NEUTRAL → 敏感 → UNSTABLE；
   raw 明显更慢 + filtered NEUTRAL → 敏感 → UNSTABLE）。
7. **v0.4.1 statistical_relation / policy_decision 形式分离**:
   CI 边界（下界 > 1 → FASTER；上界 < 1 → SLOWER；含 1.00 —— 含
   恰好触到 1.00 的边界 —— 或缺失 → UNRESOLVED，严格不等式）+ 5%
   阈值独立性（CI [1.001, 1.04] → 统计 FASTER，但 median 1.01 < 1.05
   → policy NEUTRAL，两者可背离）+ `classify_cell` 双字段（KEEP 格
   FASTER+KEEP；背离格 FASTER+NEUTRAL；早退路径 UNRESOLVED/None）。

## 6. 回归门（Phase 5，RoPE 之前）— **PASS**（2026-09-20）

v2.3 必须**不推翻**已知的 v2.2 结论（若推翻，先查 evaluator 再继续）：

- RMSNorm v4_vec_reg vs v1_vec，128×4096 FP16，streaming：v2.2/v0.3.1
  给出 ~0.949（REJECT）/ 0.959（NEUTRAL），方向 v4 ≥ v1 附近；
- Softmax softmax_baseline vs softmax_vec4，128×4096 FP16，streaming：
  必须仍为 vec4 **KEEP ~1.6×**（v2.2 参考 1.6772 / 1.6890；矩阵
  10.109/6.015 ≈ 1.69）。若 1.68× 塌成 ~1.05× 或方向翻转 →
  停 RoPE，先查 evaluator。

结果落 `benchmarks/v2.3_regression/`（新目录；历史目录不动）。

**实测结果**（`gate_summary.json`，4 条 pair 记录 + summary +
repeat 记录，全部 9/9 valid rounds）：

| case | v2.3 结果 | v0.3.1 参考 | 判定 |
|---|---|---|---|
| Softmax baseline vs vec4, streaming | **1.6745** [1.6727, 1.6793]，9/9 更快，raw=filtered，rejected 0/0 | 1.6772 (SFM-0001) / 1.6890 (final_reval) | **PASS** — ~1.6× 精确复现，无 filter sensitivity |
| RMSNorm v4 vs v1, streaming | 1.0375 [1.0279, 1.0990]，9/9 更快，raw=filtered，rejected slow=1 | 0.9489 REJECT（2026-09-20 早些时候） | 见下方调查 |
| RMSNorm v4 vs v1, streaming 复跑 | 0.9576 [0.9509, 0.9623]，0/9 更快，rejected slow=51，raw≈filtered（差 0.0013） | 同上 | 见下方调查 |
| RMSNorm v4 vs v1, hot | 0.9923 [0.9893, 1.0106]，2/9 更快，rejected slow=19 | 0.9710 NEUTRAL | 带内 |

RMSNorm 方向在数分钟内翻转一次（1.0375 → 0.9576）的调查证据
（gate 判定为**环境微态漂移，非 evaluator 缺陷**）：
(1) 第 1 次 streaming run raw==filtered（guard 未介入，翻转不是
guard 造成）；(2) 复跑 run 的 51 个慢样本由**对称** guard 拒绝且
raw≈filtered（差 0.0013 << log(1.10)≈0.0953），全程可审计——
这正是 v2.3 双轨记录设计的价值；(3) Softmax 对照在 v2.3 下精确
复现（1.6745 vs 1.6772）；(4) 该机器 idle 微态漂移已在
`evaluator_hardening_v0.3.md` 记录（v0.3.1 时代 RMSNorm hot 亦有
同类点估计漂移）。

## 7. 不变量

- 不锁时钟、不改 power limit；round 内不做 nvidia-smi 采样（v2.2 起）。
- guard 阈值全部固定并落记录；parent/candidate 同规则。
- 统计单位仍是独立 round；bootstrap seed 20260919、n=10000。
- 历史 JSON 永不修改；新记录以 `harness: "paired-streaming-v2.3"`
  自标注。
