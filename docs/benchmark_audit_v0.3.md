# CUDALab v0.3 基准方法学独立审计

- 日期: 2026-09-19
- 分支: `v0.3-softmax`（审计覆盖至 commit `e39d559`）
- 审计方式: 独立 subagent，只读（READ-ONLY）、纯 CPU，未触碰 GPU；
  对全部实验 JSON 逐项重算并与 MD 记录交叉核对，对 pair JSON 重跑
  `decide_v2` 精确复现决策；重构保真度以 `git show dfe9e9b`（v0.2.1）
  逐文件对比。
- 格式先例: `docs/benchmark_audit_v0.2.md`（v0.2 独立审计）。

## 1. 重构保真度（evaluator/bench.py vs v0.2.1@dfe9e9b）

- `cudalab/evaluator/stats.py` 与 `decision.py` 与 v0.2.1 **逐字节相同**
  （仅 docstring 头不同）；decide_v2 阈值与 bootstrap（seed 20260919，
  n_boot=10000，n<3→None）未变。
- `cudalab/evaluator/bench.py`（553 L，"paired-streaming-v2.2"）相对
  `git show dfe9e9b:cudalab/bench_v2.py`（494 L）的差异全部落在已声明
  变更内：v2.1 "warmup ≥150 launches AND ≥300ms wall burn"；v2.2 去掉
  轮内 nvidia-smi 采样（v0.2 的 3 采样/轮 DVFS guard 移除），改为逐
  样本 spike guard（SPIKE_FACTOR=1.5, MIN_CLEAN_SAMPLES=50）+ 跨块
  中位数 guard（CROSSBLOCK_WARMUP=3, CROSSBLOCK_FACTOR=1.15，只标记
  变慢块，失败块也入历史）；记录去掉 `clocks` 字段、保留整次运行的
  gpu_state_before/after（bench.py L31-72 头部文档）。
- `profiler.py` 为 operator-agnostic 重构（driver_src/kernel_regex/
  out_path 由 adapter 提供），METRICS 列表与 stall 解析不变；
  `gpu.py` 原样抽出；`scripts/bench_v2.py` 仅新增对缺 `clocks` 的
  v2.2 记录的宽容处理（旧记录仍可打印时钟区间，已静态验证）；
  stats.py 的 `check_dvfs_*` 保留为死代码但仍有测试覆盖
  （tests/test_evaluator_cpu.py）；`cudalab/bench_v2.py`（77 L）兼容
  shim 保留 v0.2 公共 API（bench_pair/bench_matrix/pytorch_ref_latency
  签名不变）。**未发现未声明的逻辑改动。**

## 2. 算子 adapter 公平性

- rmsnorm 与 softmax 共用同一引擎与同一轮结构（9 rounds × 100 iters ×
  32 batch、同预分配张量、A→B/B→A 交替、同一 warmup 与 guard）。算子
  专属仅：shape 矩阵、pool 内容、kernel regex、algorithmic_bytes。
- bytes 口径正确：softmax = M*H*es*2（读入+写出，operators/softmax.py）；
  rmsnorm = (M*H+H+M*H)*es（输入+权重+输出，operators/rmsnorm.py L78-79）。
- rmsnorm pool 生成与 v0.2 `_make_pool` 声明逐字节一致（seed 1234、相同
  x→w 抽取顺序）；vec4/ilp2 的 fallback 路径（H%4≠0 或未对齐）与
  baseline 共享同一份 scalar 内核（kernels/softmax/vec4 L77-98 →
  softmax_scalar.h），无偏置。

## 3. Softmax 基准完整性

- 逐项重算：SFM-0001..0004 四个 MD 的全部数字（median/CI/faster-rounds/
  逐轮 p/c）与 pair_*.json 一致，对 JSON 跑 decide_v2 精确复现记录决策：
  SFM-0001 KEEP（hot 1.2916 [1.2700,1.2946] 9/9；streaming 1.6772
  [1.6568,1.6794] 9/9）；SFM-0002 NEUTRAL；SFM-0003 NEUTRAL；
  SFM-0004 REJECT（streaming 0.7752 0/9；hot 0.6752 0/9）。
  **未发现篡改。**
- 正确性：5 个变体全部 72/72 通过且容差完全相同（fp16 atol 2e-3/
  rtol 5e-3/row_sum 5e-3；fp32 1e-5/1e-4；rel_eps_guard 0.001；
  correctness/v0.3/*.json），无按变体放宽；invalid_inputs 14/15（1 skip）。
- 失败实验保留：SFM-0004 REJECT 内核仍在库中；各 commit
  （6965ca7/47e10b4/5f45466/edf4662/85faeca）均把 kernel+MD+result+pair
  JSON+correctness 同提交。
- 报告范围：benchmarks/softmax/ 为 base_ 与 inc_ 各 54 格完整矩阵
  （9 shape × 2 dtype × 2 mode），无 best-shape-only 报告。
- 数值稳定：所有变体均做 max 减法（scalar.h L68-82；vec4 L77-98；
  online L124-140；hsplit2 的 (m,l) 合并）。
- 计时区：hsplit2 的 cudaMalloc 在 hs_init 一次性完成（kMaxScratchM=
  8192，hsplit2 L61），在计时区外；每 launch 的 cudaMemsetAsync(cnt)
  在计时 stream 上但属真实 launch path，与头部 L33-39 文档（"计时区域
  内无 cudaMalloc"）一致，不构成操纵。

## 4. 统计合理性

- n=9 的 percentile bootstrap CI 较宽；decide_v2 固定阈值（KEEP:
  median≥1.05 且 faster≥70% 且 CIlo>1.00；REJECT: median≤0.95 且
  faster≤30% 且 CIhi<1.00；<5 有效轮 UNSTABLE；correctness FAIL
  无条件 REJECT 优先）对所有实验一致适用。
- 边界案例：SFM-0002 streaming 1.0413 [1.0377,1.0436]、faster 9/9、
  CI 全在 1.0 之上，但 median 距 1.05 差 0.0087 → NEUTRAL。按既定
  规则该判定正确；它是在规则未改的前提下最接近翻转的案例（若阈值为
  1.04 则 KEEP），无证据表明规则被临时调整。
- 跨块 guard 盲区：1.15 系数抓不到 <15% 的整体漂移，且前 3 块不检查。
  实例：SFM-0001 hot r4 candidate 5.627 vs 运行中位数 ~5.06（比值 1.11）
  未被标记；R7 streaming r1 parent 块 6.912（vs ~5.45）亦未标记
  （CROSSBLOCK_WARMUP=3）。两者在 docs/evaluator_hardening_v0.3.md §6
  已列为残余风险。

## 5. NCU 方法学

- 与 v0.2.1 修正后语义一致：profiles/softmax/baseline_M128_H4096_
  ncu_comparison.json 的原始命令行 `--cache-control all/none --clock-
  control base --launch-skip 2 --launch-count 4 -k regex:softmax`；
  cc=none 的 raw 输出含 "Running with uncontrolled GPU caches" 警告，
  cc=all（flush/reset）与 none（不 flush）语义标注正确；metrics 经
  --query-metrics 验证；raw 存 raw/。两模式差异（DRAM 17.97% vs
  22.51%，L2 read hit 2.56%，long_scoreboard 60.6%）方向合理。

## 6. 结论：**PASS WITH CAVEATS**

记录数据逐项可复现、决策与规则一致、重构忠实、未见完整性问题；
但存在以下按严重度排序的保留意见：

1. **[最严重] SFM-0001 hot KEEP 未复现**：同日 final_reval 得 0.9865
   [0.9318,0.9960]、faster 1/9 → NEUTRAL（final_reval/final_reval_
   result.json），且同变体同日相隔 1 分钟两次运行 hot 绝对时间相差
   ~45%（inc 矩阵 4.684µs@21:42 vs final_reval 6.797µs@21:41）。
   streaming KEEP 稳定复现（1.6772→1.6894；矩阵 10.176/6.016≈1.69
   三重一致），incumbent 链 baseline→vec4 因此站得住，但记录的
   "hot 1.29×" 大概率高估，且（审计时）无任何 MD 讨论此矛盾。
   依赖 hot cell 的后续决策会被推翻。
2. **小样本 + guard 盲区使边界决策脆弱**：n=9 bootstrap 配固定
   1.05/0.95 阈值，再叠加 <15% 漂移盲区，使近界决策（SFM-0002
   streaming；final v4_vs_v1 hot 1.0144、9/9 faster → NEUTRAL）对
   单轮质量敏感。当前规则下均不翻转，但两方向功效都不足。
3. **机器状态漂移（已声明）+ 轻微记录问题**：v0.2 RMSNorm "hot
   REJECT 0.9327" 今日重跑 NEUTRAL（benchmarks/v0.3_regression/
   R7/final），跨运行绝对时间不可比，仅运行内配对有决策意义；
   SFM-0004.md 日期 "2026-07-19" 应为 2026-09-19。

未发现任何按正确方法重算后会翻转的 KEEP/REJECT/NEUTRAL；最接近的
边界是 SFM-0002 streaming（median 差 0.0087）。

## 7. 复核方响应（Lead，审计完成后）

| # | finding | 处置 |
|---|---|---|
| 1 | SFM-0001 hot KEEP 未复现，无 MD 讨论 | 已在 `experiments/softmax/SFM-0001.md` 新增 §6 复核注记：记录 hot 0.9865 NEUTRAL 与 ~45% 绝对漂移，声明 KEEP 以 primary 判据 streaming（1.6890 稳健复现）为准，hot 数字保留为记录值但不作为后续决策依据；最终报告 Q2 同步披露 |
| 2 | 边界决策脆弱（SFM-0002 streaming 等） | 维持 NEUTRAL（规则未改、按既定规则判定正确）；残余风险（n=9 功效 + <15% guard 盲区）已在 `docs/evaluator_hardening_v0.3.md` §6 列出，最终报告 Q6 同步披露 |
| 3a | 机器状态漂移 | 已声明（v2.2 文档 + 本分支各处）；最终报告 Q6 作为 Evaluator Generalization Verdict 的限定条件之一 |
| 3b | SFM-0004.md 日期笔误 | 已修正为 2026-09-19 |

## 8. v0.3.1 补记（2026-09-20，merge review 修复后）

本审计的 findings（#1–#3）结论不变。v0.3.1 修复 4 项 merge review
finding（hsplit2 隔离、统计语义澄清、带宽表述更正、evaluator 局限
登记），**未改动任何基准数据或判定**：

- Fix 1–3 不触碰数值/判定记录：hsplit2 原判 REJECT，本就不在
  best.json 链上，隔离只改 dispatch 面；统计语义与带宽更正为措辞层。
- 新增 KNOWN LIMITATION（对 caveat #2 的补充）：spike / cross-block
  guard **不对称，可能偏好性拒绝慢 excursion**
  （"KNOWN LIMITATION: spike / cross-block guards are asymmetric and
  may preferentially reject slow excursions."）。证据核查：SFM-0001
  primary streaming 记录 `invalid_spikes_rounds=0`、
  `invalid_crossblock_rounds=0`（hot 同为 0）；v0.3.1 RMSNorm 回归
  （`benchmarks/v0.3.1_regression/`，v4_vec_reg vs v1_vec，(128,4096)
  fp16，2 模式 × 9 rounds，2026-09-20）18/18 valid，
  invalid_environment/spikes/crossblock 全为 0——guard 不对称性在
  决定性 run 与回归 run 上均未触发，未影响任何已发布数字。
- 后续：Evaluator v2.3 TODO（对称阈值或 log-latency 稳健偏差）已登记
  于 `docs/evaluator_hardening_v0.3.md` §7 与最终报告 Q6 §8；属
  evaluator 演进项，不在 v0.3.1 范围。
