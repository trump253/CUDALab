# CUDALab v0.2 Benchmark 方法学审计（独立）

## 审计范围与审计者

- **审计者**：独立 subagent（由主 agent 委托的只读审计任务），未参与 v0.2 任何代码编写；立场为"假设结论有错、主动找证据"。
- **审计时间戳**（`date -Iseconds` 实测）：`2026-09-19T13:06:28+08:00`
- **审计方式**：只读数据文件与源码 + 交叉核对数值；未执行任何 GPU 命令、未跑 benchmark、未改动任何文件、未 commit。
- **实际读取的数据文件**：
  - `benchmarks/v0.2/` 下 **43** 个 JSON（任务描述写 42，实际清点为 28 个 `v02_full_*` 矩阵 + 13 个 `v02_pair_*` 配对 + `shape_winners.json` + `pytorch_ref_M128_H4096_fp16.json` = 43；逐文件读取核对）；
  - `experiments/rmsnorm/EXP-0008.json`、`EXP-0007.json`、`best_v0.1.json`、`best.json`；
  - `experiments/rmsnorm/correctness/v0.2/` 全部 6 个 JSON（5 变体 + `invalid_inputs.json`）；
  - `profiles/rmsnorm/v0.2/v02_profile_comparison.json` 与 `raw/` 下的 ncu 原始输出与驱动脚本（`*.ncu.txt`、`*_drv.py`，含 `$ ncu cmd` 命令行头）；
  - v0.1 对照：`benchmarks/bench_exp_v4_vec_reg_vs_v1_vec.json`（含逐条 `gpu_state_before/after`）、`benchmarks/bench_final_all_variants.json`、`profiles/rmsnorm/*.json`（5 个）；
  - 源码：`cudalab/bench_v2.py`、`cudalab/stats.py`、`cudalab/decision.py`、`cudalab/reference.py`、`cudalab/correctness.py`、`cudalab/negative_suite.py`、`kernels/rmsnorm/*.cu`（5 变体）、`kernels/rmsnorm/bindings.cpp`、`cudalab/profiler.py`（ncu 参数段）、`tests/test_evaluator_cpu.py`。

## 审计清单结论表

| # | 条目 | 结论 | 关键证据（文件 → 数值） |
|---|------|------|------------------------|
| 1 | 各变体计算量等价 / algorithmic_bw 口径 | **PASS（带 CONCERN）** | 计算等价（5 变体 76/76，见 #10）；访存模式不同：baseline/v1/v3 两遍重读 x，v2/v4 单遍寄存器驻留；`algorithmic_bw_gbps` 按变体无关的最小有用 IO 计算，非"按变体真实访问量"（已在 harness 文档中如实声明） |
| 2 | 相同输入 | **PASS** | `bench_v2.py` `_make_pool`：单 `Generator` seed=1234；每个 pair/matrix run 只建一次 pool，parent/candidate 全程用同一组 `xs/w/outs` 对象；streaming 每次 `_measure_block` 都从 buffer 0 起按同一顺序轮换 |
| 3 | 预分配 tensor / 分配在计时外 | **PASS** | pool 在 rounds 循环之前构建；计时区域只有 `forward_into`（`bindings.cpp` L68-79：仅验证 + launch，无分配无拷贝）；warmup 150 次不计时 |
| 4 | 同步正确 | **PASS** | `_measure_block`：warmup 后 `torch.cuda.synchronize()`；每样本 32 连发后 `synchronize()`；CUDA event 计时，100 样本取中位数 |
| 5 | 顺序无偏 | **PASS（带小注）** | paired：slot 偶数 parent 先 / 奇数 candidate 先，逐 round 记录 `order`（数据 9/9 严格交替）；matrix：`variants[slot%n:]+variants[:slot%n]` 真 round-robin，9 轮内每变体每位置出现 1–2 次，均衡 |
| 6 | 时钟可比（DVFS guard） | **PASS（带局限）** | 阈值 5%；本次全部 run 9/9 valid、采样恒 1350 MHz；1350 vs 1905 场景有单元测试 `test_dvfs_pair_extreme_clocks_invalid` 断言判 INVALID_DVFS；nvidia-smi 轮询为 kernel 区间外采样，是代理指标（必须列为局限） |
| 7 | 无效轮透明丢弃 | **PASS** | 代码层面：round 记录（含 `valid`/`invalid_reason`）无条件 append 进 `rounds[]`，统计只用 valid 子集；数据层面：v0.2 全部 41 个 run 均 0 个无效轮，故 JSON 中无 invalid 条目可展示——机制由代码 + 单元测试覆盖，未被真实数据触发 |
| 8 | 缓存模式标注准确 | **PASS** | hot：pool_size=1，working set 2,105,344 B（fp16）/ 4,210,688 B（fp32）< L2 5.5 MB，`working_set_gt_l2=false`；streaming：pool 16，33,562,624 B = 33.5 MB > L2，`true`；harness 明确"不声称完全 cold"；ncu cc=all 即 ncu 默认 = cache flush/reset（每 replay 前失效缓存，v0.2.1 修正此前写反的"热缓存"标注），v0.1 未传 `--cache-control`（其 profile JSON 无 `cache_control` 字段）走默认 all（= 失效/flush），v0.1 "cold L2" 说法与默认配置一致（v0.2 曾误判"无配置依据/实际热"，v0.2.1 更正）——v0.2 双 cc 模式实测小工作集下差异 <2% |
| 9 | 统计基于独立 round | **PASS** | bootstrap 作用于 9 个 round 级 speedup 数组（非 100-iter 内样本）；round 内样本仅用于稳健中位数；seed 固定 20260919、n_boot=10000、n<3 返回 None；`test_bootstrap_reproducibility` 验证确定性 |
| 10 | candidate 是否走捷径（容差/参考） | **PASS** | 实际容差 fp16 atol=2e-3 / rtol=5e-3，fp32 atol=1e-5 / rtol=1e-4（**不是** 2e-2/2e-2），5 变体逐字节一致、与 v0.1 相同；参考为显式 FP32 公式实现，不依赖 PyTorch 版本相关融合算子；REL_EPS_GUARD 只影响报告口径不影响 pass/fail；负例套件 27/28（1 例单 GPU 环境安全跳过），全部 launch 前拒绝 + post-check（v0.2.1：扩至 30 例 29/30，新增 2 例 v4 FP32 H=1024 对齐回归） |
| 11 | 有无 cherry-picking | **PASS** | 28 单元格（7 shape × 2 dtype × 2 模式）全矩阵 + 13 组 targeted pair 全部落盘；`shape_winners.json` 如实包含对"优化变体"不利的单元格：baseline 在 3 格获胜（(1,4096) fp32 hot、(16,4096) fp32 hot、(128,1024) fp16 streaming），v1 在 3 格、v3 在 6 格获胜 |
| 12 | 矩阵模式离群敏感（额外审查） | **CONCERN（已在文件中声明的局限）** | (16,4096) fp32 hot round 2：5 变体中 3 个升高（v2_reg 10.295 / baseline 11.374 / v4_vec_reg 12.805 µs；v1_vec 6.272、v3_wideblock 7.052 正常——注意"全体变体 10-13µs"的表述与原始数据不符，实为 3/5）；该轮 DVFS 采样恒 1350 MHz 未被 guard 拦截；winner ratio 0.9912、CI 跨 1.0。**结论：矩阵模式 winner-vs-runner CI 仅指示性，最终判定以 paired A/B 为准** |

## 逐条证据

### 1. 等价计算量与带宽指标口径 — PASS（带 CONCERN）

- **计算等价**：5 变体在 `correctness/v0.2/*.json` 均为 76/76 通过，`max_abs_error` 全部 ≤ 0.00390625、`max_rel_error` ≤ 0.00097656，容差字段五份 JSON 完全一致（见 #10）。参考 `reference.py::rmsnorm_ref` 为显式公式：`x.float()` → `pow(2).mean(dim=-1)`（FP32 累加）→ `rsqrt(var+eps)` → `*weight.float()` → 回 cast，与 kernel 的 FP32 累加策略一致。
- **访存模式核对**（读 `kernels/rmsnorm/*.cu`）：
  - `rmsnorm_baseline.cu`：两个 `for (i = tid; i < H; i += nthreads)` 全局读循环（L34、L62）→ **两遍，x 读两次**；
  - `rmsnorm_v1.cu`：L84 第一遍求 sqsum，L109-112 第二遍 `sqsum_and_unpack(xv[i], f)` 再次读 x → **两遍**（源码注释自认"仍是两遍（第二遍重读 x）"）；
  - `rmsnorm_v2.cu`：单遍，x 经 PER 循环一次性载入寄存器后输出 → **一遍**；
  - `rmsnorm_v3.cu`：L36、L60 两个全局读循环 → **两遍**（512 线程）；
  - `rmsnorm_v4.cu`：初始向量化加载入 `buf[]` 寄存器，输出遍只读寄存器 → **一遍**。
- **algorithmic_bw 口径**：`bench_v2.py` L215/L321：`algo_bytes = (M*H + H + M*H) * es`，**对全部变体相同**，即"读 x 一次 + 读 w 一次 + 写 y 一次"的最小有用 IO，**不是按变体真实访问量**。harness docstring（L32-36）如实声明这是逻辑算法流量、非实测 DRAM 吞吐，并说明修复了 v0.1 `effective_bw_gbps` 的 fp32 按 2B 计问题。
  - **CONCERN**：两遍变体（baseline/v1/v3）在指令/缓存层级确实多读一次 x；但单次 launch 的工作集（主目标 fp16 为 1 MB）< L2 5.5 MB，第二遍读大概率 L2 命中，因此该指标对本工作负载是合理归一化，不夸大两遍变体的劣势。真实 DRAM 行为以 ncu 为准：`v02_profile_comparison.json`（cc=none）dram%：baseline 19.35 / v1 36.00 / v2 42.52 / v3 28.86 / v4 36.09。

### 2. 相同输入 — PASS

- `bench_v2.py::_make_pool`（L138-154）：单个 `torch.Generator(device="cuda")`，`manual_seed(1234)`（SEED 常量，所有 JSON 中 `seed=1234` 一致）；streaming 模式 16 个 `xs` 由同一生成器顺序产生（确定性），`w = randn*0.5+1.0`（正值，分布固定）。
- `bench_pair`：pool 在整个 9 轮之前构建一次（L213），parent 与 candidate 在每一轮内使用**同一组张量对象**；`bench_matrix` 同理（L319）。
- streaming 轮换：每个 `_measure_block` 调用内部 `state={"i":0}` 从 buffer 0 开始按 `(i+1)%16` 轮换（L171-176），故同一 round 内两变体经历**相同的 buffer 序列**（含 warmup 150 次同样轮换）。

### 3. 预分配与计时边界 — PASS

- pool（`xs`、`w`、`outs`，`torch.empty_like`）在 rounds 循环前分配（L213/L319）；计时区域内无任何 malloc / 随机数 / copy。
- `bindings.cpp::rmsnorm_forward_into`（L68-79）：验证（`validate_common`/`validate_out`）+ 调用 launcher + `C10_CUDA_KERNEL_LAUNCH_CHECK()`，**不分配新张量、不拷贝**，直接写预分配 `out`。
- warmup 150 次 launch 不计时（L178-180）。

### 4. 同步 — PASS

- `_measure_block`（L178-191）：warmup 后 `torch.cuda.synchronize()`；每个样本 = 32 次连续 launch，`start.record()` … `stop.record()` 后 `torch.cuda.synchronize()`；`times.append(start.elapsed_time(stop)*1e3/batch)`；round 值 = 100 个样本的 `statistics.median`。与 v0.1 已验证的批量方案一致（WARMUP/ITERS/BATCH 同值）。

### 5. 顺序无偏 — PASS（带小注）

- **paired**：`order = [parent, candidate] if slot % 2 == 0 else [candidate, parent]`（L241，确定性交替），且逐 round 记录 `"order"`。数据验证（`v02_pair_v4_vec_reg_vs_v1_vec_M128_H4096_fp16_streaming.json`）：9 轮 order 为 v4+v1, v1+v4, v4+v1, … 严格交替。
- **matrix**：`order = variants[slot % n:] + variants[:slot % n]`（L345，真 round-robin）。数据验证（`v02_full_M128_H4096_float16_streaming.json`）：9 轮内每个变体在每个位置出现 1–2 次（如 baseline 位置计数 [2,1,2,2,2]），位置与时间漂移解耦。
- 小注：交替是确定性而非随机化；若漂移呈与 slot 奇偶相关的锯齿形态，交替不能完全去偏——但逐 round 时钟采样（#6）与 round-level 统计对此有覆盖。

### 6. 时钟可比 — PASS（带局限，局限已如实列出）

- 阈值 `DVFS_TOL = 0.05`（bench_v2.py L77）。paired：每 round 3 次 nvidia-smi 采样（pair 前 / 第一个 variant 后 / 第二个后），每 variant 有效时钟 = 区间前后两次采样均值，相对差 >5% → `INVALID_DVFS`，重试 ≤3 次，valid <5 → UNSTABLE（`check_dvfs_pair`，stats.py L85-101）。matrix：全变体有效时钟极差/均值 >5% → 无效（`check_dvfs_matrix`，L104-112）。
- **本次数据**：41 个 run（28 矩阵 + 13 配对）全部 `valid_rounds=9/9`、`invalid_dvfs_rounds=0`，所有采样恒为 **1350 MHz**（温度 33-42°C，`gpu_state_before/after` 一致），即 v0.2 全程运行在稳定低频状态，guard 无需拦截。
- **1350 vs 1905 场景可拦截的证据**：`tests/test_evaluator_cpu.py::test_dvfs_pair_extreme_clocks_invalid` 直接断言 `check_dvfs_pair(1350, 1905, 1905, parent_first=True, tol=0.05)` → `not ok and reason == "INVALID_DVFS"`（注释即标明"EXP-0007 类型场景"）；另有 `test_dvfs_pair_stable_valid`（1900/1905/1902 → valid）、`test_dvfs_pair_missing_data_invalid`（缺采样 → no_clock_data）、`test_dvfs_matrix`（矩阵版混频判无效）。
- **局限（必须声明）**：nvidia-smi 是轮询采样，采样点位于 kernel 执行区间之外（每 round 仅 3 点，round 时长 ~100-300 ms），是锁频不可用容器下的最佳代理，**不是**逐 kernel 时钟。实证失效案例：#12 的 (16,4096) fp32 hot round 2 干扰事件期间时钟读数恒 1350，guard 未能识别（它只覆盖时钟分歧，不覆盖瞬态调度/干扰事件）。

### 7. 无效轮透明丢弃 — PASS

- 代码：`bench_pair` 中 `res`（含 `"valid"`、`"invalid_reason"`、`"retries"`、逐采样时钟）在重试循环结束后**无条件** `rounds_out.append(res)`（L254-273），`record["rounds"] = rounds_out` 全量落盘；统计只用 `[r for r in rounds_out if r["valid"]]`（L275）。`bench_matrix` 同构（L356-370、L372）。
- 数据：v0.2 全部 run 无无效轮，因此 JSON 中不存在 invalid 条目可供展示——"透明丢弃"在本批数据中未被真实触发，其正确性由代码路径 + 单元测试（#6）保证。审计口径：**机制 PASS，数据层"未触发"**。

### 8. 缓存模式标注 — PASS

- `hot`：`pool_size=1`；主目标 (128,4096) fp16 working set = 2,105,344 B（2 MB）< L2 5.5 MB，JSON 中 `working_set_gt_l2=false`（13 个 pair JSON 逐一核对）。
- `streaming`：`pool_size=16`；主目标 fp16 working set = 33,562,624 B = **33.5 MB** > L2，`working_set_gt_l2=true`（fp32 为 67,125,248 B = 67.1 MB）。计时区域内 kernel 轮换 buffer，无 malloc/随机数/copy（#2/#3）。
- harness 明确**不声称"完全 cold cache"**（docstring L29-30："rotating-buffer / cache-cold-ish"）。
- **ncu 侧核对（v0.2.1 修正语义，此前写反）**：raw 命令行头（`profiles/rmsnorm/v0.2/raw/*.ncu.txt` 首行）证实 v0.2 显式传 `--cache-control none --clock-control base`（cc=none 输出含 ncu 自身警告 `==WARNING== Note: Running with uncontrolled GPU caches`，即"缓存不受控"）；cc=all 为 ncu 默认 = cache flush/reset（每个 replay pass 前失效全部缓存，确定性 flushed 状态）。`v02_profile_comparison.json` 的 note 已更新为"cc=all 为 ncu 默认 = cache flush/reset profiling…；cc=none = no-flush profiling…"（v0.2.1 更正）。
- **v0.1 "cold L2" 说法与默认配置一致（v0.2.1 更正，此前误判为"推翻/无配置依据"）**：v0.1 的 5 份 profile JSON（`profiles/rmsnorm/*.json`）**没有 `cache_control` 字段**，即走 ncu 默认 `all`（= cache flush/reset，每 replay 前失效缓存）——因此 v0.1 "cold L2" 的描述与默认配置**一致**（flushed L2）；v0.2 早期文档（`cudalab/profiler.py` 旧注释、EXP-0008 旧 note）据此误判"无配置依据/实际热"，现更正（v0.1 raw 数据未改动）。v0.2 双 cc 实测：1 MB 工作集下 cc=all 与 cc=none 的 duration 差异 <2%（baseline 14.336/14.560、v1 6.264/6.136、v2 6.040/5.952、v3 9.392/9.312、v4 7.040/7.136 µs），即对本负载 ncu 单次 launch 的缓存状态（flushed vs 不受控）不敏感，真正的缓存效应杠杆是 bench 的 hot/streaming buffer 策略。

### 9. 统计独立性 — PASS

- `stats.py` docstring 明确：500 个连续 event sample 不能当独立样本；统计单位是独立 round。round 内 100 样本只用于 `statistics.median`（`_measure_block` 返回值）。
- `bench_pair` L276-279：`paired_speedups`（9 个 round 级 parent/candidate 比值）→ `summarize` + `bootstrap_ci`。bootstrap：percentile 法、`n_boot=10000`、**seed=20260919**（`BOOTSTRAP_SEED`，所有 JSON `bootstrap` 字段一致）、对 median 重抽样、n<3 返回 None（`test_bootstrap_too_few_samples` 验证）；`test_bootstrap_reproducibility` 验证同输入同 CI（确定性）。
- 决策（`decision.py::decide_v2`）同样只消费 round 级量：KEEP 需 median≥1.05 且 faster 占比≥70% 且 CI 下界>1.00；REJECT 对称；correctness FAIL 无条件 REJECT；valid<5 → UNSTABLE。

### 10. candidate 捷径 / 容差 / 参考 — PASS

- **任务描述质疑的"容差 2e-2/2e-2"不成立**：5 份 `correctness/v0.2/*.json` 的 `tolerances` 字段均为 **fp16 atol=2e-3 / rtol=5e-3，fp32 atol=1e-5 / rtol=1e-4**，与 `cudalab/correctness.py::TOLERANCES`（L33-36）一致，也与 v0.1 相同——**没有放宽**。
- 容差对所有变体完全一致（五份 JSON 逐字段相同）；`correctness.py` docstring 声明"绝不为某个候选单独放宽"。
- 参考实现 `reference.py::rmsnorm_ref` 为显式公式、FP32 累加、不依赖随 PyTorch 版本变化的融合算子；pass/fail 用 `torch.allclose(atol, rtol)` + NaN/Inf 检查；`REL_EPS_GUARD=1e-3` 只用于报告的 max_rel_error 分母钳位，不影响判定。
- 通过情况：5 变体 × 76 例（11 shape × 3 seed × 2 dtype + 5 edge × 2 dtype）全部 76/76；`max_abs_error` 最大 0.00390625（= 2^-8，fp16 在 4.0 附近 1 ulp 量级），满足 allclose 判据（|a-b| ≤ atol + rtol·|b|）。
- **负例套件**（`invalid_inputs.json`，negative-v0.2）：28 例 = 25 reject + 2 pass control + 1 skip（多设备用例，单 GPU 环境安全跳过）；27/28 pass、`all_pass=true`。所有 reject 均 `status=rejected` 且 `post_check_ok=true`（拒绝后上下文健康）；关键回归用例（H=1025/4100 曾致 v0.1 静默误算、8B 偏移破坏 16B 对齐）消息匹配 `msg_match` 验证通过；2 个对齐 control（16B 偏移）确认不误拒。
- **v0.2.1 增补**：负例套件扩至 30 例 = 26 reject + 3 pass control + 1 skip，29/30 符合预期（新增 2 例 v4 FP32 H=1024 对齐回归：4B offset 基址必须被拒、16B offset 对照组必须 PASS）；上文引用的 v0.2 审计数字保持原样。

### 11. cherry-picking — PASS

- **全矩阵落盘**：28 个 `v02_full_*.json`（7 shape × 2 dtype × 2 模式）完整存在，`shape_winners.json` 逐格给出 winner / runner-up / 全 5 变体中位数 / round 级 ratio + CI / valid_rounds，**没有只报 global best**。
- **不利单元格如实存在**（`shape_winners.json` 原始数据）：
  - `[1,4096] fp32 hot`：**winner=baseline**（6.097 µs vs runner v4_vec_reg 6.275，ratio 0.9811，CI [0.9428, 1.2178]）；
  - `[16,4096] fp32 hot`：**winner=baseline**（6.336 µs vs runner v3_wideblock 6.528，ratio 0.9912，CI [0.9655, 1.0133]）；
  - `[128,1024] fp16 streaming`：**winner=baseline**（6.94 µs vs runner v1_vec 6.976，ratio 1.0052，CI [0.9261, 1.0739]）；
  - 其余非"v4 通吃"的单元格：v1 胜 3 格（(16,4096) fp16 streaming、(1,1024) fp16 streaming、(1,1024) fp32 hot），v3 胜 6 格（多为 M=1 / 小 H），v2 胜 8 格（含全部 fp32 大 shape 与 (128,8192)），v4 胜 8 格。
- 13 组 targeted pair 中同样包含对 incumbent 不利或中性的结果：v4 vs v1 streaming 中 v1 快 1.1%（9/9 轮）→ NEUTRAL（incumbent 未显著胜出）、v4 vs v2 两模式 NEUTRAL（统计平局）；v4 vs v1 hot 为 **REJECT v1**（candidate v1 慢于 v4：median v4/v1 = 0.9327、0/9 轮 v1 更快，v4 快约 7%）——全部如实记录于 EXP-0008 `paired_results` 与 `decision_summary`。

### 12. 矩阵模式离群敏感（额外审查）— CONCERN

- **原始数据**（`benchmarks/v0.2/v02_full_M16_H4096_float32_hot.json`）：round 2 中 **3/5** 变体升至 10-13 µs 区间（v2_reg 10.295、baseline 11.374、v4_vec_reg 12.805 µs；v1_vec 6.272、v3_wideblock 7.052 正常）。任务简报"全体变体 10-13µs"与数据不符，按数据如实记录。同文件还有分散的单元异常点（round 4 v1_vec 8.804、round 5 v1_vec 8.13、round 6 v4_vec_reg 10.689 µs）。
- 该轮所有 nvidia-smi 采样恒 1350 MHz，DVFS guard 判 valid=True——guard 只覆盖时钟分歧，**无法拦截时钟稳定下的瞬态干扰**（见 #6 局限）。
- 后果量化：该格 winner=baseline（中位数 6.336）vs runner=v3（6.528），round 级 ratio 中位数 **0.9912**、CI [0.9655, 1.0133] 跨 1.0——若离群事件落在 winner 一侧，winner 排序可能翻转；中位数本身对 9 轮中的 1 轮离群稳健，但 **winner-vs-runner 的 per-round ratio 统计与 CI 对离群敏感**。
- **必须明确的结论（写入文档要求）**：矩阵模式 winner-vs-runner CI **仅指示性**（用于选形状特定 champion 的候选），**最终判定以 paired A/B 为准**（9 轮交替、同 buffer、DVFS guard、bootstrap CI、decision.py 规则）。EXP-0008 的最终决策全部来自 13 组 paired 数据，未使用矩阵 ratio。

## 对 v0.1 结论的复核

### EXP-0007 的 1.231× 为何被判定 REVISED

- **v0.1 原始事件**（`benchmarks/bench_exp_v4_vec_reg_vs_v1_vec.json`，harness `cuda-event-batched-v1`，非配对、5 轮）：
  - parent v1_vec @(128,4096) fp16：median **5.828 µs**，`gpu_state_before/after` 的 `sm_clock_mhz` = **1350/1350**；
  - candidate v4_vec_reg @同 shape：median **4.736 µs**，`sm_clock_mhz` = **1905/1905**；
  - 5.828/4.736 = **1.2306×** → KEEP（`EXP-0007.json`，`best_v0.1.json` 据此记 v4、median_us_target 4.736）。
- **混频量化**：时钟比 1905/1350 = **1.411**。若 v4 在 1350 MHz 下测量，其时间约放大 1.411× → 4.736 × 1.411 ≈ **6.68 µs > 5.828 µs**，即同频下 v4 应**慢于** v1——与 v0.2 复测方向一致。
- **v0.2 同频（全程稳定 1350 MHz）paired 复验**（9/9 valid，pair JSON 原始数据）：
  - **streaming**：v4 6.967 vs v1 6.897 µs → median speedup 1.0113（CI [1.0017, 1.0193]），**v1 反而快 1.1%**，9/9 轮 candidate 更快 → **NEUTRAL**；
  - **hot**：v4 6.11 vs v1 6.591 µs → median v4/v1 0.9327（CI [0.8373, 0.9658]），v4 快约 7%（1/0.9327），0/9 轮 v1 更快 → **REJECT v1**（v1 不能取代 incumbent）。
- 结论：**1.231× 在两种同频缓存模式下均不可复现**（streaming 甚至反向），EXP-0008 判定 **REVISED** 成立，根因是 v0.1 非配对框架下 v1 恰好测于 1350 MHz、v4 恰好测于 1905 MHz 的 DVFS 混频膨胀（v0.1 数据中 v1 后 3 个 shape 已升至 1905、v4 全程 1905，`bench_exp_v4_vec_reg_vs_v1_vec.json` 逐条 gpu_state 可查）。

### 其余 v0.1 结论状态（EXP-0008 `v01_revalidation`，paired 同频数值）

| v0.1 结论 | v0.2 判定 | paired 证据（median speedup，parent/candidate） |
|---|---|---|
| v4 为 fp16 主目标 best（1.231× over v1） | **REVISED**（见上） | streaming 1.0113 NEUTRAL / hot 0.9327 REJECT v1 |
| v4 保留为主目标 incumbent | **CONFIRMED_WITH_CAVEATS** | v4 vs v2 fp16：hot 1.0215（CI [0.9880, 1.1715]）、streaming 1.0186（CI [0.9647, 1.0759]）→ 统计平局 NEUTRAL，incumbent 保留 |
| （v0.1 未区分 dtype 的 fp32 最佳） | **REVISED** | v4 vs v2 fp32 主目标：**streaming 1.3666**（CI [1.3466, 1.3682]）/ **hot 1.6177**（CI [1.5922, 1.6292]），9/9 轮，均 KEEP → fp32 最佳 = v2_reg |
| （v0.1 未单独验证 H=8192） | **REVISED** | (128,8192) fp16：v4 vs v2 streaming **1.3803**（CI [1.3489, 1.3995]）KEEP → v4 在 H=8192 退化（per=32 寄存器压力），最佳 = v2_reg |
| 全部优化变体 vs baseline 大幅领先 | **CONFIRMED** | fp16 主目标 paired median（vX vs baseline）：0.5274 / 0.5680 / 0.5701 / 0.5781 → 倒数 **1.73–1.90×**，CI 均不跨 1.0，9/9 轮方向一致 |

`best.json`（v0.2 后）已同步：`v4_vec_reg`（主目标 fp16 hot，1350 MHz 语境，median 6.61 µs）+ `variant_fp32=v2_reg` + `variant_h8192_fp16=v2_reg`，reason 明确"1.231× 为 DVFS 混频膨胀，已 REVISED"。

## 局限与未决问题

1. **nvidia-smi 轮询是代理指标**：每 round 仅 3 次采样且位于 kernel 执行区间之外，非逐 kernel 时钟；本次 9/9 valid 证明"采样粒度内无 DVFS 分歧"，不能证明"kernel 执行瞬间时钟恒同"。实证盲区：(16,4096) fp32 hot round 2 的干扰事件在时钟恒 1350 MHz 下未被 guard 识别（#12）。
2. **ncu 锁频无正面证据**：v0.2 ncu 显式传 `--clock-control base`，但 raw 报告中既无锁频警告（`clock_lock_warnings={}`）也无锁频成功的正面确认；容器内 `nvidia-smi` `default_applications=[N/A]`，实际是否锁到 base clock 未验证（EXP-0008 `ncu_profile.note` 已如实记录）。缓解：全部变体同一 clock-control 设置下相对比较有效；ncu 数据只作方向性/机制性证据，不用于绝对性能判定。
3. **矩阵模式 round-ratio 对离群 round 敏感**：单轮瞬态干扰即可扭曲 winner-vs-runner 的 per-round ratio 与 CI（#12 实证）。矩阵 winner 表仅指示性，最终判定必须来自 paired A/B。
4. **streaming 工作集仍不足以考察 DRAM 带宽饱和区**：主目标 streaming 工作集 33.5 MB（16 × (1 MB x + 1 MB out) + 8 KB w），ncu dram% 峰值仅 42.52%（v2_reg cc=none），全部变体处于延迟/指令受限而非带宽饱和；矩阵最大 shape (1024,4096) kernel 也只有 ~33 µs。**对 M > 1024 的带宽扩展性结论未建立**，v0.2 结论限于已测 7 个 shape。
5. **M=1 区域为 launch-bound**：M=1 各格 5 变体中位数挤在 ~6.0–7.6 µs（kernel 本体远短于 launch/调度开销），差异普遍落在 CI 跨 1.0 的噪声带内，该区域的 winner（v3/v1/baseline 交替胜出）应视为平局而非真实优劣。
6. **结论条件于本次时钟状态**：v0.2 全部 run 恰运行在稳定 1350 MHz 低频态（guard 零拦截）；相对排名在该时钟状态下成立，未在其他 boost 状态下交叉验证（v0.1 的 1905 MHz 数据为唯一跨状态旁证，且证明排名对缓存模式敏感）。

## 总体结论

**PASS（带限定）**——12 条清单全部 PASS 或 PASS/CONCERN（#1 带宽指标口径、#6 时钟代理、#12 矩阵离群敏感三处 CONCERN 均已在数据文件中如实声明且带缓解措施），零 FAIL、零不可追溯数字：全部 43 个 JSON、ncu raw 输出、v0.1 对照数据与源码逐值核对一致，容差未被放宽、无 cherry-picking、无效轮机制透明、v0.1 的 1.231× 被同频 paired 数据证伪并正确 REVISED。
