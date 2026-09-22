# CUDALab — 状态

**日期：** 2026-09-21
**阶段：** v0.6 — INT8 Weight-Only GEMV（第五算子）
**状态：** 分支 `v0.6-qgemv`（基线 main = v0.5.1 = d635903），**不 merge 回 main**（用户指定）；停止条件全部达成：两层正确性（kernel vs 量化参考 50/50 × 5 变体，固定算术误差界 + finiteness 门；保真度 vs FP16 **report-only 不作门**）+ per-variant 负例 29/29 × 5（含 3 例逐位一致回退回归）、三口径（API / native kernel loop / NCU kernel duration）分开记录且一致性已调查（无冲突：API vs native <2.5%，NCU +14.8% = observed NCU profiling
perturbation（本 workload 观测差 ~4.4–4.7 µs，不声称跨 workload 固定常数）+
base 锁频贡献 ~0.8%，已按 v0.5 clkbase/clknone 受控对解释）、4 个自主实验（QGEMV-0001..0004，1 KEEP + 3 NEUTRAL，失败/中性实验全部保留）、全形状矩阵（向量化变体 10/10 全胜 baseline）、政策 incumbent `qgemv_vec16_row`（QGEMV-0001 KEEP 起未变）+ best-observed 候选 `qgemv_warp_vec16` 两次 fresh head-to-head CI95 同向（统计快 ~1%，policy NEUTRAL）+ 9-round 复核 **2.6937×**、五算子 smoke 回归 **all PASS**、最终报告 `docs/report_v0.6_result.md`（12 节，§10 含独立 review 结论与处置表）定稿；两个独立 subagent review（CUDA correctness + Benchmark methodology）——结论与处置见报告 §10；`v0.6-qgemv` 已 push 等外部 review。**不 merge main、不 force push。**

## v0.6 完成摘要（2026-09-21）

核心问题：**权重从 FP16 2B → INT8 1B 后，能否把 v0.5 已接近 DRAM
ceiling 的 GEMV 继续加速？** 用户指定 out of scope：INT4、group-wise、
GPTQ/AWQ、activation quant、Tensor Core GEMM。最终报告：
`docs/report_v0.6_result.md`（定稿，12 节，§10 含独立 review 结论与
逐条处置表）。

| 项目 | v0.6 结果 |
|---|---|
| QGEMV 算子 | `kernels/qgemv/` 5 变体：`qgemv_baseline`（标量，每行一 block 256 线程，两级归约）/ `qgemv_vec16_row`（16B 向量 load：uint4 int8×16，x 同步向量化）/ `qgemv_vec16_scale`（x·q 累加后行末一次 ×scale，省 K 次 fp32 mul）/ `qgemv_warp_vec16`（8 warp/block，warp-per-row，ILP=2 双累加器 stride 64 向量，5 步 warp shuffle，**无 shared/barrier**）/ `qgemv_warp_vec16_ilp4`（ILP=4 stride 128）；`qgemv_common.h` 单一来源 `qgemv_scalar_kernel`（回退与 baseline 位级一致）+ `qgemv_vec_contract_ok`。`W_q int8 [N,K]` + `scale fp32 [N]` + x fp16 `[K]` → y fp16 `[N]`，FP32 累加；每行对称 INT8（zero_point=0，`scale=amax/127`，`q=clamp(round(W/scale),±127)` round-half-even，scale=0 行安全）；**量化在池构造期完成、严格位于计时区外**（`cudalab/qgemv_quantize.py`）。主目标 (4096,4096)，5 形状 {(1024,4096),(4096,1024),(4096,4096),(11008,4096),(4096,11008)} |
| 对齐契约 | W_q 基址 16B ∧ x 基址 16B ∧ K%16==0；host 侧 launch 前显式检查，不满足 → 标量回退（不拒绝调用、无静默向量化路径）；负例含 3 例 per-variant 回退位级一致回归（W_q 错位 / x 错位 / K=13） |
| 正确性（层 a，门） | 5 变体各 **50/50**（5 输入模式 × (4 形状 + 4 边界 (1,1)/(1,4096)/(4096,1)/(16,13)/(8,7) 中 5 组合)；**固定算术误差界** tol = 2·(3K·2⁻²⁴·S_n + 0.5·ulp16(|y|) + 0.5·ulp16(|exact|)，S_n = Σ\|W_q\|·scale·\|x\| **双因子取绝对值**——开发期发现 x 未取绝对值 bug 已修复；finiteness 门 y ∧ ref；非 per-variant 调参） |
| 保真度（层 b，report-only 不作门） | 每元素误差 ≤ 0.5·量化步长（实测 0.500002 = round-half-even 上界，scale 无关）；主目标 normal：cos 0.9999617（5 形状 min 0.9999594）/ max_abs 2.57 / RMSE 0.551（\|y\| ≈ 4027）；suite 最差 mixed_sign：cos **0.9998970**；5 形状 × 2 模式全档记录（`quantization_fidelity.json`） |
| 负例 | 5 变体各 **29/29**（23 reject + 6 pass，含 3 位级一致回退回归 + 3 control；post_check_ok 验证 CUDA context 无污染） |
| 三口径 | 分开记录不混用（"算法带宽 ≠ 实测 DRAM BW"）。主目标 µs：API 84.96（baseline）/ **31.513**（warp_vec16）；native w200 104.575 / 32.185、w5000 84.389 / 32.103；NCU@base ccall 109.152（dram 27.18%）/ **36.16**（dram 87.24%）、ccnone 35.88（86.57%）。**一致性（用户要求冲突必查）**：API vs native 差 <2.5%；NCU +14.8% = observed NCU profiling perturbation（本 workload 观测差 ~4.4–4.7 µs，不声称跨 workload 固定常数；v0.5 ~3.8 µs 仅作量级参照，无专门 cross-workload 实验）+ base 锁频 ~0.8%（v0.5 受控对：DRAM-bound clkbase/clknone +0.77%、latency-bound +15.9%；负载下实测 boost 1815–1920 MHz；与 v0.5 口径模型同构，非测量冲突）；warp_vec16 无 post-idle DVFS ramp 效应（w200 ≈ w5000，DRAM 饱和对时钟不敏感，同 v0.5 vec4_row 行为） |
| Baseline NCU | (4096,4096) @base：kernel 109.152 µs、DRAM 27.18%、long_scoreboard 77.2%——标量 int8 load 延迟受限（每线程在途仅 1–2 B） |
| 自主优化实验 | **4/4**（paired v2.3 streaming 9r，链式 parent）：QGEMV-0001 `qgemv_vec16_row` **KEEP** 84.96→31.8–31.9 µs，**2.6614×** [2.657,2.669] 9/9（dram 27.2%→86.02%）；QGEMV-0002 `qgemv_vec16_scale` **NEUTRAL 0.9901×**（统计 SLOWER）[0.9885,0.9906] 0/9——假设证伪：计算削减反噬 occupancy（82.62%→68.42%），dequant 乘加在延迟路径外；QGEMV-0003 `qgemv_warp_vec16` **NEUTRAL 1.0107×（统计 FASTER）** [1.0099,1.0108] 9/9——barrier 7.3%→0、dram 87.24%、occ 90.43% 全变体最佳；QGEMV-0004 `qgemv_warp_vec16_ilp4` **NEUTRAL 0.9984×（统计 SLOWER）** [0.9977,0.9990] 0/9——ILP4 假设证伪（dram/occ 持平，regs 41→43，指令开销）。**政策 incumbent = `qgemv_vec16_row`**（自 QGEMV-0001 KEEP 起未变）；`qgemv_warp_vec16` = 主目标 best-observed 候选：两次 fresh head-to-head（QGEMV-0003 轮 1.0107× + final_incumbent 轮 1.0094× [1.0086,1.0101]）CI95 均排除 1.0 且方向一致 → 统计 FASTER 但仅 ~1%，**policy NEUTRAL**（未达 ≥5% 实质性阈值, 不替换 incumbent; 旧稿「CI95 显著性优先于 5% 政策带」为事后规则, 已移除, 两次数据保留为统计证据） |
| Winner NCU | warp_vec16（ccall @base）：DRAM **87.24%**、occ **90.43%**、barrier 0、long_scoreboard 84.1%（18.53 cyc/iss）、regs 41；ccnone：35.88 µs / 86.57%——DRAM 饱和形态 |
| 主目标 best-observed 复核 | fresh 9-round（`final_incumbent_reval.json`）：baseline 84.885 → warp_vec16 **31.513 µs = 2.6937×** CI95 [2.688,2.700] 9/9；算法带宽 533.4 GB/s = **86.6%** 规格峰值（616 GB/s），理想 27.29 µs 的 1.16× |
| INT8 vs FP16（核心问题） | 31.513 vs 59.933 µs（`gemv_vec4_row` 本 session 重测，vs 其 baseline 91.927 = 1.5343×）= **1.90×**——逻辑 IO 减半（16.81 vs 33.57 MB）的收益拿到 **95% of 理论 2.0×**；瓶颈仍在 DRAM（dram 27%→87.24%），dequant 乘加免费（QGEMV-0002 反证） |
| 全矩阵 | 5 形状 × {hot,streaming} × 4 向量化变体：vs baseline **2.2–2.9× 全胜 10/10**；长 K 格变体间差 ≤~2%，短 1-dim 格差异达 regime 级 2.05×。`classify_cell` 逐格重新派生, 三口径拆分（报告 §9.1, `v0.6_shape_winners.json` 已重新生成）：**最低观测中位数: warp_vec16 5/10、vec16_row 4/10（两个 11008 形状 CI 显著 +0.4–0.6%, **已披露**）、ilp4 1/10**；**统计显著更快（CI95 排除 1.00）: 8/10**（(1024,4096) streaming 与 (4096,1024) streaming UNRESOLVED——后一格 ilp4 最低中位数 7.355 但 CI [0.9817,1.0164] 跨 1.0, **不宣称 winner**）；**policy 显著胜者（≥5%）: 0/10**（全部格 policy NEUTRAL / NO_UNIQUE_WINNER, incumbent 不变）。(4096,1024) warp 对 vec16_row **2.0×** 结构性优势——短 K 下 block-per-row 75% 线程空转 |
| PyTorch 参照 | `torch.mv` 61.235 µs（fp16 W）/ 62.297 µs（dequant W），(4096,4096)——context only 不决策；INT8 QGEMV 31.5 µs ≈ 1.95–1.98× 于两者 |
| Evaluator | v2.3 decision/profiler 语义零改动；merge-review 轮仅改 `bench.py::analyze_shape_winners` —— 逐格 winner 派生统一委托 `experiment.classify_cell`（paired 聚合约定单一来源 + statistical_relation/policy_decision/status 双字段; 旧版 raw 侧"跨 round 中位数之比"混合口径废止; 测量/决策语义不变） |
| CPU 测试 | test_evaluator_v23_cpu 37/37（+1: analyze_shape_winners→classify_cell 委托钉死）、test_evaluator_cpu 18/18、test_softmax_cpu 20/20、test_dispatch 6/6（无需 GPU，推送前复验） |
| Smoke 回归 | **5 算子 all PASS**（`experiments/regression/v0.6/`，append-only）：gemv 100/100×4 + 24/24×4；rmsnorm 76/76×5 + 29/30×5；softmax 72/72×4 + 14/15×4；rope 384/384×5 + 36/37×5；qgemv 50/50×5 + 29/29×5（每算子 skipped=1 为预存环境 skip，all_pass 仍 True）；隔离审计：gemv_splitk4 / softmax_hsplit2 正常变体列表缺席、all_variants 在列，qgemv 隔离集为空（机制保留） |
| 独立 review | 2 个独立 subagent（CUDA correctness + Benchmark methodology）——结论与逐条处置见报告 §10 |
| git | 分支 `v0.6-qgemv`（12 commit）：4fab683（算子 + 量化器 + 适配 + 50/50×5 + 29/29×5）→ 7ff05db（baseline Phase 4 记录）→ c62cdd6（QGEMV-0001/0002）→ 3e4009a（QGEMV-0003）→ d52f0b2（QGEMV-0004，4/4 实验完）→ 10c3000（final incumbent + 复核 + 三口径）→ 5e5880c（FP16 参照重测）→ 5d80d4e（全矩阵 + winner）→ 5fa4e4b（五算子回归 + 隔离审计）→ 9f6f3ec（报告定稿 + README/STATUS + review 处置）→ (merge-review 轮 2026-09-21, 2 commit) fix: restore qgemv incumbent policy semantics（政策 incumbent = vec16_row; 移除事后「CI95 优先」规则; NCU 措辞改 observed perturbation; native_timing 元数据）+ fix: regenerate shape winners with paired classification（analyze_shape_winners 委托 classify_cell; `v0.6_shape_winners.json` 纯派生重新生成, raw 记录零改动; +1 CPU 测试）；**不 merge main、不 force push** |
| 未做 / v0.7 | INT4、group-wise 量化（报告 §11 Q7 结论：per-row scale 在 4-bit 下保真度不足，group-wise(128/256) 为共需项；IO 16.81 → 8.93 MB / 下界 27.29 → 14.50 µs，理论 1.88×，效率折损后预期再 ~1.7–1.8×，值得作为单个 v0.7 项目）、GEMM、Attention、CUDALM 集成（用户指定 out of scope） |

## v0.5 完成摘要（2026-09-21）

核心问题：**闭环能否迁移到带宽受限的第四算子（FP16 GEMV，y = W @ x，
W [N,K]、x [K]、y [N]，FP32 累加），并在三口径计时（API 路径 / native
kernel loop / NCU kernel duration）下给出一致、可审计的结论。** 用户
指定 out of scope：不做 GEMM / quantization / Attention / CUDALM 集成。
最终报告：`docs/report_v0.5_result.md`（定稿，12 节，§12 含独立 review
结论与逐条处置表）。

| 项目 | v0.5 结果 |
|---|---|
| GEMV 算子 | `kernels/gemv/` 5 变体：`gemv_baseline`（标量，每行一 block）/ `gemv_vec4_row`（fp16x8 向量化 load + fp32 累加）/ `gemv_warp_vec4_b256` / `gemv_warp_vec4_b512`（warp 级列切分）/ `gemv_splitk4`（partials + combine 双 kernel）；`gemv_common.h` 单一来源 `gemv_scalar_kernel`（回退与 baseline 位级一致）+ `gemv_vec_contract_ok`。FP16 主 + FP32 顺带；主目标 (4096,4096) fp16，5 形状 {(1024,4096),(4096,1024),(4096,4096),(11008,4096),(4096,11008)} |
| 对齐契约 | W 基址 16B ∧ x 基址 16B ∧ K%epv==0（fp16 epv=8 / fp32 epv=4）；host 侧 launch 前显式检查，不满足 → 标量回退（不拒绝调用、无静默向量化路径）；负例含 3 例 per-variant 回退位级一致回归 |
| 正确性 | 5 变体各 **100/100**（固定容差非 per-variant；random/zeros/small/large/mixed-sign × 5 个 N/K；max_abs/max_rel/NaN-Inf 门）。审查 MINOR-2：mixed_sign 原为 (-1)^k 零抵消，x 改独立 Bernoulli 符号流后全套重录（100/100 ×5）。max_arith 5 变体一致 0.249954783156（K=1 元素算术界）；max_abs 4.0/8.0 = 1 ulp @ |y|∈[4096,8192)/[8192,16384) binade；max_rel 0.0124–0.0217 |
| 负例 | 5 变体各 **24/24**（18 reject + 6 pass，含 3 位级一致回退）。审查 **MAJOR-1**（negative 套件此前从未 per-variant 运行，CLI 只调用 baseline 默认）→ `run_negative(ext, variant)` 管线修复（base.py 协议 + gemv.py + 统一 CLI test/optimize + revalidate 脚本）+ 5 变体重跑归档（`invalid_inputs.json` + 4 × `invalid_inputs_<variant>.json`）+ GEMV-0001..0004 记录 additive `negative_note`（原数字不变） |
| 三口径 | 分开记录不混用（审查 MINOR-7："<4%" 表述与自身数据矛盾，已改 "API vs native-w5000 <1.5%；NCU@none 残余 ~7%"）。主目标 baseline/vec4_row（µs）：API 93.88/59.89；native w200 109.824/60.416、w5000 91.353/59.649；NCU@base 114.76/64.192、NCU@none 98.848/63.704。**冲突根因 = post-idle DVFS ramp**（base 1350 MHz → boost 1890 MHz；native w200 warmup ~22ms 落在 ramp 内：baseline w200 109.153 µs @1350 MHz vs w5000 91.339 µs @1890 MHz，probe 核实）；NCU@none ≈ API+7% = cache flush + profiling 隔离（已解释）。DVFS probe 采样稀疏（n=1 时钟样本/相位）已披露（§5.2/§11.10）；baseline NCU@base 与 @none 剖自 75c1ccd 重构前后不同二进制、代码逐行相同（审查 diff 核实）已披露（§5） |
| Baseline NCU | (4096,4096) fp16 @base：kernel 114.76 µs、DRAM 49.34%（逻辑 ~304 GB/s）、long_scoreboard 79.3% —— 标量 load 延迟受限形态 |
| 自主优化实验 | **4/4**（paired v2.3 streaming 9r，parent=gemv_baseline，NCU 证据驱动）：GEMV-0001 `gemv_vec4_row` **KEEP** 92.34→59.890 µs，1.5416 [1.5398,1.5517]；GEMV-0002 `gemv_warp_vec4_b256` KEEP 1.2539（NCU lg_throttle 84.4% / 61 regs / occ 81.25%）；GEMV-0003 `gemv_warp_vec4_b512` KEEP 1.2407；GEMV-0004 `gemv_splitk4` **REJECT** 0.8784（partials 132.128 µs DRAM 42.96% + combine 2.432 µs；barrier stall 71.9%/71.2% @ccall、72.4%/72.1% @ccnone；失败实验保留）。**incumbent = `gemv_vec4_row`** |
| Winner NCU | vec4_row DRAM 87.89%（@none 90.2%）vs baseline 49.34% —— 带宽受限形态；主目标 ~91% 2080 Ti 616 GB/s 规格峰值（33,570,816 B 算法流量，理想 @616 GB/s ≈ 54.5 µs，实测 59.89 µs） |
| 全矩阵 | 5 形状 × {fp16,fp32} × {hot,streaming}（20 full + 20 base）：**gemv_vec4_row 10 格 fp16 + 10 格 fp32 全胜**（fp16 1.51–1.63×、fp32 1.06–1.11× vs baseline）；(11008,4096)/(4096,11008) 575–576 GB/s（93.4–93.5%）；fp32 492–548 GB/s |
| 最终复核 | v1（已发布，`gemv_vec4_row_revalidation.json`）：streaming 92.754→59.879 CI [1.5406,1.5521] KEEP / hot 92.736→59.840 CI [1.5422,1.5501] KEEP + torch.mv 61.229 µs。**v2**（审查后新进程、新文件 `_revalidation_v2.json`，含 incumbent per-variant 负例 + mixed_sign 修复后正确性）：streaming 91.826→59.840 CI [1.5266,1.5378] KEEP / hot 92.809→59.779 CI [1.5521,1.5602] KEEP；run 间点估计漂移 ~1% 属已确立机器态特性，结论只用 run 内 paired 比值 |
| PyTorch 参照 | torch.mv 61.2 µs（主目标，context only 不决策）——incumbent 59.8–59.9 µs ≈ 打平 |
| Evaluator | v2.3 decision/bench **自 4eb520b 起逐字节未动**（未提前变成 evaluator 重构项目）；唯一 evaluator 改动 = `profiler.py` additive（NCU 多 kernel summary 修复：kernels[] + multi_kernel_note，splitk4 双 kernel 解析；CPU 36/36 无回归，解析器 CPU 单测缺失登记为 §11.12 gap） |
| CPU 测试 | test_evaluator_v23_cpu 36/36、test_evaluator_cpu 18/18、test_softmax_cpu 20/20、test_dispatch 6/6（无需 GPU） |
| Smoke 回归 | **6/6 PASS**（rmsnorm baseline / v4_vec_reg 76/76 + 29/30；softmax baseline / vec4 72/72 + 14/15 ×2；rope baseline / v3_half2 384/384 + 表核对 + 36/37 ×2；每算子 skipped=1 为预存环境 skip，跨变体一致）；重录各算子 correctness/negative 记录。此前 rmsnorm 挂起事故（orphaned FileBaton lock）已 root-cause 并修复，见"值得注意的事故" |
| 独立 review | 2 个独立 subagent 双双 **PASS WITH CAVEATS**（CUDA：1 MAJOR + 2 MINOR + 2 NIT；Benchmark：0 MAJOR + 5 MINOR + 8 NIT）；无记录造假类发现（byte 级核对 84 个新增 0 修改记录文件 + git 全分支 diff + 独立复算 4 个 pair 中位数 / GEMV-0001 bootstrap CI / 矩阵抽核格）。全部 findings 处置（报告 §12 处置表）：per-variant 负例归档、mixed_sign 修复、报告数字修正（33,570,816 B / 304 GB/s / 1.54× / barrier 71.9% 等 11 项）。历史 benchmark/profile JSON 逐字节未改；修正 = 报告修正 + 重录 correctness/negative + 实验记录 additive note（v0.4 先例） |
| git | 分支 `v0.5-gemv`（9 commit）：50d6c0c（GEMV 算子）→ 4521540（make_bench_pool 未定义 M 修复）→ de150bc（baseline Phase 4 记录）→ 75c1ccd（候选内核 + 负例 + 回退）→ 652f4f1（GEMV-0001..0004）→ 25bd9ab（native + NCU + 多 kernel 修复 + 口径）→ 2e4836f（全矩阵）→ 8f2b944（复核 v1）→ (末) 本报告 + README/STATUS + 独立审查处置；**不 merge main、不 force push** |
| 未做 / v0.6 | GEMM、quantization、Attention、CUDALM 集成（用户指定 out of scope）；v0.6 仅建议：**Quantized GEMV**（用户指定优先级），另见报告 §10 |

## v0.5 Merge Review 修复摘要（2026-09-21）

外部 merge review 后的 4 项修复（只修 finding：不新增 kernel、不重跑
full matrix、不做 Quantized GEMV；3 个 fix commit，见报告 §12
"v0.5 merge review 处置"）：

| # | 修复 | 处置 |
|---|---|---|
| 1 | **splitk4 隔离**（MAJOR）：`static at::Tensor g_splitk_partials` 进程级 workspace 多 stream 并发 race + 跨 device workspace 设备风险 | `kernels/gemv/bindings.cpp` 新增 `quarantined_set = {"gemv_splitk4"}`（UNSAFE_HISTORICAL_EXPERIMENT / REJECTED / NOT_FOR_NORMAL_DISPATCH）：`ext.variants()` 正常列表移除、`quarantined_variants()` 新入口、未知变体错误列出隔离变体；CLI test/benchmark/optimize/profile 全路径拒绝（既有门禁 + bench 引擎 `_require_normal_variant`）；`gemv_splitk4.cu` 头注释隔离标记（**不重写内核**）；GEMV-0004.json 追加 `quarantine_note`；显式 `forward("gemv_splitk4", ...)` 保留为受控历史审计入口 |
| 2 | **历史 artifact 不可变**（MAJOR）：v0.5 smoke 误覆盖 7 个 main 历史 correctness 记录（rmsnorm v0.3_regression ×3 / softmax v0.3 ×3 / rope v0.4 ×1） | 7 个文件按 main（4eb520b）版本恢复；本轮 smoke 结果迁移至新目录 `experiments/regression/v0.5/{rmsnorm,softmax,rope,gemv}/`（append-only，含 README 来源说明 + 不可变约定）；rmsnorm/softmax/rope 三算子默认输出目录改指 regression 目录；隔离后 4 正常变体的 GEMV 再验证存 `experiments/regression/v0.5/gemv/`（官方 v0.5 记录不动） |
| 3 | **negative 套件 variant 语义统一**（MINOR）：Softmax/RoPE `run_negative` 忽略 variant（CLI 指定候选未被实际测试）；RMSNorm cross-variant 无显式声明 | Softmax/RoPE 改 per-variant（套件本体早已参数化，补管线 + per-variant 归档命名 + `--neg-out-dir`）；4 个套件记录新增 `negative_suite_scope`（GEMV/Softmax/RoPE = **per-variant**；RMSNorm = **cross-variant**，docstring 更新，不再描述为 per-variant）；base.py 协议 docstring 更新 |
| 4 | **文档小修正**（NIT）：`2^-24 ≈ 6.1e-5` 错误；错位 W/x 用例被错误描述为 splitk4 的"标量回退契约" | `gemv_correctness.py` docstring 改 **2^-24 ≈ 5.96e-8**（半步 ≈3e-8；6.1e-5 = 2^-14 是次正规**边界**，rope 处用法正确未动）；明确**对齐不是 splitk4 的约束**（标量访存，唯一契约 K%4==0）：`gemv_negative.py` 按 variant 重构（splitk4 下 2 个错位用例 = finite-only control、K=13 用例保留 bit-identical = K%4 契约回退）、归档 `invalid_inputs_gemv_splitk4.json` 追加 `note_addendum`、报告 §3 表格更新 |

验证（2026-09-21，GPU 0）：GEMV 4 正常变体（baseline / vec4_row /
warp b256 / warp b512）正确性 100/100 + 负例 24/24（append 至
`experiments/regression/v0.5/gemv/`）；Softmax incumbent `softmax_vec4`
负例 PASS（per-variant 首跑）；RoPE incumbent `rope_v3_half2` 负例
PASS；RMSNorm cross-variant 负例 PASS；CLI 对 `gemv_splitk4` 的
test/benchmark/optimize/profile 全部拒绝（隔离消息）+ 显式 forward
审计入口仍可用 + `variants()`/`quarantined_variants()` 列表正确；
CPU evaluator 测试 36/36（+18/18、20/20、6/6）。**不重跑 GEMV full
matrix / GEMV-0001 / NCU**；incumbent 仍为 `gemv_vec4_row`（隔离不
改变任何决策证据，splitk4 本为 REJECT）。

## v0.4 完成摘要（2026-09-20）

核心问题：**(1) evaluator 从 v2.2 → v2.3（对称 guard + raw/filtered 双轨 +
filter-sensitivity）**；**(2) 闭环迁移到第三算子 interleaved RoPE**。
真实目标排序：evaluator v2.3 更可信 > 第三算子自然接入 > agent 从
profiler 证据形成有效实验——**不追求 RoPE 一定优化成功**（baseline 近
下限时全 NEUTRAL 是 PASS，不是失败）。最终报告：`docs/report_v0.4_result.md`
（定稿，含独立 review 与 Lead 审计 §17）。

| 项目 | v0.4 结果 |
|---|---|
| Evaluator v2.3 | `paired-streaming-v2.3`（`cudalab/evaluator/`）：对称 log 空间 guard（\|log(t/ref)\|>log(F)：spike 1.5× / cross-block 1.15×，快慢同因子，parent/candidate 完全同规则）+ raw/filtered 双轨记录（每 round raw/filtered 中位数 + raw_speedup + rejected_samples{fast,slow} + environment_guard 自描述块）+ filter-sensitivity（方向翻转或 \|log(filtered/raw)\|>log(1.10) → 敏感；敏感 → `apply_filter_gate` 把最终 policy_decision 一律降级 UNSTABLE 并记录 original_decision——**v0.4.1 起 KEEP/REJECT/NEUTRAL 均降级**，v0.4 实际记录 0 敏感、无历史判定受影响）。guard 逻辑纯 CPU 函数（stats.apply_spike_guard/block_stats/crossblock_flag/filter_sensitive）+ tests/test_evaluator_v23_cpu.py **36/36**（review 后 +1：raw 侧聚合约定钉死；v0.4.1 +7：gate 收紧 2（NEUTRAL+敏感 → UNSTABLE 双向必测）+ statistical_relation/policy_decision 形式分离 5）。详见 `docs/evaluator_v2_3.md` |
| v2.3 回归硬门 | **PASS**（RoPE 之前，`benchmarks/v2.3_regression/`，4 条 pair + gate_summary，全部 9/9 valid）：Softmax baseline vs vec4 streaming **1.6745** [1.6727,1.6793] 9/9 更快（v2.2 参考 1.6772/1.6890 精确复现，raw=filtered，rejected 0/0）；RMSNorm v4 vs v1 streaming 1.0375 [1.0279,1.0990] → 复跑 0.9576 [0.9509,0.9623]（数分钟内方向翻转，调查定性为**环境微态漂移、非 evaluator 缺陷**：raw==filtered、对称 guard 全程可审计、Softmax 对照精确复现、idle 微态漂移有前科——四项证据见 evaluator_v2_3.md §6）；hot 0.9923 [0.9893,1.0106] 带内 |
| RoPE 算子 | interleaved RoPE（a=x[2i], b=x[2i+1], c=cos[pos,i], s=sin[pos,i]；y[2i]=a*c−b*s, y[2i+1]=a*s+b*c；FP32 中间，输出原 dtype；base=10000，max_seq_len=4096）。`kernels/rope/` 5 变体（baseline + v1_2pair + v2_4pair + v3_half2 + v4_8pair）；`cudalab/operators/rope.py` adapter（9 形状矩阵 (1,64)…(4096,128)，主目标 (1024,128)，16 缓冲池 + 共享 cos/sin 2 MiB，NCU driver，rope_ref）；FP16 主 + FP32，禁 BF16；sm_75 / CUDA 11.8 |
| 正确性 / 负例 | baseline + 4 候选全部 **384/384**（finiteness + double-rounding 算术界 K=2 vs fp64 精确旋转 + norm 保持；与 torch 参考 allclose **报告不门控**——fp32 大值抵消 FMA 工件已文档化）；新增**独立表值核对门**（表 vs fp64 独立求值全网格，固定误差界，§17）；negative **36/37 执行、all_pass=true**（37 例 = 31 基础（28 必须拒绝 + 2 预期 PASS 对照 + 1 跳过）+ 3 per-variant 整除性（review 新增）+ 3 v0.4.1 half2 对齐回归（v0.4.1 新增）；执行 = 31 reject_ok + 5 pass_ok；launch 前 TORCH_CHECK + 启动后 C10_CUDA_KERNEL_LAUNCH_CHECK） |
| 同步验证修复 | **首跑 baseline 28.8/29.8 µs 定位为验证路径缺陷**（positions 值域检查的同步 D2H 拷贝逐 launch 强制流同步，~25–30 µs）。修复：验证拆为 meta（host 元数据，始终）+ range（`validate` 门控，默认 true）；forward/forward_into 新增 validate 参数；benchmark pool + NCU driver 传 validate=False（池契约文档化）；正确性/negative/正常调用保持默认。修正后 baseline (1024,128) fp16：**streaming 6.981 µs / 113.8 GB/s、hot 6.637 µs / 119.7 GB/s**（9/9 × 双模式）；修正前记录保留 `*_presyncfix_archive.json`（审计痕迹，不删除） |
| Baseline NCU | (1024,128) fp16，ccall+ccnone，--clock-control base（1755MHz）：kernel 4.0 µs、dram 23.25%、sm 11.61%、occupancy 78%、long_scoreboard 69.3% —— 稳态流内 launch 发射速率受限形态 |
| 自主优化实验 | **4/4，全部 NEUTRAL**（paired v2.3，(1024,128) fp16 streaming，parent=rope_baseline；NCU 证据驱动设计）：ROPE-0001 `rope_v1_2pair` 1.0000 [0.9849,1.0160]（NCU 单 launch −7.6% 但稳态流内无效——kernel 时长不是瓶颈）；ROPE-0002 `rope_v2_4pair` 1.0010 [0.9331,1.0211]（NCU +25%，occupancy 21.5%，波坍缩开始）；ROPE-0003 `rope_v3_half2` 0.9974 [0.9888,1.0025]（指令数削减控制——**成功的阴性对照**）；ROPE-0004 `rope_v4_8pair` 0.9922 [0.9861,1.0055]，rejected fast=39（review 后更正定性：r1 锚定偏差计数——~60 个环境慢态样本锚定块中位数，恢复后 ~40 个合法稳态样本被拒为 fast spike；pair 判定稳健（剔 r1 → 0.9933）、filter_sensitive=false；真实 fast 侧 guard 行为展示在矩阵 11 个 cross-block flag，报告 §9/§11/§14.4；NCU +104%，occupancy 11.9%，波坍缩灾难区）。**结论**（v0.4.1 限定范围）：在当前 Python → pybind → PyTorch C++ extension → CUDA launch 的 benchmark submission path 下，主目标表现出明显 launch/host-issuance sensitivity（paired API-path ≈ 6.4 µs vs ≈ 6.4 µs；NCU kernel-only baseline ≈ 4.00 µs, v1 ≈ 3.70 µs）；因此不能直接推断: 未来原生 C++ CUDALM 中 v1 也无收益。MLP 甜区 1–2 pairs/thread；≥4 pairs 波坍缩。NCU 诊断 vs paired 决策分工成立（v1_2pair 的 NCU −7.6% 未转化为流内 ≥5% 优势，NEUTRAL 是正确决策）。**全 NEUTRAL = PASS 结局** |
| 全矩阵 | 36 格（9 形状 × {fp16,fp32} × {hot,streaming}）× 5 变体（全部候选保留；v2/v4 的 D%8==0 / D%16==0 约束在 9 形状上全部满足）+ shape winners（`benchmarks/rope/rope_v04_matrix_*`）。31/36 格 9/9 valid、4 格 8/9、1 格 7/9（拒轮透明记录，全部 ≥7）；fast cross-block flag 共 11 个（2 个 M1_H64 streaming 格 3 轮，全部整轮作废 → 零 variant 偏置，报告 §11）；**0 格 filter-sensitive**；**无 policy 层面唯一胜出格**（36 格 winner/runner-up 比值 0.994–1.033，全部 <1.05 KEEP 线；per-cell winner 分布 baseline 16 / v1_2pair 15 / v2 3 / v3 2 / v4 0，仅指示性）。主目标 (1024,128) fp16 矩阵值（run 内）：streaming baseline 6.540 µs / hot 6.214 µs；PyTorch 2.4.1 **无内置 fused RoPE op** → Python 参考（fp16 262.1 µs / fp32 180.3 µs，`pytorch_ref_M1024_H128.json`）仅作 implementation context，不产生 "X× faster than PyTorch" headline |
| CPU 测试 | test_evaluator_v23_cpu 36/36（v0.4.1）、test_evaluator_cpu 18/18、test_softmax_cpu 20/20、test_dispatch 6/6（无需 GPU） |
| git | 分支 `v0.4-rope`（8 commit）：a591447（eval v2.3）→ c051c96（v2.3 回归门记录 + RoPE 算子/baseline/套件）→ 76a1546（validate 拆分 + 重基准）→ 4bf5f67（evaluator_v2_3.md 回归结果）→ f30d805（ROPE-0001..0004 实验）→ 85a7f24（全矩阵 + PyTorch context + v0.4 文档）→ review 修复 commit（见报告 §15）；**不 merge main、不 push 到 main** |
| 独立 review + 最终审计 | 3 个独立 subagent review 全部交付并处置：**CUDA Correctness = PASS WITH CAVEATS**（2 minor：kernel 头注释 load 计数算术错误 v1/v2/v4 → 已修；v3 "位级同数学" 过度声明 → fp32 路径改为不做位级声明）；**Benchmark Methodology = PASS**（"evaluator v2.3 对称、每记录判定可完全复现；修 4 项 docs/next-commit 级 finding、无需重录" → raw 侧聚合约定与 filtered 侧对齐 + 钉死测试）；**RoPE Math = PASS WITH CAVEATS**（F1 表构造约定措辞 → 保留 FP32 构造 + 新增独立表值核对门；F2–F5 界推导/ulp 文档/舍入位 → 已修）；Lead 最终审计（seed 20260919 复现 8 pair 记录、报告数字交叉核对、对称规则 bit 级核对 v2.2、D2H 修复量级核对、记录完备性）通过。修复后重录：正确性 5×384/384 + 表核对全过、negative 33/34 all_pass（2026-09-20T23:42）；CPU 套件 29/29/18/20/6。working tree clean，`v0.4-rope` 已 push（**不 merge main**） |

## v0.4.1 Merge Fix 摘要（2026-09-21）

外部 review 后的 4 项合并修复（只修 finding：不加 RoPE variant、不做新
优化、不开始 GEMV、不重跑 full 36-cell matrix）：

| # | 修复 | 处置 |
|---|---|---|
| 1 | `rope_v3_half2` fp16 路径 `reinterpret_cast<const __half2*>` 需要 4B 基指针对齐，`is_contiguous()` 不保证（view 奇数 half 存储偏移即可触发） | 显式检查 x/out 基指针 4B 对齐；未对齐 → 回退到与 baseline 逐语句同数学的标量 fp16 kernel（位级一致），不拒绝调用。负例套件 +3 例对齐回归（x 2B 未对齐回退 / x 4B 对齐走 half2 / out 2B 未对齐经 forward_into 回退），37 例 36/37 all_pass |
| 2 | FILTER_SENSITIVE gate 收紧 | `filter_sensitive == true` → 最终 policy_decision 一律 UNSTABLE（KEEP/REJECT/NEUTRAL 均降级；原决策记 original_decision + filter_sensitive_reason）。v0.4 全部记录 0 敏感 → 历史判定零影响；CPU 双向必测（raw 快 / raw 慢 → NEUTRAL+敏感 → UNSTABLE） |
| 3 | statistical_relation / policy_decision 形式分离 | schema 分别记录 `statistical_relation`（FASTER/SLOWER/UNRESOLVED，只基于 CI95：下界>1 / 上界<1 / 否则）与 `policy_decision`（KEEP/REJECT/NEUTRAL/UNSTABLE；5% 阈值只影响后者）。decision 纯函数 + classify_cell 双字段 + CLI 双输出 + ROPE-0001..0004 元数据追加（全部 UNRESOLVED/NEUTRAL，由已存 CI95 推导，原始数字未动）+ experiment.py "NEUTRAL = 无统计显著差异" 错误措辞更正 |
| 4 | 文档范围限定 | (a) launch-bound 结论不再泛化为"RoPE 已达到 kernel launch 下限"：限定在当前 benchmark submission path 下的 launch/host-issuance sensitivity，附 NCU kernel-only（baseline ≈ 4.00 µs, v1 ≈ 3.70 µs）vs paired API-path（≈ 6.4 µs vs ≈ 6.4 µs）双口径 + "不能外推到未来原生 C++ CUDALM"；(b) `operators/rope.py` wave 计数笔误更正（65536/(30×2048) ≈ 1.07 waves ≈ 107%） |

验证（2026-09-21）：v3 正确性 384/384 + 表核对 True（重录
`experiments/rope/correctness/v0.4/rope_v3_half2.json`）；负例 36/37
all_pass（重录 `invalid_inputs.json`）；baseline/v3 smoke pair
(1024,128) fp16 streaming 9/9、0.969977 [0.959281, 0.977719]、
raw==filtered、filter_sensitive=false（harness 完整性检查，非
ROPE-0003 重跑；跨 run 绝对时间不可比）；CPU 36/36 + 18/18 + 20/20
+ 6/6。ROPE-0001..0004 原始 benchmark/profile 数字未修改；full matrix
未重跑；历史 benchmark/profile JSON 逐字节未改。3 commit（fix: harden
rope half2 alignment / fix: make filter-sensitive decisions unstable /
docs: separate statistical relation and policy semantics），`v0.4-rope`
已 push，**不 merge main，等待外部最终 merge review**。

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
| best.json | `experiments/softmax/best.json`（classify_cell，v0.2.1 语义；v0.3.1 语义澄清）：36 格全部 decision=NEUTRAL → **NO_UNIQUE_WINNER**（winner/runner-up 比值 0.9972–1.0479，全部 <1.05 KEEP 线）；17 格 INCUMBENT 标签（vec4 在 top-2，被 policy 保留）/ 19 NO_UNIQUE_WINNER。"无唯一胜出者"是策略层面结论，不是"统计平局"断言（36 格中 21 格 winner 对 runner-up CI95 > 1.0，统计显著但 <5%；详见报告 Q4 与 summary.note） |
| NCU | baseline vs 4 候选（双 cache-control，v0.2.1 语义）+ incumbent 复验（<1% 漂移）；per-launch µs 与 NCU dur 的时钟/L2 语义差异已在报告说明 |
| 独立审计 | `docs/benchmark_audit_v0.3.md`：**PASS WITH CAVEATS**（数据逐项可复现、决策与规则一致、重构忠实；caveat #1 SFM-0001 hot 漂移已在 SFM-0001.md §6 + 报告披露，caveat #3b 日期笔误已修正） |
| 机器态限定 | (128,4096) fp16 **hot** 模式跨运行配对结果漂移（vec4 vs baseline hot 两次 committed 运行：1.2916 @21:03（SFM-0001 主 paired）→ 0.9865 @21:41（final_reval），约 40 分钟内跨越 KEEP 线；streaming 1.68 稳定）；RMSNorm 回归 hot 1.0144 NEUTRAL（v0.2 为 0.9327 REJECT）/ streaming 0.9419 REJECT（v0.2 为 0.9592 NEUTRAL）——点估计漂移、机制不变，定位为机器态而非 evaluator 缺陷 |
| 未做 | dispatcher 默认不做（无 paired 确认的 per-shape 路由证据）；不 merge main；不开始 v0.4 |

## v0.2 完成摘要（2026-09-19）

分支 `v0.2-evaluator-hardening`，基线 `main` @ 94179b6（v0.1 完成状态），
10 个阶段全部完成，10 个增量提交。v0.1 数据（EXP-0001…0007、`benchmarks/` 顶层、
`profiles/rmsnorm/`、`correctness/` 顶层）原样保留；`best_v0.1.json` 存档了
v0.1 的 best，`best.json` 为 v0.2 当前最佳。

| 项目 | v0.2 结果 |
|---|---|
| 正确性 | 5/5 变体 76/76 PASS（`correctness/v0.2/`）+ 负例套件 29/30 符合预期（1 多 GPU 用例单卡环境跳过） |
| 基准 harness | `paired-streaming-v2`：配对 A/B 交替、DVFS guard（>5% 拒轮）、hot/streaming 双模式、预分配 16-buffer 池（streaming 是否 > L2 取决于 shape，以 `working_set_gt_l2` 为准；v0.2 主形状 (128,4096) fp16 = 33.5 MB > 5.5 MB L2） |
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
- `algorithmic_bw_gbps` 为逻辑算法流量（算法 IO / 时间），非实测 DRAM 带宽，
  不能由它断言"饱和 / 未达饱和"（v0.3.1 更正：2080 Ti 规格峰值 616 GB/s）；
  M=1 区域 launch-bound。
- compute-sanitizer 不可用（未做越界/竞态检查）。
- 分发表（v0.2.1）仅在 3 个实测格路由优化变体（2 个 paired-evidence + 1 个
  incumbent-fallback），其余实测/未实测组合一律 baseline（evidence > coverage）。
- v0.1/v0.2 数字跨版本不可直接比较（harness/时钟/缓存策略均不同）。
- DVFS probe 采样稀疏（nvidia-smi 启动开销 → 有效采样周期 ~100ms，每相位
  n=1 时钟样本）：口径冲突调查中时间比为主证据、时钟采样为辅（v0.5 披露，
  报告 §5.2/§11.10）。
- NCU 多 kernel summary（splitk4 类双 kernel launch）解析器无 CPU 单测
  （v0.5 报告 §11.12 登记 gap；该路径由 GPU 记录逐条人工核对过）。

## 值得注意的事故（已记录，未隐藏）

- **EXP-0002**（v0.1）：单发事件计时（约 6 µs 启动噪声）把 v1 误判 REJECT，
  与 ncu 结论相反。修复 → `cuda-event-batched-v1`；EXP-0001/0002 标记 superseded。
- **EXP-0007**（v0.1 → v0.2 复核）：batched 框架下变体间非配对测量遭遇 DVFS
  混频（v1@~1350 MHz vs v4@~1905 MHz），1.231× 加速比虚高。v0.2 paired
  harness + DVFS guard 复验后 REVISED 为 1.011×（streaming）/ v4 快约 7%（hot，median 0.9327）。
  两条教训共同支撑客观层设计：独立于智能体、完整保留作废记录、结论可复现可审计。
- **rmsnorm smoke 挂起**（v0.5，2026-09-21）：smoke 回归首跑
  `test rmsnorm --variant baseline` 挂起 56+ 分钟、GPU 0% 且无进程记录。
  root-cause（faulthandler step-through 逐段定位）= `/root/.cache/torch_extensions/
  cudalab_rmsnorm/lock` 的 **orphaned 0 字节 FileBaton lock**：先前一个构建进程
  被 SIGKILL 未释放锁，`FileBaton.wait()` 为 `while os.path.exists: sleep(0.1)`
  无限等待，进程卡在 `cpp_extension.load` 路径（还没到任何 CUDA 调用），与
  "9 秒 CPU / 56 分钟"、GPU 空闲、新 context 可用等全部症状吻合。lsof 确认
  无 fd 持有者后删锁；同路径 87 s 完成（84 s 冷编译 + 正确性 76/76 + 负例
  all_pass）。教训：未来 torch_extensions 构建挂起，先查 orphaned `lock`
  baton 文件（`lsof` 验证无持有者再 `rm -f`）。
