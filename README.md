# CUDALab

**自主 CUDA 内核优化实验室。**

CUDALab 闭环自动化内核优化：

```
参考实现 → 正确性 → 基准测试 → GPU 剖析 → 瓶颈分析
→ 优化假设 → 内核修改 → 编译 → 正确性
→ 基准测试 → 采纳 / 拒绝 → 实验记录
```

外层 LLM 智能体（开发者的编码代理）提供优化假设与内核代码；**客观、非 LLM 的评估层** —— 正确性校验框架、配对 CUDA 事件基准框架（含 DVFS guard）、Nsight Compute 集成、以及固定的 KEEP/REJECT/NEUTRAL/UNSTABLE 判定规则 —— 提供证据。智能体不能自封胜者；只有框架的数字才算数。见 [docs/design.md](docs/design.md)。

## 当前状态：v0.6（INT8 Weight-Only GEMV）

v0.6 回答一个问题：**权重从 FP16 2B → INT8 1B 后，能否把 v0.5 已接近
DRAM ceiling 的 GEMV 继续加速？** 分支 `v0.6-qgemv`（基线 main =
v0.5.1 = d635903），**不 merge 回 main**（用户指定）。最终报告：
`docs/report_v0.6_result.md`。

**CUDALab Operators：RMSNorm / Softmax / RoPE / GEMV / QGEMV**。QGEMV：
`y = (W_q · scale) @ x`（`W_q int8 [N,K]` 行主序 + `scale fp32 [N]`，
x fp16 `[K]`，y fp16 `[N]`，**FP32 累加**；每行对称 INT8，zero_point=0，
`scale[n] = max|W[n,:]|/127`，`q = clamp(round(W/scale), ±127)`，
round-half-even，scale=0 行安全；**量化在池构造期完成、严格位于计时区
外**）。测试形状 (N,K)：1024×4096 / 4096×1024 / **4096×4096（主目标）** /
11008×4096 / 4096×11008。范围排除：INT4、group-wise、GPTQ/AWQ、
activation quant、Tensor Core GEMM（用户指定）。

v0.6 交付：
- **4 个自主优化实验**（NCU 证据 → 假设，paired v2.3，(4096,4096)
  streaming，parent 链式）：

  | 实验 | 变体 | 假设（来自剖析） | 判定 |
  |---|---|---|---|
  | QGEMV-0001 | `qgemv_vec16_row` | baseline DRAM 27.2% / long_scoreboard 77% → 16B 向量 load（uint4 int8×16，在途字节 16×） | **KEEP 2.6614×** [2.657,2.669] 9/9；NCU DRAM 86.02% = DRAM 饱和 |
  | QGEMV-0002 | `qgemv_vec16_scale` | 计算减半（x·q 累加后一次 ×scale，省 K 次 fp32 mul） | **NEUTRAL 0.9901×**（统计 SLOWER）[0.9885,0.9906] 0/9——计算削减反噬 occupancy（68.42%），dequant 乘加本就在延迟路径外 |
  | QGEMV-0003 | `qgemv_warp_vec16` | warp-per-row + ILP=2 + 纯 warp shuffle：消除 shared/barrier（barrier 7.3%→0） | **NEUTRAL 1.0107×（统计 FASTER）** [1.0099,1.0108] 9/9；DRAM 87.24%、occ 90.43% 全变体最佳 |
  | QGEMV-0004 | `qgemv_warp_vec16_ilp4` | ILP 2→4（在途字节再翻倍） | **NEUTRAL 0.9984×（统计 SLOWER）** [0.9977,0.9990] 0/9——假设被证伪（DRAM/occ 持平，regs 41→43，指令开销） |

  失败/中性实验全部保留。
- **最终 incumbent = `qgemv_warp_vec16`**：两次 fresh head-to-head
  （vec16_row vs warp_vec16）CI95 均排除 1.0 且方向一致（1.0107× /
  1.0094× [1.0086,1.0101]，均在 5% 政策带内）→ 最终裁决规则
  **CI95 显著性优先于 5% 政策带**（已在决策链中明文记录）。
  主目标三口径分开报告：API-path **31.513 µs**（vs baseline 84.885 =
  **2.6937×** CI95 [2.688,2.700] 9/9；533.4 GB/s = **86.6%** 规格峰值
  616 GB/s，理想 27.29 µs 的 1.16×）/ native kernel-loop 32.185（w200）+
  32.103（w5000）/ NCU kernel duration 36.16 µs（base 锁频，DRAM 87.24%、
  occ 90.43%、barrier 0、long_scoreboard 84.1%）+ 35.88 µs（@none）。
  三口径一致：API vs native 差 <2.5%；NCU +14.8% = NCU profiling 固定
  开销（~4.4 µs 量级，同 v0.5 ~3.8 µs）+ base 锁频贡献 ~0.8%（v0.5
  clkbase/clknone 受控对：DRAM-bound +0.77%；负载下实测 boost
  1815–1920 MHz），与 v0.5 口径模型一致（冲突已调查，未挑数字）。
- **INT8 vs FP16（核心问题答案）**：31.513 vs 59.933 µs（gemv_vec4_row
  本 session 重测）= **1.90×**——逻辑 IO 减半的收益拿到 95%（理论 2.0×）；
  瓶颈仍在 DRAM（dram 27.2% → 87.24%），dequant 乘加"免费"（QGEMV-0002
  证明削减它只会伤 occupancy）。
- **两层正确性**：(a) kernel vs 量化参考（`W_q.float()*scale @ x.float()`
  转 fp16）：**50/50 × 5 变体**（5 输入模式 × (4 形状 + 4 边界形状)；
  **固定算术误差界** tol = 2·(3K·2⁻²⁴·S_n + 0.5·ulp16(|y|) + 0.5·ulp16
  (|exact|)，S_n 双因子取绝对值——开发期发现并修复 x 符号 bug；finiteness
  门；非 per-variant 调参）；(b) 保真度 vs 原始 FP16 `W@x`：**report-
  only 不作门**——per-element 误差 ≤ 0.5·量化步长（实测 0.500002，
  round-half-even 上界），主目标 cosine 0.9999617（normal；5 形状 min 0.9999594）/
  0.9998970（mixed_sign，suite 最差），max_abs 2.57 / RMSE 0.551（|y| ≈
  4027），相对精度 scale 不变。
- **负例 29/29 × 5 变体**（23 reject + 6 pass，含 3 个 per-variant
  标量回退**逐位一致**回归：W_q 错位 / x 错位 / K=13——向量化契约
  16B∧16B∧K%16，不满足 → 与 baseline 同一份 `qgemv_scalar_kernel`）。
- **全 shape matrix**（5 形状 × {hot,streaming}，4 向量化变体 vs
  baseline 2.2–2.9× 全胜 10/10；长 K 格变体间差 ≤~2%，短 1-dim 格差异
  达 regime 级 2.05×（(4096,1024) streaming））：按格 winner
  **warp_vec16 6/10、vec16_row 4/10**（两个 11008 形状 vec16_row
  CI 显著 +0.4–0.6%，已披露；(4096,1024) warp 对 vec16_row **2.0×**
  结构性优势——短 K 下 block-per-row 75% 线程空转）、ilp4 无统计显著
  胜格。
- **PyTorch context**：`torch.mv` 61.235 µs（fp16 W）/ 62.297 µs
  （dequant W，(4096,4096)）——仅参照，不作决策依据（INT8 QGEMV 31.5 µs
  ≈ 1.9–2.0× 于两者）。
- **回归**（`experiments/regression/v0.6/`，append-only）：gemv /
  rmsnorm / softmax / rope / qgemv 五算子 **all PASS**（历史 artifact
  不可变 + 隔离审计：gemv_splitk4 / softmax_hsplit2 仍在隔离、qgemv
  隔离集为空）；CPU 测试 36/36 + 18/18 + 20/20 + 6/6。
- **独立 review**：2 路独立 subagent（CUDA correctness + benchmark
  methodology）双双 **PASS WITH CAVEATS**（benchmark：2 MAJOR + 4 MINOR +
  6 NIT；CUDA：0 MAJOR + 1 MINOR + 2 NIT）；无记录造假类发现；逐条处置
  见报告 §10（benchmark 路 MAJOR/MINOR 全部已修，CUDA 路 NIT-1 已修
  docstring、MINOR-1 记为已知边界）。

## 当前状态：v0.5（FP16 GEMV 优化）

v0.5 回答一个问题：**闭环能否在第四算子 GEMV（内存受限）上做出真实的
优化收益？** 分支 `v0.5-gemv`（基线 main = v0.4.1 = 4eb520b），
**不 merge 回 main**（用户指定）。最终报告：`docs/report_v0.5_result.md`。

**CUDALab Operators：RMSNorm / Softmax / RoPE / GEMV**。GEMV：
`y = W @ x`（W `[N,K]` 行主序，x `[K]`，y `[N]`），主路径
**W/x/output = FP16，累加 = FP32**；ref `torch.mv(W.float(), x.float())
.to(torch.float16)`；FP32 输入/输出为支持路径。测试形状 (N,K)：
1024×4096 / 4096×1024 / **4096×4096（主目标）** / 11008×4096 /
4096×11008（LLM hidden/MLP 形状）。范围排除：GEMM、quantization、
Attention、CUDALM 集成（用户指定）。

v0.5 交付：
- **4 个自主优化实验**（NCU 证据 → 假设，paired v2.3，(4096,4096) fp16
  streaming，parent=gemv_baseline）：

  | 实验 | 变体 | 假设（来自剖析） | 判定 |
  |---|---|---|---|
  | GEMV-0001 | `gemv_vec4_row` | long_scoreboard 79% / DRAM 49% → 16B 向量 load（在途字节 4×） | **KEEP 1.5416×** [1.5398,1.5517] 9/9 → **incumbent**；NCU DRAM 87.9%（@none 90.2%）= DRAM 饱和 |
  | GEMV-0002 | `gemv_warp_vec4_b256` | warp-per-row + ILP=4（消除 shared/barrier） | **KEEP 1.2539×** [1.2454,1.2555] 9/9；瓶颈搬到 LSU issue（lg_throttle 84%） |
  | GEMV-0003 | `gemv_warp_vec4_b512` | block 512 杠杆 | **KEEP 1.2407×** [1.2396,1.2446] 9/9（同族机理） |
  | GEMV-0004 | `gemv_splitk4` ⚠隔离 | split-K×4 并行度 | **REJECT 0.8784×** [0.8755,0.8804] 0/9（每线程 4 元素，MLP 不足 + 二次 launch）；**v0.5 merge review 后已隔离**（见下） |

  失败实验全部保留。正确性 **100/100 × 5 变体**（2 dtype × 10 形状 ×
  5 模式；**固定算术误差界容差**，非 per-variant 调参；allclose 只记录
  不门控）+ 负例 **24/24 × 5 变体**（18 拒绝 + 6 通过，含 3 个标量
  回退**逐位一致**回归——向量化变体显式 16B 对齐契约 + host 侧检查 +
  回退到与 baseline 同一份 `gemv_scalar_kernel`；**对齐不是标量变体的
  约束**：baseline 与隔离的 splitk4 均为标量访存，splitk4 唯一契约
  K%4==0）。

  **v0.5 merge review 修复（2026-09-21, 3 个 fix commit, 推送前）**：
  (1) **splitk4 隔离**（UNSAFE_HISTORICAL_EXPERIMENT / REJECTED /
  NOT_FOR_NORMAL_DISPATCH）：`static at::Tensor g_splitk_partials`
  进程级 workspace 有多 stream 并发 race + 跨 device workspace 风险；
  从 `ext.variants()` 与 CLI test/benchmark/optimize/profile 正常路径
  移除（显式请求报隔离错误），源码与全部 bench/NCU 历史保留，显式
  `forward("gemv_splitk4", ...)` 为受控历史审计入口；(2) **历史
  artifact 不可变**：恢复被 v0.5 smoke 误覆盖的 7 个 main 历史
  correctness 记录（rmsnorm/softmax/rope），本轮验证一律追加到
  `experiments/regression/v0.5/`（append-only 目录，含 README），
  三个旧算子默认输出目录改指该处；(3) **negative 套件语义统一**：
  Softmax/RoPE 的 `run_negative` 改为 per-variant（CLI 指定的候选被
  实际测试），四个套件记录带 `negative_suite_scope` 字段（GEMV/
  Softmax/RoPE = per-variant，RMSNorm = cross-variant 单跑设计，
  不再描述为 per-variant）。细节：报告 §12 "merge review 处置"。
- **主目标最终结果**（`gemv_vec4_row`，三种口径分开报告，不混用）：
  API-path（paired v2.3 streaming）59.88 µs / native kernel-loop
  （1 次 Python 调用 → C++ 连续 launch → CUDA events /N，w5000 warmup）
  59.65 µs / NCU kernel duration 64.19 µs（base 锁频）/ 63.70 µs
  （@none）。vs baseline 93.9–94.2 µs = **1.54×**，算法带宽 560 GB/s =
  **91% 规格峰值**（616 GB/s），理想 54.5 µs 的 1.10×。独立复核（新进程
  9-round 双模式 + 正确性/负例重跑）KEEP/KEEP。
- **全 shape matrix**（5 形状 × {hot,streaming}）：**vec4_row 10/10 格
  胜出**（fp16 1.51–1.63× vs baseline；fp32 子集 1.06–1.11×）。
- **三口径冲突调查**（用户要求记录并调查）：API < native-w200 < NCU@base
  的排序差已定位 = idle 缺口后 DVFS 从 base 1350 MHz 爬向 boost 1890 MHz，
  native 默认 200-launch warmup 落在爬坡内（3 s 空闲 + 时钟采样探测定量
  复现：w200 109.2 µs @1350 MHz vs w5000 91.3 µs @1890 MHz）；DRAM 饱和
  的 vec4_row 对时钟不敏感（API 与两档 native 差 <1.3%）；控制时钟状态后
  API 与 native-w5000 差 <1.5%，NCU@none 仍有 ~7% 残余（cache flush +
  剖析隔离，已解释，§5.4）。与 v0.4 RoPE（launch 受限 → NCU 快于
  API）方向相反，是瓶颈资源不同导致的口径关系翻转。
- **PyTorch context**：`torch.mv` 61.23 µs（(4096,4096) fp16）——仅参照，
  vec4_row 与 cuBLAS 同级（~1.02×），不作决策依据。
- **Evaluator v2.3 决策/基准零改动**；唯一 evaluator 侧改动 = profiler
  对多内核算子（splitk4 两阶段）新增逐内核 `kernels[]` 摘要（增量、由
  真实数据触发、顶层字段不变、splitk4 记录修复后重录）；CPU 36/36。
- RMSNorm / Softmax / RoPE smoke 回归 **6/6 PASS**；两个独立 subagent
  review（CUDA correctness + benchmark methodology）双双 **PASS WITH
  CAVEATS**（CUDA 1 MAJOR + 2 MINOR + 2 NIT；Benchmark 5 MINOR + 8 NIT），
  全部 findings 已处置（报告 §12 处置表）：per-variant 负例归档 +
  mixed_sign 构造修复 + 记录重录 + 报告数字修正，历史 benchmark/profile
  记录逐字节未动。

## 当前状态：v0.4（Evaluator v2.3 + RoPE 泛化）

v0.4 回答两个问题：**(1) evaluator 能否从 v2.2 升级到 v2.3**（对称
guard + raw/filtered 双轨 + filter-sensitivity，修复 v0.3.1 登记的
不对称 guard 已知局限）；**(2) 闭环能否迁移到第三个算子 RoPE**。
分支 `v0.4-rope`（基线 main = v0.3.1 = 86bd871），**不 merge 回
main、不开始 v0.5**（已 push，等待外部 review）。
最终报告：`docs/report_v0.4_result.md`（定稿，含独立 review 与 Lead
最终审计 §17）。

**CUDALab Operators：RMSNorm / Softmax / RoPE**（interleaved RoPE：
a=x[2i], b=x[2i+1], c=cos[pos,i], s=sin[pos,i]；y[2i]=a*c−b*s，
y[2i+1]=a*s+b*c；FP32 中间，输出原 dtype；base=10000）。

v0.4 交付：
- **Evaluator v2.3**（`paired-streaming-v2.3`，
  [docs/evaluator_v2_3.md](docs/evaluator_v2_3.md)）：对称 log 空间
  guard（|log(t/ref)|>log(F)，快慢同因子：spike 1.5× / cross-block
  1.15×，parent/candidate 完全同规则）+ raw/filtered 双轨记录
  （每 round raw/filtered 中位数 + raw_speedup + rejected_samples{fast,slow}
  + environment_guard 自描述块）+ filter-sensitivity（方向翻转或
  |log(filtered/raw)|>log(1.10) → 敏感；敏感 → `apply_filter_gate`
  把最终 policy_decision 一律降级 UNSTABLE——v0.4.1 起 KEEP/REJECT/
  NEUTRAL 均降级，记录 original_decision）。guard 逻辑抽为纯 CPU
  函数（stats.apply_spike_guard/block_stats/crossblock_flag），
  tests/test_evaluator_v23_cpu.py 36/36（review 后 +1：raw 侧聚合
  约定钉死；v0.4.1 +7：gate 收紧 2（NEUTRAL+敏感 → UNSTABLE 双向
  必测）+ statistical_relation/policy_decision 形式分离 5）。
- **v2.3 回归硬门 PASS**（RoPE 之前，`benchmarks/v2.3_regression/`）：
  Softmax baseline vs vec4 streaming **1.6745** [1.6727,1.6793] 9/9
  （v2.2 参考 1.6772/1.6890 精确复现，raw=filtered，rejected 0/0）；
  RMSNorm v4 vs v1 streaming 1.0375 → 复跑 0.9576（数分钟内方向翻转 =
  环境微态漂移、非 evaluator 缺陷，四项调查证据在 evaluator_v2_3.md §6）；
  hot 0.9923 带内。
- **新算子：interleaved RoPE**（`kernels/rope/`，5 个注册变体；
  主目标 (1024,128)；9 形状矩阵 (1,64)…(4096,128)；FP16 主 + FP32，
  禁 BF16；sm_75 / CUDA 11.8）。baseline 正确性 384/384（finiteness +
  double-rounding 算术界 K=2 vs fp64 精确旋转 + norm 保持；与 torch
  参考 allclose 报告不门控；review 后新增独立表值核对门：表 vs
  θ(pos,i)=pos·10000^(−2i/D) 的 fp64 独立求值全网格核对 + 固定误差界，
  5 变体共享、折叠进 all_pass）+ negative 36/37 执行、all_pass=true
  （37 例 = 31 基础 + 3 per-variant 整除性 + 3 v0.4.1 half2 对齐
  回归；5 预期 PASS + 1 跳过）。
- **同步验证修复（v0.4 最重要的工程发现之一）**：首跑 baseline
  28.8/29.8 µs 被定位为验证路径缺陷——positions 值域检查（0≤p<L）
  的**同步** D2H 拷贝逐 launch 强制流同步（~25–30 µs）。修复：
  验证拆为 meta（host 元数据，始终执行）+ range（`validate` 门控，
  默认 true）；benchmark pool 与 NCU driver 传 `validate=False`
  （池契约：构造期全量预验证，positions=0..M-1<4096 必然成立）。
  修正后 baseline：streaming 6.981 µs / hot 6.637 µs（9/9 × 双模式）；
  修正前记录保留为 `*_presyncfix_archive.json`（审计痕迹）。
- **4 个自主优化实验**（profiler → hypothesis，v2.3 判定，**全部
  NEUTRAL —— PASS 结局**，见下）：

  | 实验 | 变体 | 假设（来自剖析） | 判定（(1024,128) fp16 streaming） |
  |---|---|---|---|
  | ROPE-0001 | `rope_v1_2pair` | long_sb 69% → 1 线程→2 pairs（MLP 杠杆） | NEUTRAL（1.0000 [0.9849,1.0160]）；NCU 单 launch −7.6% 但稳态流内无效 |
  | ROPE-0002 | `rope_v2_4pair` | MLP 到 4 pairs（D%8==0） | NEUTRAL（1.0010 [0.9331,1.0211]）；NCU +25%，波坍缩开始 |
  | ROPE-0003 | `rope_v3_half2` | 发射侧非瓶颈 → 指令数削减控制（fp16 `__half2` 打包） | NEUTRAL（0.9974 [0.9888,1.0025]）—— 成功的阴性对照 |
  | ROPE-0004 | `rope_v4_8pair` | MLP 边界（D%16==0） | NEUTRAL（0.9922 [0.9861,1.0055]，rejected fast=39 = r1 锚定偏差计数，见报告 §9）；NCU +104%，波坍缩灾难区 |

  结论（v0.4.1 限定范围）：在当前 Python → pybind → PyTorch C++
  extension → CUDA launch 的 benchmark submission path 下，主目标
  (1024,128) 表现出明显 launch/host-issuance sensitivity——paired
  API-path: baseline ≈ 6.4 µs vs v1 ≈ 6.4 µs；NCU kernel-only:
  baseline ≈ 4.00 µs, v1 ≈ 3.70 µs（−7.6%）。因此不能直接推断:
  未来原生 C++ CUDALM 中 v1 也无收益（submission path 会变）。MLP
  甜区 1–2 pairs/thread；≥4 pairs 波坍缩。**"不要追求 RoPE 一定
  优化成功"**——真实目标是 evaluator 更可信 + 第三算子自然接入 +
  从 profiler 证据形成有效实验；该 submission path 下 baseline 已
  近下限时全 NEUTRAL 是 PASS，不是失败。NCU 用于诊断、paired bench
  用于决策（v1_2pair 的 NCU −7.6% 未转化为流内 ≥5% 优势，NEUTRAL
  是正确决策）。
- **全矩阵**：36 格（9 形状 × 2 dtype × hot/streaming）× 5 变体
  全部保留（`benchmarks/rope/rope_v04_matrix_*` + `rope_v04_matrix_shape_winners.json`）。
  质量：31/36 格 9/9 valid、4 格 8/9、1 格 7/9（spike/cross-block 拒轮透明
  记录，全部 ≥7 推荐值）；fast cross-block flag 共 11 个（2 个 M1_H64
  streaming 格 3 轮，整轮作废 → 零 variant 偏置，报告 §11）；
  **0 格 filter-sensitive**（raw 与 filtered 结论一致；raw 侧聚合约定
  review 后与 filtered 侧对齐为 per-round 比值中位数）。
  **无一格存在 policy 层面的唯一胜出者**：36 格 winner/runner-up 比值
  0.994–1.033，全部 <1.05 KEEP 线（per-cell winner 分布 baseline 16 /
  v1_2pair 15 / v2_4pair 3 / v3_half2 2 / v4_8pair 0 —— 矩阵 winner 仅
  指示性，最终判定以 paired A/B 为准）。主目标 (1024,128) fp16 矩阵
  值（run 内）：streaming baseline 6.540 µs / hot 6.214 µs。
  PyTorch 2.4.1 **无内置 fused RoPE op** → Python 参考（rope_ref，多
  kernel）仅作 implementation context（主目标 fp16 262.1 µs / fp32
  180.3 µs，`benchmarks/rope/pytorch_ref_M1024_H128.json`），不产生
  "X× faster than PyTorch" headline。
- 独立 review：3 个 subagent（CUDA Correctness = PASS W/CAVEATS /
  Benchmark Methodology = PASS / RoPE Math = PASS W/CAVEATS）+ Lead
  最终审计，全部交付并处置（kernel 头注释 load 计数、v3 位级声明、
  独立表值核对门、negative per-variant 整除性、raw 侧聚合约定；
  正确性/negative 记录重录）。详见 `docs/report_v0.4_result.md` §17。

## v0.3 + v0.3.1 合并修复（Evaluator Generalization + Softmax 自主优化，历史保留）

v0.3 回答一个问题：**v0.2 的闭环（正确性 → 配对 bench → 统计 → 决策 →
剖析 → 实验史）能否原样迁移到第二个算子？** 分支 `v0.3-softmax`
（基线 main = v0.2.1 = dfe9e9b），已发布为 tag v0.3.1（main = 86bd871）。
最终报告：[docs/report_v0.3_result.md](docs/report_v0.3_result.md)。

v0.3.1（合并修复，2026-09-19；不新增内核、不重跑全矩阵）：
- **隔离 `softmax_hsplit2`**（UNSAFE_HISTORICAL_EXPERIMENT / REJECTED /
  NOT_FOR_NORMAL_DISPATCH）：其跨 block spin-wait 合并依赖 CUDA 调度模型
  不保证的并发驻留假设（liveness 风险），且 HsGlobal scratch 为进程级
  共享状态（多 stream / 多 device race 风险）。正常 `variants()` 列表
  移除，全部 SFM-0004 历史数据保留。见 [experiments/softmax/SFM-0004.md](experiments/softmax/SFM-0004.md) §6；
- **统计语义澄清**：区分统计关系（FASTER/SLOWER/UNRESOLVED）与政策决策
  （KEEP/REJECT/NEUTRAL/UNSTABLE）——"CI 排除 1.0 但 <5%" 是政策
  NEUTRAL，不再表述为"统计平局"；移除"结构最优 / 设计空间闭合"表述；
- **带宽表述更正**：2080 Ti 规格峰值 **616 GB/s**（此前误写 550）；
  `algorithmic_bw_gbps` 是逻辑算法流量，不是实测 DRAM 吞吐；
- **Evaluator v2.2 已知局限登记**：spike / cross-block guard 不对称
  （只拒绝异常慢状态），Evaluator v2.3 TODO；
- 无效轮计数字段更名 `invalid_environment_rounds`（旧字段名 `invalid_dvfs_*`
  保留为 legacy alias，旧 JSON 兼容）。

v0.3 交付：
- **Evaluator 通用化**（不做大规模重写）：`cudalab/evaluator/`（bench v2.2、
  stats/decision/profiler/negative/experiment，operator-agnostic）+ 算子
  adapter `cudalab/operators/{rmsnorm,softmax}.py`；stats.py/decision.py 与
  v0.2.1 逐字节相同（独立审计确认）；统一 CLI `scripts/cudalab.py
  test|benchmark|profile|optimize {rmsnorm,softmax}`；
- **harness v2.2**（[docs/evaluator_hardening_v0.3.md](docs/evaluator_hardening_v0.3.md)）：
  移除 round 内 nvidia-smi 采样（其 ~40ms 空闲间隙会把 GPU 推入性能退化态，
  v2.1 DVFS guard 的偏离作为"基于证据的机器态适配"如实记录）→ 时间基准
  burn（≥150 launches 且 ≥300ms）+ 逐样本 spike guard（1.5× 运行中干净
  中位数）+ 跨块一致性 guard（block 中位数 > 运行中 median×1.15 →
  INVALID_CROSSBLOCK）；
- **新算子：row-wise Softmax**（`kernels/softmax/`，5 个注册变体；
  v0.3.1: `softmax_hsplit2` 被隔离为 NOT_FOR_NORMAL_DISPATCH，正常
  `variants()` 列表 = 4 个，见 SFM-0004.md §6；FP32 内部，输出原
  dtype；FP16 主 + FP32，禁 BF16；sm_75 / CUDA 11.8）；正确性（v0.3
  历史数据）5 变体 × 72/72（容差逐变体相同）+ negative 14/14+1 skip
  （launch 前 `TORCH_CHECK`）；
- **4 个自主优化实验**（profiler → hypothesis，v0.3 语义；失败内核全部
  保留作参考实现）：

  | 实验 | 变体 | 假设（来自剖析） | 判定 |
  |---|---|---|---|
  | SFM-0001 | `softmax_vec4` | 标量小事务是瓶颈（long_scoreboard 60.6%）→ 4 宽向量化 | **KEEP** → incumbent |
  | SFM-0002 | `softmax_online` | 3 读 1 写 → 2 读 1 写（online (m,l) 单遍） | NEUTRAL（瓶颈是延迟不是带宽） |
  | SFM-0003 | `softmax_vec4_ilp2` | 每线程在飞 load 加倍隐藏延迟 | NEUTRAL（ILP 不是杠杆） |
  | SFM-0004 | `softmax_hsplit2` | occupancy 44% → 86%（H 对半分 2 块/行） | **REJECT**（barrier stall 31–35%，不 occupancy-bound）；v0.3.1 **隔离**：UNSAFE_HISTORICAL_EXPERIMENT / NOT_FOR_NORMAL_DISPATCH（跨 block 并发假设 + 进程级 scratch race，见 SFM-0004.md §6） |

  主目标 (128,4096) fp16（v0.2.1 语义，paired v2.2）：SFM-0001 **streaming
  1.6772 [1.6568,1.6794] 9/9 → KEEP**（final_reval 1.6890 稳健复现）；
  hot 记录值 1.2916 存在机器态漂移（final_reval 0.9865 NEUTRAL，
  [SFM-0001.md §6](experiments/softmax/SFM-0001.md) 披露），KEEP 以
  primary 判据 streaming 为准。四个设计维度（宽度/流量/每线程 ILP/
  块级并行）全部测完 ≠ 设计空间穷尽（v0.3.1 措辞更正）：
  **`softmax_vec4` 是当前 acceptance policy 下的 incumbent；后续候选
  尚未达到 ≥5% 的替换门槛**（NEUTRAL 是 policy_decision，不是"统计
  平局"或"结构最优"——统计语义见 v0.3.1 澄清与报告 Q3/Q4）。
- **best.json**（`experiments/softmax/best.json`，classify_cell，
  v0.2.1 语义；v0.3.1 语义澄清）：36 格（9 形状 × 2 dtype）全部 decision=
  NEUTRAL → **NO_UNIQUE_WINNER**（winner/runner-up 比值 0.9972–1.0479，
  全部 <1.05 KEEP 线）；17 格 INCUMBENT 标签（vec4 在 top-2，被 policy
  保留）/ 19 NO_UNIQUE_WINNER。NO_UNIQUE_WINNER 是**策略层面**的"无
  唯一胜出者"，不是"统计平局"断言：36 格中 21 格 winner 对 runner-up
  CI95 全在 1.0 之上（统计显著更快但 <5% → policy NEUTRAL），15 格 CI95
  跨 1.0（统计不可区分）——详见报告 Q4 与 summary.note。
- **RMSNorm 回归硬门 PASS**（eae07bb + 最终复跑 85faeca）：CPU tests +
  negative 29/30+1 skip + 正确性 76/76 × 2 + paired 全兼容 v0.2 结论
  （hot 1.0144 NEUTRAL / streaming 0.9419 REJECT，点估计漂移、机制不变，
  机器态而非 evaluator 缺陷）。
- 独立方法学审计 [docs/benchmark_audit_v0.3.md](docs/benchmark_audit_v0.3.md)：
  **PASS WITH CAVEATS**（数据逐项可复现、决策与规则一致、重构忠实；
  caveat #1 hot 漂移已在 SFM-0001.md §6 + 报告披露，caveat #3b 日期笔误
  已修正）。

### v0.3 机器态限定（如实记录）

(128,4096) fp16 **hot** 模式的配对结果对机器态敏感（2 MB 热工作集的
L2 驻留 + SM 时钟爬坡 1350→1890 MHz 的差异）：vec4 vs baseline hot
两次 paired 运行 1.2916（21:03，SFM-0001 记录）→ 0.9865（21:41，
final_reval，NEUTRAL）；vec4 hot 绝对时间在相隔约 90 秒的两组运行间
漂移 ~45%（4.684 µs @21:42 inc 矩阵 vs 6.797 µs @21:41 final_reval，
baseline 稳定 ~6.7 µs）；**streaming 稳定**（1.6772 → 1.6890，
矩阵口径 10.109/6.015 ≈ 1.68 一致）。跨运行绝对时间不可比，仅运行内
配对有决策意义。详见报告 Q6。

## v0.2（评估器加固与完整复验，历史保留）

v0.2 **没有新增内核**（5 个变体原样保留），而是修复 v0.1 代码审查发现的问题，
重建评估层，并在同一套可信方法下重新得出性能结论。v0.1 全部历史数据
（EXP-0001…0007、`benchmarks/`、`profiles/rmsnorm/`、`correctness/` 顶层文件）
原样保留，未做任何改写。

v0.2 交付：
- **API 加固**（输入验证 Finding A–D + 启动后错误检查）与 **30 例非法输入负例套件**
  （v0.2.1 增补 2 例 v4 FP32 H=1024 对齐回归）；
- **配对基准 harness `paired-streaming-v2`**：A/B 交替顺序去偏、每轮 3 次 SM 时钟
  采样的 DVFS guard（>5% 相对差判 invalid round）、hot/streaming 双缓存模式、
  预分配缓冲池（streaming 工作集是否 > L2 取决于 shape，以记录中的
  `working_set_gt_l2` 为准；v0.2 主形状 (128,4096) fp16 = 33.5 MB > 5.5 MB L2）；
- **round-level 统计与决策引擎**（纯 CPU、可单测）：paired speedup 中位数 +
  bootstrap 95% CI（固定种子 20260919）+ KEEP/REJECT/NEUTRAL/UNSTABLE 四态判定；
- **NCU 方法学审计**：`--cache-control` 语义修正（v0.2.1，此前写反）——
  `all`（默认）= cache flush/reset profiling（每 replay 前失效缓存，确定性
  flushed 状态）、`none` = no-flush profiling（不失效，状态不受控，ncu 警告
  "Running with uncontrolled GPU caches"）；v0.1 走默认 `all`（= 失效/flush），
  其 "cold L2" 说法与默认配置一致（v0.2 曾误判为"无配置依据/实际热"，已更正）；
  v0.2 双缓存模式显式剖析并记录；
- **完整复验**：7 形状 × {fp16,fp32} × {hot,streaming} × 5 变体（28 组矩阵，
  全部 9/9 valid rounds）+ 13 组主目标配对精度测量 + 双模式 NCU 剖析；
- **形状/dtype 分发表**（v0.2.1 证据政策 evidence > coverage：仅 2 个 paired 确认格
  + 1 个显式 incumbent 格路由优化变体，其余一律 baseline，`cudalab/dispatch.py`）；
- 独立方法学审计文档 [docs/benchmark_audit_v0.2.md](docs/benchmark_audit_v0.2.md)。

### v0.2 正确性

| 套件 | 结果 |
|---|---|
| 合法输入套件 × 5 变体（76 例/变体） | **380/380 PASS**（`experiments/rmsnorm/correctness/v0.2/`） |
| 非法输入负例套件（30 例） | **29/30 符合预期，1 跳过**（多 GPU 用例，单 GPU 环境安全跳过） |

max_abs_error = 3.91e-3 / max_rel_error ≈ 9.7e-4（fp16 最坏值，全变体一致）。
负例套件覆盖：非对齐/非法 H（1023/1025/4095/4097/4100）、w 长度与 dtype 错配、
CPU 输入、非连续输入、out 张量错配、bf16、eps=NaN/负、v1/v4 指针 8B 未对齐
（+ 16B 对齐对照组 PASS）、v4 FP32 H=1024 对齐回归（PER=4 亦按 float4 要求 16B：
4B offset 拒绝、16B offset PASS，v0.2.1 新增）、多 GPU 混布（跳过）。
所有非法输入在 **kernel 启动前** 被拒（`TORCH_CHECK`），启动后 `C10_CUDA_KERNEL_LAUNCH_CHECK()`。

### v0.2 主目标性能（M=128, H=4096, fp16，主形状）

`paired-streaming-v2`，9 rounds，全部 valid，有效 SM 时钟稳定 1350 MHz。
中位延迟（µs）与相对 baseline 加速比：

| 变体 | hot（µs） | hot 加速比 | streaming（µs） | streaming 加速比 |
|---|---|---|---|---|
| baseline | 10.334 | 1.00× | 13.158 | 1.00× |
| v1_vec | 6.652 | 1.55× | 7.182 | 1.83× |
| v2_reg | 6.723 | 1.54× | 7.419 | 1.77× |
| v3_wideblock | 7.040 | 1.47× | 8.768 | 1.50× |
| **v4_vec_reg** | **6.610** | **1.56×** | **7.039** | **1.87×** |

- **hot** = 固定预分配 buffer（L2 热）；**streaming** = 16 组 buffer 逐 launch
  轮转（是否 > L2 取决于 shape，以 `working_set_gt_l2` 为准；v0.1 主形状
  (128,4096) fp16 = 33.5 MB > L2 5.5 MB，每次 launch 面对冷 L2）。streaming
  延迟更高是**真实访存成本**，不是测量缺陷；两种模式都如实报告。
- 配对判定（9 rounds，bootstrap CI95）：
  - v4 vs baseline：streaming 1.896× [1.860, 1.908]、hot 1.740× [1.680, 1.771] → **KEEP v4**
  - v1 vs baseline：streaming 1.730×、hot 1.761× → v1 显著快于 baseline
  - v4 vs v1：hot v4/v1 median 0.9327 [0.8373, 0.9658]、0/9 轮 v1 更快（v4 快约 7%，REJECT v1）；streaming 平局（1.011×，NEUTRAL，9/9 轮 v1 微快）
  - v4 vs v2：两模式均 **NEUTRAL**（CI 均含 1.0，1–2% 差距在噪声内）
- **v0.2 主形状 fp16 无统计唯一胜出者（NO_UNIQUE_WINNER，v0.2.1 语义修正）**：
  streaming（primary 模式）v1/v2/v4 两两 NEUTRAL；hot（secondary 模式）v4 对 v1
  显著更快（REJECT v1）、对 v2 平局。`v4_vec_reg` 保留为 v0.1 incumbent，
  但**并非经统计确认的唯一最佳**（v1/v2 仍为竞争性变体）。

### v0.2 形状/dtype 结论（不是单一 global best）

完整 28 单元格 winner 表：`benchmarks/v0.2/shape_winners.json`。要点：

| 单元格 | 最佳 | 证据 |
|---|---|---|
| (128,4096) fp16 | NO_UNIQUE_WINNER（incumbent `v4_vec_reg`） | streaming：v1/v2/v4 两两 NEUTRAL；hot：v4 vs v1 REJECT v1；v4 保留 v0.1 incumbent（非统计确认唯一最佳）；vs baseline 1.56×/1.87× |
| (128,4096) fp32 | **v2_reg** | paired 1.37×（streaming）/ 1.62×（hot），CI 不跨 1.0 → KEEP |
| (128,8192) fp16 | **v2_reg** | paired 1.38×（CI 不跨 1.0）→ KEEP；**v4 在 H=8192 退化** |
| (128,8192) fp32 | v2_reg | 1.15–1.23×，两模式显著 |
| (1024,4096) 两 dtype | v4_vec_reg | 1.28–1.30× vs baseline，两模式 CI 不跨 1.0 |
| (1,4096) fp16 | v3_wideblock | launch-bound 区域；hot 1.11× vs runner CI 不跨 1.0 |
| (1,·)/(16,·)/(·,1024) 其余 | baseline | 优化变体对 baseline 无显著优势（如实回退） |

**v0.1 结论复核**（详见 EXP-0008）：

| v0.1 结论 | v0.2 判定 |
|---|---|
| EXP-0007：v4 比 v1 快 1.231× | **REVISED** —— 同频 1350 MHz 下 v4 vs v1 仅 1.011×（streaming，v1 反微快）/ v4 快约 7%（hot，median 0.9327）；1.231× 来自 v1@~1350MHz vs v4@~1905MHz 的 DVFS 混频，不可复现 |
| v4 = 主形状 fp16 最佳 | **CONFIRMED（附保留）→ v0.2.1 语义修正为 NO_UNIQUE_WINNER** —— streaming 无唯一胜出者（v1/v2/v4 两两 NEUTRAL），v4 保留 incumbent（hot：对 v1 显著更快、对 v2 平局），非统计确认唯一最佳 |
| 优化变体全面快于 baseline | **CONFIRMED** —— 主形状 1.7–1.9×，CI 不跨 1.0 |
| （v0.1 未区分）fp32 路径 | **REVISED（新发现）** —— fp32 上 v2_reg 是最佳（1.37–1.62× 快于 v4）；v4 的 fp32 寄存器路径明显弱 |

### v0.2 NCU 剖析（M=128×H=4096 fp16，双缓存模式）

`profiles/rmsnorm/v0.2/`（cc=all 为 ncu 默认 = cache flush/reset，cc=none = no-flush；
v0.2.1 修正语义，此前写反）：

| 变体 | µs (all/none) | DRAM % (all/none) | L2 read hit % | L1 hit % | 寄存器 |
|---|---|---|---|---|---|
| baseline | 14.34 / 14.56 | 14.6 / 19.4 | 41.3 | 45.5 | 16 |
| v1_vec | 6.26 / 6.14 | 31.3 / 36.0 | 38.7 | 35.0 | 22 |
| v2_reg | 6.04 / 5.95 | 34.6 / 42.5 | 36.6 | 32.3 | 50 |
| v3_wideblock | 9.39 / 9.31 | 22.3 / 28.9 | 41.5 | 45.4 | 16 |
| v4_vec_reg | 7.04 / 7.14 | 30.5 / 36.1 | 36.1 | **57.8** | 30 |

- **方法学审计结论（v0.2.1 修正，此前写反）**：ncu 2022.3 `--cache-control`
  默认 `all` = **cache flush/reset profiling**（每个 replay pass 前失效全部
  缓存，确定性 flushed 状态）；`none` = **no-flush profiling**（不失效缓存，
  状态不受控，ncu 警告 "Running with uncontrolled GPU caches"）。v0.1 未传
  该参数，走默认 `all`（= 失效/flush），其 "cold L2" 说法与默认配置一致
  （v0.2 曾误判为"无配置依据/实际热"，已更正）。v0.2 起该参数显式记录。
- 本 kernel 工作集 ~1 MB，cc=all 与 cc=none 的 duration/DRAM%/hit rate 几乎无
  差异：NCU 单次 launch 的缓存状态（flushed vs 不受控）对这种小工作负载不
  敏感；真正的缓存效应杠杆是基准层的 hot/streaming buffer 策略。
- `--clock-control base`（ncu 默认）未报告锁频警告，但也无锁频成功的正面证据
  （容器内 `nvidia-smi` default_applications=[N/A]）；所有变体同一设置下相对比较有效。
- v4 的 L1 命中率最高（57.8%）：寄存器驻留设计让 x 的第二次访问留在 L1。

### 形状分派（Phase 9，v0.2.1 证据政策修订）

`cudalab/dispatch.py`：`select_variant(M,H,dtype)` 纯 CPU 选择器，
**evidence > coverage**（v0.2.1 修订，review Finding 4）：

| 路由 | 单元格 | 证据 |
|---|---|---|
| `v2_reg`（paired-evidence） | (128,4096) fp32 | paired 1.3666× streaming / 1.6177× hot，均 KEEP |
| `v2_reg`（paired-evidence） | (128,8192) fp16 | paired 1.3803× streaming KEEP；v4 在 H=8192 退化 |
| `v4_vec_reg`（incumbent-fallback） | (128,4096) fp16 | NO_UNIQUE_WINNER → 显式保留 v0.1 incumbent（非统计确认唯一最佳） |
| baseline（matrix-only） | 其余 11 个实测单元格 | 矩阵 winner 未经 paired 验证不构成路由证据；含 hot/streaming 冲突格 (16,4096) fp16（hot winner=v4 / streaming winner=v1，不声称 v4 稳定） |
| baseline（baseline-fallback） | 所有未实测 (M,H) | 不做无证据外推（旧版对 fp16 M≥16 的 v4、fp32 M≥128 的 v2 外推已移除） |

`dispatch_info()` 返回上述四类 `evidence_source`（paired-evidence /
incumbent-fallback / matrix-only / baseline-fallback）与逐格理由，可审计。
6 个单元测试通过。**不改变任何内核**，只是选择器。

## 评估方法（演进：v1 → v2 → v2.2 → v2.3）

evaluator 是一个**持续修正的系统**：每一版都由真实事件（EXP-0002 单发噪声、
EXP-0007 DVFS 混频、v0.3 hot 机器态漂移、v0.3.1 登记的 guard 不对称）驱动
升级，历史版本与数据原样保留、按其各自 harness 版本解释。

- **v1（v0.1，`cuda-event-batched-v1`）**：32 连发 + 事件计时。修复了单发
  噪声（EXP-0002），但变体间**非配对**测量在 DVFS 活跃环境下仍产生虚高
  （EXP-0007 1.231× → 复验 1.011×）。
- **v2（v0.2，`paired-streaming-v2`）**：配对 A/B + 预分配池 + 每轮 SM 时钟
  DVFS guard（>5% 拒轮）+ round-level 统计 + 四态决策 + UNSTABLE。
- **v2.2（v0.3，`paired-streaming-v2.2`）**：移除 round 内 nvidia-smi 采样
  （其 ~40ms 空闲间隙推 GPU 入性能退化态）→ 时间基准 burn（≥150 launches
  且 ≥300ms）+ 逐样本 spike guard（1.5×）+ 跨块一致性 guard（1.15×）。
  已知局限：guard **只拒慢不拒快**（不对称 → 潜在选择偏差 + 不可审计）。
- **v2.3（v0.4，`paired-streaming-v2.3`，当前）**：对称 log 空间 guard
  （|log(t/ref)|>log(F)：spike 1.5× / cross-block 1.15×，快慢同因子，
  parent/candidate 完全同规则）+ **raw/filtered 双轨记录**（每 round
  raw/filtered 中位数 + raw_speedup + rejected_samples{fast,slow} +
  environment_guard 自描述块）+ **filter-sensitivity**（raw vs filtered
  方向翻转或 |log(filtered/raw)|>log(1.10) → 敏感；敏感 →
  `apply_filter_gate` 把最终 policy_decision 一律降级 UNSTABLE，
  记录 original_decision——v0.4.1 起 KEEP/REJECT/NEUTRAL 均降级）。
  guard 逻辑为纯 CPU 函数（`stats.apply_spike_guard/block_stats/
  crossblock_flag`），36 个确定性 CPU 单测钉死
  （`tests/test_evaluator_v23_cpu.py`；v0.4.1 新增 7 个：2 个
  NEUTRAL+敏感 → UNSTABLE 双向必测 + 5 个 statistical_relation /
  policy_decision 形式分离）。
  详见 [docs/evaluator_v2_3.md](docs/evaluator_v2_3.md)（含 v2.3 回归门
  结果与 RMSNorm 方向翻转调查）。

**v0.3 变更（v2.2，详见 [docs/evaluator_hardening_v0.3.md](docs/evaluator_hardening_v0.3.md)）**：
round 内 nvidia-smi 采样移除（其 ~40ms 空闲间隙会把 GPU 推入性能退化态——
v2.1 DVFS guard 的偏离，已作为"基于证据的机器态适配"如实记录并写入
最终报告 verdict），替换为时间基准 burn（≥150 launches 且 ≥300ms）+
逐样本 spike guard（1.5× 运行中干净中位数）+ 跨块一致性 guard
（block 中位数 > 运行中 median×1.15 → INVALID_CROSSBLOCK，重试 ≤3）。
统计/决策引擎（`stats.py`/`decision.py`）与 v0.2.1 逐字节相同；
harness 位于 `cudalab/evaluator/bench.py`（`cudalab/bench_v2.py` 为兼容 shim）。

### 配对基准 harness（`cudalab/bench_v2.py`，`paired-streaming-v2`）

- **配对 A/B**：每 round 内 parent 与 candidate 交替测量（slot 奇偶决定顺序），
  消除系统性顺序偏置（v0.1 的变体间先后顺序差异是 EXP-0007 混频事件的一部分）。
- **DVFS guard**：每个测量区间前后各 1 次 + 之后 1 次 nvidia-smi SM 时钟采样，
  有效时钟 = 均值；paired round 两变体有效时钟相对差 >5% → 该轮 invalid
  （1350 vs 1905 MHz 类失配必被拒）；矩阵 round 全变体 spread >5% → invalid；
  无时钟数据 → invalid。invalid 轮透明重试（≤3 次）并**完整保留**在
  `rounds[].invalid_reason` 中；valid rounds < 5 → 结论 UNSTABLE，不强行判 KEEP/REJECT。
- **预分配缓冲池**：计时区域外分配 16 组 x/out（streaming）或 1 组（hot）；
  streaming 工作集 33.5 MB（128×4096 fp16）> 5.5 MB L2。
- **计时**：每块 = 150 次不计时预热 + 100 次迭代 × 32 batch 连发，
  每 batch 一次 synchronize，取中位数单 launch 延迟（µs）。
- **统计**：decision 在 **9 个独立 round** 的 paired speedup 上计算（非 iter 级
  伪样本）：median + 10000 次 bootstrap 95% CI（纯 Python、固定种子 20260919，
  跨平台可复现）。判定：KEEP 需 median ≥1.05 且 ≥70% rounds 更快且 CI 下界 >1.0；
  REJECT 需 median ≤0.95 且 ≤30% rounds 更快且 CI 上界 <1.0；correctness FAIL
  无条件 REJECT；valid rounds <5 → UNSTABLE；其余 NEUTRAL。
- **算法带宽**（`algorithmic_bw_gbps`）：按变体真实访问量计
  （baseline/v1 读 x 两次：`(2·M·H + H + M·H)·elem_size/t`；v2/v3/v4 读一次：
  `(M·H + H + M·H)·elem_size/t`）——修正 v0.1 `effective_bw_gbps` 的 fp32 按
  2 字节/元素计数、以及忽略 v1/v3 双读的问题。仍是逻辑流量，非实测 DRAM（以 NCU 为准）。
- **PyTorch 参照**：`F.rms_norm`（torch 2.4.1，非融合路径）66.2 µs（主形状
  fp16）——仅作 implementation context 记录，不参与判定（非公平 fused-kernel 对比）。

### DVFS 事件与发现

本 GPU 的 SM 时钟范围 300–2100 MHz，轻载下稳定在 1350 MHz，持续负载可升至
~1905 MHz（v0.1 观察值）。v0.1 的 EXP-0007 恰好在 v1 于 1350、v4 于 ~1905 的
条件下比较，产生 1.231× 的虚高加速比。v0.2 的全部 28 组矩阵 + 13 组配对共
41 个 run、369 个 paired/matrix rounds **全部 9/9 valid，有效时钟稳定 1350 MHz，
0 个 invalid round** —— 本次复验没有发生频率漂移，DVFS guard 的价值体现在
**把这类事件从"污染结论"变成"被记录并拒绝的数据"**。guard 的行为有单元测试
覆盖（`tests/test_evaluator_cpu.py`：1350 vs 1905 → INVALID_DVFS）。

局限：nvidia-smi 轮询是 kernel 区间外的代理采样，不能捕捉区间内瞬时降频；
这是已记录的残余风险（见审计文档）。

## 范围（v0.5：四个算子）

- **算子 4：GEMV**（v0.5 新增；`y = W @ x`，W `[N,K]` 行主序、x `[K]`、
  y `[N]`；主路径 fp16 输入/输出 + FP32 累加，FP32 支持路径；ref
  `torch.mv(W.float(), x.float()).to(torch.float16)`；主目标 (4096,4096)
  fp16，5 形状矩阵见上。向量化变体（vec4_row / warp b256 / warp b512）
  显式对齐契约（W/x 基指针 16B 对齐 ∧ K % 16B内元素数 == 0，fp16:8/
  fp32:4），不满足 → 标量回退（与 baseline 同一份 `gemv_scalar_kernel`，
  逐位一致），不拒绝合法输入。**标量变体（baseline / splitk4）无对齐
  约束**（splitk4 唯一契约 K%4==0）。排除：GEMM、quantization、
  Attention、CUDALM 集成。）
  ⚠ `gemv_splitk4` 在 v0.5 merge review 后被**隔离**
  （UNSAFE_HISTORICAL_EXPERIMENT / REJECTED / NOT_FOR_NORMAL_DISPATCH：
  进程级 static workspace 多 stream 并发 race + 跨 device workspace 风险）：
  不在 `ext.variants()` 正常列表与 CLI test/benchmark/optimize/profile
  正常路径；源码与全部 bench/NCU 历史保留，显式 `forward("gemv_splitk4",
  ...)` 为受控历史审计入口（报告 §6/§12）。
- **算子 1：RMSNorm**（`y = x * rsqrt(mean(x², dim=-1) + eps) * w`，默认
  `eps=1e-5`，FP32 累加）—— v0.1/v0.2 历史算子，v0.3/v0.4/v0.5 仅做回归硬门。
- **算子 2：row-wise Softmax**（v0.3 新增；`y = exp(x − rowmax)/Σexp(x −
  rowmax)`，FP32 内部计算，输出原 dtype；ref `torch.softmax(x.float(),
  dim=-1).to(x.dtype)`）。连续输入；baseline 任意 H；vec4/ilp2 要求
  H%4==0 且 8B（fp16）/16B（fp32）对齐否则回退同一份标量核；online 任意 H；
  hsplit2 要求 H%8==0 且对齐、M≤8192、奇数 wave 容量回退（否则走 vec4
  回退核 / 标量核）。
- **算子 3：RoPE（interleaved）**（v0.4 新增；旋转位置编码，interleaved
  约定）：对每行 `x (D,)`、位置 `p`，取 `a=x[2i]`, `b=x[2i+1]`,
  `c=cos[p,i]`, `s=sin[p,i]`，输出 `y[2i]=a*c−b*s`, `y[2i+1]=a*s+b*c`
  （**FP32 中间计算**，输出原 dtype；`cos/sin` 表 = `make_rotary_table`，
  `base=10000`、`max_seq_len=4096`，launch 前 cast 到 dtype，计时区外）。
  连续输入；`D` 必须为偶数（pair 粒度）；`positions ∈ [0, 4096)`。
  变体 D 约束：baseline/v1/v3 任意偶数 D；**v2_4pair 要求 D%8==0；
  v4_8pair 要求 D%16==0**（启动前 `TORCH_CHECK`）。9 形状矩阵 (1,64)…
  (4096,128) 的 D∈{64,128} 全部满足两约束。
- 硬件：NVIDIA RTX 2080 Ti（Turing，**sm_75**），CUDA 11.8，PyTorch 2.4.1+cu118。
- dtype：**fp16 为主**，支持 fp32，**禁 BF16**（v0.4 范围）。
- RMSNorm 各变体支持的 H（v0.2 已加启动前显式校验）：
  baseline 任意 H；v1 H%8==0（fp16）/ H%4==0（fp32）；
  **v2 H/256 ∈ {2,4,8,16,32}（H ∈ {512…8192}）；v3 H%512==0；
  v4 H/256 ∈ {4,8,16,32}（H ∈ {1024,2048,4096,8192}）**。
- 主要优化目标形状：RMSNorm/Softmax 共用 **M=128, H=4096, fp16**；
  RoPE 主目标 **M=1024, D=128, fp16**（`D` = 每行维度，此处即 `H`）。
- 完整基准矩阵始终测量并保存 —— 不做形状挑拣。

## 优化实验（v0.1 历史 + v0.2 复验 + v0.3 Softmax + v0.4 RoPE）

### v0.4 — RoPE（`experiments/rope/`，ROPE-xxxx，全部保留）

| 实验 | 变体 | 判定（(1024,128) fp16 streaming，paired v2.3） | 备注 |
|---|---|---|---|
| ROPE-0001 | `rope_v1_2pair` | NEUTRAL（1.0000 [0.9849,1.0160]，4/9） | 1 线程→2 pairs（grid 减半、8 loads 提前，MLP 杠杆）；NCU 单 launch **−7.6%**（3.696 vs 4.000 µs）但稳态流内无效——**kernel 时长不是瓶颈，launch 发射速率才是**（NCU 诊断 vs paired 决策分工的实例） |
| ROPE-0002 | `rope_v2_4pair` | NEUTRAL（1.0010 [0.9331,1.0211]，5/9） | 4 pairs/thread（D%8==0）；NCU **+25%**（5.008 µs，occupancy 21.5%）——波坍缩开始 |
| ROPE-0003 | `rope_v3_half2` | NEUTRAL（0.9974 [0.9888,1.0025]，2/9） | fp16 `__half2` 打包 load/store + FP32 旋转（指令数削减控制，数学与 baseline 位级一致）；**成功的阴性对照**——发射侧非瓶颈的预测被证实，验证评估器拒绝灵敏度 |
| ROPE-0004 | `rope_v4_8pair` | NEUTRAL（0.9922 [0.9861,1.0055]，3/9；rejected fast=39） | 8 pairs/thread（D%16==0）；NCU **+104%**（8.176 µs，occupancy 11.9%）——波坍缩灾难区；rejected fast=39 = r1 锚定偏差计数（环境恢复后合法样本被拒，pair 判定稳健，报告 §9）；v2.3 快侧 guard 的真实行为展示 = 矩阵 11 个 cross-block flag（§11） |

四个候选 384/384 正确性全部通过。**全部 NEUTRAL 是 PASS 结局**（"不要
追求 RoPE 一定优化成功"）：在当前 Python → pybind → PyTorch C++
extension → CUDA launch 的 benchmark submission path 下，主目标表现出
明显 launch/host-issuance sensitivity（paired API-path: baseline
≈ 6.4 µs vs v1 ≈ 6.4 µs；NCU kernel-only: baseline ≈ 4.00 µs,
v1 ≈ 3.70 µs；dram 23.25% / long_sb 69.3%），因此不能直接推断:
未来原生 C++ CUDALM 中 v1 也无收益；MLP 杠杆甜区在 1–2 pairs/thread，
≥4 pairs 进入波坍缩；
没有候选达到 ≥5% 替换门槛，incumbent 保持 `rope_baseline`。完整记录：
`experiments/rope/ROPE-000{1..4}.json`（paired v2.3 pair 记录在
`benchmarks/rope/`，NCU 在 `profiles/rope/`）。

### v0.3 — Softmax（`experiments/softmax/`，SFM-xxxx，全部保留）

| 实验 | 变体 | 判定（(128,4096) fp16，paired v2.2） | 备注 |
|---|---|---|---|
| SFM-0001 | `softmax_vec4` | **KEEP**（streaming 1.6772 9/9；hot 记录 1.2916，final_reval 0.9865 NEUTRAL，hot 机器态漂移已披露） | 4 宽向量化（fp16 8B/fp32 16B），回退共享标量核；**新 incumbent** |
| SFM-0002 | `softmax_online` | NEUTRAL（hot 0.9775 / streaming 1.0413） | online (m,l) 单遍 + block merge 恒等；docs/softmax_algorithm.md + 5 CPU 恒等测试门禁；瓶颈是内存延迟不是 DRAM 带宽 |
| SFM-0003 | `softmax_vec4_ilp2` | NEUTRAL（hot 1.0108 8/9 / streaming 0.9815 0/9） | 2 路展开，与 vec4 逐位一致；寄存器 19→28、barrier stall 上升抵消收益；每线程 ILP 不是杠杆 |
| SFM-0004 | `softmax_hsplit2` | **REJECT**（hot 0.6752 0/9 / streaming 0.7752 0/9）；v0.3.1 起 **UNSAFE_HISTORICAL_EXPERIMENT / NOT_FOR_NORMAL_DISPATCH**（隔离，见 §6 与下文） | H 对半分 2 块/行 + (m,l) 跨块合并（单 launch）；occupancy 44.5%→85.7% 达成但 barrier stall 5.6%→31–35% → **不 occupancy-bound**；v0.3.1 更正：其"无死锁"论证依赖 CUDA 调度模型不保证的跨 block 并发驻留假设（另见进程级 scratch race 风险），四个正交维度测完 ≠ 设计空间穷尽 |

**v0.3.1 隔离说明（`softmax_hsplit2`）**：SFM-0004 不仅性能 REJECT，
后续 review 还发现其依赖未被 CUDA 调度模型保证的跨 block 并发假设
（每行 2 个普通 thread block + 全局 scratch → atomicAdd → spin-wait，
CUDA 不保证不同 thread block 的调度顺序或并发驻留 → 设备繁忙时存在
死锁 / liveness 风险）；第二已知风险：HsGlobal scratch 为进程级共享
状态，多 CUDA stream / 多 device 并发调用存在 race 风险。v0.3.1 处置：
标记 UNSAFE_HISTORICAL_EXPERIMENT / REJECTED / NOT_FOR_NORMAL_DISPATCH，
从默认 `ext.variants()` 正常列表移除（`bindings.cpp` quarantine 集），
统一 CLI 与基准引擎显式请求时明确拒绝；内核源码与全部 SFM-0004 实验 /
bench / NCU 数据原样保留（历史证据）；显式 `ext.forward("softmax_hsplit2", x)`
为受控历史审计入口。完整记录：
[experiments/softmax/SFM-0004.md](experiments/softmax/SFM-0004.md) §6。

### v0.1/v0.2 — RMSNorm

| 实验 | 变体 | 判定 | 备注 |
|---|---|---|---|
| EXP-0001 | baseline | KEEP（基线） | 旧框架，已作废（superseded） |
| EXP-0002 | v1_vec | REJECT | **方法论事故** —— 单发事件计时噪声（见下） |
| EXP-0003 | baseline | KEEP（基线） | cuda-event-batched-v1 |
| EXP-0004 | v1_vec | KEEP | 1.212×，5/5 轮更快 |
| EXP-0005 | v2_reg | NEUTRAL | 0.989×（±5% 带内） |
| EXP-0006 | v3_wideblock | REJECT | 0.768×，5/5 轮更慢 |
| EXP-0007 | v4_vec_reg | KEEP | 1.231× —— **v0.2 判定为 DVFS 混频膨胀（REVISED）** |
| EXP-0008 | 全部 5 变体 | 复验 | v0.2 paired-streaming-v2 完整复验（见上文） |

**方法论事故（保留在案，未隐藏）——两条教训：**

1. **EXP-0002（单发事件噪声）**：单发 cuda-event 框架引入约 6 µs 启动噪声，
   把 v1 判为比 baseline 更慢（0.844×），而 ncu 同时显示 v1 内核快 2.35×。
   促使改用 `cuda-event-batched-v1`（每样本 32 连发 + 同步）并对所有变体重测。
2. **EXP-0007（批量框架仍不够）**：batched 框架解决了 launch 噪声，但
   **变体间非配对测量**在 DVFS 活跃的环境下仍会产生虚假结论（1350 vs 1905 MHz
   混频 → 1.231× 虚高）。v0.2 的 `paired-streaming-v2` 针对的正是这类问题：
   配对顺序去偏 + 每轮时钟校验 + round-level 统计 + UNSTABLE 状态。
   两次事故共同支撑 CUDALab 的核心设计：客观层独立于智能体、全部被拒/作废
   实验完整保留、性能结论必须可复现可审计。

完整记录：[`experiments/rmsnorm/`](experiments/rmsnorm/)（EXP-0008 为 v0.2 复验记录；
`best_v0.1.json` 为 v0.1 最佳存档，`best.json` 为 v0.2 当前最佳）。

## 架构（v0.5：evaluator 核心 + 四算子 adapter）

```
cudalab/
  evaluator/            通用评估核心（operator-agnostic，v0.4 = v2.3）
    bench.py            paired-streaming-v2.3（burn + 对称 spike/cross-block guard
                        + raw/filtered 双轨 + filter-sensitivity）
    stats.py            round-level paired 统计 + bootstrap CI + v2.3 纯 CPU guard
                        函数（apply_spike_guard / block_stats / crossblock_flag /
                        filter_sensitive，对称化；round 统计部分不变）
    decision.py         KEEP/REJECT/NEUTRAL/UNSTABLE + statistical_relation
                        （FASTER/SLOWER/UNRESOLVED，v0.4.1）+ classify_cell +
                        apply_filter_gate（敏感 → 最终 policy_decision
                        一律 UNSTABLE，v0.4.1 起含 NEUTRAL；
                        记录 original_decision）
    profiler.py         通用 NCU --csv 集成（driver_src/kernel_regex 由 adapter 提供）
    negative.py         通用负例运行器
    experiment.py       实验记录 + 判定 + best.json 生成
    gpu.py / correctness.py
  operators/            算子 adapter（v0.5：四算子）
    base.py             adapter 接口
    rmsnorm.py          RMSNorm adapter（matrix/pool/bytes/正确性/负例/NCU）
    softmax.py          Softmax adapter（同上 + ncu_kernel_regex="softmax"）
    rope.py             RoPE adapter（9 形状池 + 共享 cos/sin 表 + rope_ref +
                        ncu driver；计时 launch 用 validate=False，池契约见 docstring）
    gemv.py             GEMV adapter（5 形状池 + bytes + 正确性(100 例)/
                        负例(24 例) + NCU driver + native_timing 入口）
  build.py              扩展构建 + 内容哈希缓存（build(op="rmsnorm"|"softmax"|"rope"|"gemv")，
                        四扩展独立）
  reference.py          显式 FP32 累加的 RMSNorm 参考实现
  softmax_correctness.py / softmax_negative.py   Softmax 正确性(72 例)/负例(15 例)套件
  rope_correctness.py   RoPE 正确性(384 例：finiteness + double-rounding 算术界
                        K=2 vs fp64 精确旋转 + norm 保持；allclose 报告不门控)
  rope_negative.py      RoPE 负例(37 例 = 34 review 版 + 3 v0.4.1 对齐回归，
                        launch 前 TORCH_CHECK + 启动后检查)
  dispatch.py           RMSNorm 形状/dtype 分发表（v0.2.1 证据政策，v0.4 未改）
  benchmark.py          v0.1 批量 cuda-event 框架（保留，历史对照）
  stats.py / decision.py / profiler.py / bench_v2.py / …   v0.2 导入路径兼容 shim
kernels/rmsnorm/
  rmsnorm_common.h + bindings.cpp + rmsnorm_baseline.cu … rmsnorm_v4.cu   5 变体（v0.3 未改动）
kernels/softmax/          v0.3 新算子
  softmax_common.h  自注册变体注册表 + el_to_float/el_from_float
  softmax_scalar.h  标量 3 遍内核（回退共享实现）
  softmax_baseline.cu      1 块/行 3 遍（参照）
  softmax_vec4.cu          4 宽向量化（**incumbent**，H%4≠0/未对齐回退标量核）
  softmax_online.cu        online (m,l) 单遍 + block merge
  softmax_vec4_ilp2.cu     vec4 + 2 路展开（与 vec4 逐位一致）
  softmax_hsplit2.cu       H 对半分 2 块/行 + (m,l) 跨块合并
                           （**v0.3.1 隔离**: UNSAFE_HISTORICAL_EXPERIMENT /
                           REJECTED / NOT_FOR_NORMAL_DISPATCH；历史证据保留）
  bindings.cpp      PyTorch 扩展入口（统一 launch 前 TORCH_CHECK + 启动后检查；
                    v0.3.1: quarantine 集 → variants() 正常列表 / all_variants()
                    全量 / quarantined_variants() 隔离）
kernels/rope/             v0.4 新算子（interleaved RoPE）
  rope_common.h     自注册变体注册表 + el_to_float/el_from_float + validate 契约
                    （rope_forward/rope_forward_into 入口声明只在 bindings.cpp）
  rope_baseline.cu  1 线程→1 pair（grid M×D/2，block 128，FP32 旋转；**incumbent**）
  rope_v1_2pair.cu  1 线程→2 pairs（grid M×D/4，8 loads 提前，fp16 在途 8B→16B，MLP 杠杆）
  rope_v2_4pair.cu  1 线程→4 pairs（grid M×D/8，16 loads 提前，fp16 在途 8B→32B；要求 D%8==0）
  rope_v3_half2.cu  fp16 `__half2` 打包 load/store + FP32 旋转（指令数削减控制；
                    fp16 路径与 baseline 位级一致 192/192；fp32 路径 = baseline 标量，
                    跨 build 存在 nvcc codegen/FMA 位级漂移，不做位级声明，见报告 §6/§14.7）
  rope_v4_8pair.cu  1 线程→8 pairs（grid M×D/16，32 loads 提前，fp16 在途 8B→64B；要求 D%16==0）
  bindings.cpp      PyTorch 扩展入口（forward/forward_into 带 validate 参数，默认
                    true；validate=false 跳过 positions 值域 D2H 同步，仅供预验证
                    基准池/NCU driver；正确性/negative/正常调用保持完整验证）
kernels/gemv/             v0.5 新算子（y = W@x，fp16 主 + fp32，FP32 累加）
  gemv_common.h     自注册变体注册表 + el_to_float/el_from_float + **共享
                    `gemv_scalar_kernel`**（所有向量化变体的回退 = 与 baseline
                    同一份源码 → 逐位一致）
  gemv_baseline.cu  一行一 block，256 线程，2B 标量 load，warp-shuffle+shared
                    归约（GEMV-0000；**简单形式参照**）
  gemv_vec4_row.cu  GEMV-0001：16B 向量 load × 8 half/次（`uint4`），结构不变
                    —— **incumbent**（1.54×，NCU DRAM 87.9%）
  gemv_warp_vec4_b256.cu  GEMV-0002：warp-per-row + ILP=4，无 shared/barrier
  gemv_warp_vec4_b512.cu  GEMV-0003：同 0002，block 512
  gemv_splitk4.cu   GEMV-0004：split-K×4 两阶段（(N,4) partials + per-row
                    combine；K%4≠0 回退标量；主目标 REJECT 0.878×）
  bindings.cpp      PyTorch 扩展入口（launch 前对齐契约检查 + 统一
                    TORCH_CHECK + 启动后检查）+ **`native_timing`**（1 次
                    Python 调用 → C++ 连续 launch × N → CUDA events / N；
                    三口径计时的 native kernel-loop 表面）
scripts/
  cudalab.py            统一 CLI：test|benchmark|profile|pytorch|optimize
                        {rmsnorm,softmax,rope,gemv}
  bench_gemv_full.py    GEMV 全矩阵驱动（fp16 全 5 变体 / fp32 子集；per-dtype
                        winners tag）
  revalidate_gemv_incumbent.py  最终 incumbent 独立复核（新进程 9r 双模式 +
                        正确性/负例重跑 + PyTorch context）
  dvfs_probe.py / caliber_warmup_probe.py   三口径冲突调查（DVFS 爬坡探测，
                        线程化 nvidia-smi --id=0 时钟采样）
  bench_v2.py / profile_v2.py / test_rmsnorm.py / …   v0.2 入口（保留）
tests/
  test_evaluator_cpu.py     stats/decision 纯 CPU 单元测试（v0.3 全过）
  test_evaluator_v23_cpu.py v2.3 对称 guard / raw-filtered / filter-sensitive
                            + v0.4.1 gate 收紧 / relation-policy 分离
                            确定性单测（36/36）
  test_softmax_cpu.py       Softmax 数值 + online (m,l) merge 恒等测试（20/20）
  test_invalid_inputs.py / test_dispatch.py
docs/
  softmax_algorithm.md          online (m,l) 推导 + CPU 门禁清单
  evaluator_hardening_v0.3.md   v2.2 变更、DVFS guard 偏离说明、残余风险
  evaluator_v2_3.md             v2.3 对称 guard + raw/filtered + filter-sensitivity
                                + v2.3 回归门结果 + RMSNorm 方向翻转调查
  benchmark_audit_v0.2.md / benchmark_audit_v0.3.md   独立方法学审计
  report_v0.5_result.md         v0.5 GEMV 最终报告（12 节：三口径分离 + 冲突
                                调查 + 4 实验 + 全矩阵 + 复核 + review）
tools/env.sh        环境变量的唯一事实来源
experiments/rmsnorm/   EXP-*.json + correctness/ + best.json / best_v0.1.json（v0.2 冻结）
experiments/softmax/   SFM-0001…0004（MD + result/pair JSON）+ correctness/v0.3/
                       + final_reval/ + best.json（36 格 classify_cell）
                       + v0.3.1/（合并修复验证：4 变体正确性 + negative + 隔离验证记录）
experiments/rope/      ROPE-0001…0004（result/pair JSON）+ correctness/v0.4/
                       （baseline + 4 候选 384/384 + 表核对 + invalid_inputs 36/37）
experiments/gemv/      GEMV-0001…0004（result/pair JSON）+ correctness/v0.5/
                       （5 变体 100/100 + invalid_inputs 24/24）+
                       revalidation/（最终 incumbent 独立复核）
benchmarks/softmax/    base_*/inc_*/full5_* 36 格 × 多组 + pair_*（v0.2 路径原样保留）
benchmarks/v0.3_regression/   RMSNorm/Softmax 回归硬门记录（v2.2 协议）
benchmarks/v2.3_regression/   v2.3 回归硬门记录（gate_summary + 4 pair + repeat）
benchmarks/rope/       base_main_{streaming,hot} + ROPE-000{1..4}_pair_* +
                       rope_v04_matrix_*（36 格 × 5 变体）+ *_shape_winners +
                       *_presyncfix_archive（D2H 同步 bug 审计痕迹）
benchmarks/gemv/       gemv_base_*（Phase 4 baseline 20 条）+
                       gemv_GEMV-000{1..4}_pair_* +
                       gemv_full_*（全矩阵 20 条）+
                       gemv_full_shape_winners_{float16,float32}.json +
                       native_timing/（w200 + w5000 双档）
profiles/softmax/      baseline vs 4 候选 NCU 对比 + per-variant 双 cache-control + raw/
profiles/rope/         baseline + 4 候选 NCU（双 cache-control @1755MHz）+ raw/
profiles/gemv/         5 变体 NCU（cc all/none × clk base/none）+ raw/ +
                       caliber_probe/（DVFS/warmup 冲突调查记录）
```

新增内核变体 = 新增一个 `.cu` 文件（自注册；无需改动绑定层）。
**v0.4 约束**：不复制成熟 kernel 源码（全部从零编写）；失败实验永久保留；
dispatcher 默认不做（无 paired 确认的 per-shape 路由证据时不路由）；
RoPE 基准/NCU 计时路径用 `validate=False`（池契约：构造期全量预验证，
positions=0..M-1<4096 必然成立），正确性/negative/正常调用路径保持
完整验证（`validate=True`）。

## 环境

| 项目 | 取值 |
|---|---|
| GPU | 2× NVIDIA RTX 2080 Ti（Turing，CC 7.5）；使用 GPU 0；SM 时钟范围 300–2100 MHz |
| CUDA 工具链 | 11.8（`/usr/local/cuda`） |
| Python | `/root/miniconda3/envs/pytorch/bin/python`（3.10） |
| PyTorch | 2.4.1+cu118 |
| 剖析器 | Nsight Compute 2022.3（`/usr/local/bin/ncu`） |
| compute-sanitizer | **不可用**（容器内未安装；未用作证据） |
| 编译参数 | `-O3 -lineinfo --use_fast_math -gencode=arch=compute_75,code=sm_75` |

## 正确性方法论

- 主参考：显式公式
  `y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)`
  `* w.float()`，再转回输入 dtype（FP32 累加，与 PyTorch 版本无关）。
- 固定容差（**所有变体一致**）：fp16 atol=2e-3 / rtol=5e-3；fp32 atol=1e-5 / rtol=1e-4。
- 每变体 76 例：11 形状 × 3 种子 × 2 dtype + 5 个边界用例 × 2 dtype。
- **负例套件**：30 例非法/未对齐输入（v0.2 28 例 + v0.2.1 新增 2 例 v4 FP32 H=1024
  对齐回归），全部预期在 kernel 启动前被拒（`TORCH_CHECK` 中文报错），
  对照组（16B 对齐的 4B 对齐用例）预期 PASS。
- 正确性 FAIL 的变体无条件 REJECT，永远不可能成为"最佳"。

## 如何复现（统一 CLI；v0.2 入口保留）

```bash
cd /root/code/cuda
source tools/env.sh          # 设置 CUDA_HOME、PATH、PYTHON、架构列表

# 构建（有缓存；冷启动约 1 分钟，热启动几乎瞬时；三算子独立扩展）
$PYTHON cudalab/build.py                  # rmsnorm 扩展
$PYTHON -c "from cudalab.build import build; build('softmax')"   # softmax 扩展
$PYTHON -c "from cudalab.build import build; build('rope')"      # rope 扩展

# 统一 CLI（算子无关；--help 可查全部子命令）
$PYTHON scripts/cudalab.py test softmax --variant softmax_baseline   # 单变体正确性（72 例）+ negative
$PYTHON scripts/cudalab.py test rmsnorm --variant v4_vec_reg         # 单变体正确性（76 例）+ negative
$PYTHON scripts/cudalab.py test rope --variant rope_baseline         # 单变体正确性（384 例 + 表核对）+ negative（37 例）
# 注: 隔离变体（softmax_hsplit2）在 CLI 各入口被拒绝（NOT_FOR_NORMAL_DISPATCH）
$PYTHON scripts/cudalab.py benchmark pair softmax \
    --parent softmax_baseline --candidate softmax_vec4 \
    --M 128 --H 4096 --dtype float16 --mode streaming --rounds 9
$PYTHON scripts/cudalab.py benchmark full softmax  # 36 格全矩阵
$PYTHON scripts/cudalab.py profile softmax         # NCU（双 cache-control）
$PYTHON scripts/cudalab.py pytorch softmax         # PyTorch 参照（仅记录）
$PYTHON scripts/cudalab.py optimize softmax        # 实验脚手架
$PYTHON scripts/cudalab.py test rope --variant rope_v1_2pair   # RoPE 候选正确性
$PYTHON scripts/cudalab.py benchmark full rope --tag rope_v04_matrix   # RoPE 36 格 × 5 变体
$PYTHON scripts/cudalab.py profile rope --M 1024 --H 128       # RoPE NCU（主目标）
$PYTHON scripts/cudalab.py pytorch rope --M 1024 --H 128 --dtype float16   # Python 参照（仅 context）
# 注: RoPE 基准/NCU 计时路径用 validate=False（池契约，见 cudalab/operators/rope.py
#     make_bench_pool docstring）；test / 正常调用路径保持完整验证（validate=True）。

# v0.2 正确性（5 变体 → experiments/rmsnorm/correctness/v0.2/）
$PYTHON - <<'EOF'
import sys; sys.path.insert(0, "scripts")
from _common import get_ext
from cudalab.correctness import run_suite, summarize, save_results
from pathlib import Path
ext = get_ext(); out = Path("experiments/rmsnorm/correctness/v0.2")
for v in ext.variants():
    r = run_suite(v, ext, dtypes=("float16","float32"), edge=True)
    print(v, summarize(r)); save_results(r, out / f"{v}.json")
EOF

# v0.2 负例套件
$PYTHON tests/test_invalid_inputs.py

# v0.2 配对基准（主形状，9 rounds）
$PYTHON scripts/bench_v2.py pair --parent v4_vec_reg --candidate v1_vec \
    --M 128 --H 4096 --dtype float16 --mode streaming --rounds 9 \
    --tag v02_pair_v4_vs_v1_M128_H4096_fp16_streaming

# v0.2 全矩阵（7 形状 × 2 dtype × 2 模式 × 5 变体，约 15 分钟）
$PYTHON scripts/bench_v2.py full

# v0.2 双缓存 NCU 剖析 → profiles/rmsnorm/v0.2/
$PYTHON scripts/profile_v2.py

# CPU 单元测试（无需 GPU）
$PYTHON tests/test_evaluator_cpu.py
$PYTHON tests/test_evaluator_v23_cpu.py  # v0.4.1: 36/36（对称 guard / raw-filtered / filter-sensitive / raw 侧约定 / gate 收紧 / relation-policy 分离）
$PYTHON tests/test_dispatch.py
$PYTHON tests/test_softmax_cpu.py        # v0.3: 20/20（数值 + online merge 恒等）
```

产物（v0.4）：`experiments/rope/`（ROPE-0001…0004 + correctness/v0.4/）、
`benchmarks/rope/`（base_main + pair + 36 格矩阵 + shape_winners +
presyncfix 审计痕迹）、`profiles/rope/`、`benchmarks/v2.3_regression/`
（v2.3 回归门）。
产物（v0.3）：`experiments/softmax/`（SFM-0001…0004 + correctness/v0.3/ +
final_reval/ + best.json）、`benchmarks/softmax/`、`profiles/softmax/`、
`benchmarks/v0.3_regression/`（RMSNorm 回归）。
产物（v0.2，保留）：`experiments/rmsnorm/EXP-0008.json`、`benchmarks/v0.2/`
（含 `shape_winners.json`）、`profiles/rmsnorm/v0.2/`、
`experiments/rmsnorm/correctness/v0.2/`。

## 局限（v0.5 更新）

- **v0.5 新增（GEMV 三口径计时）**：
  - **native kernel-loop 默认 warmup=200 对时钟敏感 kernel 系统性偏慢**：
    本 GPU idle 缺口后从 base 1350 MHz 爬向 boost 1890 MHz 需数百 ms；
    200 次 warmup（~22 ms）落在爬坡内，延迟受限的 gemv_baseline 在
    w200 下读 109.8 µs vs w5000 91.3 µs（+20%）。v0.5 全部 native 记录
    同时提供 w200 与 w5000 两档，报告统一用 w5000（boost 稳态）；
    DRAM 饱和 kernel（vec4_row）API/native 差 <1.3%（时钟不敏感；NCU
    残余 ~7% 与下一条同源，与时钟无关）。详见报告 §5 与
    `profiles/gemv/caliber_probe/`。
  - NCU@none 比 API-path 高 ~7%（cc=all 每 replay 前 flush 缓存 + 剖析
    隔离），方向已解释、幅度已量化、不影响 paired 决策（报告 §5.4）。
  - **多内核算子的 NCU 顶层标量是跨内核平均**：v0.5 起 profiler 新增
    逐内核 `kernels[]` 分解 + `multi_kernel_note`（由 splitk4 两阶段
    数据触发；顶层字段语义不变、向后兼容）。读旧记录无此问题（均单
    内核算子）。
  - **splitk4 static workspace（已隔离）**：`gemv_splitk4` 的进程级
    `static at::Tensor g_splitk_partials` 在多 stream / 多线程并发下 race、
    且固定留在首次调用设备（跨 device 风险）—— 正常 dispatch 不安全，
    v0.5 merge review 后正式隔离（NOT_FOR_NORMAL_DISPATCH）：移出
    `ext.variants()` 与 CLI test/benchmark/optimize/profile 正常路径，
    源码与全部 bench/NCU 历史保留，显式 `forward("gemv_splitk4", ...)`
    为受控历史审计入口（报告 §6/§12）。
- **历史 experiment artifact 不可变（v0.5 merge review 新约定）**：
  main（4eb520b）上提交的 experiment 记录一律不再改写；新验证只
  append 到 `experiments/regression/<版本>/<op>/`（含 README 说明
  来源与约定）；RMSNorm/Softmax/RoPE 三算子的默认输出目录已改指
  该处，防止再覆盖历史文件。
- 环境有 **2× 2080 Ti**（nvidia-smi 可见 index 0/1）；v0.5 全部测量固定
  GPU 0（`CUDA_VISIBLE_DEVICES=0` + 时钟采样 `nvidia-smi --id=0`）。
- 四个算子（RMSNorm + row-wise Softmax + interleaved RoPE + GEMV）；单 GPU
  （GPU 0）；仅连续输入；fp16/fp32（**禁 BF16**）。
- 变体 H/D 支持约束如上（baseline 完全通用；softmax 非对齐/小 H 回退
  共享标量核；RoPE v2_4pair 要求 D%8==0、v4_8pair 要求 D%16==0）。
- **RoPE 的 `validate` 契约（v0.4 新增，需理解后再复用）**：positions
  值域检查（0≤p<L）需要一次**同步** D2H 拷贝（逐 launch ~25–30 µs 流
  同步），因此 `forward`/`forward_into` 带 `validate` 参数（默认 true，
  完整验证）；基准池与 NCU driver 用 `validate=False`（池构造期已全量
  预验证 + positions=0..M-1<4096 必然成立）。**误用 validate=False 于
  未预验证的输入会跳过值域检查**——正常 API 调用/测试/negative 一律
  保持默认 true。这是 v0.4 首跑 baseline 28.8 µs（后修正为 6.98 µs）
  的根因，详见 evaluator_v2_3.md 与 report_v0.4。
- **PyTorch 2.4.1 无内置 fused RoPE op**：`pytorch rope` 子命令用 Python
  参考（rope_ref，多 kernel 的索引 + 旋转）作 implementation context，
  **不是公平 kernel 对比**，不产生 "X× faster than PyTorch" headline
  （与 Softmax 的 `F.softmax` 参照同语义：仅记录）。
- **机器态敏感性（v0.3 确认，v0.4 再证，最重要）**：(128,4096) fp16 的
  **hot** 模式配对结果跨运行漂移（vec4 vs baseline hot：1.2916@21:03 →
  0.9865@21:41 两次 paired 运行；绝对时间 4.684 µs@21:42 vs 6.797 µs@21:41，
  相隔约 90 秒漂移 ~45%，baseline 稳定 ~6.7 µs；streaming 稳定 1.68）；
  v0.2 时代的跨运行结论在 v0.3/v0.4 机器态下重跑会出现点估计漂移
  （RMSNorm hot 0.9327 REJECT → 1.0144 NEUTRAL；v2.3 回归门 RMSNorm
  streaming 1.0375 → 0.9576 数分钟内方向翻转，调查定性为环境微态漂移
  非 evaluator 缺陷，见 evaluator_v2_3.md §6）。**跨运行绝对时间不可比，
  仅运行内配对有决策意义**；依赖 hot cell 的结论需以 streaming 或复跑
  确认。
- 容器内无法锁定 GPU 时钟 → v2.3 的对称 spike/cross-block guard 只能
  **检测并拒绝**异常轮（快慢双向），不能预防；v2.1 的 round 内
  nvidia-smi 采样已被移除（其 ~40ms 空闲间隙会推入性能退化态，见
  evaluator_hardening_v0.3.md）。
- NCU `--clock-control base` 是否真正锁频在容器内无正面证据（无警告也无确认）。
- 矩阵模式（round-robin，非配对）的 round-level ratio 对离群干扰轮敏感
  （如 (16,4096) fp32 hot 的 round 2 有 3/5 变体升至 10–13 µs，且该轮时钟恒
  1350 MHz、DVFS guard 未拦截）：矩阵 winner 仅指示性，最终判定以 paired A/B 为准。
- `algorithmic_bw_gbps` 是逻辑吞吐（算法 IO / 时间），不是实测 DRAM 带宽，
  不能由它得出"饱和 / 接近饱和"结论（v0.3.1 措辞更正：2080 Ti 规格峰值
  616 GB/s，(1024,4096) fp16 streaming 逻辑吞吐 ≈485 GB/s = 78.7% 规格峰值，
  DRAM 饱和未建立，是否真正达到饱和需对应 NCU 验证）；M=1 区域是
  launch-bound，绝对延迟无意义。
- `compute-sanitizer` 不可用，未做越界/竞态检查（v0.1/v0.2 均如此）。
- 分发表（v0.2.1 证据政策）仅在 3 个实测格路由优化变体（2 个 paired-evidence +
  1 个 incumbent-fallback）；其余实测格（matrix-only，含 hot/streaming 冲突格）
  与所有未实测组合一律 baseline —— evidence > coverage。
- v0.1 数据保留在案但**已被 v0.2 取代**：跨版本数字不可直接比较
  （harness、时钟条件、缓存策略均不同）。

## 路线图（v0.5 建议；v0.3/v0.4 已完成项以 ~~删除线~~ 标出）

- ~~新内核（Softmax、RoPE）复用 v0.2 客观层~~ —— **Softmax 已在 v0.3 完成、
  RoPE 已在 v0.4 完成**（第三算子自然接入验证成功；evaluator 同期升级到
  v2.3）。
- ~~v0.5 候选：GEMV~~ —— **已在 v0.5 完成**（分支 `v0.5-gemv`：
  baseline + 4 候选、vec4_row 1.54× KEEP、全矩阵 10/10、三口径计时、
  失败实验保留；不 merge main）。
- ~~v0.6 建议：Quantized GEMV~~ —— **已在 v0.6 完成**（分支
  `v0.6-qgemv`，基线 main = v0.5.1 = d635903，不 merge main，等待外部
  review）：INT8 每行对称量化 + 内核内 dequant，final incumbent
  `qgemv_warp_vec16` 31.513 µs = **2.69× vs INT8 baseline、1.90× vs
  FP16**（95% of 理论 2×），4 实验（1 KEEP + 3 NEUTRAL，保留）、
  50/50×5 + 29/29×5 两层正确性、三口径无冲突、全矩阵 10/10 全胜
  baseline。
- **v0.7 建议：INT4 + group-wise 量化**（报告 §11 Q7 结论：per-row scale
  在 4-bit 下保真度不足，group-wise（128/256）scale 为共需项；逻辑 IO
  16.81 → 8.93 MB / 下界 27.29 → 14.50 µs，理论 1.88×，效率折损后
  预期再 ~1.7–1.8×）。
- 支持锁频的环境（裸机/特权容器）下重跑 paired harness，验证 v2.3 对称
  guard 在零失配条件下的噪声下限；重点复验 (128,4096) fp16 **hot** 模式
  （v0.3 已确认其机器态敏感性，streaming 结论稳健）。
- 引入 compute-sanitizer（越界/竞态）作为正确性的第二道门。
- fp32 路径专项：v4 的 fp32 寄存器路径是已知弱点（v2 快 1.37–1.62×），
  允许新变体时优先做 fp32 向量化重设计。
- 更大 M（8192/16384）矩阵，覆盖带宽敏感区（DRAM 饱和判断需对应 NCU 验证）。
