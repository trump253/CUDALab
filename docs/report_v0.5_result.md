# CUDALab v0.5 Result — FP16 GEMV 优化

**分支**: `v0.5-gemv`（基线 `main` = v0.4.1 = 4eb520b）
**日期**: 2026-09-21（UTC+08:00）
**硬件/环境**: NVIDIA RTX 2080 Ti × 2（**固定 GPU 0**；Turing, sm_75, 30 SM, L2 5.5 MB,
规格峰值 DRAM 带宽 616 GB/s；实测 boost SM 1890 MHz / base 1350 MHz），
CUDA 11.8，PyTorch 2.4.1+cu118，NCU 2022.3，容器化（无 compute-sanitizer）。
**分支状态**: 未 merge 回 main（用户指定）。

---

## 1. Status

**v0.5 完成（PASS 结局）。GEMV 主目标 (N=4096, K=4096) fp16：baseline 93.8–94.2 µs
（API-path, ~360 GB/s, 58% 峰值）→ 最终 incumbent `gemv_vec4_row` 59.9 µs
（~560 GB/s, **91% 峰值**），paired 1.54×（9/9 轮, CI95 [1.540, 1.552]），
独立复核再次 KEEP（streaming + hot 双口径）。**

- 4 个完整证据链实验（GEMV-0001..0004）：vec4_row **KEEP 1.5416×**、warp_vec4_b256
  **KEEP 1.2539×**、warp_vec4_b512 **KEEP 1.2407×**、splitk4 **REJECT 0.8784×**
  （失败实验全部保留为证据）。
- 正确性 **100/100 × 5 变体**（固定算术误差界容差，非 per-variant 调参）+
  负例 **24/24 × 5 变体**（含 3 个向量化回退**逐位一致**回归门）。
- 三种计时口径（API-path / native kernel-loop / NCU kernel duration）按用户要求
  **分开记录、不混用**；三者在主目标上出现系统性排序差（API < native < NCU），
  **已定位根因并定量复现**：idle 缺口后的 DVFS 爬坡（base 1350 MHz vs boost
  1890 MHz）+ native 表面 200-launch warmup 落在爬坡内（§5）。控制时钟状态后
  API 与 native-w5000 差 <1.5%；NCU@none 仍有 ~7% 残余
  （cache flush + 剖析隔离，方向已解释，不影响任何 paired 决策）。
- 全 shape matrix（5 形状 × {hot, streaming}）：**vec4_row 10/10 格胜出**
  （fp16 1.51–1.63× vs baseline；fp32 子集 1.06–1.11×）。
- RMSNorm / Softmax / RoPE smoke 回归通过（§10）；evaluator v2.3 决策/基准路径
  **零改动**，仅 profiler 摘要一处增量修复（多内核算子，§10）。
- 独立 review（CUDA correctness + benchmark methodology）：双双 **PASS WITH
  CAVEATS**，全部 findings 已处置（§12 处置表）。

**停止条件核对**（全部满足）：v0.4.1 已 release（main=4eb520b，本次未动 main）✓；
GEMV baseline ✓；正确性+负例 ✓；baseline NCU ✓；API+native 计时 ✓；≥4 实验 ✓；
全 shape matrix ✓；hot+streaming ✓；statistical_relation/policy_decision 分列 ✓；
最终 incumbent 复核 ✓；三算子 smoke 回归 ✓；README/STATUS/本报告 ✓；
独立 CUDA + benchmark review ✓（§12）；git 树干净 ✓；推送 v0.5-gemv ✓（未 merge）。

---

## 2. 算子定义与范围

**问题**: `y = W @ x`，W `[N, K]` 连续行主序，x `[K]`，y `[N]`；每行
`y[n] = Σ_k W[n,k]·x[k]`，主路径 **W/x/output = FP16，累加 = FP32**；
参考实现 `torch.mv(W.float(), x.float()).to(torch.float16)`。FP32 输入/输出
为支持路径（非主目标）。

**形状**（(N, K)，LLM hidden/MLP 典型）: (1024, 4096), (4096, 1024),
**(4096, 4096) 主目标**, (11008, 4096), (4096, 11008)。

**显式排除（用户指定）**: 不做 GEMM、quantization、Attention 或 CUDALM 集成。
PyTorch `torch.mv` 仅作 framework context（§9），**不是候选、不是决策基准**；
决策 = incumbent vs candidate 的 paired 证据。

**复杂度定位**: GEMV 算术强度 ≈ 1 FLOP/byte，纯 DRAM 带宽受限。
主目标算法字节数 = (N·K + K + N)·2 = 33,570,816 B；616 GB/s 下理想
= **54.5 µs**（全版本对照基准）。

---

## 3. 正确性

**容差（固定算术误差界，全变体同一标准，非 per-variant 调参）**:
每个输出格子的算术上界

```
tol = TOL_K(2) · ( 2K · 2^-24 · S_n                 # FP32 累加 FMA 误差界（S_n = Σ|w·x|）
                       + 0.5·ulp_dtype(|y|)          # 输出舍入
                       + 0.5·ulp_dtype(|exact|) )    # 参考舍入
```

**门 = arith_max_ratio ≤ 1 且 y 有限且 ref 有限**；`torch.allclose` 数值只
**记录报告、不作门**（避免 1e-3 量级 allclose 对大数值 fp16 假阳性/假阴性）。
max_abs/max_rel/NaN/Inf 全部逐项记录。

**测试矩阵**: 2 dtype（fp16/fp32）× 10 (N,K) × 5 模式（random / zeros /
small / large / mixed-sign）= **每变体 100 项**，覆盖不同 N/K。
**large 模式 scale=10**（y 的 σ = scale²·√K：K=4096 → 6400，8σ ≈ 51200 < 65504；
K=11008 → σ ≈ 10492，4.5σ ≈ 47000 < 65504，固定 seed 数据上验证全有限）。

**Phase 1 发现并修复的数据生成缺陷（保留为教训）**: W 与 x 同 seed 同 device
生成时走同一段随机流，x 成为 W flat 布局的**前缀**（x[k] == W[0,k]），
large 模式下 y[0] = Σ W[0,k]² ≈ K·scale² 溢出 fp16（94/100 → 6 项 ref_has_inf
FAIL）。修复：W seed=SEED、x seed=SEED+1（**独立随机流**），100/100 全有限。

**负例（每变体 24 项, 对每个受测变体运行并归档全套 —— v0.5 独立
审查 MAJOR-1 修复）**: 18 项必须拒绝（shape / dtype / device / 非连续 /
out 形状·dtype·设备 / 未知变体 / x 长度不匹配，全部要求 launch 前显式异常 +
CUDA context 无污染后置检查）+ 6 项合法输入必须通过，其中 3 项为
**标量回退逐位一致回归**（对每个向量化变体自身执行；对隔离的
gemv_splitk4 仅 K=13 一项适用，见下表）：

| 用例 | 构造 | 向量化变体语义 | splitk4 语义（隔离后） |
|---|---|---|---|
| fallback_W_misaligned | W = 大块[2:2+N·K].view(N,K)，基址偏移 4B | 16B 对齐不满足 → 标量回退 | **无对齐约束**：直接走 split-K 路径（finite-only control） |
| fallback_x_misaligned | x = 大块[1:1+K]，基址偏移 2B | 同上 | 同上 |
| fallback_K_not_mult8 | N=16, K=13（K%8=5 且 K%4=1） | 向量契约不满足 → 标量回退 | 唯一契约 K%4==0 不满足 → 标量回退（bit-identical 真契约） |

回退门要求：不拒绝、成功执行、**输出与 gemv_baseline `torch.equal` 逐位一致**。
逐位一致由构造保证：所有回退与 baseline 调用**同一份** `gemv_scalar_kernel`
（`kernels/gemv/gemv_common.h` 单一来源）。**splitk4 例外**（v0.5 merge
review 澄清）: splitk4 为标量访存、**无对齐契约** —— K%4==0 时的错位
W/x 直接走 split-K 路径（归约顺序与 baseline 不同，bit-identical 不
保证）; v0.5 归档记录中该两例的 bit-identical 是种子数据巧合，套件
代码已按此重构（`cudalab/gemv_negative.py`），归档记录
`invalid_inputs_gemv_splitk4.json` 追加 `note_addendum` 说明。

**向量化对齐契约**（所有 16B load 变体显式声明，host 侧 launch 前检查）:
W 基指针 16B 对齐 ∧ x 基指针 16B 对齐 ∧ K % (16B 内元素数) == 0
（fp16: 8, fp32: 4）。不满足 → 标量回退，**永不拒绝合法输入**。
out 为标量 2B/4B 写，无对齐检查。`is_contiguous()==true` 不保证对齐
（offset view 用例钉死此语义）。**对齐契约不适用于标量变体**：
gemv_baseline 与（隔离的）gemv_splitk4 均为标量访存，无对齐约束
（splitk4 的唯一契约是 K%4==0）。

**结果**（v0.5 独立审查修复后重录归档）: 5/5 变体 正确性 100/100 +
负例 24/24。记录：`experiments/gemv/correctness/v0.5/` 下 5 个
`<variant>.json` + `invalid_inputs.json`（baseline 规范套件）+ 4 个
`invalid_inputs_<variant>.json`（向量化变体 per-variant 套件, MAJOR-1）。
baseline max_arith_max_ratio=0.25（误差界的 25%）、max_abs=4.0（= 1 ulp,
|y| ∈ [4096, 8192) binade, large 模式 fp16）、max_rel=0.0189；
各变体同级（warp 双 b 变体 max_abs=8.0 = 1 ulp, |y| ∈ [8192, 16384)
binade），跨变体 max_rel 区间 0.0124–0.0217（修复前记录为 0.0122–0.022, 量级不变）。

**mixed_sign 构造修复（MINOR-2, 前后对照）**: 原构造 W 符号 (i+j)%2 ×
x 符号 k%2 → 乘积符号 = (-1)^i **逐行恒定**，|y| = Σ|W·x|，该模式
抵消深度为零（真实抵消覆盖此前来自 random 模式）；已修复为 x 独立
Bernoulli 符号流（乘积符号独立, 行求和出现真实正负抵消）, 并**全套
重跑 5 变体正确性重录**（修复前记录保留于 75c1ccd git 历史）。

**v0.5 merge review 后（隔离 + append-only 约定）**: gemv_splitk4 被
隔离（NOT_FOR_NORMAL_DISPATCH，见 §6 GEMV-0004 行与 §11）后，正常
变体集为 4 个（gemv_baseline / gemv_vec4_row / gemv_warp_vec4_b256 /
gemv_warp_vec4_b512）。此后所有再验证**只追加、不覆盖**上述 v0.5
官方记录 —— 隔离后的 4 变体正确性 / 负例重录存于
`experiments/regression/v0.5/gemv/`（append-only 约定见该目录
README）。

---

## 4. Baseline 与 NCU 诊断

`gemv_baseline`（GEMV-0000，用户指定简单形式）: 一行一 block、256 线程
（8 warp）、**2B 标量 load**、每线程 strided FP32 累加、5 步 warp shuffle +
shared 8 槽 + warp0 3 步二级归约、lane0 写回。regs=16，无 shared 压力。

**NCU（主目标 4096×4096 fp16, `--clock-control base`, `--cache-control all`）**:

| 指标 | baseline |
|---|---|
| kernel duration | **114.76 µs** |
| DRAM throughput | **49.34%**（≈304 GB/s @base clock） |
| SM throughput | 25.86% |
| achieved occupancy | 94.16% |
| L2 read hit / L1 hit | 2.28% / 48.73% |
| regs / shared | 16 / 32 B |

**warp 停顿**: long_scoreboard **22.37 cyc/issue（79.3%）**、wait 6.9%、
selected 3.5%、barrier 2.9%、not_selected 1.9%、short_scoreboard 1.6%。

**诊断**: DRAM **延迟受限 / 内存级并行（MLP）不足**——2B 标量 load 使每线程
在途字节太少，长记分板停顿占绝对主导；**不是** occupancy（94%）/ SM（26%）/
barrier（2.9%）受限。优化方向 = 增大每线程在途字节（向量 load / 提高 ILP），
而非堆 occupancy 或改 block 布局。

---

## 5. 三种计时口径与冲突调查

按用户要求，三种口径**分开测量、分开报告、不混为同一指标**。
主目标 (4096,4096) fp16：

| 口径 | 定义 | baseline | vec4_row | warp_b256 | warp_b512 | splitk4 |
|---|---|---|---|---|---|---|
| **API-path**（paired-streaming-v2.3, 9r, 300ms 时间制 burn, 32-launch 块） | streaming 中位数 | 93.88 µs | 59.89 µs | 74.68 µs | 75.24 µs | 106.89 µs |
| **native kernel-loop**（1 次 Python 调用 → C++ 连续 64 次 raw launch × 10 windows → CUDA events → /64；warmup=200） | window 中位数 | **109.82 µs** | 60.42 µs | 90.20 µs | 80.27 µs | 108.59 µs |
| **native kernel-loop**（同上，warmup=5000 ≈550ms） | window 中位数 | **91.35 µs** | 59.65 µs | 72.08 µs | 72.67 µs | 103.57 µs |
| **NCU kernel duration**（base clock 锁频, cc=all） | kernel-only | **114.76 µs** | 64.19 µs | 103.02 µs | 103.46 µs | partials 132.13 + combine 2.43 |
| **NCU kernel duration**（不锁频, cc=all） | kernel-only | 98.85 µs | 63.70 µs | — | — | — |

> **记录溯源注记（独立审查 MINOR-3）**: baseline 的两条 @base 记录
> （114.76 / 114.74 µs）剖于 75c1ccd 标量重构**前**的 `gemv_baseline_kernel`
> （de150bc 构建, 旧 schema 摘要, 无 kernels[]）; baseline 的两条 @none 记录
> （99.03 / 98.85 µs）为 25bd9ab 重录的 `gemv_scalar_kernel` —— 两个内核**代码
> 本体逐行相同**（`gemv_baseline.cu`@de150bc vs `gemv_common.h`, 独立审查
> diff 核实）, 时长比 1.160 与 DVFS 解释（1350→1890 MHz）自洽。

**冲突**: 同一形状上 API(93.9) < native-w200(109.8) < NCU@base(114.8)——
三口径排序一致但与 v0.4 RoPE（NCU 快于 API）方向相反。按用户要求**记录并
调查**，结论如下（探测脚本与原始样本：`profiles/gemv/caliber_probe/`）：

1. **NCU 档差 = 刻意的 base 锁频**。`--clock-control base` 把 GPU 锁在
   1350 MHz；不锁频 NCU@none = 98.85 µs，落回 native/API 区间。
2. **native-w200 档差 = idle 缺口后的 DVFS 爬坡，被短 warmup 捕获**。
   - 持续/突发负载对比探测（sustained 6.3k 连续 launch vs 32-launch 块 +
     Python 间隙，各 ~1 s）：两模式 SM 时钟**均稳态 1890 MHz**、功耗
     ~253 W、温度 47–51 °C → **排除**"持续满载降压"假设。
   - warmup 探测（3 s 空闲缺口后分别以 200 / 5000 次 warmup 进入计时）：
     200 次（≈22 ms）时**采样 SM 时钟 = 1350 MHz**，window 中位数
     **109.15 µs**；5000 次（≈550 ms）时 1890 MHz，**91.34 µs**。
      （时钟采样稀疏：nvidia-smi 进程启动开销使有效采样周期 ~100 ms，
      每相位仅 n=1 样本 —— 时钟值只作佐证，**主证据是 warmup 长度与
      时长的定量关系**；独立审查 NIT-5 披露。）
     即 native 表面默认 200-launch warmup 落在 GPU 从 base 爬向 boost 的
     窗口内（v2.3 文档早已记录该态衰减 8–10 ms 至 >150 ms；API-path 的
     ≥300 ms 时间制 burn 正是为此设计）。
3. **时钟敏感度解释变体间差异**：baseline 是延迟受限（49% DRAM），对 SM
   时钟敏感 → 1350 MHz 时 109.2 µs vs 1890 MHz 时 91.3 µs（+19.6%）；
   vec4_row 已 DRAM 饱和（88–90% DRAM），对 SM 时钟**不敏感** → w200 60.42 ≈
   w5000 59.65 ≈ API 59.89（三口径差 <1.3%）。
4. **控制时钟状态后 API / native 两口径一致（<1.5%），NCU 残余 ~7% 已
   解释**：1890 MHz 下 API 92.75（复核 run）/ native-w5000 91.35；
   NCU@none 98.85 仍高 ~7%，方向可解释（cc=all 每次 replay 前 flush 全部
   缓存，L2 常驻的 x/out 变冷 + 剖析隔离开销），幅度小、**不影响任何
   paired 决策**（决策在单一口径内完成）。

**与 v0.4 方向相反的说明**: v0.4 RoPE 主目标 (1024,128) 是 host 发射受限
（kernel ~4 µs，launch 间隙占主导），NCU kernel-only 自然快于 API-path；
GEMV 是 DRAM 受限（kernel ≥60 µs，launch 开销被隐藏），API-path ≈ 真实
kernel 执行时间，两档差变成**时钟状态**问题。两版本结论不矛盾，是瓶颈资源
不同导致的口径关系翻转。

---

## 6. 优化实验（GEMV-0001..0004）

主目标 (4096,4096) fp16，streaming（决策口径），9 rounds，parent = baseline，
paired-streaming-v2.3。每个实验含完整证据链：correctness 100/100 +
negative 24/24（gate 当时运行 baseline 规范套件; v0.5 独立审查
MAJOR-1 后已补齐 per-variant 负例归档, 见 §3 与实验记录 additive note）
+ paired bench +（KEEP/REJECT 后）NCU 机理。
`filter_sensitive = false`（4/4）。实验记录：`experiments/gemv/GEMV-000*.json`。

| ID | 候选 | 杠杆（假设） | 改动 | paired（parent → cand, µs） | speedup | CI95 | 更快轮 | policy |
|---|---|---|---|---|---|---|---|---|
| **GEMV-0001** | `gemv_vec4_row` | 向量化（保持 block-per-row 结构）：16B load × 8 half/次，每线程在途字节 4× | W/x load → `uint4` 16B 向量（`#pragma unroll 4`），reduction 结构不变；契约不满足回退标量 | 92.34 → **59.89** | **1.5416** | [1.5398, 1.5517] | 9/9 | **KEEP** |
| **GEMV-0002** | `gemv_warp_vec4_b256` | warp-per-row + 16B load + ILP=4：消除 shared reduction 与 barrier | 1 warp/行，lane stride-128 覆盖，4 独立累加链，5 步 warp shuffle，无 shared/barrier；block 256 | 93.25 → 74.68 | 1.2539 | [1.2454, 1.2555] | 9/9 | **KEEP** |
| **GEMV-0003** | `gemv_warp_vec4_b512` | block size 杠杆（同 0002 结构，16 行/block） | block 512，其余同 0002 | 93.75 → 75.24 | 1.2407 | [1.2396, 1.2446] | 9/9 | **KEEP** |
| **GEMV-0004** ⚠隔离 | `gemv_splitk4` | split-K×4 并行度杠杆（预期主目标收益有限） | 两阶段：(N,4) grid 归约 K/4 段 → fp32 partials [N][4] + 每行 1 线程 combine；标量 load（结构实验） | 94.08 → 106.89 | 0.8784 | [0.8755, 0.8804] | 0/9 | **REJECT**（已隔离 ⚠） |

⚠ **隔离说明（v0.5 merge review, 2026-09-21）**: `gemv_splitk4` 标记为
UNSAFE_HISTORICAL_EXPERIMENT / REJECTED / NOT_FOR_NORMAL_DISPATCH ——
其 `static at::Tensor g_splitk_partials` 进程级 workspace 在**多 stream
并发调用时存在 race**（一次调用的 combine 可能读到另一次的 partials），
且 workspace 固定在首次调用的 device（**跨 device 调用拿到错误设备的
workspace**）。该内核为标量访存（无对齐契约，唯一契约 K%4==0），静态
workspace 无任何收益；决策 REJECT 也使其永非正常 dispatch 目标。处置：
从 `ext.variants()` 与 CLI test/benchmark/optimize/profile 正常路径移除
（`kernels/gemv/bindings.cpp` `quarantined_set`；显式请求报隔离错误）；
源码、本行全部 bench/NCU 历史数据与实验记录原样保留（实验记录含
additive `quarantine_note`）；显式 `forward("gemv_splitk4", ...)` 保留为
受控历史审计入口（单 stream / 单 device / 单线程）。上述 paired/NCU
数字为单 stream 单 device harness 下的有效证据，不受隔离影响。

**NCU 机理解释**（base clock, cc=all）:

| 变体 | duration | DRAM | 主导停顿 | 解读 |
|---|---|---|---|---|
| baseline | 114.76 µs | 49.3% | long_scoreboard 79.3% | 2B load MLP 不足，延迟受限 |
| **vec4_row** | **64.19 µs** | **87.9%** | long_scoreboard 82.8%（30.7 cyc） | 16B load 在途字节 4× → **DRAM 饱和**（停顿占比不变但 DRAM 吞吐翻倍，延迟被吸收） |
| warp_b256 | 103.02 µs | 53.9% | **lg_throttle 84.4%**（52.4 cyc） | ILP=4×16B 撞 **LSU issue 饱和**；61 regs → occupancy 81.3%；barrier 消除的收益被 LSU 瓶颈吃掉 |
| warp_b512 | 103.46 µs | 54.7% | lg_throttle（同族） | block size 512 未改机理（occ 85.3%） |
| splitk4 | partials 132.13 + combine 2.43 µs | 43.0%（partials） | barrier 71.9/71.2%（partials @ccall; @ccnone 72.4/72.1） | 每线程仅 K/4/256 = **4 元素** → MLP 严重不足；另加 64 KB partials 写 + 二次 launch |

**结论**: 单杠杆"向量化"（0001）在保持 baseline 结构下即达 DRAM 饱和，是
主目标最优；warp-per-row 的 ILP 设计在 sm_75 上把瓶颈从延迟搬到 LSU issue
（次优但 KEEP，>5% 门槛）；split-K 在 N=4096（行已饱和 30 SM）+ K 分段后
每线程工作过少，主目标 REJECT——与其假设一致（其价值在小 N，见 §8 讨论）。

---

## 7. 主目标最终结果（最终 incumbent）

**`gemv_vec4_row` @ (N=4096, K=4096) fp16**（三种口径分开报告）:

| 口径 | 延迟 | 算法带宽 | 占 616 GB/s 峰值 |
|---|---|---|---|
| API-path（streaming, 9r, 复核 run） | **59.879 µs**（hot 59.840） | 560.6 GB/s | **91.0%** |
| native kernel-loop（w5000） | 59.649 µs | 562.8 GB/s | 91.4% |
| NCU kernel duration @base / @none | 64.19 / 63.70 µs | —（DRAM 87.9% / 90.2%） | — |

- vs baseline：paired **1.5406–1.5521×（streaming）/ 1.5422–1.5501×（hot）**
  —— 独立复核（新进程、新 9-round run、correctness 100/100 + negative 24/24
  重跑）：**KEEP / KEEP**，`statistical_relation = FASTER`，
  `policy_decision = KEEP`（双字段分列记录）。
- vs 理想（54.5 µs @ 100% 峰值）：**1.098× 差距**（91% → 峰值余量 ~9%；
  DRAM 88–90% 已接近该硬件 GEMV 的带宽天花板）。
- 复核记录：`experiments/gemv/revalidation/gemv_vec4_row_revalidation.json`。
- v0.5 独立审查后复核 v2（新进程、含 incumbent per-variant 负例 +
  mixed_sign 修复后正确性; 原记录不覆盖, 新文件
  `gemv_vec4_row_revalidation_v2.json`）: streaming 91.826 → 59.840 µs
  CI95 [1.5266, 1.5378] **KEEP** / hot 92.809 → 59.779 µs CI95
  [1.5521, 1.5602] **KEEP**; 两次复核的 run 间点估计漂移（baseline
  92.75 → 91.83 µs, ~1%）属 §11.3 已确立的机器态特性, 结论只用
  run 内 paired 比值。

---

## 8. 全 Shape Matrix

`benchmarks/gemv/gemv_full_M*_H*_<dtype>_<mode>.json`（5 形状 × {hot,
streaming}，9 rounds，paired-streaming-v2.3；winners：
`gemv_full_shape_winners_float16/float32.json`）。

**FP16（全 5 变体；median µs，run 内同形状同模式可比）**:

| (N,K) | baseline | **vec4_row** | warp_b256 | warp_b512 | splitk4 | winner（vs baseline） |
|---|---|---|---|---|---|---|
| (1024,4096) | 25.52/25.84 | **16.90/16.93** | 20.60/21.03 | 20.76/21.07 | 29.39/29.76 | vec4_row **1.51×** |
| (4096,1024) | 28.15/28.16 | **17.27/17.26** | 21.25/21.47 | 21.54/21.67 | 48.83/48.97 | vec4_row **1.63×** |
| (4096,4096) | 94.16/94.21 | **59.94/59.93** | 76.29/76.42 | 76.60/76.68 | 107.89/107.87 | vec4_row **1.57×** |
| (11008,4096) | 248.06/248.07 | **156.64/156.66** | 196.03/196.41 | 199.80/200.01 | 283.18/282.76 | vec4_row **1.58×** |
| (4096,11008) | 254.80/254.96 | **156.74/156.86** | 200.06/200.07 | 200.11/200.03 | 260.22/260.14 | vec4_row **1.62×** |

（每格 "hot/streaming"；vec4_row 带宽：(4096,4096) 560 GB/s 91%；
(11008,4096)/(4096,11008) 575–576 GB/s **93.4–93.5%**；(1024,4096) 496 GB/s 80.5%；
(4096,1024) 487 GB/s 79.0%——小形状占比低是 ~2–4 µs 固定 launch/tail 开销
在 17 µs 级 kernel 上的相对放大，非带宽退化。）

- **vec4_row 10/10 格胜出**；runner-up 恒为 warp_b256/b512（对 winner
  1.22–1.28×）；splitk4 全部垫底或次差（(4096,1024) 对 baseline 0.58×）——
  K=1024 时 split-4 段仅 256 元素/线程段，MLP 更差，符合 §6 机理。
- hot vs streaming 差异 <1%（W 32–90 MB ≫ L2 5.5 MB，x/out 池不改变 DRAM
  主导流量）。
- 所有格 `filter_sensitive=false`；winners 的 winner/runner-up CI95 全部
  显著（>1.05 KEEP 线），**无 UNRESOLVED 格**。

**FP32 子集（baseline + vec4_row；fp32 "natural if supported"）**: vec4_row
**10/10 格胜**（1.06–1.11×）。fp32 baseline 本身已达 492–548 GB/s（80–89%
峰值，Phase 4 记录），向量化余量小，与 fp16 的 1.51–1.63× 形成对照——
2B load 的 MLP 惩罚在 4B 元素上减半。

**split-K 定位（诚实记录）**: 主目标 REJECT（§6），全矩阵亦无一格胜出；
其假设价值（小 N 提高并行度）在本轮测试形状（最小 N=1024，已够 30 SM 用）
中**未被验证到收益**，不作为 v0.5 结论外推，保留内核与记录供后续小 N
（N < 30×rows/block）场景实验。

---

## 9. PyTorch 参考（framework context）

`torch.mv(W, x)`（cuBLAS Gemv；fp16 输入 FP32 compute type）@ (4096,4096)
fp16：**median 61.23 µs**（200 samples，`pytorch_ref` 记录）。

按用户指定，**仅作实现参照/带宽上下文，不作决策依据**（决策 = incumbent vs
candidate paired 证据）：vec4_row 59.88 µs 与 cuBLAS 同级（~1.02×），
baseline 94.2 µs 落后 cuBLAS **1.54×**（94.2/61.23）。v0.5 不复制 cuBLAS 实现（用户指定
baseline 保持简单形式；向量化采用标准 16B 对齐 load 惯用法并显式契约化）。

---

## 10. 回归与 Evaluator 状态

**三算子 smoke 回归**（v0.5 改动后，逐算子 correctness + negative，
baseline + 当前 incumbent 双变体）:

| 算子 | 变体 | correctness | negative | 结果 |
|---|---|---|---|---|
| RMSNorm | baseline / v4_vec_reg | 76/76 ×2 | 29/30 ×2（1 skipped） | PASS |
| Softmax | softmax_baseline / softmax_vec4 | 72/72 ×2 | 14/15 ×2（1 skipped） | PASS |
| RoPE | rope_baseline / rope_v3_half2 | 384/384 ×2 + 表值核对 | 36/37 ×2（1 skipped） | PASS |

（6/6 全部 all_pass；记录：`experiments/{rmsnorm,softmax,rope}/...`
对应 v0.5 smoke 时点文件, 本次重录。skipped 为各套件既有环境性跳过
用例, 跨变体一致。）

**smoke 运行环境事件（诚实记录）**: smoke 首跑曾被一次 56 分钟挂起
阻塞 —— 根因是 `/root/.cache/torch_extensions/cudalab_rmsnorm/lock`
孤立 torch 构建 baton 锁（09-20 23:32 一个构建进程被 SIGKILL 后未释放;
`FileBaton.wait()` 对存在即无限自旋, 与"GPU 0% / 无 nvidia-smi 进程 /
主线程 poll_schedule_timeout"现象完全吻合）。删除孤立锁后 6/6 全通;
其余 3 个扩展构建目录无此锁。环境事件, 非代码/驱动问题。

**Evaluator v2.3 状态**: `paired-streaming-v2.3` 的基准引擎与决策函数
（bench.py / decision.py / stats）**零改动**（用户指定"不要提前把 v0.5 变成
evaluator 重构项目"——v0.5 数据未触发其已知限制）。CPU 单测
`tests/test_evaluator_v23_cpu.py` **36/36 通过**。

**唯一 evaluator 侧改动（增量、由 GEMV 数据触发、记录在此）**:
`cudalab/evaluator/profiler.py` 的 NCU 摘要对**一次算子调用发射多个内核**
的情况（splitk4：partials + combine）原来只保留最后一个内核名，且
`kernel_duration_us` 等顶层标量是**跨内核平均**（[131.9 µs, 2.4 µs] →
67.2 µs，方向性误导）。修复：新增逐内核 `kernels[]` 分解（name / n_launches /
duration_us / dram%）+ `multi_kernel_note`；**顶层标量字段原样保留**（向后
兼容），单内核算子行为不变。所有 splitk4 profile 记录在该修复**之后**重录，
无"误导性历史值"留档。此改动不影响任何决策（决策走 paired bench，不走 NCU
摘要）。

---

## 11. 已知限制与开放问题

1. **native kernel-loop 默认 warmup=200 对时钟敏感 kernel 系统性偏慢**
   （§5：baseline 109.8 vs 91.3 µs，+20%）。用户口径定义未规定 warmup，
   默认值保留；所有 v0.5 native 记录**同时**提供 w200 与 w5000 两档，
   报告与对比统一使用 w5000（boost 稳态）。
2. **NCU@none 比 API-path 高 ~7%**（cache flush + 剖析隔离），幅度已量化、
   方向已解释、不影响决策（§5.4）。
3. **跨 run 点估计漂移**（v0.4 已确立的机器态特性）：如 baseline
   (4096,4096) hot 93.76（Phase 4）vs 94.16（Phase 7 run），0.4% 量级。
   一切结论只用 **run 内 paired** 比值；不做跨 run 绝对时间比较。
4. **splitk4 的 NCU 顶层标量是跨内核平均**——已由 `kernels[]` 修复并
   重录（§10）；阅读旧格式记录（v0.4 及以前）无此问题（均单内核算子）。
5. **双 GPU 容器**：本环境有 2× 2080 Ti，所有测量固定 GPU 0
   （`CUDA_VISIBLE_DEVICES=0` + nvidia-smi `--id=0` 采样）。
6. **小 N 场景 split-K 未验证**：最小测试 N=1024 仍远超 SM 饱和所需行数，
   split-K 的并行度价值未被本轮形状覆盖（§8）；不作为限制声明，仅记录
   实验边界。
7. **12 节报告结构**：按用户 v0.5 指令的 12 节要求组织（本报告的节序即
   交付结构）；节内内容完整覆盖用户列出的所有报告项。
8. **splitk4 的 partials 为 static 设备缓冲**（`g_splitk_partials`）,
   假设单流使用 —— 多流并发 forward_into(splitk4) 会在 partials 上竞争;
   本项目全部单流测量, 不受影响（独立审查 NIT-4, 记录不改动内核）。
   **v0.5 merge review 处置**: 该风险 + 跨 device workspace 风险（静态
   缓冲固定在首次调用设备）已升级为正式隔离 —— `gemv_splitk4` 标记
   UNSAFE_HISTORICAL_EXPERIMENT / REJECTED / NOT_FOR_NORMAL_DISPATCH，
   从 `ext.variants()` 与全部 CLI 正常路径移除（`quarantined_set`，
   见 §6 GEMV-0004 行注）；内核源码与全部历史数据保留，显式
   `forward("gemv_splitk4", ...)` 为受控历史审计入口。
9. **baseline NCU@base 与 @none 记录的内核名不同**（`gemv_baseline_kernel`
   重构前 vs `gemv_scalar_kernel` 重构后, 代码逐行相同, 独立审查 diff
   核实）—— 已在 §5 注记披露（MINOR-3）。
10. **DVFS 探针时钟采样稀疏**（n=1/相位, 有效周期 ~100 ms, nvidia-smi
    进程启动开销）—— 时钟值仅作佐证, 主证据为 warmup 长度 × 时长定量
    关系（§5.2, 独立审查 NIT-5）。
11. **native_timing 记录的 `window_median_us` 字段实为逐窗口中位数
    列表**（"median" 指每窗口内的中位数, 非跨窗口标量）; 报告与对比
    统一使用该列表的中位数（独立审查 NIT-6）。
12. **NCU 多内核摘要解析器无 CPU 单测**（splitk4 真实记录为实证;
    evaluator 决策/基准 36/36 CPU 单测不覆盖 profiler 解析路径, 独立
    审查 NIT-7）—— 记录为已知缺口, v0.6 补。
13. **14 条 GEMV NCU 摘要记录中 12 条含 `kernels[]`** —— 2 条
    baseline@base 为 de150bc 旧 schema（profiler 修复前, 单内核算子,
    顶层标量准确）, 其余 12 条（含全部 splitk4 重录）均含 kernels[]
    （独立审查 NIT-8 核对结果）。
14. **历史 experiment artifact 不可变（v0.5 merge review 新约定）**:
    历史上提交于 main（4eb520b）的 experiment artifact（如
    `experiments/rmsnorm/correctness/v0.3_regression/`、
    `experiments/softmax/correctness/v0.3/`、
    `experiments/rope/correctness/v0.4/`）一律**不可再改写** ——
    v0.5 smoke 曾误覆盖其中 7 个文件，已按 main 版本恢复，本轮结果
    迁移至新的 append-only 目录 `experiments/regression/v0.5/`（含
    README 说明约定与来源）。三个旧算子的默认输出目录已改指
    `experiments/regression/v0.5/<op>/`，防止复发。同版本记录
    （`experiments/gemv/correctness/v0.5/`）本轮同样不改写，新验证
    一律追加到 regression 目录。
15. **negative 套件 scope 语义（v0.5 merge review 新约定）**: 套件记录
    的 `negative_suite_scope` 字段明确声明其语义 ——
    **per-variant**（GEMV / Softmax / RoPE: 套件主体对请求的 variant
    运行, CLI 指定的候选变体即被实际测试）或 **cross-variant**
    （RMSNorm: 单次运行跨多变体覆盖算子级共享契约, 用例自带
    variant 字段; 保持单跑设计, **不**改描述为 per-variant）。

---

## 12. Git、复现与 v0.6 建议

**分支与提交**（`v0.5-gemv`，基于 main=4eb520b=v0.4.1；**未 merge main**，
无 force-push）:

| commit | 内容 |
|---|---|
| 50d6c0c | GEMV 算子：baseline kernel、adapter、correctness/negative 套件（Phase 1） |
| 4521540 | fix: make_bench_pool 未定义 M（del M 后引用） |
| de150bc | Phase 4：baseline 20 条 bench 记录 + NCU（cc all/none）+ native w200 |
| 75c1ccd | GEMV-0001..0004 候选内核（含对齐契约回退）+ 负例 +3 逐位一致回归 + 5 变体全套记录 |
| 652f4f1 | 4 个优化实验完整记录（0001/0002/0003 KEEP, 0004 REJECT） |
| 25bd9ab | 候选 native（w200+w5000）+ NCU（含多内核摘要修复）+ 三口径冲突调查记录 |
| 2e4836f | 全 shape matrix（fp16 5 变体 + fp32 子集）+ per-dtype winners |
| 8f2b944 | 最终 incumbent 独立复核记录 |
| (末) | 本报告 + README/STATUS 更新 + 独立审查处置（per-variant 负例归档 + mixed_sign 修复 + 记录重录 + 本报告修正） |
| (merge review ×3) | v0.5 merge review 修复（见下）: `fix: quarantine unsafe splitk4 experiment` / `fix: preserve historical regression artifacts` / `fix: unify negative-suite variant semantics` |

**复现**:

```bash
source tools/env.sh   # $PYTHON, CUDA 11.8 环境
# 正确性 / 负例（每变体 100 + 24）
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/cudalab.py test gemv --variant gemv_vec4_row
# 主目标实验（paired 9r streaming + 决策）
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/cudalab.py optimize gemv --id GEMV-0001 \
  --parent gemv_baseline --candidate gemv_vec4_row \
  --hypothesis "..." --changes "..." \
  --M 4096 --H 4096 --dtype float16 --mode streaming --rounds 9
# NCU 诊断（base 锁频 + 双 cache-control）
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/cudalab.py profile gemv \
  --variants gemv_vec4_row --M 4096 --H 4096 --clock-control base
# 全矩阵 + winners（fp16 全变体 / fp32 子集）
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/bench_gemv_full.py --dtype float16 --tag gemv_full
# 最终 incumbent 独立复核
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/revalidate_gemv_incumbent.py
# 口径冲突探测
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/dvfs_probe.py
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/caliber_warmup_probe.py
```

**独立 review**（推送前完成，模式同 v0.4）: 双双 **PASS WITH
CAVEATS**。CUDA 侧 1 MAJOR + 2 MINOR + 2 NIT; benchmark 侧 0 MAJOR +
5 MINOR + 8 NIT。无 benchmark/profile 记录造假类发现; 全部 findings
处置如下（原始数字一律不改写, 修正 = 本报告修正 + 重录 correctness /
negative 记录 + 实验记录 additive note, v0.4 先例）:

| # | 来源 | 级别 | finding | 处置 |
|---|---|---|---|---|
| 1 | CUDA | **MAJOR** | negative 套件从未 per-variant 运行（CLI 只调用 baseline 默认, 4 个向量化变体的契约/回退证据未归档） | `run_negative(ext, variant)` 管线修复（base.py 协议 + gemv.py + 统一 CLI test/optimize + revalidate 脚本）; 5 变体各重跑 24 例并归档（baseline 规范 `invalid_inputs.json` + 4 个 `invalid_inputs_<variant>.json`, 均 24/24）; GEMV-0001..0004 记录追加 `negative_note`（原数字不变）; §3/§6 更新 |
| 2 | CUDA | MINOR | mixed_sign 构造零抵消（乘积符号逐行恒定 (-1)^i） | x 改独立 Bernoulli 符号流; 5 变体正确性全套重跑重录（100/100 ×5）; 代码与 gemv_correctness.py docstring 更新 |
| 3 | CUDA | MINOR | §3 max_abs ulp 标注错（4.0/8.0 的 binade） | 改为 1 ulp @ \|y\|∈[4096,8192)/[8192,16384) |
| 4 | CUDA | NIT | splitk4 单流 workspace 假设 | §11.8 记录（不改内核） |
| 5 | CUDA | NIT | gemv_negative.py docstring 过期（Phase-1 叙述） | 已刷新为当前 per-variant 状态 |
| 6 | Bench | MINOR | §2/§7 字节数 33,574,848 算错 | 改 33,570,816 B（(4096·4096+4096+4096)·2; 代码与全部记录原本正确） |
| 7 | Bench | MINOR | §1 "三口径相差 <4%" 与自身数据矛盾 | 改 "API vs native-w5000 <1.5%; NCU 残余 ~7%（已解释）"（§5.4 同改） |
| 8 | Bench | MINOR | baseline NCU@base 与 @none 剖自不同二进制（75c1ccd 标量重构前后） | 代码逐行相同（审查 diff 核实）, §5 注记披露 |
| 9 | Bench | MINOR | §9 cuBLAS 比值 1.57× 算错 | 改 1.54×（94.2/61.23） |
| 10 | Bench | MINOR | DVFS 探针 n=1 稀疏采样未披露 | §5.2 + §11.10 |
| 11 | Bench | NIT×8 | 97.8/97.4%（最终文本已无此数, 早期草稿修正消化）; barrier 72.0%→71.9/71.2%; max_rel 区间下限; 302→304 GB/s; fp32 下限 494→492; `window_median_us` 命名; 解析器无 CPU 单测; "全部 NCU 记录含 kernels[]" 实为 12/14 | 逐条处置: §4/§6/§8 数值修正, §11.11/§11.12/§11.13 记录 |

**v0.5 merge review 处置**（2026-09-21, 推送前第二轮审查, 4 项, 无重跑
full matrix）:

| # | 级别 | finding | 处置 |
|---|---|---|---|
| M1 | MAJOR | splitk4 的 `static at::Tensor g_splitk_partials` 进程级 workspace: 多 stream 并发 race + 跨 device workspace 设备风险 | 正式隔离（UNSAFE_HISTORICAL_EXPERIMENT / REJECTED / NOT_FOR_NORMAL_DISPATCH）: `bindings.cpp` `quarantined_set` 将其移出 `ext.variants()`, CLI test/benchmark/optimize/profile 全路径拒绝（显式报隔离错误）; 源码/GEMV-0004/bench/NCU 历史全部保留; 显式 forward 保留为受控历史审计入口; GEMV-0004.json 追加 `quarantine_note`; 内核头注释标记（§6/§11.8） |
| M2 | MAJOR | v0.5 smoke 覆盖了 main 历史 artifact（rmsnorm v0.3_regression ×3, softmax v0.3 ×3, rope v0.4 ×1） | 7 个文件按 main（4eb520b）版本恢复; 本轮结果迁移至新目录 `experiments/regression/v0.5/`（append-only, 含 README 来源说明）; 三个旧算子默认输出目录改指 regression 目录; 新约定: 历史 experiment artifact 不可变, 新验证只 append（§11.14） |
| M3 | MINOR | negative 套件 variant 语义不统一: Softmax/RoPE 的 `run_negative` 忽略 variant 参数（CLI 指定的候选未被实际测试）; RMSNorm 套件实为 cross-variant 却无显式声明 | Softmax/RoPE `run_negative` 改为 per-variant（套件本体早已参数化, 补管线 + per-variant 归档命名）; 四个套件记录新增 `negative_suite_scope` 字段（GEMV/Softmax/RoPE = per-variant, RMSNorm = cross-variant 并更新 docstring, 不再描述为 per-variant）; base.py 协议 docstring 更新（§11.15） |
| M4 | NIT | 文档 2^-24 ≈ 6.1e-5 错误; splitk4 的错位用例被错误描述为其"标量回退契约" | `gemv_correctness.py` docstring 修正为 2^-24 ≈ 5.96e-8（半步 ≈3e-8; 6.1e-5 = 2^-14 次正规**边界**, 另一处用法正确未动）; 明确**对齐不是 splitk4 的约束**（标量访存, 唯一契约 K%4==0）: 套件代码按 variant 重构（错位用例在 splitk4 下为 finite-only control, K=13 用例保留 bit-identical = K%4 契约回退）, 归档记录追加 `note_addendum`, §3 表格更新 |

审查范围（benchmark 侧, 逐字核对）: 报告全文、bindings.cpp、
kernels/gemv（含 de150bc 历史版本 diff）、bench.py / profiler.py、
两个探针脚本、4 pair + 2 revalidation + 5 correctness JSON、10
native_timing、14 NCU 摘要 + splitk4 raw、2 探针 JSON、winners×2 +
6 个抽核 full/base JSON、git 全分支 diff。独立复算: 4 个 pair 的
per-round 中位数、GEMV-0001 bootstrap CI（独立重跑 [1.539819,
1.551701] 与记录一致）、shape-matrix 抽核格（59.927/76.418/1.2752
逐位一致）、20+ 组数字对照。

**v0.6 建议（仅建议，不在 v0.5 范围）**: **Quantized GEMV**（用户指定优先序）
—— W 量化（INT8/FP8）+ 内核内 dequant FMA，字节量减半 → 理论带宽余量 2×；
直接复用 v0.5 的向量 load 契约、回退门与三口径计时表面。次选：小 N split-K
验证（§11.6）或 half2 2 元素/16B 的更低 ILP 变体（LSU 压力介于 vec4_row
与 warp_b256 之间）。
