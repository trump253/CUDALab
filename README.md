# CUDALab

**自主 CUDA 内核优化实验室。**

CUDALab 闭环自动化内核优化：

```
参考实现 → 正确性 → 基准测试 → GPU 剖析 → 瓶颈分析
→ 优化假设 → 内核修改 → 编译 → 正确性
→ 基准测试 → 采纳 / 拒绝 → 实验记录
```

外层 LLM 智能体（开发者的编码代理）提供优化假设与内核代码；**客观、非 LLM 的评估层** —— 正确性校验框架、配对 CUDA 事件基准框架（含 DVFS guard）、Nsight Compute 集成、以及固定的 KEEP/REJECT/NEUTRAL/UNSTABLE 判定规则 —— 提供证据。智能体不能自封胜者；只有框架的数字才算数。见 [docs/design.md](docs/design.md)。

## 当前状态：v0.2（评估器加固与完整复验）

v0.2 **没有新增内核**（5 个变体原样保留），而是修复 v0.1 代码审查发现的问题，
重建评估层，并在同一套可信方法下重新得出性能结论。v0.1 全部历史数据
（EXP-0001…0007、`benchmarks/`、`profiles/rmsnorm/`、`correctness/` 顶层文件）
原样保留，未做任何改写。

v0.2 交付：
- **API 加固**（输入验证 Finding A–D + 启动后错误检查）与 **28 例非法输入负例套件**；
- **配对基准 harness `paired-streaming-v2`**：A/B 交替顺序去偏、每轮 3 次 SM 时钟
  采样的 DVFS guard（>5% 相对差判 invalid round）、hot/streaming 双缓存模式、
  预分配缓冲池（streaming 工作集 33.5 MB > 5.5 MB L2）；
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
- **形状/dtype 分发表**（仅基于实测显著证据的保守 dispatch，`cudalab/dispatch.py`）；
- 独立方法学审计文档 [docs/benchmark_audit_v0.2.md](docs/benchmark_audit_v0.2.md)。

### v0.2 正确性

| 套件 | 结果 |
|---|---|
| 合法输入套件 × 5 变体（76 例/变体） | **380/380 PASS**（`experiments/rmsnorm/correctness/v0.2/`） |
| 非法输入负例套件（28 例） | **27/28 符合预期，1 跳过**（多 GPU 用例，单 GPU 环境安全跳过） |

max_abs_error = 3.91e-3 / max_rel_error ≈ 9.7e-4（fp16 最坏值，全变体一致）。
负例套件覆盖：非对齐/非法 H（1023/1025/4095/4097/4100）、w 长度与 dtype 错配、
CPU 输入、非连续输入、out 张量错配、bf16、eps=NaN/负、v1/v4 指针 8B 未对齐
（+ 16B 对齐对照组 PASS）、多 GPU 混布（跳过）。所有非法输入在 **kernel 启动前**
被拒（`TORCH_CHECK`），启动后 `C10_CUDA_KERNEL_LAUNCH_CHECK()`。

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
  轮转（工作集 33.5 MB > L2 5.5 MB，每次 launch 面对冷 L2）。streaming 延迟更高
  是**真实访存成本**，不是测量缺陷；两种模式都如实报告。
- 配对判定（9 rounds，bootstrap CI95）：
  - v4 vs baseline：streaming 1.896× [1.860, 1.908]、hot 1.740× [1.680, 1.771] → **KEEP v4**
  - v1 vs baseline：streaming 1.730×、hot 1.761× → v1 显著快于 baseline
  - v4 vs v1：hot v4/v1 median 0.9327 [0.8373, 0.9658]、0/9 轮 v1 更快（v4 快约 7%，REJECT v1）；streaming 平局（1.011×，NEUTRAL，9/9 轮 v1 微快）
  - v4 vs v2：两模式均 **NEUTRAL**（CI 均含 1.0，1–2% 差距在噪声内）
- **v0.2 主形状 fp16 最佳 = `v4_vec_reg`**（与 v2 统计平局，保留 v0.1 incumbent）。

### v0.2 形状/dtype 结论（不是单一 global best）

完整 28 单元格 winner 表：`benchmarks/v0.2/shape_winners.json`。要点：

| 单元格 | 最佳 | 证据 |
|---|---|---|
| (128,4096) fp16 | v4_vec_reg | incumbent；v2 平局 NEUTRAL；1.56× vs baseline |
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
| v4 = 主形状 fp16 最佳 | **CONFIRMED（附保留）** —— 与 v2 统计平局，incumbent 保留 |
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

### v0.2 形状分派（Phase 9）

`cudalab/dispatch.py`：28 个实测单元格的分发表（每条目带证据理由）+
保守 fallback（未实测组合只外推 (128,8192) 的 v2 证据与 fp16 incumbent v4，
其余回退 baseline；H 不支持时回退 baseline）。`select_variant(M,H,dtype)` 纯
CPU 可单测；5 个单元测试 + 端到端验证通过。**不改变任何内核**，只是选择器。

## 评估方法（v0.2）

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

## 范围（不变）

- **一个内核：RMSNorm**（`y = x * rsqrt(mean(x², dim=-1) + eps) * w`，默认
  `eps=1e-5`，FP32 累加）。
- 硬件：NVIDIA RTX 2080 Ti（Turing，**sm_75**），CUDA 11.8，PyTorch 2.4.1+cu118。
- dtype：**fp16 为主**，支持 fp32。连续（contiguous）输入。
- 各变体支持的 H（v0.2 已加启动前显式校验）：
  baseline 任意 H；v1 H%8==0（fp16）/ H%4==0（fp32）；
  **v2 H/256 ∈ {2,4,8,16,32}（H ∈ {512…8192}）；v3 H%512==0；
  v4 H/256 ∈ {4,8,16,32}（H ∈ {1024,2048,4096,8192}）**。
- 主要优化目标形状：**M=128, H=4096, fp16**。
- 完整基准矩阵始终测量并保存 —— 不做形状挑拣。

## 优化实验（v0.1 历史 + v0.2 复验）

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

## 架构

```
cudalab/
  reference.py      显式 FP32 累加的 RMSNorm 参考实现
  build.py          扩展构建 + 内容哈希缓存管理
  correctness.py    固定容差正确性框架（76 例套件）
  negative_suite.py 非法输入负例套件（28 例，v0.2）
  benchmark.py      v0.1 批量 cuda-event 框架（保留，历史对照）
  bench_v2.py       v0.2 配对基准 harness + 矩阵 + shape winners + PyTorch 参照
  stats.py          round-level paired 统计 + bootstrap CI + DVFS 校验（纯 CPU）
  decision.py       KEEP/REJECT/NEUTRAL/UNSTABLE 决策规则（纯 CPU）
  dispatch.py       v0.2 形状/dtype 分发表（纯 CPU）
  profiler.py       ncu --csv 集成（v0.2: cache_control/clock_control 显式化）
  experiment.py     实验记录 + 判定规则
kernels/rmsnorm/
  rmsnorm_common.h  自注册变体注册表 + 对齐辅助
  bindings.cpp      PyTorch 扩展入口（v0.2: 统一输入验证 + 启动检查）
  rmsnorm_baseline.cu … rmsnorm_v4.cu   5 个变体（v0.2 未改动计算逻辑）
scripts/
  test_rmsnorm.py / benchmark_rmsnorm.py / profile_rmsnorm.py / optimize_rmsnorm.py
  bench_v2.py           v0.2 基准入口（pair/matrix/full/winners）
  profile_v2.py         v0.2 双缓存模式剖析入口
tests/
  test_invalid_inputs.py   负例套件 CLI（27/28 + 1 跳过）
  test_evaluator_cpu.py    stats/decision 纯 CPU 单元测试（18/18）
  test_dispatch.py         分发表单元测试（5/5）
tools/env.sh        环境变量的唯一事实来源
experiments/        EXP-*.json + correctness/{,v0.2/} + best.json / best_v0.1.json
benchmarks/         v0.1 bench_*.json|csv + v0.2/（43 个 v0.2 数据文件）
profiles/rmsnorm/   v0.1 剖析 + v0.2/（双 cache-control，raw/ 被 git 忽略）
```

新增内核变体 = 新增一个 `.cu` 文件（自注册；无需改动绑定层）。
**v0.2 约束：不新增内核/变体** —— 评估器优先。

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
- **v0.2 负例套件**：28 例非法/未对齐输入，全部预期在 kernel 启动前被拒
  （`TORCH_CHECK` 中文报错），对照组（16B 对齐未对齐变体）预期 PASS。
- 正确性 FAIL 的变体无条件 REJECT，永远不可能成为"最佳"。

## 如何复现

```bash
cd /root/code/cuda
source tools/env.sh          # 设置 CUDA_HOME、PATH、PYTHON、架构列表

# 构建（有缓存；冷启动约 1 分钟，热启动几乎瞬时）
$PYTHON cudalab/build.py

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
$PYTHON tests/test_dispatch.py
```

产物：`experiments/rmsnorm/EXP-0008.json`、`benchmarks/v0.2/`（含
`shape_winners.json`）、`profiles/rmsnorm/v0.2/`、`experiments/rmsnorm/correctness/v0.2/`。

## 局限（v0.2 更新）

- 仅 RMSNorm；单 GPU（GPU 0）；仅连续输入。
- 变体 H 支持约束同上（baseline 完全通用）。
- 容器内无法锁定 GPU 时钟 → DVFS guard 只能**检测并拒绝**失配轮，不能预防；
  nvidia-smi 轮询是区间外代理采样，不捕捉瞬时降频（已记录为残余风险）。
- NCU `--clock-control base` 是否真正锁频在容器内无正面证据（无警告也无确认）。
- 矩阵模式（round-robin，非配对）的 round-level ratio 对离群干扰轮敏感
  （如 (16,4096) fp32 hot 的 round 2 有 3/5 变体升至 10–13 µs，且该轮时钟恒
  1350 MHz、DVFS guard 未拦截）：矩阵 winner 仅指示性，最终判定以 paired A/B 为准。
- streaming 工作集 33.5 MB 仍不足以让 DRAM 带宽完全饱和（M=1024 行才接近）；
  M=1 区域是 launch-bound，绝对延迟无意义。
- `compute-sanitizer` 不可用，未做越界/竞态检查（v0.1/v0.2 均如此）。
- 分发表只覆盖 14 个实测 (M,H) 组合；其余组合走保守 fallback（多为 baseline）。
- v0.1 数据保留在案但**已被 v0.2 取代**：跨版本数字不可直接比较
  （harness、时钟条件、缓存策略均不同）。

## 路线图（v0.3 建议）

- 支持锁频的环境（裸机/特权容器）下重跑 paired harness，验证 DVFS guard
  在零失配条件下的噪声下限。
- 引入 compute-sanitizer（越界/竞态）作为正确性的第二道门。
- 新内核（Softmax、RoPE）复用 v0.2 客观层（该层已刻意做成内核无关）。
- fp32 路径专项：v4 的 fp32 寄存器路径是已知弱点（v2 快 1.37–1.62×），
  允许新变体时优先做 fp32 向量化重设计。
- 更大 M（8192/16384）矩阵，覆盖 DRAM 带宽饱和区。
