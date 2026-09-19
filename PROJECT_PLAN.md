# CUDALab v0.1 — 项目计划

自主 CUDA 内核优化实验室。v0.1 范围：**仅 RMSNorm**，
硬件为 2× NVIDIA RTX 2080 Ti（Turing，sm_75），CUDA 11.8，PyTorch 2.4.1+cu118。

## 流水线

```
参考实现 → 正确性 → 基准测试 → GPU 剖析 → 瓶颈分析
→ 优化假设 → 内核修改 → 编译 → 正确性
→ 基准测试 → 采纳/拒绝 → 实验记录
```

## 阶段

### 阶段 0 — 仓库初始化（按定义即起点）
- 清理 `/root/code/cuda`，`git init -b main`，repo-local git 身份。
- `.gitignore`（排除构建产物/`.so`/剖析原始输出；保留基准 CSV/JSON、
  实验元数据、报告）。
- `tools/env.sh` 作为环境变量的唯一事实来源
  （`CUDA_HOME=/usr/local/cuda`、`PYTHON=/root/miniconda3/envs/pytorch/bin/python`、
  `TORCH_CUDA_ARCH_LIST=7.5`、`CUDA_VISIBLE_DEVICES=0`）。
- `PROJECT_PLAN.md`、`STATUS.md`。
- ✅ 已完成

### 阶段 1 — RMSNorm 参考实现
- `cudalab/reference.py`：显式 FP32 累加公式
  `y = x * rsqrt(mean(x², -1) + eps)`，默认 `eps=1e-5`。
- ✅ 已完成

### 阶段 2 — 基线 CUDA 内核
- `kernels/rmsnorm/rmsnorm_baseline.cu`：每行一个 block，
  warp 归约 + 共享内存归约，FP16 进出，FP32 累加。
- `kernels/rmsnorm/bindings.cpp` + `cudalab/build.py`，使用
  `torch.utils.cpp_extension.load()`，显式 `sm_75`，并采用内容哈希
  构建目录：源文件不变绝不重编译，每个候选变体有独立缓存。
- ✅ 已完成

### 阶段 3 — 正确性框架
- `cudalab/correctness.py`：max_abs_error、max_rel_error、NaN/Inf、allclose。
  固定且已记录的容差（fp16: atol=2e-3, rtol=5e-3；fp32: atol=1e-5,
  rtol=1e-4 —— 对所有候选一致，绝不按候选放宽）。
- 矩阵：M ∈ {1,16,128,1024} × H ∈ {1024,2048,4096,8192} 子集 + 边界用例
  （全零、微小值、多尺度、多种子）。
- JSON + 人类可读输出。
- ✅ 已完成

### 阶段 4 — 基准框架
- `cudalab/benchmark.py`：`torch.cuda.Event` 计时，预热 ≥ 100，
  迭代 ≥ 200，≥ 5 个独立轮次，中位数为主指标，
  p50/p95/min/max，有效 DRAM 带宽（读 x + 权重、写 y）。
- 每套运行前后做 GPU 状态快照（nvidia-smi：时钟、温度、功率、利用率）。
- 每个（形状, dtype）的输入张量固定，所有变体共享。
- 所有变体的基准矩阵，CSV + JSON。
- ✅ 已完成

### 阶段 5 — 剖析器集成
- `cudalab/profiler.py`：在小型专用剖析驱动程序上运行 `ncu`，
  映射 Nsight-Compute 2022.3 在 sm_75 上的指标名（通过
  `ncu --query-metrics` 发现），输出结构化 JSON 摘要
  （内核时长、DRAM 吞吐、SM 吞吐、占用率、
  每线程寄存器、共享内存、warp 停顿）。取不到的字段为 null。
- 兜底方案：保存真实的 ncu 错误，记录到 STATUS.md，改用
  nsys / torch.profiler 作为替代。
- ✅ 已完成（或兜底已记录）

### 阶段 6 — 实验追踪
- `cudalab/experiment.py`：`experiments/rmsnorm/EXP-NNNN.json` 记录，
  含假设、改动、正确性、相对当前最佳的基准、剖析观察、
  判定（KEEP/REJECT/NEUTRAL）。
- 判定规则（v0.1）：
  - 正确性 FAIL → REJECT（无条件）；
  - 中位数加速 ≥ 5% 且多数轮次更快 → KEEP；
  - ±5% 以内 → NEUTRAL；明显更慢 → REJECT。
  - 完整基准矩阵始终保存；不做形状挑拣。
- `scripts/optimize_rmsnorm.py`：构建 → 正确性 → 基准 → 剖析
  → 候选 vs 当前最佳评估 → 判定 → 记录。
- ✅ 已完成

### 阶段 7 — 第一轮优化循环
- ≥ 3 个真实、数据驱动的优化实验（对照 baseline）：
  1. 向量化加载（float4/half2）—— 内存带宽瓶颈假设；
  2. half2 + 减少同步 / 单遍重构（若剖析数据支持）；
  3. 一个"预期会失败"的假设以验证 REJECT 分支
     （如更大的 grid-stride / 不同块大小）—— 仅在数据真正支持时做。
- 主要目标形状：**M=128, H=4096, FP16**；完整矩阵仍全部报告。
- ✅ 已完成

### 阶段 8 — 文档与最终验证
- 对最终最佳内核重跑完整正确性套件 + 完整基准矩阵。
- `README.md`（GitHub 质量、只含真实数字）、`docs/design.md`、
  更新 `STATUS.md`、定稿 PROJECT_PLAN。
- 每个阶段一个 git 提交；最终工作树干净 + 提交。
- ✅ 已完成

# v0.2 — Evaluator Hardening & Revalidation（开始于 2026-09-19）

目标：修复 v0.1 code review 发现的 evaluator、benchmark、API correctness 与方法学
问题，使性能结论更可信、可重复、可审计。v0.1 数据全部保留为历史，不删除、不修改。
**v0.2 不新增算子、不新增 kernel 变体**（除非修复 correctness bug 所必须）。

优先级：Correctness > Benchmark validity > Reproducibility > Statistical confidence
> Performance > New features。

## v0.2 Phases

### Phase 1 — API correctness hardening ✅ 已完成
（ce5eac1：bindings 统一 validate_common/validate_out + 5 变体 TORCH_CHECK +
C10_CUDA_KERNEL_LAUNCH_CHECK；v1/v4 显式对齐契约）

原计划:
- v2/v4: PER switch 之前验证 `H % 256 == 0`（修复非法 H 静默错误，Finding A）
- bindings: `forward` / `forward_into` 共享同一 validation helper（dim/size/dtype/
  contiguous/CUDA/device 一致/eps 有限且非负，Finding B）
- 每个 kernel launch 之后 `C10_CUDA_KERNEL_LAUNCH_CHECK()`
  （宏已在本机 torch 2.4.1 头文件 `c10/cuda/CUDAException.h` 中核实存在，Finding C）
- v1/v4: 向量化加载的显式对齐契约 —— 指针 16B/4B 对齐验证 + 清晰报错
  （策略 1：显式 validation，不静默执行未对齐的 float4 加载，Finding D）

### Phase 2 — Negative correctness tests ✅ 已完成
（d1a4fd8：cudalab/negative_suite.py 28 例，27/28 符合预期 + 1 多 GPU 跳过；
全部在 kernel 启动前被拒，结果在 experiments/rmsnorm/correctness/v0.2/invalid_inputs.json）

原计划:
- 非法 H（1023/1025/4095/4097/4100 × v2/v4）、w 长度/dtype/device 错误、
  non-contiguous x/w、错误 out shape/dtype、不支持 dtype、eps 非有限/负、
  未对齐指针（storage offset 破坏 16B 对齐）
- 要求：kernel launch 之前以明确异常拒绝（而非静默计算或 CUDA 运行时错误）
- 结构化结果保存至 `experiments/rmsnorm/correctness/v0.2/`

### Phase 3 — Benchmark redesign: paired benchmark ✅ 已完成
（f9a98ab：cudalab/bench_v2.py bench_pair/bench_matrix，slot 奇偶交替 + round-robin，
预分配缓冲池，seed/每轮顺序完整记录）

原计划:
- `bench_pair(parent, candidate, ...)`: A/B 时间上相邻、执行顺序轮换并记录、
  同一 round 使用完全相同的张量（不重新生成输入）
- 全矩阵采用 shape 内 round-robin 轮换顺序，平衡 variant 位置与
  热/DVFS 漂移的相关性；记录 seed 与每轮顺序

### Phase 4 — DVFS / clock stability guard ✅ 已完成
（stats.py check_dvfs_pair/check_dvfs_matrix，>5% 判 INVALID_DVFS，重试 ≤3，
valid<5 → UNSTABLE；1350 vs 1905 拦截行为有单元测试覆盖；复验 41 run 全部 9/9 valid）

原计划:
- 每个 paired round 记录 SM clock / mem clock / 温度 / 功耗（A 前、A 后、B 后）
- A/B 有效 SM clock 相对差 > 5% → 该 round `INVALID_DVFS`，不进入统计
- 无效 round 最多重试 3 次；仍不足最小有效轮数 → 最终 `UNSTABLE`
- 不修改 power limit / 不锁时钟（容器不允许）；仅记录 + 判无效

### Phase 5 — hot / streaming cache modes ✅ 已完成
（POOL_SIZE=16 轮换 buffer；(128,4096) fp16 streaming working_set 33.5 MB > L2 5.5 MB；
pool_size/working_set_bytes/element_size 均记录于 JSON）

原计划:
- `hot`: 沿用 v0.1 设计（单 x/w/out 缓冲、连续启动 = cache 友好稳态），
  但不再表述为"唯一真实推理场景"
- `streaming`: 预分配轮换缓冲池（timing 区域内无 malloc/copy/随机数）；
  M=128 H=4096 fp16 下 pool working set >> L2 (5.5 MB)；记录 pool_size 与
  working_set_bytes；不声称"完全 cold cache"（rotating-buffer / cache-cold-ish）

### Phase 6 — Statistical decision redesign ✅ 已完成
（stats.py paired_speedups/summarize/bootstrap_ci + decision.py 四态决策；
tests/test_evaluator_cpu.py 18/18 通过）

原计划:
- 统计单位 = 独立 benchmark round（不是 500 个连续 event sample）
- round-level paired speedup: median / mean / min / max / faster-round 计数
- 95% bootstrap CI 基于 round-level speedup（固定 seed，确定性可复现）
- 决策规则: KEEP / REJECT / NEUTRAL / **UNSTABLE**（新增）；全部 CPU 单元测试

### Phase 7 — Profiler methodology audit ✅ 已完成
（aca86a6/517d548 初版；v0.2.1 修正 --cache-control 语义（此前写反）：默认
all = cache flush/reset（每 replay 前失效缓存，确定性 flushed 状态）、none =
no-flush（不失效，状态不受控，ncu 警告 "Running with uncontrolled GPU caches"）
→ v0.1 走默认 all（= 失效），其 "cold L2" 说法与默认配置一致（v0.2 曾误判
"无配置依据"，已更正）；profiler 显式化 cache_control/clock_control +
L1/L2 命中率；scripts/profile_v2.py 双模式剖析 → profiles/rmsnorm/v0.2/）

原计划:
- 用本机 ncu 2022.3 真实验证 `--cache-control {all|none}`（已确认存在，默认 all；
  v0.2.1 依据 --help + raw 输出 + NVIDIA 文档核实语义并修正此前写反的标注）
- 双模式剖析（cache flush/reset vs no-flush）；README 只保留已验证的描述，
  用 cache-flush / no-flush 术语（不称 hot/cold L2，除非可从 replay 配置严格推出）

### Phase 8 — Full v0.2 revalidation ✅ 已完成
（5 变体 76/76 ×5 + 负例复跑 + 28 组全矩阵（369/369 valid）+ 13 组主形状/关键形状
配对精测 + shape_winners.json + pytorch_ref + EXP-0008 + best_v0.1.json 存档/
best.json 更新；v0.2 best: fp16 主形状 v4（v2 平局），fp32 与 (128,8192) v2）

原计划:
- 5 变体重测：合法正确性 + negative suite + hot/streaming 完整矩阵（fp16，
  fp32 若成本可接受）+ 主目标 paired 精测
- 由 v0.2 evaluator 重新确定 best（不预设 v4 胜出）；shape-specific winner 矩阵
- 新实验记录 EXP-0008+（v0.2 schema）；旧 EXP-0001…0007 原样保留

### Phase 9 — Optional shape dispatcher ✅ 已完成（实现）
（fb22967：复验显示稳定且显著的 shape/dtype 专属优势（fp32→v2 1.37–1.62×、
(128,8192)→v2 1.38×），满足实现条件；cudalab/dispatch.py 28 单元格实测分发表 +
保守 fallback，5 个 CPU 测试 + 端到端验证）

原计划:
- 仅在 v0.2 revalidation 完成且 shape winner 稳定、收益明显时实现
- 否则记录不实现的理由（非强制 Stop Condition）

### Phase 10 — Documentation and final audit ✅ 完成

原计划:
- README（v0.2 结果 + 方法学 + v0.1 历史，保留 EXP-0002 事故）
- STATUS / PROJECT_PLAN 更新；`docs/benchmark_audit_v0.2.md` 独立审计
- 按阶段 commit；不 push（除非用户另行要求）

**已完成（2026-09-19）**：README 全面重组（v0.2 结果 + 方法学 + v0.1 双事故史 + 复现命令）；
STATUS / PROJECT_PLAN 更新；`docs/benchmark_audit_v0.2.md` 独立审计（subagent 只读审计，
12 项清单零 FAIL，总体 PASS 带限定；Lead 逐项复核引用数值后修正了 2 处表述偏差）。
按阶段 commit（9 个实现/数据 commit + 1 个文档 commit）；未 push（用户未要求）。

## v0.2 状态

**已完成（2026-09-19）**：Phase 1–10 全部完成。完整摘要见 `STATUS.md` 的 v0.2 段落。

## 不可妥协的规则
- 不伪造任何数字；每个报告的指标都来自真实执行。
- 编译时间绝不计入内核计时；先基准、后剖析。
- 所有变体使用相同输入；权重乘法绝不跳过。
- 失败的实验保留并报告。

## 状态
实时进展见 `STATUS.md`。
