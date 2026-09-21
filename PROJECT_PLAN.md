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
（v0.2.1：增补 2 例 v4 FP32 H=1024 对齐回归 → 30 例，29/30 符合预期 + 1 跳过；
fp32 路径 PER=4 亦按 float4 要求 16B 对齐，见 kernels/rmsnorm/rmsnorm_v4.cu v4_precheck）

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
是否 > L2 取决于 shape，以 JSON 中的 `working_set_gt_l2` 字段为准；
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
best.json 更新；v0.2 best: fp16 主形状无唯一胜出者（v0.2.1 修正，v4 保留 incumbent），fp32 与 (128,8192) v2）

原计划:
- 5 变体重测：合法正确性 + negative suite + hot/streaming 完整矩阵（fp16，
  fp32 若成本可接受）+ 主目标 paired 精测
- 由 v0.2 evaluator 重新确定 best（不预设 v4 胜出）；shape-specific winner 矩阵
- 新实验记录 EXP-0008+（v0.2 schema）；旧 EXP-0001…0007 原样保留

### Phase 9 — Optional shape dispatcher ✅ 已完成（实现）
（fb22967：复验显示稳定且显著的 shape/dtype 专属优势（fp32→v2 1.37–1.62×、
(128,8192)→v2 1.38×），满足实现条件；cudalab/dispatch.py 28 单元格实测分发表 +
保守 fallback，5 个 CPU 测试 + 端到端验证）
（v0.2.1 修订，review Finding 4：evidence > coverage —— 移除对未实测 (M,H) 的
v4/v2 外推与 hot/streaming 冲突格的硬编码（(16,4096) fp16 hot winner=v4 /
streaming winner=v1 → baseline）；现仅 paired 证据格（(128,4096) fp32、
(128,8192) fp16 → v2_reg）+ 显式 incumbent 格（(128,4096) fp16 → v4，
NO_UNIQUE_WINNER）路由优化变体；dispatch_info 四类 evidence_source，
6 个 CPU 测试）

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

# v0.3 — Evaluator Generalization + Softmax Autonomous Optimization（2026-09-19 起）

核心问题：**v0.2 的闭环（正确性 → 配对 bench → 统计 → 决策 → 剖析 → 实验史）
能否原样迁移到第二个算子？** 算子：row-wise Softmax
（`y = exp(x − rowmax)/Σexp(x − rowmax)`，FP32 内部，输出原 dtype，
ref `torch.softmax(x.float(), dim=-1).to(x.dtype)`；FP16 主 + FP32，禁 BF16；
sm_75 / CUDA 11.8）。分支 `v0.3-softmax`（基线 main = v0.2.1 = dfe9e9b），
**不 merge 回 main，不开始 v0.4**。

## v0.3 Phases

### Phase 1 — v0.2.1 发布 ✅
`v0.2-evaluator-hardening` ff-merge 到 main + smoke + push + annotated tag
v0.2.1（= dfe9e9b）；从 main 拉 `v0.3-softmax`。

### Phase 2 — Evaluator 通用化 + RMSNorm 回归硬门 ✅
`cudalab/evaluator/{bench,stats,decision,profiler,negative,experiment,gpu,correctness}.py`
+ 算子 adapter `cudalab/operators/{rmsnorm,softmax}.py`（不做大规模重写）。
harness 升级到 v2.2（`docs/evaluator_hardening_v0.3.md`）：移除 round 内
nvidia-smi 采样（其 ~40ms 空闲间隙会把 GPU 推入性能退化态）→ 时间基准
burn（≥150 launches 且 ≥300ms）+ 每样本 spike guard（1.5× 运行中干净中位数）
+ 每变体 cross-block 一致性 guard（block 中位数 vs 运行中 median×1.15）。
RMSNorm 回归硬门 **PASS**（commit eae07bb）：CPU tests + dispatch + negative +
correctness + (128,4096) fp16 paired 全部与 v0.2 结论兼容；发现的"冲突"
（v0.2 hot REJECT → 今天 NEUTRAL）定位为机器态漂移而非 evaluator 缺陷。

### Phase 3 — Softmax baseline + 正确性/负例 ✅
`kernels/softmax/`（softmax_common.h 注册表 + scalar 共享内核 + baseline +
bindings 统一验证）；72 项正确性（2 dtype × (9 形状 × 3 seed + 9 edge)）+
15 项负例（launch 前 TORCH_CHECK + 启动后 C10_CUDA_KERNEL_LAUNCH_CHECK）。

### Phase 4 — Baseline 全矩阵 + NCU + PyTorch context ✅
36 格矩阵（9 形状 × 2 dtype × hot/streaming）全 9/9 valid；NCU 双
cache-control（v0.2.1 语义，先 --query-metrics 验证）；PyTorch 参照仅记录。

### Phase 5 — ≥4 个自主优化实验 ✅（4/4，profiler→hypothesis 驱动）
| 实验 | 假设（来自剖析） | 结果 |
|---|---|---|
| SFM-0001 `softmax_vec4` | 标量小事务是瓶颈（long_scoreboard 60.6%）→ 4 宽向量化 | **KEEP**（hot 1.2916 / streaming 1.6772，9/9）→ incumbent |
| SFM-0002 `softmax_online` | 3 读 1 写 → 2 读 1 写（online (m,l) + merge 恒等；先 docs/softmax_algorithm.md + 5 个 CPU 恒等测试门禁） | NEUTRAL（bottleneck 是延迟不是带宽） |
| SFM-0003 `softmax_vec4_ilp2` | 每线程在飞 load 加倍隐藏延迟 | NEUTRAL（寄存器/屏障代价抵消） |
| SFM-0004 `softmax_hsplit2` | occupancy 44%→86%（H 对半分 2 块/行 + (m,l) 跨块合并，单 launch） | **REJECT**（barrier stall 5.6%→31%；不 occupancy-bound）；v0.3.1 起**隔离**（UNSAFE_HISTORICAL_EXPERIMENT / NOT_FOR_NORMAL_DISPATCH，见 SFM-0004.md §6） |

失败实验全部保留（NEUTRAL/REJECT 内核留在仓库作参考实现与历史证据）。
**四个设计维度（宽度/流量/每线程 ILP/块级并行）全部测完**（v0.3 原文称
"四轴设计空间闭合"，v0.3.1 措辞更正：测完 4 个正交维度 ≠ 设计空间穷尽）。
**`softmax_vec4` 是当前 acceptance policy 下的 incumbent；后续候选尚未
达到 ≥5% 的替换门槛**（NEUTRAL 是 policy_decision，不是"统计平局"或
"结构最优"）。
dispatcher：默认不做（v0.3 无 paired 确认的 per-shape 路由证据）。

### Phase 6 — 最终完整重验 ✅
incumbent 正确性 72/72 + 负例 14/14+1 skip + 36 格全矩阵（9/9 valid）+
主目标 paired 复跑（streaming 1.6890 KEEP 稳健；hot 0.9865 NEUTRAL ——
机器态敏感，已如实记录）+ RMSNorm 回归复跑（v2.2 协议）+ NCU 复验
（<1% 漂移）+ 全 5 变体 36 格矩阵 → `experiments/softmax/best.json`
（classify_cell：36 格全部 NO_UNIQUE_WINNER；17 格 INCUMBENT 标签）。

### Phase 7 — 文档、审计、发布
- 独立 benchmark methodology review（subagent 只读审计）；
- README / STATUS / PROJECT_PLAN 更新；`docs/softmax_algorithm.md`、
  `docs/evaluator_hardening_v0.3.md`；
- 最终中文报告 `docs/report_v0.3_result.md`（Q1–Q6 + Evaluator Generalization
  Verdict，含 v2.2 偏离与机器态漂移的全部限定）；
- 分小 commit、working tree clean、push `v0.3-softmax`（**不 merge main**）。

# v0.4 — Evaluator v2.3 + RoPE Generalization（2026-09-20 起）

核心问题：**闭环能否迁移到第三个算子（RoPE），同时 evaluator 从 v2.2 升级到
v2.3（对称 guard + raw/filtered 双轨 + filter-sensitivity）？** 算子：
**interleaved RoPE**（a=x[2i], b=x[2i+1], c=cos[pos,i], s=sin[pos,i]；
y[2i]=a*c−b*s, y[2i+1]=a*s+b*c；FP32 中间，输出原 dtype；base=10000，
max_seq_len=4096；FP16 主 + FP32，禁 BF16；sm_75 / CUDA 11.8）。
分支 `v0.4-rope`（基线 main = v0.3.1 = 86bd871），**不 merge 回 main、
不开始 v0.5**。真实目标排序：evaluator v2.3 更可信 > 第三算子自然接入 >
agent 从 profiler 证据形成有效实验——**不追求 RoPE 一定优化成功**
（baseline 已近硬件/发射下限时，全 NEUTRAL 是 PASS 结局）。

## v0.4 Phases

### Phase 0 — v0.3.1 发布 ✅
main 打 annotated tag v0.3.1（= 86bd871）；从 main 拉 `v0.4-rope`。

### Phase 1 — Evaluator v2.3（paired-streaming-v2.3）✅
对称 log 空间 guard（|log(t/ref)| > log(F)，快慢同因子）：per-sample
spike 1.5×（新增 fast 侧）+ cross-block 1.15×（新增 fast 侧，3-block
warmup）；raw/filtered 双轨记录（每 round raw/filtered 中位数、raw_speedup、
rejected_samples{fast,slow}、environment_guard 自描述块）；filter-sensitivity
（方向翻转或 |log(filtered/raw)| > log(1.10) → 敏感；KEEP/REJECT + 敏感 →
`apply_filter_gate` 降级 UNSTABLE，记录 original_decision）；guard 逻辑纯
CPU 函数（stats.apply_spike_guard / block_stats / crossblock_flag）+
tests/test_evaluator_v23_cpu.py 28/28。
详见 [docs/evaluator_v2_3.md](docs/evaluator_v2_3.md)。

### Phase 2 — v2.3 回归硬门（RoPE 之前）✅ PASS
v2.3 不得推翻 v2.2 已知结论：Softmax baseline vs vec4 streaming
**1.6745** [1.6727,1.6793] 9/9（v2.2 参考 1.6772/1.6890，精确复现，
raw=filtered，rejected 0/0）；RMSNorm v4 vs v1 streaming 1.0375 →
复跑 0.9576（数分钟内方向翻转 = 环境微态漂移，**非 evaluator 缺陷**，
四项调查证据见 evaluator_v2_3.md §6：raw==filtered、对称 guard 全程可审计、
Softmax 对照精确复现、idle 微态漂移有前科）；hot 0.9923 带内。
记录 `benchmarks/v2.3_regression/`。

### Phase 3 — RoPE 算子 + baseline + 正确性/负例 ✅
`kernels/rope/`（rope_common.h 自注册表 + rope_baseline.cu（1 线程→1 pair，
grid M×D/2，FP32 旋转）+ bindings.cpp 统一 launch 前验证 + 启动后检查）；
`cudalab/operators/rope.py` adapter（9 形状矩阵 (1,64)…(4096,128)、主目标
(1024,128)、bench pool（16 缓冲轮换，cos/sin 共享 2 MiB）、NCU driver、
Python rope_ref）；`cudalab/rope_correctness.py` 384 项（finiteness +
double-rounding 算术界 K=2 vs fp64 精确旋转 + norm 保持；与 torch 参考的
allclose **报告不门控**——fp32 大值抵消 FMA 工件已文档化）；
`cudalab/rope_negative.py` 31 例。baseline 384/384 + negative 30/31
（1 例为预期 PASS 对照）。

### Phase 4 — Baseline bench + NCU（含同步验证修复）✅
**首跑 baseline 28.8/29.8 µs 定位为验证路径缺陷**：positions 值域检查
（0≤p<L）的同步 D2H 拷贝逐 launch 强制流同步（~25–30 µs）。修复：
验证拆为 meta（host 元数据，始终执行）+ range（validate 门控）；
forward/forward_into 新增 `validate` 参数（默认 true）；benchmark pool
与 NCU driver 传 validate=False（池契约文档化：池构造期全量预验证，
positions=0..M-1<4096 必然成立）；正确性/negative/正常调用路径默认不变。
修正后 baseline (1024,128) fp16：streaming 6.981 µs / 113.8 GB/s、hot
6.637 µs / 119.7 GB/s（9/9 × 双模式）；修正前记录保留为
`*_presyncfix_archive.json`（审计痕迹）。
baseline NCU（ccall+ccnone，--clock-control base 1755MHz）：kernel 4.0 µs、
dram 23.25%、sm 11.61%、occupancy 78%、long_scoreboard 69.3% —— 稳态
流内 launch 发射速率受限形态。

### Phase 5 — ≥4 个自主优化实验 ✅（4/4，全部 NEUTRAL —— PASS 结局）
| 实验 | 变体 | 假设（来自剖析） | 结果（(1024,128) fp16 streaming，paired v2.3） |
|---|---|---|---|
| ROPE-0001 | `rope_v1_2pair` | baseline 65536 线程、long_sb 69% → 1 线程→2 pairs（grid 减半、8 loads 提前，MLP 杠杆） | NEUTRAL（median 1.0000 [0.9849,1.0160]，4/9）；NCU 单 launch 快 7.6%（3.696 vs 4.000 µs）但稳态流内无效 —— kernel 时长不是瓶颈 |
| ROPE-0002 | `rope_v2_4pair` | MLP 杠杆继续到 4 pairs（需 D%8==0） | NEUTRAL（1.0010 [0.9331,1.0211]，5/9）；NCU +25%（5.008 µs，occupancy 21.5%）—— 波坍缩开始 |
| ROPE-0003 | `rope_v3_half2` | NCU 显示发射侧非瓶颈 → 指令数削减控制（fp16 `__half2` 打包 load/store + FP32 旋转，数学与 baseline 位级一致） | NEUTRAL（0.9974 [0.9888,1.0025]，2/9）—— **成功的阴性对照**，验证评估器拒绝灵敏度 |
| ROPE-0004 | `rope_v4_8pair` | MLP 杠杆边界（需 D%16==0） | NEUTRAL（0.9922 [0.9861,1.0055]，3/9；rejected fast=39 —— v2.3 快侧 spike 防护真实工作）；NCU +104%（8.176 µs，occupancy 11.9%）—— 波坍缩灾难区 |

四个候选 384/384 全部通过。结论（v0.4.1 限定范围）：在当前 Python →
pybind → PyTorch C++ extension → CUDA launch 的 benchmark submission
path 下，主目标表现出明显 launch/host-issuance sensitivity（paired
API-path: ≈ 6.4 µs vs ≈ 6.4 µs；NCU kernel-only: baseline ≈ 4.00 µs,
v1 ≈ 3.70 µs）；因此不能直接推断: 未来原生 C++ CUDALM 中 v1 也无
收益。MLP 杠杆甜区在 1–2 pairs/thread；≥4 pairs 波坍缩。NCU 用于诊断、
paired bench 用于决策 —— 正是本 lab 方法论的价值体现（v1_2pair 的
NCU −7.6% 没有转化为流内 ≥5% 优势，NEUTRAL 是正确决策）。

### Phase 6 — 全矩阵 + PyTorch context ✅
36 格矩阵（9 形状 × {fp16,fp32} × {hot,streaming}）× 5 变体（全部候选
保留；v2/v4 的 D%8==0 / D%16==0 约束在 9 形状上全部满足）+ shape
winners；PyTorch 2.4.1 **无内置 fused RoPE op** → Python 参考（rope_ref，
多 kernel）仅作 implementation context，不产生"X× faster than PyTorch"
headline。

### Phase 7 — 文档、审计、发布
- 3 个独立 subagent review（CUDA Correctness / Benchmark Methodology /
  RoPE Math）+ Lead 最终审计；
- README（CUDALab Operators: RMSNorm/Softmax/RoPE；evaluator 演进
  v1→v2→v2.2→v2.3 + "持续修正的系统" 定位）/ STATUS / PROJECT_PLAN 更新；
- 最终报告 `docs/report_v0.4_result.md`（# CUDALab v0.4 Result）；
- 分小 commit、working tree clean、push `v0.4-rope`（**不 merge main**）。

## v0.4.1 Merge Fix（2026-09-21，外部 review 后）

只修 4 项 finding：不加 RoPE variant、不做新优化、不开始 GEMV、不重跑
full 36-cell matrix。

1. **`rope_v3_half2` 对齐加固**：fp16 路径 `reinterpret_cast<const
   __half2*>` 的 4B 基指针对齐契约显式化（`is_contiguous()` 不保证
   4B）；x/out 任一未 4B 对齐 → 回退到与 baseline 逐语句同数学的标量
   fp16 kernel（位级一致），不拒绝调用。负例 +3 例对齐回归（37 例，
   36/37 all_pass）。
2. **FILTER_SENSITIVE gate 收紧**：filter_sensitive → 最终
   policy_decision 一律 UNSTABLE（KEEP/REJECT/NEUTRAL 均降级，原决策
   记 original_decision）。v0.4 记录 0 敏感，历史零影响。
3. **statistical_relation / policy_decision 形式分离**：CI95 判定的
   统计陈述（FASTER/SLOWER/UNRESOLVED）与 5% 阈值的 acceptance policy
   （KEEP/REJECT/NEUTRAL/UNSTABLE）分别入 schema（decision 纯函数 +
   classify_cell + CLI + ROPE-0001..0004 元数据，由已存 CI 推导，
   原始数字未动）；experiment.py 错误措辞更正。
4. **文档范围限定**：launch-bound 结论限定在当前 benchmark submission
   path（NCU kernel-only vs paired API-path 双口径 + 不外推到未来
   原生 C++ CUDALM）；`operators/rope.py` wave 计数笔误更正
   （≈ 1.07 theoretical-residency waves ≈ 107%）。

验证：v3 正确性 384/384 + 表核对、负例 36/37、baseline/v3 smoke pair
9/9（0.969977，harness 完整性检查）、CPU 36/36+18/18+20/20+6/6。
3 commit + push `v0.4-rope`（**不 merge main**，等待外部最终 merge
review）。

## 不可妥协的规则
- 不伪造任何数字；每个报告的指标都来自真实执行。
- 编译时间绝不计入内核计时；先基准、后剖析。
- 所有变体使用相同输入；权重乘法绝不跳过。
- 失败的实验保留并报告。

## 状态
实时进展见 `STATUS.md`。
