# CUDALab v0.4 Result

**分支**: `v0.4-rope`（基线 `main` = v0.3.1 = 86bd871）
**日期**: 2026-09-20（所有时间戳 UTC+08:00）
**硬件/环境**: NVIDIA RTX 2080 Ti × 1（Turing, sm_75, 30 SM, L2 5.5 MB, 规格峰值带宽 616 GB/s），
CUDA 11.8，PyTorch 2.4.1+cu118，容器化（无时钟锁定、无 compute-sanitizer）。
**分支状态**: 未 merge 回 main，等待外部 review。

## 1. Status

**v0.4 完成（PASS 结局）。3 个独立 review（CUDA Correctness / Benchmark
Methodology / RoPE Math）全部交付（PASS / PASS / PASS WITH CAVEATS），
review 发现已全部处置（代码修复 + 文档更正 + 记录重录），分支已推送。**

- Evaluator **v2.3（`paired-streaming-v2.3`）** 实现并通过回归硬门：对称 guard、
  raw/filtered 双轨、filter-sensitivity 全部落地，29/29 CPU 单测（review 后
  新增 1 个约定钉死测试），两条已知回归（Softmax KEEP / RMSNorm 方向敏感对）
  均 PASS。方法论 review 结论：*"evaluator v2.3 is sound, symmetric per
  spec, and every recorded decision is fully reproducible"*（§17）。
- 第三算子 **RoPE（interleaved 约定）** 完整接入：baseline + 4 个候选变体、
  正确性 384/384 × 5 变体 + 独立表值核对（review 新增门）、负例
  36/37 执行 all_pass（1 项跳过、5 项预期 PASS、31 项必须拒绝；review
  后扩至 34 例 + v0.4.1 新增 3 例 half2 对齐回归 = 37）、
  baseline bench（含一次真实缺陷的发现与修复）、NCU、4 个自主优化实验
  （全部 NEUTRAL）、36 格全矩阵 + shape winners、PyTorch context。
- 诚实结论：**(1024,128) 主目标上 baseline 已贴近稳态流内 launch 发射下限，
  四个候选全部 NEUTRAL —— 按预设判据这是 PASS 结局，不是失败。**
  NCU 单 launch 上 v1_2pair 快 7.6%，但没有转化为流内 ≥5% 优势，
  评估器正确判为 NEUTRAL。没有为项目故事挑选好看的数字。

## 2. v0.3.1 Release

- `main` 打 annotated tag **v0.3.1 = 86bd871**（v0.3 全部成果：evaluator v2.2、
  Softmax 自主优化闭环、文档与复现链）。
- `v0.4-rope` 从 main 拉出（6 个功能 commit + review 修复 commit，见 §15），
  working tree clean。
- v0.3 的已知结论作为 v2.3 回归硬门的参考基线：
  Softmax baseline vs vec4 streaming 1.6772（SFM-0001）/ 1.6890（final_reval）KEEP；
  RMSNorm v4 vs v1 streaming 0.948892 REJECT（v2.2，2026-09-20 同日早些时候）/
  hot 0.970954 NEUTRAL。

## 3. Evaluator v2.3

代码：`cudalab/evaluator/{bench.py,stats.py,decision.py}`，
设计文档 `docs/evaluator_v2_3.md`，CPU 单测 `tests/test_evaluator_v23_cpu.py`
（29/29，其中 1 个为 v0.4 review 后新增的矩阵 raw 轨聚合约定钉死测试，§17）。

**动机**：v2.2 的 guard 只拒绝"异常慢"的样本（不对称）——环境偶发抖动若恰好
打在 candidate 侧的**快**边（或 parent 侧的慢边），中位数估计会静默偏置，
且无审计痕迹。v0.3 的 RMSNorm 复跑（0.9489 → 0.9710 方向漂移）暴露了
"guard 行为不可审计"的问题。

**三项变更**：

1. **对称 log 空间 guard**（纯 CPU 函数 `stats.apply_spike_guard` /
   `stats.block_stats` / `stats.crossblock_flag`）：
   - per-sample spike：`|log(t/ref)| > log(1.5)`，ref = 该变体最近 50 个
     **已接受**样本的中位数 —— 快慢两侧同因子拒绝（fast 侧为新增）；
   - cross-block：`|log(med/running_med)| > log(1.15)`，running_med = 该变体
     本轮已出块的 filtered 中位数（3-block warmup 后启用）—— fast 侧 flag 新增；
   - parent 与 candidate 用完全相同的规则与阈值；被拒样本分
     `rejected_samples{fast, slow}` 计数留痕。
2. **raw/filtered 双轨记录**：每个 block 同时记录 raw_median_us（guard 前）
   与 median_us（guard 后）、样本数、raw/accepted 样本数组；pair 记录增加
   `raw_speedup` / `filtered_speedup`（各自轨道的 per-round 比值中位数）+
   raw/filtered 各自的跨 round 中位数与 bootstrap CI95；记录含自描述
   `environment_guard{method, symmetric, spike_factor, spike_window,
   crossblock_factor, crossblock_warmup, min_accepted_samples,
   filter_sensitive_log_delta, raw_and_filtered_recorded}` 块。
   矩阵记录同步：per-round `us`（filtered）+ `us_raw`，每变体
   `median_us` + `raw_median_us`，记录级 `winner_raw`。
3. **filter-sensitivity**（`stats.filter_sensitive` + `decision.apply_filter_gate`）：
   若 raw 与 filtered 两轨 speedup 在 1.0 两侧方向翻转，或
   `|log(filtered/raw)| > log(1.10)`（≈9.53%），标记 `filter_sensitive=true`；
   **KEEP/REJECT + filter_sensitive → 降级 UNSTABLE**，原判定记入
   `original_decision`；NEUTRAL/UNSTABLE 只留标记。矩阵格同理（raw 与
   filtered 两套 winner 排序不同即敏感）。

判定引擎（decide_v2）语义不变：KEEP 需 median≥1.05 且 ≥70% round 更快且
CIlo>1.00；REJECT 需 median≤0.95 且 ≤30% 更快且 CIhi<1.00；有效 round <5 →
UNSTABLE；bootstrap seed 20260919，n_boot=10000。
**statistical_relation（FASTER/SLOWER/UNRESOLVED）与 policy_decision
（KEEP/REJECT/NEUTRAL/UNSTABLE）保持形式分离**（v0.2 起的原则，v2.3 延续）。

## 4. Evaluator Regression（v2.3 回归硬门）

记录：`benchmarks/v2.3_regression/`（4 条 pair 记录 + `gate_summary.json`，
generated 2026-09-20T17:34:50+08:00）。全部 9/9 有效 round。

| 用例 | v2.3 结果 | raw/filtered | 拒绝样本 | v0.3.1 参考（v2.2） | 判定 |
|---|---|---|---|---|---|
| Softmax baseline vs vec4, M128 H4096 fp16 streaming | **1.674508** [1.672655, 1.679296]，9/9 更快 | raw=filtered | 0/0 | 1.6772 / 1.6890 KEEP | **PASS**（精确复现，差异 <0.25%） |
| RMSNorm v4 vs v1, M128 H4096 fp16 streaming | **1.037543** [1.027895, 1.098996]，9/9 更快 | raw=filtered | slow=1 | 0.948892 REJECT（同日早些时候） | 见下 |
| 同对复跑（streaming repeat2） | **0.957568** [0.950919, 0.96232]，0/9 更快 | raw 0.958856 / filtered 0.957568（差 0.0013 ≪ log(1.10)） | slow=51 | —（v0.3.1 仅跑过一次 streaming） | 见下 |
| RMSNorm v4 vs v1, hot | **0.99227** [0.989298, 1.010596]，2/9 更快 | raw=filtered | slow=19 | 0.970954 NEUTRAL | **PASS**（NEUTRAL 带内） |

**RMSNorm 方向翻转调查结论：环境微态漂移，不是 evaluator 缺陷**（四项证据，
详见 `docs/evaluator_v2_3.md` §6）：
1. run 1 的 raw==filtered 完全一致 —— guard 没有参与改变结果；
2. repeat2 的 51 个慢样本是被**对称** guard 拒绝的，raw 与 filtered 估计
   仅差 0.0013（远低于 log(1.10)≈0.0953 的敏感阈值），全程可审计；
3. Softmax 对照在 v2.3 下精确复现（1.6745 vs 1.6772/1.6890）——
   引擎本身稳定；
4. 该机器 idle 微态漂移有前科（`docs/evaluator_hardening_v0.3.md` 记录过
   同对复跑 0.9489→0.9710）。
即：**这台机器上 ±5% 量级的 RMSNorm 对本来就在噪声带内**，v0.3.1 的
REJECT（0.9489，CIhi<1.00）与 v2.3 的两次 1.0375/0.9576 都落在"方向敏感
对"区间内；v2.3 的价值在于把这种漂移**显式化**（raw/filtered 双轨 +
拒绝计数 + filter_sensitive 标记），而不是消除它。

## 5. RoPE Definition

**interleaved RoPE**（本 lab 约定，非 NeoX half-split）：

```
对每行 x ∈ R^D（D 偶）、位置 p：
  a = x[2i], b = x[2i+1]
  c = cos[p, i], s = sin[p, i]        i = 0..D/2−1
  y[2i]   = a·c − b·s
  y[2i+1] = a·s + b·c
```

- FP32 中间计算，输出 dtype = 输入 dtype；fp16 主、fp32 对照（禁 BF16）。
- cos/sin 表：θ[pos,i] = pos · base^(−2i/D)，**base=10000，max_seq_len=4096**，
  表形状 (4096, D/2)，**FP32 构建后 cast 到工作 dtype**（仓库自文档化的构造
  约定；math review 曾按审计简报中的"fp64 构建"表述标记偏差，仓库内部
  自洽，处置见 §17：保留 FP32 构建 + 新增独立 fp64 核对门把它变成界内
  可验证）；基准路径上内核与 Python 参考消费同一张 cast 表，表值本身由
  正确性套件的独立表值核对判定（§7 门 3）。
- 矩阵 9 形状 (M,H) ∈ {(1,64),(1,128),(32,64),(32,128),(128,64),(128,128),
  (1024,64),(1024,128),(4096,128)}；**主目标 (1024,128)**。
- 所有 D 满足 v2 的 D%8==0 与 v4 的 D%16==0 约束（64/128）。

## 6. RoPE Implementations

全部新写，自注册于 `kernels/rope/rope_common.h`，统一入口
`kernels/rope/bindings.cpp`（`ext.forward(variant, x, positions, cos_t, sin_t,
validate=True)` / `forward_into(...)` / `variants()`）：

| 变体 | 结构 | grid | 约束 | 假设来源 |
|---|---|---|---|---|
| `rope_baseline` | 1 线程→1 pair，标量，FP32 旋转，block 128 | M×D/2 | D 偶 | 参考实现 |
| `rope_v1_2pair` | 1 线程→2 pairs，8 个 load 全部 hoist（MLP 杠杆） | M×D/4 | D 偶 | NCU：long_scoreboard 69.3%、DRAM 23.25%、SM 11.61% |
| `rope_v2_4pair` | 1 线程→4 pairs，16 个 load hoist（8 x + 4 cos + 4 sin，fp16 32B） | M×D/8 | D%8==0 | MLP 扫描续 |
| `rope_v3_half2` | 1 pair/thread；fp16：`__half2` 打包 load（4B）+ FP32 旋转 + 打包 store（4B）；fp16 路径输出与 baseline **逐位一致**（192/192 fp16 用例）；fp32 路径 = baseline 标量，**不作逐位一致声明**（见下） | M×D/2 | — | 指令数削减对照（NCU 显示发射端不紧） |

**fp32 跨构建逐位漂移（review F5 更正）**：v3 头注释原声称 fp32 与
baseline"逐位一致"，被记录数据推翻 —— 5 个变体正确性 JSON 中 110/192 个
fp32 用例的误差值互不相同（如 (1024,64) fp32 pos_max_seq_len−1：baseline
max_abs 4.8e-7 vs 候选 2.4e-7，arith ratio 0.4438 vs 0.4831，全部在界内
全过）；fp16 路径 192/192 完全一致（v3 唯一实质改动处）。归因：nvcc
逐函数 codegen/FMA 收缩漂移（device 源码相同的函数在不同构建/变体间
不保证逐位相同）。故正确性合同只声明"固定 arith 界内"，不声明逐位一致
（§14.9）。
| `rope_v4_8pair` | 1 线程→8 pairs，32 个 load hoist（16 x + 8 cos + 8 sin，fp16 64B） | M×D/16 | D%16==0 | MLP 扫描上限 |

**同步验证修复（本分支核心工程发现）**：首跑 baseline 28.817/29.768 µs
（streaming/hot），异常地比 6.98 µs 量级高 ~22 µs。定位为验证路径缺陷：
positions 值域检查（0≤p<L）需要 `positions.to(at::kCPU)` 同步 D2H 拷贝，
每次 launch 强制流同步（~25–30 µs）。修复：验证拆为 meta（host 元数据，
始终执行）+ range（`validate` 门控）；`forward`/`forward_into` 新增
`validate` 参数（默认 true）；**benchmark pool 与 NCU driver 传
validate=False**（契约文档化：池构造期全量预验证，positions=0..M-1<4096
必然成立）；正确性/负例/正常调用路径默认行为不变。修正前记录保留为
`benchmarks/rope/rope_base_main_*_presyncfix_archive.json`（审计痕迹）。

## 7. Correctness

**正确性门**（`cudalab/rope_correctness.py`，384 项，SEED=0）：
1. 有限性（无 NaN/Inf）；
2. **双舍入算术界**：对精确 float64 旋转，逐元素误差 ≤ K=2 ulp 误差模型
   （arith_max_ratio ≤ 1）；fp16 ulp 用解析 binade 公式
   2^(floor(log2|v|)−10)（torch.nextafter 对 CUDA Half 未实现），
   fp32 ulp 用 torch.nextafter；
3. **范数保持**：逐 pair ||y||²/||x||²−1 ≤ NRM_REL_TOL（fp16 5e-3 / fp32 1e-5），
   分母下限 fp16 4e-9 / fp32 1e-30。
- 与 Python 参考 `rope_ref` 的 elementwise allclose **只报告不判定**
  （fp32 大值 + a·c≈b·s 抵消下，直接 fp32 参考与双舍入界定的精确参考
  合法地差最后几位 —— FMA/舍入工件，已文档化并经 math review 独立复现：
  scale 1000、M=8192 下双舍入 vs FMA 两路径 77,290/524,288 元素不同，
  3 元素超 allclose 容差（~0.6 ulp of |y|≈0.166），两路径均在 K=2 界内
  max ratio 0.28，排除 allclose 出判定门成立，§17）。

**独立表值核对（review 新增第 3 道门，判表不判核）**：门 2/3 都以表的
**存储值**为输入（共同模式）——表自身的构造错误（base 取错、指数符号、
i 轴张错、cos/sin 互换、相位符号）会让 kernel 与 ref 一起错、三门全过。
新增 `check_table_independence`：把套件实际使用的表对数学定义
θ(pos,i)=pos·10000^(−2i/D) 的 fp64 独立求值做**全网格**核对（4096×D/2
点 × 2 信号，无采样），固定误差界
`err ≤ 8·2⁻²³·|angle64| + 2⁻²² + ulp_dtype(|value|)`（FP32 构造链角误差
~5·2⁻²³·|angle| 取 8 留 1.8x；fp32 cos/sin 实现误差 ≤2ulp；cast 按 K=2
记整 1 ulp），另判 pos0 行 cos≡1/sin≡0 逐位精确与正交性 c²+s²−1
（固定阈值 fp16 2e-3 / fp32 1e-6）。5 变体共享同一核对结果，折叠进
all_pass（每份 JSON 的 `table_independence_check` 字段）。实测
（2026-09-20 重录记录，fp16 最紧）：

| dtype, D | n 点 | max abs err (cos/sin) | max err/bound | c²+s²−1 max | passed |
|---|---|---|---|---|---|
| fp16, 128 | 262,144 | 4.05e-4 / 3.81e-4 | **0.4996**（~2x 余量） | 6.87e-4 | ✅ |
| fp16, 64 | 131,072 | 3.34e-4 / 3.39e-4 | 0.4995 | 6.87e-4 | ✅ |
| fp32, 128 | 262,144 | 2.30e-4 / 2.39e-4 | 0.1129 | 8.95e-8 | ✅ |
| fp32, 64 | 131,072 | 1.46e-4 / 1.44e-4 | 0.1121 | 8.95e-8 | ✅ |

pos0 逐位精确 4/4 组合通过；base/指数/i 轴/符号类错误在此界下以 O(1)
量级违例（如 base 误取 1000 → 角偏差 ~O(|angle|) ≫ 界 ~1e-3），该类错误
从"结构不可见"变为门内可捕获。

**结果**（`experiments/rope/correctness/v0.4/`，review 后重录，
generated 2026-09-20T23:42）：

| 变体 | 通过 | max_arith_max_ratio | max_norm_rel_error | 表核对 |
|---|---|---|---|---|
| rope_baseline | **384/384** | 0.50543169 | 1.7614e-3 | ✅ |
| rope_v1_2pair | **384/384** | 0.48309729 | 1.7614e-3 | ✅ |
| rope_v2_4pair | **384/384** | 0.48309729 | 1.7614e-3 | ✅ |
| rope_v3_half2 | **384/384** | 0.48309729 | 1.7614e-3 | ✅ |
| rope_v4_8pair | **384/384** | 0.48309729 | 1.7614e-3 | ✅ |

（12 位精度为 review F6 修复：旧 8 位小数把 tiny 用例的审计值舍成 0.0。）

**负例**（`cudalab/rope_negative.py`，**34 例** = 31 例基础 + 3 例
per-variant 整除性（review 新增：v1/v2/v4 的 D%4/D%8/D%16 TORCH_CHECK
此前未覆盖），generated 2026-09-20T23:42:32）：
31 例必须 launch 前拒绝（均被对应 variant 的 TORCH_CHECK 拒绝，
post_check 验证拒绝未污染 CUDA 上下文且**拒绝消息含该 variant 自己的
约束**）+ 2 例预期 PASS 对照 + 1 例跳过（baseline 标量访存无对齐
契约）= **33/34 执行通过，all_pass=true**。

**CPU 测试套件**：test_evaluator_v23_cpu 29/29（review 后 +1 约定钉死
测试）、test_evaluator_cpu 18/18、test_softmax_cpu 20/20、test_dispatch 6/6。

## 8. Baseline NCU

`profiles/rope/*_M1024_H128_ccall_clkbase.json`（--clock-control base →
1755 MHz，cache flush，M=1024 H=128 fp16；NCU 仅作诊断，不参与判定）：

| 变体 | kernel 时长 | DRAM | SM | occupancy | 寄存器 | long_scoreboard |
|---|---|---|---|---|---|---|
| baseline | 4.000 µs | 23.25% | 11.61% | 78.0% | 16 | 69.3%（19.4 cyc/issue） |
| v1_2pair | 3.696 µs（−7.6%） | 23.74% | 8.74% | 42.9% | 16 | 63.4%（12.3） |
| v2_4pair | 5.008 µs（+25%） | 17.38% | 5.88% | 21.5% | 24 | 49.2%（10.0） |
| v3_half2 | 3.920 µs（−2%） | 22.92% | 11.60% | 75.4% | 16 | 69.7%（18.9） |
| v4_8pair | 8.176 µs（+104%） | 11.02% | 3.54% | 11.9% | 41 | 25.9%（8.8） |

形态解读：baseline 是**稳态流内 launch 发射速率受限**（DRAM 23%、SM 11.6%，
long_scoreboard 主导 stall）。单 launch NCU 时长（~4 µs）与稳态流内
每 launch 成本（~6.4–7 µs）不同口径 —— 这一差异正是后续实验的判读关键。

## 9. Optimization Experiments

4 个实验，profiler→hypothesis 驱动，全部 paired streaming v2.3
（(1024,128) fp16，parent=rope_baseline，9/9 有效，generated 18:32–18:34）：

| 实验 | 变体 | 假设（来自 NCU 证据） | paired 结果 | NCU 单 launch | 判定 |
|---|---|---|---|---|---|
| ROPE-0001 | v1_2pair | long_sb 69%、发射受限 → 1 线程 2 pairs（grid 减半、8 loads 在途） | median 1.0000 [0.984914, 1.016031]，4/9 更快，rejected 0/3 | −7.6% | **NEUTRAL** |
| ROPE-0002 | v2_4pair | MLP 扫描续（16 loads 在途，fp16 32B） | 1.000985 [0.933102, 1.021143]，5/9，rejected 0/0 | +25%（occ 21.5%） | **NEUTRAL** |
| ROPE-0003 | v3_half2 | 指令数削减对照（`__half2` 打包，位级同数学） | 0.99737 [0.988764, 1.002525]，2/9，rejected 0/2 | −2% | **NEUTRAL**（成功的阴性对照） |
| ROPE-0004 | v4_8pair | MLP 扫描上限（32 loads 在途，fp16 64B，8192 线程 ≈0.133 wave） | 0.992155 [0.986084, 1.005505]，3/9，**rejected fast=39 / slow=7** | +104%（occ 11.9%，41 regs） | **NEUTRAL** |

- 四个候选 384/384 全部通过正确性门。
- **ROPE-0001 是本分支方法论的核心展示**：NCU 单 launch 快 7.6%（>5%
  单看会想 KEEP），但稳态流内 9 轮 paired 只有 4/9 更快、median 1.0000、
  CI 含 1.00 → NEUTRAL 是正确判定 —— kernel 时长不是流内瓶颈，
  launch 发射/流同步成本才是。
- **ROPE-0003 是成功的阴性对照**：位级同数学、仅指令形态不同 →
  按证据预测 NEUTRAL/REJECT，实际 NEUTRAL（2/9 更快），验证评估器
  对"真无差异"的拒绝灵敏度。
- **ROPE-0004 的 fast 侧计数：review 后更正叙述**（原稿"v2.2 下这类样本
  会静默污染中位数"被方法论 review 推翻，§17）。39 个 "fast" 拒绝样本是
  v4 的**合法稳态样本**，不是异常：round 1 前半段环境处于 ~2× 慢态
  （~60 个 @~12.4 µs 样本被接受并成为块中位数锚点），环境恢复后 v4 回到
  真稳态 ~6.2 µs（r2–r9 全部 6.15–6.26 µs）；这批恢复样本因 intra-block
  首样本锚定 + 最近 50 已接受样本为参考，被拒为"fast spike"。更正要点：
  v2.2 下该块中位数**相同**（慢样本占 60/100 多数, v2.2 全收）——"静默
  污染"不成立，v2.3 在此只增加了审计计数；实际影响是 r1 candidate
  filtered median 12.416 µs（2× 抬高）、该轮 speedup 0.4985 计入，而
  pair 级判定对此稳健（剔除 r1 → 0.9933，仍 NEUTRAL），
  raw 0.993276 vs filtered 0.992155，差 0.0011 ≪ log(1.10)，
  filter_sensitive=false。**fast 侧 guard 在真实数据上的真正展示在
  矩阵**：11 个 fast cross-block flag 集中在 2 个 M1_H64 streaming 格的
  3 个 round（2 轮为 5 变体同时跌至 0.70–0.86× 的全局瞬态、1 轮为单变体
  跌落 0.76×），全部整轮作废 → 零 variant 偏置（§11）。
- 结论：MLP 杠杆甜区在 1–2 pairs/thread；≥4 pairs 波坍缩（occupancy
  21.5% → 11.9%）；指令数削减对发射受限 kernel 无收益。
  **没有任何候选达到 KEEP 阈值 —— 按预设判据，全 NEUTRAL 是 PASS 结局。**

## 10. Main Target

(1024,128) fp16 主目标，standalone baseline bench（paired-streaming-v2.3，
9/9 × 双模式，generated 18:22）：

| 指标 | streaming | hot |
|---|---|---|
| median | **6.981 µs** | **6.637 µs** |
| 有效 round | 9/9 | 9/9 |
| algorithmic 带宽（逻辑 IO，非 DRAM 实测） | 113.8 GB/s | 119.7 GB/s |
| 修正前存档（D2H 同步缺陷期） | 28.817 µs | 29.768 µs |

逻辑 IO = (M×D×2B×2) / t；该算子工作集（x+out 256KB×2 ×16 轮换 ≈ 8 MiB
> L2 5.5 MiB，streaming 模式）远未打满 616 GB/s 规格峰值 ——
发射受限形态下这是预期的，**不据此宣称任何 DRAM 饱和度**。

矩阵 run 内主目标口径（`rope_v04_matrix_M1024_H128_*`，5 变体，
median of per-round）：

| dtype × mode | baseline | v1_2pair | v2_4pair | v3_half2 | v4_8pair |
|---|---|---|---|---|---|
| fp16 streaming | 6.540 µs | 6.880 | 6.858 | 6.589 | 6.543 |
| fp16 hot | 6.214 µs | 6.208 | 6.405 | 6.296 | 6.400 |
| fp32 streaming | 6.910 µs | 6.528 | 6.528 | 6.584 | 7.809 |
| fp32 hot | 6.291 µs | 6.373 | 6.318 | 6.311 | 7.224 |

注 1：矩阵 run 内 baseline（6.540 µs）与 standalone 记录（6.981 µs）
差 ~6% —— 跨 run 绝对时间不可比（机器微态漂移），**判定只认 run 内 paired**。
注 2：fp32 streaming 格存在 round 级双峰（round 1/6 的 v1_2pair 块
8.576/7.237 µs 慢块）：ratio-of-medians = 1.059 但 median-of-per-round-ratios
= 1.0028 —— shape winners 用后者（对 round 级异常稳健），
与"矩阵 winner 仅指示性，判定以 paired 为准"的既定口径一致。

## 11. Shape Matrix

`benchmarks/rope/rope_v04_matrix_M*_H*.json`（36 格 = 9 形状 × {fp16,fp32} ×
{hot,streaming}，每格 5 变体，round-robin 轮转）+
`rope_v04_matrix_shape_winners.json`：

- **有效性**：31/36 格 9/9 有效，4 格 8/9、1 格 7/9（spike/cross-block
  拒绝所致，均 ≥7 ≥ 最小有效线 5）；
- **fast cross-block flag 共 11 个，全部集中在 2 个 M1_H64 streaming 格
  的 3 个 round，均整轮作废（任一 variant flag → 整轮 invalid，
  INVALID_CROSSBLOCK）**：M1_H64 fp16 r5（5 变体同时跌至 running median
  的 0.70–0.77×，全局瞬态）、M1_H64 fp16 r6（单 variant 0.76×）、
  M1_H64 fp32 r4（5 变体同时 0.80–0.86×，全局瞬态）—— 没有任何一个
  variant 因邻居的瞬态被单独牺牲（整轮作废 ⇒ **零 variant 偏置**），
  这是 v2.3 fast 侧 cross-block 规则在真实数据上的正确行为样本；
- **filter-sensitive 格数：0**（raw 与 filtered 两套 winner 排序在所有格一致；
  raw 侧聚合约定已按 review 更正为与 filtered 侧相同的 per-round 比值中位数，
  见 §17，全部 36 格 filter_sensitive=false 不受影响）；
- **无 policy 级 winner**：全部 36 格 winner/runner-up 比值
  **0.9941 – 1.0334**，全部 < 1.05 的 KEEP 阈值 —— 矩阵层面没有任何变体
  值得采纳；
- 指示性 winner 分布：baseline 16、v1_2pair 15、v2_4pair 3、v3_half2 2、
  v4_8pair 0；
- 主目标格 (1024,128)：fp16 hot winner=v1_2pair（1.000966）、
  fp16 streaming winner=baseline（0.998318）、fp32 hot winner=baseline
  （1.000636）、fp32 streaming winner=v1_2pair（1.002795）——
  差距全部在噪声带内。

**PyTorch context**（`benchmarks/rope/pytorch_ref_M1024_H128.json`，
generated 18:55；implementation context only，不产生 headline，不参与判定）：
PyTorch 2.4.1 **无内置 fused RoPE op** → 用 Python 参考 `rope_ref`
（索引 + cos/sin gather + 多次 kernel 启动，非公平 fused-kernel 对比）：
fp16 median 262.128 µs / fp32 180.343 µs（n=200）—— 仅作实现参考
（fused kernel ~6.5 µs 量级 vs 多 kernel Python 路径 ~200–260 µs 量级）。

## 12. What RoPE taught us

1. **launch 发射下限是可以被 NCU 单 launch 数字误导的**：v1_2pair 单 launch
   −7.6%，流内 paired 却 4/9。对发射受限（而非计算/带宽受限）的小 kernel，
   "kernel 更快"不等于"流内更快"。NCU 用于诊断、paired bench 用于决策，
   两者口径必须分开 —— 这是本 lab 方法论在第三个算子上的又一次验证。
2. **MLP 杠杆有甜区**：2 pairs/thread（grid 减半、8 loads 在途）是单 launch
   最优；4 开始波坍缩（occ 21.5%）、8 是灾难区（+104%、occ 11.9%）。
   对发射受限 kernel，减少线程数本身就有代价。
3. **阴性对照 + guard 可审计性 = 闭环可信度**：v3_half2（位级同数学）判
   NEUTRAL，验证评估器对"真无差异"的拒绝灵敏度；v4_8pair 的 round 1
   锚定偏差（~60 个慢态样本锚定块中位数，环境恢复后 ~40 个合法样本被
   首样本锚定 + 最近 50 已接受参考拒为 "fast spike"，计数留痕
   rejected fast=39，pair 判定不变 0.9933/NEUTRAL，filter_sensitive=false）
   —— guard 留下的不是"误判"而是**结构性偏差**（净方向朝高估时间、
   保守侧），其修复方案记录于 §14。
4. **验证路径也是性能路径**：一个 launch 前的同步 D2H 值域检查就能把
   6.98 µs 的 kernel 基准整体抬到 28.8 µs（×4.1），且**首跑不查就发现不了**
   —— 基准数字必须能与独立参考（NCU 单 launch 4 µs）交叉核对。
   "池契约 + validate 门控"是后续算子（GEMV）复用 bench pool 时应固化的模式。
5. **全 NEUTRAL 也是 PASS**：baseline 已贴近流内发射下限时，没有 KEEP
   是正确的科学结论；为项目故事硬造一个"赢"的变体才是事故。

## 13. Evaluator Generalization Verdict

**v2.3 通过泛化验证。** 判据与结果：

| 判据 | 结果 |
|---|---|
| 对称 guard（快慢同因子、parent/candidate 同规则） | ✅ 代码 + 29/29 CPU 单测（review 后 +1 约定钉死测试） |
| raw/filtered 双轨完整记录（pair + 矩阵 + environment_guard 自描述） | ✅ 全部 v2.3 记录含 raw 轨 |
| filter-sensitive 标记 + KEEP/REJECT 降级 UNSTABLE（original_decision 留痕） | ✅ 实现 + 单测；v0.4 实际记录 0 敏感（真实数据上无敏感发生） |
| 回归门：Softmax KEEP 精确复现 | ✅ 1.6745 vs 1.6772/1.6890 |
| 回归门：RMSNorm 方向敏感对行为可解释、可审计 | ✅ 四项证据（§4） |
| 第三算子接入不改 evaluator 代码 | ✅ rope adapter 只写了 operators/rope.py + suite；harness 零算子特化 |
| 新 guard 在真实数据上产生过可审计行为 | ✅ 11 个 fast cross-block flag（矩阵，3 轮全部整轮作废 → 零 variant 偏置，§11）；ROPE-0004 fast=39 为锚定偏差计数（合法恢复样本），pair 判定稳健（剔 r1 → 0.9933）、filter_sensitive=false |

v2.3 相对 v2.2 的本质变化：**guard 行为从"静默发生"变成"全程留痕可审计"**
（raw/filtered 双轨 + fast/slow 分侧计数 + 自描述 environment_guard 块 +
filter_sensitive 标记），而判定阈值与决策引擎不变。RMSNorm 方向翻转事件
证明：对 ±5% 噪声带内的对，任何版本都会给出方向不同的判定 ——
v2.3 不解决环境问题，它让环境问题**可见、可审计、可标记**。

## 14. Known Limitations

1. **validate 契约误用风险**：bench pool / NCU driver 传 validate=False，
   依赖"池构造期预验证 + positions=0..M-1<4096"契约。若未来调用方绕过
   池直接以 validate=False 调 forward，将跳过值域检查（其他 meta/out 检查
   仍在）。契约已文档化于 rope.py make_bench_pool docstring、bindings.cpp
   头注释、README 局限节。
2. **无 fused RoPE op 可对比**：PyTorch 2.4.1 无内置 RoPE，"vs PyTorch"
   数字（262/180 µs）是 Python 多 kernel 路径，非公平对比，仅作 context。
3. **机器微态敏感**：容器化无锁频，±5% 量级 pair 的判定跨 run 不稳定
   （RMSNorm 对已三次演示）；跨 run 绝对时间不可比，判定只认 run 内 paired。
   v2.3 的对称 guard 与双轨记录缓解**可见性**，不消除漂移本身。
4. **块内锚定偏差（已文档化、未修复）**：块级 spike guard 的构造
   （首样本必收 + 参考 = 最近 50 个已接受样本中位数）在块内发生
   >2× 环境态阶跃（slow→fast 恢复）时，会把合法恢复样本拒为
   "fast spike"；**净偏差方向朝高估时间（保守侧）**，且 raw 轨同样被
   污染、dual-track 对此盲（methodology review finding 2）。ROPE-0004 r1
   是真实实例（fast=39，pair 判定稳健、filter_sensitive=false，§9）。
   修复方案（v2.4 候选，methodology 建议）：块内检测 >2× 阶跃 →
   整轮 invalidate + retry，或将 cross-block 检查用 raw 样本扩展到
   warmup 块。
5. **crossblock flag 历史含被 flag 块**：`crossblock_flag` 无条件把被
   flag 的块追加进 running_med 历史（stats.py，v2.2 遗留）—— 被 flag 的
   慢块会抬高 running_med、削弱后续慢块的可检出性；判定本身只用
   有效轮（flag 轮整轮作废）。methodology review 实测 14 个 flag
   （11 fast + 3 slow）均为单轮事件、无数据影响；可选修正
   （flagged 中位数不入 hist）留待后续。
6. **cos/sin 表 FP32 构造约定的范围**：表按 FP32 构造后 cast 到
   目标 dtype（仓库自文档化约定；audit brief 的"fp64 构建"表述与
   实现不符，math review F1）。决策：保留 FP32 构造 + 独立表值核对
   门（固定误差界，§7）兜住该约定下的构造正确性；若未来改为
   fp64 构造，表核对界与正确性记录需重新评估、重录。
7. **fp32 跨 build 位级漂移**：fp32 正确性用例在 variant/build 间
   存在位级差异（110/192 例，nvcc codegen/FMA 漂移，差异 ~1e-7 量级，
   如 (1024,64) fp32 pos_max_seq_len−1：4.8e-7 vs 2.4e-7），
   fp16 则 192/192 逐位一致；对 K=2 界无影响，因此 v3 头注释
   不对 fp32 路径做位级一致声明（§6）。
8. **矩阵 winner 仅指示性**：round 级双峰会使 ratio-of-medians 失真
   （§10 注 2）；矩阵不产生 KEEP/REJECT，只产生"该跑哪些 paired"。
9. **NCU 单 launch 与流内口径不可互换**：已有误导实例（v1_2pair −7.6%
   单 launch vs 流内 NEUTRAL）；文档已固化"NCU 诊断、paired 决策"。
10. **正确性门的 FMA 工件**：allclose vs Python 参考只报告不判定
   （fp32 大值 + a·c≈b·s 抵消）—— math review 独立复现：scale 1000
   下双舍入 vs FMA 两路径 77,290/524,288 元素不同、3 元素破 allclose
   容差（~0.6 ulp），两路径均在 K=2 界内（max ratio 0.28）；
   若未来出现"双舍入界内但 allclose 大面积失败"的新形态，
   需人工介入判断（当前 384/384 未出现）。
11. **grid dim int 截断（仅记录，nit）**：5 个 launcher 均把
   M·D/2 强转 int 作 grid 维（`rope_baseline.cu:54` 等），M·D/2 > 2^31
   时静默截断 —— 远超出声明矩阵（max 4096×128/2 = 262144），CUDA review
   建议仅记录、不加 guard。
12. **未做**：块内锚定偏差的修复（§14.4）、v2.3 跨 run 稳定性协议
   （§18 备选）、RoPE 的 BF16 路径、与 Q/K 融合场景（attention 内）、
   更复杂的 position 模式（rotary 表外推 >4096）、GEMV、量化、
   完整 Transformer —— 均不在 v0.4 范围。

## 15. Git Commits

`v0.4-rope`（基线 86bd871 = tag v0.3.1），无 force-push / rewrite / squash：

```
65879b6 rope: review 修复（头注释 load 计数、v3 位级声明范围、独立表值核对门、
          negative 34 例、raw 侧聚合约定 + 钉死测试、审计值 12 位舍入；
          正确性 5×384/384 + negative 33/34 重录 2026-09-20T23:42）
85a7f24 rope: 全矩阵 36 格 × 5 变体 + PyTorch context + v0.4 文档（README/STATUS/PLAN）
f30d805 rope: 实验 ROPE-0001..0004（MLP 扫描 + half2 指令控制, 全部 NEUTRAL）
4bf5f67 docs: v2.3 regression gate 结果（28/28 CPU, 双回归 PASS, 附 RMSNorm 方向翻转调查）
76a1546 rope: gate positions D2H range check behind validate flag + re-bench baseline
c051c96 rope: v2.3 regression gate records + RoPE operator/baseline + suites
a591447 eval: symmetric guards + raw/filtered + filter-sensitive (paired-streaming-v2.3)
（本 commit）docs: 报告定稿（§17 独立 review + lead 最终审计 + errata）+ STATUS/README
```

## 16. Reproduce

```bash
source tools/env.sh          # CUDA_HOME / PYTHON / TORCH_CUDA_ARCH_LIST=7.5 / CUDA_VISIBLE_DEVICES=0
PY=$PYTHON

# CPU 测试（无 GPU；测试文件自带 __main__ runner，不依赖 pytest）
for t in test_evaluator_v23_cpu test_evaluator_cpu test_softmax_cpu test_dispatch; do
  $PY tests/$t.py
done

# RoPE 正确性 / 负例（需 GPU）
$PY scripts/cudalab.py test rope --variant rope_baseline     # 384 项
$PY scripts/cudalab.py negative rope                         # 37 例（31 基础 + 3 per-variant 整除性 + 3 v0.4.1 对齐回归）

# Baseline bench（paired streaming v2.3，9 rounds）
$PY scripts/cudalab.py benchmark pair rope \
    --parent rope_baseline --candidate rope_baseline \
    --M 1024 --H 128 --dtype fp16 --mode streaming --tag rope_base_main_streaming

# NCU 诊断（--clock-control base, cache flush）
$PY scripts/cudalab.py profile rope --variants rope_baseline,rope_v1_2pair \
    --M 1024 --H 128

# 全矩阵 + shape winners
$PY scripts/cudalab.py benchmark full rope --tag rope_v04_matrix
$PY scripts/cudalab.py benchmark winners rope --tag rope_v04_matrix

# PyTorch context
$PY scripts/cudalab.py pytorch rope --M 1024 --H 128 --dtype fp16
```

注意：bench pool 与 NCU driver 内部传 validate=False（池契约：构造期
预验证 + positions=0..M-1）；直接 API 调用保持默认 validate=True。
所有记录 JSON 已入库，复现应产生同结构（不同绝对值）的记录。

## 17. 独立 Review 与 Lead 最终审计

3 个独立 subagent review（各只看分支代码 + 入库记录 + 报告草稿，互不
共享结论；lead 逐条裁决处置）+ Lead 最终审计 + errata 汇总。

### 17.1 CUDA Correctness（subagent 6e5cb049）— **VERDICT: PASS WITH CAVEATS**

| # | 级别 | Finding | 处置 |
|---|---|---|---|
| 1 | minor | negative suite 固定 `rope_baseline`；v1/v2/v4 的 D%4/D%8/D%16 TORCH_CHECK（存在且 launch 前执行）无用例覆盖 | **修复**：suite 按 variant 参数化 + 3 个 per-variant 整除性用例（34 例），重录 2026-09-20T23:42（31 reject_ok + 2 pass_ok + 1 skipped，all_pass） |
| 2 | minor | arith gate 的 exact 参考与内核共用同一张存储 cos/sin 表（common mode）：表构造 bug（base/指数/频率符号）会在 5 变体上 384/384 静默通过 | **修复**：新增独立表值核对门 `check_table_independence`（表 vs fp64 独立求值全网格 + 固定误差界 + pos0 逐位 + 正交性），折叠进 all_pass，记录重录（§7） |
| 3 | nit | validate=False 时唯一 OOB 途径是 position 值；现有两个调用方（bench pool / NCU driver）契约成立 | 维持现状（契约已文档化，§14.1 / README） |
| 4 | nit | launcher 把 M·D/2 强转 int 作 grid dim，> 2^31 时静默截断（远超出声明矩阵） | 仅记录（§14.11），不加 guard |

review 未要求重跑任何记录（数字与代码自洽；代码审查 + 已落盘记录）。

### 17.2 Benchmark Methodology（subagent 334994ae）— **VERDICT: PASS**

> *"evaluator v2.3 is sound, symmetric per spec, and every recorded
> decision is fully reproducible. Fix findings 1–4 in docs/next commit
> (2 minor, 2 nit), no record reruns needed."*

| # | 级别 | Finding | 处置 |
|---|---|---|---|
| 1 | minor | 报告 ROPE-0004 叙述颠倒：39 个 "fast" 拒绝样本是 v4 的**合法稳态**（~6.2 µs），异常是被 guard 留在块中位数里的 12.4 µs 慢态（r1 块内 2× 台阶）；且 v2.2 下同块中位数同样 ≈12.4 µs（慢样本 60/100 多数），"v2.2 下会静默污染中位数"不成立 | **修复**：§9 重写（v2.2 median 相同 → v2.3 只增加审计计数；pair 稳健 0.9933、filter_sensitive=false；fast 侧真实行为展示 = 矩阵 11 个 cross-block flag，§11） |
| 2 | minor | 块内过渡锚定偏差（stats.py:42-62）：块首样本恒接受 + ref = 最近 50 已接受中位数 ⇒ 块中位数钉在块前半段状态；净偏差偏慢；raw 轨同样被污染、dual-track 对此盲 | 文档化（§14.4）+ v2.4 修复候选（块内 >2× 台阶检测 → invalidate + retry，或 cross-block 检查用 raw 样本扩展 warmup 块）；不重跑现有记录 |
| 3 | nit | crossblock hist 无条件追加被 flag 块（stats.py:136，v2.2 遗留）：被 flag 慢块抬高 running_med、削弱后续慢块可检出性；实测 14 个 flag 均为单轮事件 | 文档化（§14.5）；可选修正（flagged 中位数不入 hist）留待后续 |
| 4 | nit | 矩阵 raw 轨用 ratio-of-medians、filtered 轨用 median-of-per-round-ratios（experiment.py:147-148），与 bench.py 两轨均 ratio-of-medians 不一致；双峰轮两口径可差 >5%（fp32-streaming 1.059 vs 1.0028） | **修复**：raw 侧改为与 filtered 侧相同的 per-round 比值中位数（us_raw 已落盘）+ 钉死测试 `test_matrix_raw_convention_median_of_round_ratios`（29/29）；36 格全部 fs=false，无数据影响 |

### 17.3 RoPE Math（subagent 657fc61f）— **VERDICT: PASS WITH CAVEATS**

| # | 级别 | Finding | 处置 |
|---|---|---|---|
| F1 | major | 表构造约定偏离：表以 **FP32 构建**后 cast（`operators/rope.py`），审计约定文本为"fp64 构建"；repo 内部自洽（docstring/JSON 均写 FP32 构建） | **决策：保留 FP32 构建**（repo 自洽，改 fp64 构造将作废现有 384/384 记录且不改变任何结论）+ 约定文本更正（§5/§7）+ 新增独立表值核对门兜住该约定下的构造正确性；改 fp64 构造需重新评估界与记录（§14.6） |
| F2 | minor | 三条 gate 均不约束表误差（内核与 ref 共享同一张表，错误表全过）；表正交性仅经 norm gate 间接约束（5e-3 阈值 vs 实测 1.76e-3，~2.8× 余量） | **修复**：`check_table_independence`（§7：全网格 + 固定误差界 + pos0 逐位 + 正交性） |
| F3 | minor | arith 界 docstring 漏推导中的 +2⁻²³·\|y\| 子舍入项（已证仍有效：\|y\|≤\|ac\|+\|bs\|、K=2 加倍 + 两个 0.5·ulp 覆盖，CPU 实测 max ratio 0.32） | **修复**：docstring 推导补全（模块注释 (1) 节） |
| F4 | minor | 报告称表值"经独立 fp64 核对，见 §17"，当时 §17 为空壳 | **修复**：核对数字落 §7（实测表）+ §17.4（math review 独立 CPU 对照） |
| F5 | minor | v3 声称 "bit-identical"，但 fp32 路径 110/192 用例跨构建不同（设备源码未变 ⇒ codegen/FMA 漂移）；fp16 路径 192/192 全变体一致 | **修复**：头注释只声明 fp16 路径位级一致，fp32 路径不做位级声明（§6/§14.7）；无 gate 影响 |
| F6 | nit | JSON 审计值 round(·,8) 使 tiny 用例 arith_tol_max/max_abs_error 舍成 0.0 | **修复**：共享 helper 改 12 位（`evaluator/correctness.py` / `rope_correctness.py`），记录重录 |
| F7 | nit | `_ulp` 实为**上 binade** 间距（精确 2 的幂处比 nextafter 粗 2×，对 RN 上界保守、安全），docstring"公式精确"不准确 | **修复**：`_ulp` docstring 更正 |

### 17.4 Math review 独立 CPU 验证关键数字（与 repo 代码/数据不共享）

- **pos0 逐位**：positions=0 → y==x bit-exact（fp32/fp16 均 True）；
- **fp64 范数**：‖y‖/‖x‖ 逐行 = 1.0（max 1+2.2e-16）；复合旋转
  R(θ)∘R(θ) vs R(2θ)（fp64）max abs 4.44e-16；
- **CPU 双路径 K=2 比值**（vs 存储值之 fp64-exact）：fp32 管线
  0.3224 / 0.2755 / 0.2713（scales 1/1000/1e-4）；fp16 管线
  0.2480 / 0.2487 / 0.2493（≤0.4985 ulp of |y|）；
- **FMA 工件**（scale 1000, M=8192）：524,288 元素中 77,290 个在
  dbl-round 与 FMA 路径间不同；3 个击穿 allclose(atol=1e-5, rtol=1e-4)
  （最坏 diff 3.93e-5 > tol 2.66e-5，~0.6 ulp of |y|≈0.166）——与记录
  的 7 项 fp32+large allclose 失败一致；两路径均 ≤0.28×K=2 界
  ⇒ allclose 排除出判定门成立；
- **norm gate 单独薄弱示例**：构造 2.00%/2.04% 逐元素误差且范数精确
  保持的"坏内核"：norm_rel=0.0（两档 gate 均 PASS）但
  arith_ratio=4.1e4（FAIL）——norm gate 单独不足，arith gate 兜底，
  联合门 sound；
- **表构建对照（F1 支撑）**：fp32 构建 vs fp64 构建表 max |Δcos|=1.36e-4、
  |Δsin|=1.15e-4（pos 4095, i=1）；c²+s²−1：fp32 表 ≤7.9e-8、
  fp64 表 ≤2.2e-16、fp16 表 ≤6.3e-4。

### 17.5 Lead 最终审计（lead，2026-09-20）

- **判定可复现**：seed 20260919 纯 CPU 重放 v2.3 全部 8 条 pair 记录
  （4 回归 pair + ROPE-0001..0004）+ 矩阵 shape winners 聚合：median / CI /
  判定 / rejected 计数逐项与入库 JSON 一致；
- **对称性**：v2.3 slow 侧规则与 v2.2 逐样本 bit 级一致（parent/
  candidate 同规则、快慢同因子），fast 侧为新增；
- **D2H 修复量级**：修正前 vs 修正后 Δ = 21.8 / 23.1 µs（streaming /
  hot）≈ 每 launch 一次同步 D2H 的成本，与定性定位一致；
- **记录完备性**：正确性 384 = 2×2×4×6×4（dtype×D×M×pos 模式×输入
  模式：float16/float32 × D∈{64,128} × M∈{1,32,128,1024} × 6 pos 模式
  × 输入 normal/zeros/tiny/large）×5 变体 n_fail=0；矩阵 36 格 winner 分布 baseline 16 / v1 15 / v2 3 /
  v3 2 / v4 0；cross-block flag 构成 11 fast + 3 slow，fast 侧 3 个
  触发轮（M1_H64 fp16 r5 全 5 变体 / r6 单变体 v4 / M1_H64 fp32 r4 全 5
  变体）均整轮作废（报告 §9/§11 的早期 overclaim"全部整轮全局瞬态"
  已按记录更正）；
- **重录一致性**：review 后重录的正确性记录（2026-09-20T23:42）与
  修复前数值一致（max_arith 0.505431694705 / 0.483097294008、
  max_norm 1.7614e-3 不变），差异仅为 12 位精度与新 `table_independence_check`
  字段 —— 重录未改变任何判定。

### 17.6 Errata（报告初稿更正汇总）

1. **kernel 头注释 load 计数算术**（CUDA finding 1）：v1 3→4 loads、
   v2 12→16、v4 24→32（按 kPairs×4）——头注释更正；实验 JSON 的
   hypothesis 字段保持原样（历史记录）；kernel 代码本身不受影响
   （仅注释算术错）。
2. **"表 fp64 构建"表述**（math F1）：初稿沿用审计约定文本，实际实现
   为 FP32 构建后 cast —— §5/§7 更正为实际约定，并新增独立表值核对门。
3. **ROPE-0004 叙述颠倒**（methodology finding 1）：初稿"v2.2 下这类
   样本会静默污染中位数"不成立（v2.2 同块中位数相同）—— §9 重写。
4. **v3 "bit-identical" 过度声明**（math F5）：fp32 路径跨构建位级
   漂移 110/192 —— 头注释改为仅声明 fp16 路径位级一致（§6）。
5. **§11 fast cross-block flag 构成**（lead 审计）：初稿"11 个全部是
   整轮全局瞬态/5 变体同时"—— 实际 10/11 在 2 个全 5 变体轮、1 个为
   单变体轮（M1_H64 fp16 r6, v4, 0.7624）；"整轮作废 ⇒ 零 variant
   偏置"结论不变。

## 18. Recommended v0.5

**建议下一个算子：GEMV（y = A·x，瘦矩阵，fp16 为主）。**

理由（仅建议，v0.5 未开始）：
1. **瓶颈形态互补**：RoPE 是发射受限（DRAM 23%）、RMSNorm/Softmax 是
   带宽受限（L2 敏感）；GEMV 是纯 DRAM 带宽受限（算术强度 ~1 FLOP/B，
   2080 Ti 616 GB/s 规格峰值下存在真实的 >50% 提升空间），
   能为 evaluator 提供**大提升量级**的 KEEP 判例（v0.2–v0.4 的 KEEP 最大
   量级是 Softmax 的 1.67×，GEMV baseline→tuned 有望到 3–10× 量级）。
2. **复用面最大**：bench pool 契约（validate 门控模式直接迁移）、
   v2.3 双轨 + filter_sensitive、paired 判定链、NCU 诊断链全部复用，
   再次检验"evaluator 零算子特化"。
3. **已知陷阱可预置**：GEMV 的 tile/split-K 扫描天然是 MLP/occupancy
   扫描（与 RoPE 的 pairs/thread 扫描同构），v0.4 学到的
   "甜区 + 波坍缩"判读框架直接可用。
4. **范围控制**：只做 y = A·x（含 fp16 累加 fp32 选项的正确性门），
   不做 GEMM（大 M）、不做 attention、不做量化 —— 保持"一个算子闭环"
   的 lab 纪律。

备选（若 GEMV 数据形态与 lab 目标偏离）：Softmax 的 fp16 累加精度
专项（v0.3 遗留的"正确性-性能"权衡），或 v2.3 的**跨 run 稳定性协议**
（把"±5% 噪声带内的对"形式化为 policy：连续 N 次 paired 同向才可
KEEP/REJECT —— 这是 RMSNorm 方向翻转事件的长期解）。

---

*本报告所有数字均来自入库的真实执行记录（benchmarks/、experiments/、
profiles/、docs/evaluator_v2_3.md）；无估计值、无跨 run 绝对时间对比。*
