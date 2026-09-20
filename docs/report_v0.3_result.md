# CUDALab v0.3 Result

- **日期**: 2026-09-19（UTC+8；全部时间戳为 ISO 8601 带时区）
- **分支**: `v0.3-softmax`（基线 `main` = v0.2.1 = `dfe9e9b`）；**未 merge 回 main，未开始 v0.4**
- **核心问题**: v0.2 的闭环（正确性 → 配对 bench → 统计 → 决策 → 剖析 → 实验史）
  能否**原样迁移**到第二个算子？
- **TL;DR**: 能。evaluator 核心通用化后，RMSNorm 回归硬门 PASS、Softmax 上
  完成 4 个 profiler 驱动的自主优化实验（1 KEEP / 2 NEUTRAL / 1 REJECT），
  得到 (128,4096) fp16 当前 acceptance policy 下的 incumbent `softmax_vec4`
  （streaming 1.69× vs baseline，KEEP，稳健复现；v0.3.1 措辞更正：不是
  "结构最优"——后续候选尚未达到 ≥5% 的替换门槛）；36 格全矩阵**无
  策略层面唯一胜出者**（NO_UNIQUE_WINNER，v0.2.1 语义如实记录；v0.3.1:
  其中 21 格 winner 统计显著快于 runner-up 但改进 <5%，详见 Q4）；全部数字可逐项复现（独立审计 PASS WITH CAVEATS）。最重要的
  caveat 是 **(128,4096) fp16 hot 模式的机器态敏感性**——详见 Q6。

---

## 0. 范围与方法学

| 项 | 取值 |
|---|---|
| 算子 | row-wise Softmax：`y = exp(x − rowmax) / Σexp(x − rowmax)`，**FP32 内部计算，输出原 dtype**；ref `torch.softmax(x.float(), dim=-1).to(x.dtype)` |
| dtype | FP16 主 + FP32；**禁 BF16**；连续输入 |
| 硬件 | GPU 0：RTX 2080 Ti（Turing，**sm_75**，30 SM，L2 5.5 MB，wave 容量 240×256 线程）；CUDA 11.8；torch 2.4.1+cu118；ncu 2022.3.0 |
| 工作负载 | 9 形状 {(1,128),(1,1024),(1,4096),(32,1024),(128,1024),(128,4096),(1024,1024),(1024,4096),(128,8192)} × {fp16, fp32} × {hot, streaming} = **36 格**；主目标 (128,4096) fp16 |
| harness | `paired-streaming-v2.2`（`cudalab/evaluator/bench.py`）：9 rounds × 100 iters × 32-launch 连发流，A/B 交替，时间基准 burn（≥150 launches 且 ≥300ms），逐样本 spike guard（1.5×），跨块一致性 guard（1.15×）；**per-launch µs**（持续流水下的单次 launch 延迟，见 Q6 §4） |
| 统计/决策 | `decide_v2`（与 v0.2.1 逐字节相同）：KEEP 需 median≥1.05 且 ≥70% rounds 更快且 CI95 下界>1.00；REJECT 需 median≤0.95 且 ≤30% rounds 更快且 CI95 上界<1.00；<5 有效轮 UNSTABLE；bootstrap 10000 次、固定种子 20260919 |
| 带宽口径 | `algorithmic_bytes = M×H×element_size×2`（读入+写出，逻辑流量，非实测 DRAM；实测以 NCU 为准） |
| NCU | `--cache-control all/none`（v0.2.1 修正后语义：all=flush/reset，none=no-flush）+ `--clock-control base` + `--launch-skip 2 --launch-count 4`；raw 存 `profiles/softmax/raw/` |
| PyTorch 参照 | 仅记录 implementation context，**不参与任何决策**（非公平 fused-kernel 对比） |

**evaluator 通用化方式**（不做大规模重写）：`cudalab/evaluator/`（operator-
agnostic 核心）+ `cudalab/operators/{rmsnorm,softmax}.py`（算子 adapter：
matrix/pool/bytes/正确性/负例/NCU 参数）；`stats.py`/`decision.py` 与
v0.2.1 **逐字节相同**；统一 CLI `scripts/cudalab.py test|benchmark|profile|
optimize {rmsnorm,softmax}`；v0.1/v0.2.1 全部路径与产物冻结未动。

---

## Q1. baseline 的瓶颈是什么？

NCU 剖析（`profiles/softmax/baseline_M128_H4096_ncu_comparison.json`，
(128,4096) fp16，ncu 2022.3.0，--clock-control base）：

| 指标 | cc=all（flushed） | cc=none（no-flush） | 含义 |
|---|---|---|---|
| kernel duration | 11.976 µs | 11.728 µs | — |
| `long_scoreboard`（stall 占比） | **60.6%**（9.99 cyc/issue） | 47.2% | **主导停顿 = 全局内存延迟** |
| `dram__throughput` | **17.97%** | 22.51% | 远未到 DRAM 峰值 |
| L2 read hit | 2.56% | 2.33% | flushed 下 L2 几乎不命中（预期） |
| L1 hit | 62.5% | 62.5% | L1 有效但未消除往返 |
| achieved occupancy | 46.4% | 46.3% | 中等 |
| registers/thread | 16 | 16 | 不是瓶颈 |

**结论**：baseline（1 块/行、标量访存、3 遍 = 3 读 1 写 = **4× 算法字节数**
内部流量）的瓶颈**不是 DRAM 带宽**（<23%），而是**内存指令数 / 小粒度
事务的往返延迟**：fp16 每线程每次只取 2B，每个元素 3 次独立小粒度全局读 +
1 次小粒度写。优化空间因此排序为：① 加宽访存事务（降指令数）> ② 减遍数
（降流量）> ③ 加并发（占满 SM）。—— 这个排序被后续 4 个实验逐一验证。

## Q2. 最大的性能收益是什么？为什么？

**SFM-0001 `softmax_vec4`（4 元素向量化：fp16 8B = `__half2`×2 / fp32 16B =
`float4`）→ KEEP，新 incumbent**。这是全 v0.3 唯一显著胜出（paired，
(128,4096) fp16，9/9 valid）：

| mode | parent (baseline) | candidate (vec4) | speedup median | 95% CI | faster | 决策 |
|---|---|---|---|---|---|---|
| streaming（primary） | 9.929 µs | 5.936 µs | **1.6772** | [1.6568, 1.6794] | 9/9 | **KEEP** |
| hot | 6.548 µs | 5.068 µs | 1.2916 | [1.2700, 1.2946] | 9/9 | KEEP（记录值，见下） |

逻辑带宽 211.2 → 353.3 GB/s（streaming）、320.3 → 413.8 GB/s（hot）。
NCU 复核：duration 11.976 → 6.576 µs（cc=all），`long_scoreboard` 60.6% →
51.0%，L1 hit 62.5% → 75.0%，DRAM 17.97% → 30.86%（仍非带宽受限）。

**为什么赢**：访存宽度 ×4 把每线程全局内存指令数 ÷4、sector 利用率拉满；
内部流量仍是 4× 算法（**不是**靠减流量赢的），但单位流量有效带宽显著
提升。假设（"更宽事务 → 指令发射压力与往返次数下降"）成立且超出预期
（streaming 收益大于 hot）。

**复现性与如实披露（重要）**：
- **streaming KEEP 稳健复现**：final_reval（2026-09-19T21:41:19+08:00，
  同协议）1.6890 [1.6824, 1.6924] 9/9（parent 10.102 → candidate 5.980 µs）；
  矩阵口径（base 10.109 µs @20:53 vs inc 6.015 µs @21:42）≈ 1.68。三重一致。
- **hot 记录值 1.2916 存在机器态漂移**：final_reval 同协议复跑得
  **0.9865 [0.9318, 0.9960]、1/9 → NEUTRAL**（parent 6.705 / candidate
  6.797 µs）；vec4 hot 绝对时间在相隔约 90 秒的两组运行间漂移 ~45%
  （4.684 µs @21:42:46 inc 矩阵 vs 6.797 µs @21:41:19 final_reval），
  而 baseline hot 稳定在 ~6.7 µs。机制：(128,4096) fp16 的 ~2 MB 热工作集
  的 L2 驻留状态 + SM 时钟爬坡（运行前 sm_clock 1350 MHz → 运行中 1890
  MHz，`gpu_state_before/after` 有记录）对"已驻留 L2 的小 kernel"的相对
  放大。详见 SFM-0001.md §6 复核注记与 Q6 §3。
- **政策**：本实验的 KEEP 以 primary 判据 **streaming** 为准，成立且稳健；
  hot 数字保留为记录值，**不作为后续任何决策的依据**。

数值安全：vec4 正确性 72/72（max_abs 7.63e-06，与 baseline 同一 FP32
中间路径）；H%4≠0/未对齐回退共享标量核 6/6；negative 14/14+1 skip；
baseline 重构回归（标量核移入 `softmax_scalar.h`）72/72 逐位一致。

## Q3. 哪些假设失败了？学到了什么？

四个实验按 profiler 证据逐一推进，覆盖 4 个设计轴：
**宽度 → 流量 → 每线程 ILP → 块级并行**。前 3 轴之外的第 4 轴直接证伪，
4 个正交维度全部测完（v0.3 原文称"设计空间闭合"；v0.3.1 措辞更正：
测完 4 个维度 ≠ 设计空间穷尽，见下文与 Q4）。

| 实验 | 假设 | paired 结果（(128,4096) fp16，9/9 valid） | 决策 |
|---|---|---|---|
| SFM-0002 `softmax_online` | 3 读 1 写 → 2 读 1 写（online (m,l) 单遍 + 恒等 merge；先过 docs/softmax_algorithm.md + 5 个 CPU 恒等测试门禁） | hot 0.9775 [0.9496, 1.0004] 2/9；streaming 1.0413 [1.0377, 1.0436] 9/9 | **NEUTRAL** |
| SFM-0003 `softmax_vec4_ilp2` | vec4 基础上 2 路展开，每线程在飞 load 加倍隐藏延迟 | hot 1.0108 [1.0035, 1.0944] 8/9；streaming 0.9815 [0.9797, 0.9831] 0/9 | **NEUTRAL** |
| SFM-0004 `softmax_hsplit2` | occupancy 44% → 86%：H 对半分 2 块/行 + (m,l) 跨块合并（单 launch） | hot 0.6752 [0.6492, 0.7220] 0/9（5.056→7.481 µs）；streaming 0.7752 [0.7246, 0.7796] 0/9（5.979→7.706 µs） | **REJECT** ＋ v0.3.1 隔离（UNSAFE，见下） |

**SFM-0002（流量轴）——NEUTRAL 的教训**：减一遍读（4×→3× 算法流量）没有
带来显著收益，因为瓶颈是内存指令/事务延迟而非 DRAM 带宽（DRAM 仍只有
31–39%）。streaming 1.0413 是全部实验中**最接近翻转**的案例（median 距
1.05 KEEP 线差 0.0087，CI 全在 1.0 之上，9/9 轮更快）——按既定规则判
NEUTRAL 正确，规则未被临时调整。
**v0.3.1 语义澄清（此案例是典型）**：NEUTRAL **不是**"统计平局"——
online 对 vec4 在该指标下**统计上显著更快**（CI95 全在 1.0 之上，
9/9 轮更快，statistical_relation = FASTER），但 median 改进 4.13% 低于
5% 的实质替换门槛（material threshold）→ policy_decision = NEUTRAL →
incumbent（vec4）保留。项目语义区分两层：`statistical_relation`
（FASTER / SLOWER / UNRESOLVED，CI95 是否排除 1.0）与 `policy_decision`
（KEEP / REJECT / NEUTRAL / UNSTABLE，decide_v2，KEEP 额外要求
median ≥ 5% + 多数轮更快）。

**SFM-0003（每线程 ILP 轴）——NEUTRAL 的教训**：与 vec4 逐位一致（8/8 位
模式），但寄存器 19→28，NCU：`long_scoreboard` 51.0→45.4% 改善的同时
`wait` 17.3→15.7% 变化有限、barrier 6.4→8.2% 变差，duration 6.576→6.640
持平——每线程 ILP 不是这个工作负载的杠杆。

**SFM-0004（块级并行轴）——REJECT 的教训（最重要）**：occupancy 44.5% →
**85.7%**（cc=all，杠杆机械上完全达成），但 **barrier stall 5.6%（vec4
6.4%）→ 31.0–35.3%**（跨块 spin-wait 两阶段 merge 的代价），duration
6.576→7.688 µs 变慢，paired 两模式均 0/9 轮更快 → REJECT。
**结论：(128,4096) fp16 的 softmax 不是 occupancy-bound**——加并发换不来
收益，因为每块的内存延迟链没有被并发摊薄（跨块同步反而引入了新的等待）。

**安全补记（v0.3.1 合并 review，推翻 SFM-0004.md §2 的 liveness 论证）**：
SFM-0004 不仅性能 REJECT，后续 review 还发现其依赖未被 CUDA 调度模型保证
的跨 block 并发假设（线性 wave 派发 + 同 wave 共驻），存在
deadlock/liveness 风险；HsGlobal scratch 为进程级共享状态，多 CUDA
stream / 多 device 并发调用存在 race 风险。因此 `softmax_hsplit2` 被隔离为
**UNSAFE_HISTORICAL_EXPERIMENT / REJECTED / NOT_FOR_NORMAL_DISPATCH**：
移出正常 dispatch 列表（`ext.variants()` 正常列表 4 变体），CLI / engine
路径（test / benchmark / profile / optimize）一律拒绝该名称；
`softmax_hsplit2.cu` 与全部 SFM-0004 数据原样保留，仅保留显式
`ext.forward("softmax_hsplit2", ...)` 受控历史审计入口（详见
`experiments/softmax/SFM-0004.md` §6）。

**四个设计维度全部测完（v0.3 原文称"设计空间闭合"，v0.3.1 措辞更正）**：
宽度（KEEP）→ 流量（NEUTRAL）→ 每线程 ILP（NEUTRAL）→
块级并行（REJECT）。**v0.3.1 更正**："四个正交维度测完"≠"设计空间
穷尽"——split-K、其他 launch 结构、不同块组织等方向在 v0.3 范围外未测，
不能由 4 个维度外推为整体最优。**`softmax_vec4` 是当前 acceptance
policy 下的 incumbent；后续候选尚未达到 ≥5% 的替换门槛**（NEUTRAL 是
policy_decision，不是"统计平局"，更不是"结构最优"的结论）。
所有失败内核（online/ilp2/hsplit2）**永久保留在仓库**作参考实现与
实验证据（`experiments/softmax/SFM-000*.md` + result/pair JSON 同
commit 提交；hsplit2 于 v0.3.1 隔离为 UNSAFE_HISTORICAL_EXPERIMENT，
见 SFM-0004.md §6，数据完整保留）。

## Q4. 性能在 shape/dtype 矩阵上如何扩展？

incumbent `softmax_vec4` vs baseline 的 36 格全矩阵（每格 9/9 valid；
µs/launch，逻辑带宽 GB/s 见 Q0 口径；PyTorch 参照仅记录）：

| M | H | dtype | inc hot | inc streaming | base hot | base streaming | PyTorch ref |
|---|---|---|---|---|---|---|---|
| 1 | 128 | fp16 | 4.808 | 4.927 | 4.865 | 5.096 | 10.010 |
| 1 | 128 | fp32 | 4.734 | 4.869 | 4.815 | 5.082 | 9.876 |
| 1 | 1024 | fp16 | 4.736 | 4.927 | 4.752 | 5.049 | 10.219 |
| 1 | 1024 | fp32 | 4.689 | 4.888 | 4.766 | 5.443 | 10.105 |
| 1 | 4096 | fp16 | 4.705 | 4.923 | 5.908 | 5.899 | 10.077 |
| 1 | 4096 | fp32 | 4.736 | 4.888 | 4.861 | 5.068 | 9.178 |
| 32 | 1024 | fp16 | 4.736 | 4.928 | 4.864 | 5.248 | 9.640 |
| 32 | 1024 | fp32 | 4.696 | 4.961 | 4.863 | 5.086 | 9.487 |
| 128 | 1024 | fp16 | 4.735 | 4.931 | 4.928 | 5.056 | 9.445 |
| 128 | 1024 | fp32 | 4.736 | 4.932 | 4.861 | 5.049 | 9.390 |
| **128** | **4096** | **fp16** | **4.684** | **6.016** | 6.976 | 10.176 | 9.356 |
| 128 | 4096 | fp32 | 4.845 | 9.154 | 6.016 | 9.503 | 9.408 |
| 128 | 8192 | fp16 | 6.422 | 10.240 | 11.818 | 18.981 | 10.659 |
| 128 | 8192 | fp32 | 16.954 | 19.528 | 19.199 | 21.424 | 20.285 |
| 1024 | 1024 | fp16 | 7.552 | 10.013 | 9.920 | 14.240 | 9.897 |
| 1024 | 1024 | fp32 | 17.600 | 17.683 | 17.894 | 17.983 | 16.171 |
| 1024 | 4096 | fp16 | 34.625 | 34.560 | 49.920 | 48.978 | 50.246 |
| 1024 | 4096 | fp32 | 93.488 | 93.312 | 92.793 | 92.867 | 67.072 |

（数据：`benchmarks/softmax/{base,inc}_*.json` + 两个 matrix summary；
PyTorch 参照为 200 样本中位延迟。）

**读法**：
1. **launch 下界**：M≤128 且 H≤1024 的全部 12 格（含 hot/streaming）聚集在
   4.69–4.96 µs —— 这一区域的绝对延迟由 launch/延迟下界主导，变体间差异
   在噪声内（也是 best.json 里这些格全部 NO_UNIQUE_WINNER 的原因）。
2. **主目标**：(128,4096) fp16 streaming 10.176 → 6.016（≈1.69×，与 SFM-0001
   paired 1.6772/1.6890 一致）；hot 4.684 已落在 launch 下界上。
3. **H 扩展**：(128,8192) fp16 streaming 18.981 → 10.240（≈1.85×）——
   行越长收益越大，因为每块的内存延迟链更长、向量化摊薄比例更高。
4. **M 扩展（逻辑吞吐；DRAM 饱和未建立，v0.3.1 更正）**：(1024,4096)
   fp16 34.6 µs ≈ **485 GB/s 逻辑吞吐**（16.78 MB/34.56 µs；2080 Ti 规格
   峰值为 **616 GB/s**（v0.3.1 更正，此前误写 550）→ 485/616 ≈
   **78.7%**）。`algorithmic_bw_gbps` 是最小有用算法 IO/latency 的逻辑
   流量指标，**不是**实测 DRAM 吞吐——不能据此断言"DRAM 带宽饱和 /
   进入带宽受限区"。工作集（8.4 MB/方向）超出 5.5 MB L2，该形状是高
   逻辑吞吐 / 可能更带宽敏感的方向，**是否真正达到 DRAM 饱和需要对应
   形状的 NCU 验证**（v0.3 未对 (1024,4096) 做 NCU，无 DRAM% 证据）；
   inc 与 base 差距收敛到 1.33–1.44×（与带宽敏感方向一致，但属旁证）。
   (1024,4096) fp32 两变体均 ~93 µs（字节数约为 fp16 的 2× 而时长约
   翻倍，与带宽敏感形状一致；同样无直接 DRAM% 证据，v0.3.1：不称
   "带宽墙"）。
5. **fp32 逐格更慢**（2× 字节）但相对排序与 fp16 一致；(128,4096) fp32
   streaming 9.503 → 9.154（仅 1.04×，矩阵口径）——fp32 主目标收益显著
   小于 fp16（与 RMSNorm v0.2 的 fp32 弱点观察同构）。

**`experiments/softmax/best.json`（classify_cell，v0.2.1 语义；v0.3.1
语义澄清）**：全部 36 格 decision=NEUTRAL → **NO_UNIQUE_WINNER**
（ratio 0.9972–1.0479，全部 <1.05 KEEP 线）；17 格 **INCUMBENT** 标签
（incumbent `softmax_vec4` 位于该格 top-2 且被 policy 保留）/ 19 格
NO_UNIQUE_WINNER 标签。**v0.3.1 澄清**：NO_UNIQUE_WINNER 是**策略层面**
的"无唯一胜出者"（顶部变体对 runner-up 未达到 median ≥ 5% + CI95>1.0 +
多数轮更快的 KEEP 线），**不是**"顶部两变体统计平局"的断言——36 格中
21 格 winner-vs-runner-up CI95 全在 1.0 之上（winner 统计显著更快，
statistical_relation=FASTER，但改进低于 5% 实质阈值 →
policy_decision=NEUTRAL）；其余 15 格 CI95 跨 1.0（统计不可区分）。
主目标格（(128,4096) fp16 streaming）winner=online（5.760）vs
runner-up=vec4（6.015），ratio 1.0443 [1.0393, 1.0453] 9/9 → **统计上
显著更快，但改进 4.43% < 5%** → NEUTRAL → INCUMBENT 标签（incumbent
保留，非"统计平局"）。**best.json 不声称任何 per-shape 路由证据**
（见"未做"）。注意：INCUMBENT 标签 ≠ "对 baseline 显著胜出"，vec4 对
baseline 的显著优势只存在于 paired 记录（SFM-0001/final_reval，
streaming 1.69 KEEP）。

## Q5. v0.2 的优化循环迁移了吗？两个算子的策略差异是什么？

**迁移了，且差异本身就是验证**：同一套循环、同一套规则，在两个不同起点
的算子上给出了不同但各自正确的形状——

| | RMSNorm（v0.2，历史） | Softmax（v0.3，本分支） |
|---|---|---|
| baseline 起点 | 已经历 v0.1 优化迭代（v4_vec_reg 是 v0.1 incumbent；v1/v2/v3/v4 同为 2 遍结构的不同宽度/寄存器配置） | 朴素标量 3 遍（1 块/行，2B/线程） |
| 第一刀 | 宽度/寄存器重排：v4 vs v1 streaming 1.011 NEUTRAL / hot 0.9327（REJECT v1）；vs baseline 1.740× hot / 1.896× streaming | 宽度：vec4 vs baseline **streaming 1.6772 KEEP**（hot 记录 1.2916，机器态敏感） |
| 后续空间 | 基本平坦：v1/v2/v4 两两 NEUTRAL，主形状 fp16 **NO_UNIQUE_WINNER** | 3 个后续轴：流量 NEUTRAL / ILP NEUTRAL / occupancy REJECT |
| 结局 | 无统计唯一胜出者（v4 保留 incumbent，非统计确认唯一最佳） | 1 个 KEEP（width 轴陡峭）后 3 连非 KEEP；vec4 = 当前 acceptance policy 下的 incumbent（四个维度测完 ≠ 设计空间穷尽，v0.3.1 措辞更正） |

**解读**：
- RMSNorm 进入 v0.2 循环时已接近局部最优 → 循环**正确地**得出"top-2 平坦、
  无统计唯一胜出者"，没有为产出故事而放宽规则；
- Softmax 从朴素标量起点进入 → 循环第一刀就找到陡峭的 width 轴（1.69×
  streaming），随后 3 个轴正确地平/负收场；
- 两个算子、两条完全不同的收益曲线、**同一套 decide_v2 与 UNSTABLE/
  NO_UNIQUE_WINNER 语义**——这正是"闭环可迁移"的最直接证据。

RMSNorm 回归硬门（v2.2 协议复跑，(128,4096) fp16，v4 vs v1）：
hot 1.0144 [1.0042, 1.0237] 9/9 NEUTRAL；streaming 0.9419 [0.9403, 0.9480]
0/9 REJECT；正确性 76/76 × 2；negative 29/30+1 skip —— 与 v0.2 结论
**机制一致**（v4≈v1 ≫ baseline），点估计漂移归因机器态（见 Q6 §3），
硬门 **PASS**。

## Q6. Evaluator Generalization Verdict

**判定：PASS —— v0.2 评估闭环成功迁移到第二个算子；v0.3 全部性能结论
在同一套未改写的决策规则下得出，且经独立审计逐项复现。判定附带以下
已声明的偏离与 caveat（均如实记录在案，不隐藏）：**

### 1. 结构迁移（审计确认）

独立只读审计（`docs/benchmark_audit_v0.3.md`）确认：
- `stats.py`/`decision.py` 与 v0.2.1 **逐字节相同**；`bench.py`（v2.2）相对
  v0.2.1 `bench_v2.py` 的差异**全部落在已声明变更内**，未发现未声明逻辑
  改动；
- 两算子共用同一引擎与轮结构（9 rounds × 100 iters × 32 batch、同预分配
  池、A/B 交替、同 guard），算子专属仅 matrix/pool/regex/bytes；bytes
  口径正确（softmax M·H·es·2；rmsnorm (M·H+H+M·H)·es）；
- SFM-0001…0004 全部数字与 pair JSON 一致，重跑 `decide_v2` **精确复现**
  记录决策；正确性 5 变体 72/72 且容差逐变体相同（fp16 atol 2e-3/rtol
  5e-3/row_sum 5e-3；fp32 1e-5/1e-4）；无按变体放宽、无 best-shape-only
  报告、失败实验全部保留（kernel+MD+JSON 同 commit）；
- NCU 方法学与 v0.2.1 修正后语义一致（cc=all/none 语义标注正确、metrics
  经 --query-metrics 验证、raw 在档）。

### 2. 已声明偏离：v2.2 移除 round 内 nvidia-smi 采样（必须 flag）

v0.2.1 的 DVFS guard（每测量区间前后+之后 3 次 nvidia-smi 时钟采样，
>5% 失配拒轮）在 v0.3 被**移除**，替换为：时间基准 burn（≥150 launches
且 ≥300ms 把 SM 拉到持续负载时钟）+ 逐样本 spike guard（>1.5× 运行中
干净中位数 1.5× 即剔除）+ 跨块一致性 guard（block 中位数 > 运行中
median×1.15 → INVALID_CROSSBLOCK，重试 ≤3）。
**偏离依据（基于证据的机器态适配，非规则放宽）**：nvidia-smi 调用引入
~40ms 进程级空闲间隙，实测会把 GPU 推入性能退化态、污染被采样的测量
区间本身（采样即扰动）；v2.2 的全部 36 格 × 5 变体矩阵与 12 组 paired
均 9/9 valid、0 invalid round。该偏离在
`docs/evaluator_hardening_v0.3.md` 全文记录，**并在此 verdict 中明示：
v0.3 所有数字均在 v2.2 协议下产生，与 v0.2.1（v2/v2.1）数字不构成同协议
对比**。残余风险（<15% 整体漂移盲区、前 3 块不检查、n=9 功效）已在该
文档 §6 列出。

### 3. 机器态漂移（最重要的 caveat）

- **Softmax hot 模式跨运行漂移**：vec4 vs baseline hot 两次 paired 运行
  1.2916（2026-09-19T21:03:32+08:00，SFM-0001 记录）→ 0.9865 NEUTRAL
  （21:41:19，final_reval）；vec4 hot 绝对时间 4.684 µs（21:42:46 inc
  矩阵）vs 6.797 µs（21:41:19 final_reval），相隔约 90 秒漂移 ~45%，
  baseline hot 稳定 ~6.7 µs；机制为 ~2 MB 热工作集的 L2 驻留 + SM 时钟
  爬坡（1350 → 1890 MHz，`gpu_state_before/after` 在档）。**streaming
  模式稳定**：1.6772 → 1.6890（paired）与 10.109/6.015 ≈ 1.68（矩阵）
  一致。处置：SFM-0001 KEEP 以 streaming（primary）为准，hot 记录值不作
  决策依据（SFM-0001.md §6 已披露）。
- **RMSNorm 回归点估计漂移**：hot 0.9327 REJECT（v0.2，v2 协议）→
  0.9969 NEUTRAL（v2.2，20:37）→ 1.0144 NEUTRAL（v2.2，21:41）；
  streaming 1.011 NEUTRAL（v0.2）→ 0.9592 NEUTRAL（20:37）→ 0.9419
  REJECT（21:41）。点估计在边界附近摆动、机制不变（v4≈v1 ≫ baseline），
  判定为机器态而非 evaluator 缺陷——**跨运行绝对时间不可比，仅运行内
  配对有决策意义**，这一政策适用于本分支全部结论。
- **边界决策脆弱性**（审计 caveat #2）：n=9 + 固定 1.05/0.95 阈值 + <15%
  guard 盲区，使近界决策（SFM-0002 streaming 1.0413）对单轮质量敏感；
  当前规则下无一翻转，但两方向功效都不足——已在 hardening 文档 §6 列为
  残余风险，不在此处粉饰。

### 4. 计时语义（防止误读）

- **per-launch µs**：bench 测的是持续 32-launch 流水流（负载时钟 ~1905
  MHz）中的单次 launch 延迟；**NCU dur** 在 base 时钟（~1350–1530 MHz）+
  flushed 缓存 + 串行化下测量——因此 vec4 的 NCU dur 6.576 µs **大于**
  bench 矩阵 hot 4.684 µs 是自洽的（时钟 + L2 状态差异），不是 bug。
- **streaming 逻辑吞吐可超过 DRAM 规格峰值（指标性质，非实测 DRAM
  带宽，v0.3.1 澄清）**：16-buffer 轮转使 buffer 在 5.5 MB L2 中部分
  驻留，`algorithmic_bw_gbps` 计入的逻辑流量因此可接近甚至超过规格
  峰值（616 GB/s，v0.3.1 更正）——这是逻辑流量指标的性质，**不是**
  DRAM 饱和的证据；实测 DRAM 占比以 NCU 为准（仅主目标 (128,4096)
  有实测：30.9–36.2%；(1024,4096) 无 NCU DRAM 证据）。

### 5. RMSNorm 回归硬门

**PASS**（eae07bb 首验 + 85faeca 复验）：CPU 测试全过；negative 29/30+1
skip（单 GPU 安全跳过）；正确性 v4_vec_reg / baseline 76/76 × 2；
(128,4096) fp16 paired 机制兼容 v0.2（Q5 表）。v0.1/v0.2.1 路径与产物
冻结未动。

### 6. 独立审计结论

`docs/benchmark_audit_v0.3.md`：**PASS WITH CAVEATS**（数据逐项可复现、
决策与规则一致、重构忠实、未见完整性问题）。3 项 caveat 处置：
#1（SFM-0001 hot 未复现、无 MD 讨论）→ SFM-0001.md §6 复核注记 + 本节
§3 披露；#2（边界决策脆弱）→ 如实保留为残余风险；#3（机器态漂移已声明
+ SFM-0004.md 日期笔误 2026-07-19）→ 笔误已修正。

### 7. 完整性自查（无作弊项）

固定容差（无按变体放宽）；36 格全矩阵测量保存（无 best-shape-only）；
失败实验永久保留（SFM-0002/0003/0004 内核与记录在库）；负例全部在
kernel **launch 前**被拒（`TORCH_CHECK`）；无 per-variant 特殊路径进入
计时（回退路径与 baseline 共享同一份标量核，审计确认无偏置）；
incumbent/best 由 `evaluator.experiment.classify_cell` 产生，无手工指定；
NO_UNIQUE_WINNER/REJECT 均如实记录（未为"成功故事"翻转任何判定）。

---

## 交付物与数据索引

- **分支**: `v0.3-softmax`（基线 dfe9e9b = v0.2.1 = main）；**未 merge、未 tag、未开始 v0.4**
- **evaluator 核心**: `cudalab/evaluator/{bench,stats,decision,profiler,negative,experiment,gpu,correctness}.py` + `cudalab/operators/{base,rmsnorm,softmax}.py`
- **kernels**: `kernels/softmax/{softmax_common.h,softmax_scalar.h,softmax_baseline.cu,softmax_vec4.cu（incumbent）,softmax_online.cu,softmax_vec4_ilp2.cu,softmax_hsplit2.cu（REJECT ＋ v0.3.1 隔离 UNSAFE_HISTORICAL_EXPERIMENT，历史证据保留）,bindings.cpp}`
- **实验记录**: `experiments/softmax/SFM-0001.md`…`SFM-0004.md` + 各 `SFM-000*/`（result + pair JSON）+ `correctness/v0.3/`（5 变体 × 72/72 + invalid_inputs）+ `final_reval/`（最终完整重验）+ `best.json`
- **基准**: `benchmarks/softmax/`（base_*/inc_*/full5_* 36 格 + pair_*）；`benchmarks/v0.3_regression/`（RMSNorm 回归硬门，v2/v2.1/v2.2 多轮）
- **剖析**: `profiles/softmax/`（baseline vs 4 候选 NCU 对比 + per-variant 双 cache-control + final re-verify + raw/）
- **文档**: `docs/softmax_algorithm.md`（online (m,l) 推导 + CPU 门禁）、`docs/evaluator_hardening_v0.3.md`（v2.2 + 偏离 + 残余风险）、`docs/benchmark_audit_v0.3.md`（独立审计 + 复核方响应）、README/STATUS/PROJECT_PLAN 已同步
- **CLI**: `scripts/cudalab.py {test,benchmark,profile,pytorch,optimize} {rmsnorm,softmax}`（--help 冒烟通过；GPU 子命令未在本次会话重跑以免争用）

## 未做（按 v0.3 范围约定）

- **dispatcher**：默认不做——36 格全部 NO_UNIQUE_WINNER，没有 paired 确认
  的 per-shape 路由证据；softmax 的 dispatch 留待有证据时再建。
- **不 merge `v0.3-softmax` 回 main、不打 tag、不开始 v0.4**（等外部
  reviewer）。
- 不复制任何成熟 kernel 源码（4 个候选全部从零编写）；不做 BF16；
  不做 M>8192（非范围；唯一的块级并行变体 hsplit2 已于 v0.3.1 隔离，
  不进入正常 dispatch）。

## 附录 A. 复现

```bash
cd /root/code/cuda && source tools/env.sh
PY=/root/miniconda3/envs/pytorch/bin/python
$PY -c "from cudalab.build import build; build('softmax')"          # softmax 扩展
$PY scripts/cudalab.py test softmax --variant softmax_vec4          # 单变体 × 72 例（v0.3.1: hsplit2 已隔离，正常列表 4 变体）
$PY scripts/cudalab.py benchmark pair softmax \
    --parent softmax_baseline --candidate softmax_vec4 \
    --M 128 --H 4096 --dtype float16 --mode streaming --rounds 9    # SFM-0001 主 paired
$PY scripts/cudalab.py benchmark full softmax                       # 36 格矩阵
$PY scripts/cudalab.py profile softmax                              # NCU 双 cache-control
$PY tests/test_softmax_cpu.py                                       # 20/20 CPU（含 online merge 恒等）
```

## 附录 B. 术语

- **KEEP / REJECT / NEUTRAL / UNSTABLE**：decide_v2 四态（Q0 阈值）。
  **v0.3.1 语义分层**：这是 `policy_decision`（是否达到替换/保留的
  实质门槛），与 `statistical_relation`（FASTER / SLOWER / UNRESOLVED，
  只看 CI95 是否排除 1.0）是两层独立结论："CI95 明确排除 1.0 但
  median < 1.05"= 统计显著更快但低于实质门槛 → policy NEUTRAL，
  **不是**"统计平局"。
- **NO_UNIQUE_WINNER（classify_cell）**：某格内 winner 与 runner-up 的
  round-level paired 比值未达到 KEEP 线（decision=NEUTRAL/REJECT，或
  <5 有效轮 → UNSTABLE；v0.2.1 修正）。**v0.3.1 澄清**：这是策略层面
  的"无唯一胜出者"，**不是**"无统计显著差异"的断言——v0.3 的 36 格中
  21 格 winner-vs-runner-up CI95 全在 1.0 之上（统计显著更快但 <5%，
  见 Q4）；其余 15 格 CI95 跨 1.0（统计不可区分）。也**不是** "没有
  变体快于 baseline"。
- **INCUMBENT 标签（best.json）**：实验链 incumbent（`softmax_vec4`）位于
  该格 top-2 且被 acceptance policy 保留（v0.3.1：不是"与顶部变体统计
  平局"的断言，见上条）；不暗示对 baseline 的显著优势。
- **hot / streaming**：固定预分配 buffer（L2 热）/ 16 组 buffer 轮转
  （工作集是否 > L2 取决于 shape，以记录中的 `working_set_gt_l2` 字段
  为准；主目标 (128,4096) fp16 streaming 工作集 33554432 B（≈ 33.5 MB）
  > 5.5 MB L2 → 每次 launch 面对近似冷 L2；同 shape hot 为 2 MiB，
  < L2）。
