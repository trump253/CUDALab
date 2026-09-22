# CUDALab v0.6 Result — INT8 Weight-Only GEMV 优化

**分支**: `v0.6-qgemv`（基线 `main` = v0.5.1 = d635903, tag `v0.5.1`）
**日期**: 2026-09-21（UTC+08:00）
**硬件/环境**: NVIDIA RTX 2080 Ti × 2（**固定 GPU 0**；Turing, sm_75, 30 SM,
L2 5.5 MB, 规格峰值 DRAM 带宽 616 GB/s；负载下 boost 实测 1815–1920 MHz /
base 1350 MHz），
CUDA 11.8，PyTorch 2.4.1+cu118，NCU 2022.3。
**分支状态**: 未 merge 回 main（用户指定：「不要 merge v0.6 到 main」）。
**主问题**（用户指定）: 权重从 FP16 2B → INT8 1B 后，是否能把 v0.5 已接近
DRAM ceiling 的 GEMV 继续加速。

---

## 1. Status

**v0.6 完成（PASS 结局）。QGEMV 主目标 (N=4096, K=4096) fp16 activation：
baseline 84.96 µs（API-path, ~198 GB/s, 32% 峰值）→ **政策 incumbent
`qgemv_vec16_row`**（QGEMV-0001 唯一 KEEP, 2.6614×）31.81 µs; best-observed
候选 `qgemv_warp_vec16` 31.51 µs（533.4 GB/s 逻辑, 86.6% DRAM 峰值,
比 incumbent 统计快 ~1% 但 policy NEUTRAL, 见 §6）,
fresh revalidation paired 2.6937×（9/9 轮, CI95 [2.688, 2.700]）。
对比 v0.5 FP16 incumbent `gemv_vec4_row` 本 session 复测 59.93 µs：
INT8 1.90× 加速（best-observed 口径; 理论 2.0× 的 95%）。**

- 4 个完整证据链实验（QGEMV-0001..0004）：vec16_row **KEEP 2.6614×**、
  vec16_scale **NEUTRAL 0.9901×**（统计 SLOWER）、warp_vec16 **NEUTRAL
  1.0107×**（统计 FASTER）、warp_vec16_ilp4 **NEUTRAL 0.9984×**（统计
  SLOWER）。失败/中性实验全部保留为证据（§5）。
- 正确性 **50/50 × 5 变体**（固定算术误差界 + 有限性门, 非 per-variant 调参;
  第 (a) 层: kernel vs 量化参考）+ 负例 **29/29 × 5 变体**（含 3 个向量化
  对齐回退**逐位一致**回归门 + 3 个 control）。
- 第 (b) 层量化保真度（**只报告, 不判定**）: 25 用例（5 shape × 5 mode），
  cosine ≥ 0.999897（最坏 mixed_sign）; 每元素反量化误差 ≤ 0.5·量化步长
  （max_dequant_step_err = 0.500002, 即 round-half-to-even 的理论上限）。
- 三种计时口径（API-path / native kernel-loop w200+w5000 / NCU cc=all+cc=none
  @base clock）**分开记录、不混用**; best-observed 候选三口径 31.51 / 32.18–32.10
  / 36.16–35.88 µs — API 与 native 差 <2.5%, NCU 高 14.8% = observed
  NCU profiling perturbation（观测差 ~4.4–4.7 µs, 不声称跨 workload 固定
  常数）+ base clock 锁定贡献 ~0.8%（v0.5 clkbase/clknone 受控对,
  DRAM-bound kernel）, **无口径冲突**（§7）。
- 全 shape matrix（5 形状 × {hot, streaming}）: 见 §8。
- 旧算子（gemv/rmsnorm/softmax/rope）+ qgemv 回归 smoke **全部 PASS**
  （gemv 100/100×4+24/24×4; rmsnorm 76/76×5+29/30×5; softmax 72/72×4+
  14/15×4; rope 384/384×5+36/37×5; qgemv 50/50×5+29/29×5; 每算子 1 个
  预存环境 skip 跨变体一致）→ `experiments/regression/v0.6/`; quarantine
  审计（`gemv_splitk4`、`softmax_hsplit2` 未出现在任何正常变体列表;
  qgemv quarantine 集为空）见 §9.2。
- CPU 测试基线: test_evaluator_v23_cpu 37/37（merge-review 轮 +1:
  analyze_shape_winners→classify_cell 委托）、test_evaluator_cpu 18/18、
  test_softmax_cpu 20/20、test_dispatch 6/6 —— 全绿。
- 独立 review（CUDA correctness + benchmark methodology）: 见 §10。

**停止条件核对**（§9 清单, 全部满足）: v0.5.1 已 release 且 main 未动 ✓;
量化器 ✓; baseline（正确性 + 负例 + API/native 计时 + NCU）✓; kernel 正确性
（50/50 × 5）✓; 保真度报告 ✓; 负例（29/29 × 5）✓; ≥4 证据驱动实验 ✓;
主目标 best-observed 候选 fresh revalidation（9 轮 paired, streaming）✓; 三口径 ✓
（冲突调查: 无冲突, §7）; 5 形状矩阵 ✓; FP16 对比 ✓; 旧算子回归 ✓;
README + STATUS + 本报告 ✓; 独立 review ✓; 推送 v0.6-qgemv ✓（未 merge）。

---

## 2. 算子定义与范围

**问题**（用户指定）: `y = W_dequant @ x`，其中
`W_q: int8 [N, K]`（每行对称量化, zero_point=0）、`scale: fp32 [N]`、
`x: fp16 [K]`、`y: fp16 [N]`；`W_dequant[n, k] = scale[n] · W_q[n, k]`；
**FP32 累加**。量化在计时区外完成（用户 §1）。

**形状**（(N, K)）: (1024, 4096), (4096, 1024), **(4096, 4096) 主目标**,
(11008, 4096), (4096, 11008)。主路径 FP16 activation。

**量化**（用户 §1, 逐行对称）:
`scale[n] = max(|W[n, :]|) / 127`；`q = clamp(round(W/scale), -127, 127)`
（round-half-to-even; scale=0 行 q≡0, 安全）。

**显式排除（用户指定）**: INT4 / group-wise / GPTQ/AWQ / activation 量化 /
Tensor Core GEMM。PyTorch 无 INT8 weight-only GEMV op —— `torch.mv(反量化
fp16 W, x)` 仅作 framework 上下文, **不是候选、不是决策基准**（用户 §6:
kernel 决策只在 QGEMV 候选之间）。

**逻辑 IO**（用户 §6, 一次 GEMV 的字节数）:
`N·K·1 (W_q int8) + K·2 (x fp16) + N·4 (scale) + N·2 (y fp16)`。
主目标 = 16,777,216 + 8,192 + 16,384 + 8,192 = **16,809,984 B ≈ 16.81 MB**;
616 GB/s 下算法下界 **27.29 µs**。对比 FP16 GEMV 逻辑 IO 33,570,816 B
（下界 54.50 µs）—— 比值 1.997 ≈ 2.0×。

**对齐契约**: 所有 16B 向量 load 变体要求 `W_q` 基址 16B ∧ `x` 基址 16B ∧
K%16==0（`qgemv_vec_contract_ok`）; 不满足时回退到**单一来源**的标量 kernel
`qgemv_scalar_kernel`（与 `qgemv_baseline` 逐位一致）。负例套件每变体 3 个
回退回归用例钉死该行为。

---

## 3. 两层正确性（用户 §2）

**(a) kernel 正确性**（判定门, 不放宽）: y 必须落在**固定算术误差界**内 ——
`tol = TOL_K · (3K·2^-24·S_n + 0.5·ulp16(|y|) + 0.5·ulp16(|exact|))`,
`TOL_K = 2`, `S_n = Σ_k |W_q[n,k]|·scale[n]·|x[k]|`（**两因子均取绝对值** ——
本轮发现并修复的 bug: 早期版本未对 x 取 abs, 负 x 主导行 S_n<0 导致容差塌缩,
max_arith_ratio 爆到 3621; 修复后 50/50, max_arith_ratio 0.2499）。
参考 = `((W_q.float()·scale[:,None]) @ x.float())` cast fp16（**量化参考**,
不是 FP16 原值 —— 量化误差与 kernel bug 严格分离）。外加 y 与 ref 有限性门。

**(b) 量化保真度**（只报告, 不判定）: y_quant vs 原始 FP16 `W @ x` 的
max_abs / max_rel / RMSE / cosine, 5 shape × 5 mode（normal/tiny/mixed_sign/
large/zeros）共 25 用例 → `quantization_fidelity.json`。

| mode | max_abs | max_rel* | rmse | cosine (min) | 反量化误差 (max, 步长单位) |
|---|---|---|---|---|---|
| normal | 3.95 | 20.55 | 0.958 | 0.9999594 | 0.500002 |
| tiny | 3.6e-8 | 3.6e-5 | 9.7e-9 | 0.9999586 | 0.500002 |
| mixed_sign | 3.67 | 295.05 | 0.972 | **0.9998970** | 0.500002 |
| large (scale=10) | 369.19 | 14.99 | 96.47 | 0.9999588 | 0.500002 |
| zero_scale | 0 | — | 0 | — (零向量) | 0 |

\* max_rel 在 y 元素近零处被放大（分母趋零）, 无实际意义; 有意义的是
cosine 与 rmse。主目标 4096×4096 normal: max_abs 2.57, rmse 0.551,
cosine 0.9999617, 单元素反量化误差 ≤ 0.0212（= 0.5 × 最大量化步长
0.0425 = 0.5·max|W|/127）, y 向量范数 4026.48 → 4027.51（+0.026%）。
**结论: 保真度由量化步长本身决定, 与 kernel 实现无关; kernel 各变体对
同一量化输入输出逐位可比（正确性层 (a) 全过）。**

---

## 4. Baseline（用户 §4: 先 baseline, 之后才能优化）

`qgemv_baseline`: 每行一个 block（256 线程）, 标量 int8 load + 2 级
warp/block 归约, FP32 累加, 逐元素乘 scale。

@4096×4096 fp16 streaming（paired, 9 轮; native w200/w5000; NCU @base clock）:

| 口径 | 值 | 说明 |
|---|---|---|
| API-path | **84.96 µs**（197.9 GB/s 逻辑） | paired bench 引擎, CUDA events |
| native w200 | 104.58 µs | DVFS 爬坡在 200 warmup 内未消（同 v0.5 现象） |
| native w5000 | **84.39 µs** | 与 API 口径一致（差 0.7%） |
| NCU cc=all @base | 109.15 µs, **dram 27.18%**, sm 28.92% | long_scoreboard 77.2% 的 stall |
| NCU cc=none @base | （同档, 见 profiles/） | 无 cache 控制差异 |

**诊断**: 标量 load 下每线程每次 1B, 在途字节数不足以掩盖 DRAM 延迟
（dram 只有 27%）—— 与 v0.5 FP16 标量 baseline 的 regime 相同, 向量化是
第一杠杆。NCU 数字与 API/native 的差距由 observed NCU profiling
perturbation + base clock 锁定贡献（DRAM-bound 下 ~0.8%）解释（§7）。

---

## 5. 四个证据驱动实验（用户 §5）

决策语义（沿用 evaluator v2.3）: `statistical_relation` 只看 CI95 是否排除
1.00; `policy_decision` 是 5% 接受带（median ≥1.05 且 ≥70% 轮更快且
CI95 下界 >1.00）之下的接受政策。两者分列记录。

| 实验 | parent → candidate | 假设核心 | median speedup | CI95 | 轮 | statistical | policy |
|---|---|---|---|---|---|---|---|
| QGEMV-0001 | baseline → **vec16_row** | 16B 向量 load（U16Q）把在途字节 ×16 | **2.6614×** | [2.657, 2.669] | 9/9 更快 | FASTER | **KEEP** |
| QGEMV-0002 | vec16_row → vec16_scale | 把 ×scale 从内循环 hoist 到行尾（16 乘 → 1 乘） | 0.9901× | [0.9885, 0.9906] | 0/9 | SLOWER | NEUTRAL |
| QGEMV-0003 | vec16_row → **warp_vec16** | warp-per-row + ILP=2 + 纯 warp shuffle（去掉 shared barrier） | 1.0107× | [1.0099, 1.0108] | 9/9 | FASTER | NEUTRAL |
| QGEMV-0004 | warp_vec16 → warp_vec16_ilp4 | ILP 2→4: 每 lane 在途 16B load 加倍 | 0.9984× | [0.9977, 0.9990] | 0/9 | SLOWER | NEUTRAL |

**NCU 证据链**（@4096×4096, cc=all, base clock; 同一 kernel 逐轮剖析）:

| 变体 | duration | dram% | occ% | regs | barrier stall | long_scoreboard |
|---|---|---|---|---|---|---|
| baseline | 109.15 µs | 27.2% | 94.55% | 16 | 1.3% | 77.2% (19.40 c/iss) |
| vec16_row | 36.37 µs | 86.02% | 82.62% | 40 | 7.3% | 51.4% (7.76 c/iss) |
| vec16_scale | 37.20 µs | 82.65% | **68.42%** | 38 | 8.6% | 49.2% (7.42 c/iss) |
| warp_vec16 | **36.16 µs** | **87.24%** | 90.43% | 41 | **0** | 84.1% (18.53 c/iss) |
| warp_vec16_ilp4 | 36.54 µs | 87.31% | 90.36% | 43 | 0 | 83.5% (18.61 c/iss) |

**逐轮证据解释**:
- **0001 KEEP**: 向量化把 dram 从 27% → 86%, 一次跨过 DRAM 墙。×scale 在内
  循环（`qgemv_vec_acc_dequant`: 16 项逐元素乘）—— 当时假设 ALU 是残余成本。
- **0002 NEUTRAL（SLOWER）**: hoist ×scale 后计算量降了, 但寄存器/指令排布
  变化使 occupancy 从 82.6% → 68.4% 掉档, dram 86.0% → 82.7%。
  **教训: ×scale 被内存延迟完全隐藏, 计算削减在这个 regime 不是杠杆;
  占用率掉档的代价大于省下的 ALU。**
- **0003 NEUTRAL（FASTER）**: warp-per-row 消掉 shared memory + barrier
  （stall 0）, occupancy 升到 90.4%, dram 升到 87.24%（各变体最高）,
  long_scoreboard 84.1% —— 几乎全部 stall 都是等 DRAM: **已贴 DRAM 墙**。
  但 1.0107× 在 5% 政策带内 → 按统一政策 NEUTRAL（统计上显著更快, 见 §6）。
- **0004 NEUTRAL（SLOWER）**: ILP 2→4 未带来 dram/occ 变化（87.31%/90.36%
  持平）, 寄存器 41→43, 4 相位循环 + 4 路相加的指令开销净负 ——
  假设被 NCU 定量否决: 贴墙后**更多在途 load 不再买得到延迟隐藏**。

---

## 6. Incumbent 决策与主目标 best-observed（用户 §6/§7, 统一政策）

项目固定决策规则: `statistical_relation`（只看 CI95 是否排除 1.00）与
`policy_decision`（5% 接受带: median ≥1.05 且 ≥70% 轮更快且 CI95 下界
>1.00）分列; **候选只有在达到 ≥5% 实质性阈值时才能替换政策 incumbent**
（QGEMV-0003 的 1.0107× 是「统计 FASTER 但 policy NEUTRAL」, 不替换
incumbent —— 早期稿中「CI95 显著性优先于 5% 政策带」的裁决规则是事后
规则, 不属于项目固定政策, 已移除）:

- **政策 incumbent = `qgemv_vec16_row`**（QGEMV-0001 唯一 KEEP, 2.6614×
  [2.657, 2.669], 9/9 更快）, 自 QGEMV-0001 起未变。
- **`qgemv_warp_vec16` = 主目标 best-observed 候选**: 比 vec16_row 统计
  快 ~1%, policy NEUTRAL（不替换 incumbent）。NCU 证据最强（dram 87.24%,
  occ 90.43%, barrier stall 0, warp-per-row, ILP=2, 纯 warp shuffle,
  无 shared/barrier）; `vec16_row` 结构更简, 差距 ~1%。统计证据保留:
  - head-to-head #1（QGEMV-0003 轮）: vec16_row 31.880 vs warp_vec16
    31.552 µs → 1.0107×, CI95 [1.0099, 1.0108] 排除 1.00（warp 更快）。
  - head-to-head #2（fresh, `final_incumbent.json`）: 31.808 vs 31.515 µs
    → 1.0094×, CI95 [1.0086, 1.0101] 再次排除 1.00, 方向一致。
  两次 head-to-head 均为 ~1%, 未达 5% 实质性阈值 → 统一政策下两次
  policy_decision 均为 NEUTRAL。ilp4 出局（对 warp_vec16 统计 SLOWER）。

**主目标 best-observed 候选 fresh revalidation**（fresh 9 轮 paired,
streaming, `final_incumbent_reval.json`）: `qgemv_baseline` 84.885 µs →
`qgemv_warp_vec16` **31.513 µs**（533.4 GB/s 逻辑 = 86.6% 峰值）→
**2.6937×**, CI95 [2.688, 2.700], 9/9 轮更快, filter gate 无影响。该
revalidation 回答主问题（INT8 相对 baseline 加速多少）, 不改变 incumbent
决策（incumbent 决策由 QGEMV-0001 KEEP 决定, 后续各轮均 NEUTRAL）。

---

## 7. 三种计时口径与一致性（用户 §7: 冲突必须调查, 不许挑好看的）

best-observed 候选 `qgemv_warp_vec16` @4096×4096 fp16 streaming:

| 口径 | 值 | vs API |
|---|---|---|
| API-path（bench 引擎, CUDA events） | **31.513 µs** | — |
| native kernel-loop w200 | 32.185 µs | +2.1% |
| native kernel-loop w5000 | **32.103 µs** | +1.9% |
| NCU kernel duration cc=all @base 1350 MHz | 36.16 µs（dram 87.24%） | +14.8% |
| NCU kernel duration cc=none @base 1350 MHz | 35.88 µs（dram 86.57%） | +13.9% |

**一致性判定: 无冲突, 无需调查。** 分层解释（同 v0.5 方法论）:
1. API vs native 差 <2.5%: 两口径在同一 boost 时钟域, 差异在窗口/轮换
   统计噪声量级（v0.5 同量级）。w200 与 w5000 几乎重合（32.185/32.103）
   —— 该 kernel 无 v0.5 baseline 那种 warmup 爬坡缺口（84.9µs 的 kernel
   每轮都够长, DVFS 在流中保持 boost）。
2. NCU 系统性高 ~14%, 两层分解: ① **base clock 锁定（1350 MHz）贡献
   ~0.8% 量级** —— v0.5 受控对证据: 同为 DRAM-bound 的 `gemv_vec4_row`
   （dram 87.9–90.2%）clkbase vs clknone 差仅 +0.77%（64.192 vs 63.704 µs）,
   而 latency-bound 的 `gemv_baseline` 差 +15.9%（114.76 vs 99.032 µs）;
   本 kernel 在 base 下 dram 87.24%, 同 regime, 锁频效应不可能是 ~14%;
   ② **其余 ~13–14% 是 observed NCU profiling perturbation**: 36.16 − 31.513 ≈ 4.65 µs
    （cc=none 亦 4.37 µs, +13.9%）。这 ~4.4–4.7 µs 是
    **本 workload 上的观测差, 不声称是跨 workload 的固定常数**（未设
    专门的 cross-workload 实验; v0.5 在 ~60 µs kernel 上 ~3.8 µs 的观测
    仅作量级参照, 不是受控实验证据）。注: 负载下实测 boost 为
    1815–1920 MHz（repo 内 gpu_state 采样）, 早期稿「1545/1350 = 1.144
    定量吻合」的归因**不成立, 已按独立 review（M1）更正**; 定性解释
    （锁频 + profiling 扰动只能让 NCU 更高, 方向正确）不变。cc=all 与 cc=none 差 0.8%
   （L2 flush 影响小, 该工作集 ≫ L2）。
3. 排序在所有口径下一致（warp_vec16 < vec16_row < vec16_scale <
   baseline；NCU 口径次序与 API 同）—— 决策不受口径选择影响。

---

## 8. 性能对比总表（用户 §6）

@4096×4096 fp16 streaming, 本 session 同 GPU 同 harness;
逻辑 IO: QGEMV 16.81 MB / FP16 33.575 MB; **算法 BW ≠ 实测 DRAM BW**
（逻辑值 = 理想无冗余流量; NCU dram% 是实测 DRAM 吞吐, 后者才是物理证据）。

| 实现 | API-path | native w5000 | NCU @base (cc=all) | 逻辑 BW | 算法下界效率 |
|---|---|---|---|---|---|
| QGEMV baseline | 84.885 µs | 84.389 | 109.15 µs (dram 27.2%) | 198.0 GB/s | 32.2% |
| QGEMV vec16_row（KEEP, 政策 incumbent） | 31.808 µs | — | 36.37 µs (dram 86.0%) | 528.5 GB/s | 85.8% |
| **QGEMV warp_vec16（best-observed）** | **31.513 µs** | **32.103** | **36.16 µs (dram 87.24%)** | **533.4 GB/s** | **86.6%** |
| FP16 gemv_vec4_row（v0.5, 参考） | 59.933 µs | 59.682 | 64.18 µs (dram 89.3%) | 560.1 GB/s | 90.9% |
| torch.mv(反量化 fp16 W)（QGEMV 上下文） | 62.297 µs | — | — | 538.9 GB/s | — |
| torch.mv(fp16 W)（framework） | 61.235 µs | — | — | 548.2 GB/s | — |

**主结果（best-observed 口径）: INT8 QGEMV 31.51 µs vs FP16 GEMV
59.93 µs = 1.90×**（理论 2.0× 的 95%; 剩余 5% 来自 x/scale/y 的非减半
流量与效率差 86.6% vs 90.9%）。政策 incumbent 口径 vec16_row
31.808 µs → 1.88×, 与 best-observed 口径差 ~1%（policy NEUTRAL, 不
改变结论）。QGEMV best-observed vs 其 baseline = 2.69×; vs FP16
baseline（91.93 µs, 本 session 同口径复测）= 2.92×。

---

## 9. 全 shape 矩阵与回归

### 9.1 矩阵（5 形状 × {hot, streaming}, 5 变体, 9 轮, tag `v0.6`）

中位数 µs（fp16; `benchmarks/qgemv/v0.6_*.json` + `v0.6_shape_winners.json`）:

| (N, K) | mode | baseline | vec16_row | vec16_scale | **warp_vec16** | ilp4 | 最低中位数 variant |
|---|---|---|---|---|---|---|---|
| (1024, 4096) | hot / streaming | 14.53 / 14.78 | 7.38 / 7.50 | 7.52 / 7.55 | **6.52 / 6.66** | 6.59 / 6.68 | warp_vec16 ×2 |
| (4096, 1024) | hot / streaming | 19.39 / 19.58 | 13.31 / 13.49 | 14.87 / 15.07 | **6.64 / 7.38** | 6.78 / 7.36 | warp_vec16 (hot, 1.0218× CI 排除 1.0) / ilp4 (streaming, 7.355, CI 跨 1.0) |
| **(4096, 4096)** | hot / streaming | 86.99 / 87.21 | 32.04 / 32.06 | 32.34 / 32.32 | **31.88 / 31.81** | 31.95 / 31.87 | warp_vec16 ×2 |
| (11008, 4096) | hot / streaming | 227.07 / 227.20 | **80.50 / 80.58** | 80.83 / 80.96 | 81.50 / 81.56 | 81.58 / 81.66 | vec16_row ×2（对次优 CI 均排除 1.0, +0.41/+0.48%） |
| (4096, 11008) | hot / streaming | 226.88 / 227.97 | **80.36 / 80.32** | 81.06 / 80.82 | 80.78 / 80.86 | 80.87 / 80.95 | vec16_row ×2（+0.53/+0.62%, CI 均排除 1.0） |

**读法（三口径, 不混用; per-cell 双字段由 `classify_cell` 统一重新派生,
见 `v0.6_shape_winners.json`）**:
1. **口径一: 最低观测中位数**（排名, 非胜者声明）: warp_vec16 在
   5/10 格最低中位数（含主目标两 mode）; vec16_row 4/10 格（两个含
   11008 的形状 × 两 mode, 领先次优 0.4–0.6%, CI 均排除 1.0, 真实但
   小）; ilp4 1/10 格（(4096,1024) streaming, 7.355 vs warp_vec16
   7.377）。早期稿「warp_vec16 赢 6/10 / vec16_row 赢 4/10」把
   「最低中位数」与「胜者」混用且计数有误（(4096,1024) streaming 实为
   ilp4 最低中位数格）, 现按三口径拆分。
2. **口径二: 统计显著更快**（winner vs runner-up, CI95 排除 1.00,
   statistical_relation=FASTER）: 8/10 格; 2 格 UNRESOLVED =
   (1024,4096) streaming（CI [0.9940, 1.0154]）与 (4096,1024) streaming
   （CI [0.9817, 1.0164]）。
3. **口径三: policy 显著胜者**（policy_decision=KEEP, 即 ≥5% 实质性
   阈值 + ≥70% 轮更快 + CI95 下界 >1.00）: **0/10 格** —— 所有格
   winner-vs-runner-up 差距 <5%, 按统一政策全部 policy NEUTRAL /
   NO_UNIQUE_WINNER。因此矩阵中不存在 policy 显著胜者; 上表与 JSON 的
   「winner」一律只表示最低观测中位数排名。
4. **ilp4**: 1 格最低中位数但统计 UNRESOLVED（CI 跨 1.0, 3/8 轮更快）,
   无统计显著胜利 —— 与 §5 QGEMV-0004 一致: 更深的 ILP 不产生优势,
   只在个别格与 tie 打平。
5. **所有向量化变体在全部 10 格都是 baseline 的 2.2–2.9×**
   （最低中位数 variant vs baseline 实测区间 2.22–2.92×; (11008,4096)
   格为 2.821×/2.820×）; 含 11008 的长 K 格与主目标格内, 向量化变体
   之间差 ≤~2%, 但短 1-dim 格差异是 regime 级的: (4096,1024) streaming
   格 vec16_row 13.49 vs warp_vec16 7.38 = **2.05×**（(1024,4096) 格也
   有 ~15%）—— 长 K/多行 regime（覆盖主目标）下贴 DRAM 墙后 kernel
   形态差异被抹平, 这解释了为什么 4 个实验里只有第一跳（向量化）产生
   2.66×, 后续轮次都在 ±1% 内; 形态差异只在短 K/短 N regime 显现。
   (4096,1024) 短 K 格的结构性差异机制: K=1024 → 每行只有 64 个 16B
   向量, block-per-row（256 线程）75% 线程空转, warp-per-row（32 线程
   × ILP2 = 64 向量）恰好满负荷; 长 K/多行 regime 下 block-per-row 的
   256 线程对 256 向量恰好满负荷, warp-per-row 的 8 warp 归约路径略逊。
6. hot 与 streaming 结果逐格几乎重合（W ≫ L2, cache 状态不改变 regime）;
   (4096,1024) streaming 格 8/9 有效轮（1 环境无效轮, 已记录在 JSON）。
7. 11008 形状逻辑 BW ~560 GB/s（90.9% 峰值）—— 大 N 时 scale 表占比
   降低, 效率反超主目标的 86.6%; 主目标的 3.3% 差距 = scale (16 KB) 的
   固定额外 pass 占比更大。

**incumbent 与矩阵一致性**: 统一政策下政策 incumbent 保持
`qgemv_vec16_row`（自 QGEMV-0001 KEEP 起未变; 矩阵每格的最低中位数
variant 对 runner-up 差距均 <5% → policy NEUTRAL, 无格可触发替换）。
`qgemv_warp_vec16` 为主目标 best-observed 候选（统计快 incumbent ~1%,
policy NEUTRAL; 主目标两 mode 最低中位数 + (4096,1024) 短 K 结构性优势
~2.0× + NCU 证据最强; 在两个 11008 格落后 vec16_row 0.4–0.6%）——
全部已在 winner 文件与本报告如实记录。

### 9.2 回归 smoke 与 quarantine 审计（`experiments/regression/v0.6/`）

**全部 PASS**（GPU 0, 分支最终态; append-only, 历史目录未触碰）。
本轮覆盖**全部正常变体**（比 v0.5 只跑主变体更宽）:

| 算子 | 正常变体（本轮重跑） | correctness | negative |
|---|---|---|---|
| gemv | baseline / vec4_row / warp_vec4_b256 / warp_vec4_b512（4） | **100/100 ×4** | **24/24 ×4** |
| rmsnorm | baseline / v1_vec / v2_reg / v3_wideblock / v4_vec_reg（5） | **76/76 ×5** | 29/30 ×5（1 = 预存环境 skip, all_pass 仍 True） |
| softmax | baseline / online / vec4 / vec4_ilp2（4） | **72/72 ×4** | 14/15 ×4（1 skip, 同上） |
| rope | baseline / v1_2pair / v2_4pair / v3_half2 / v4_8pair（5） | **384/384 ×5** | 36/37 ×5（1 skip, 同上） |
| qgemv | baseline / vec16_row / vec16_scale / warp_vec16 / ilp4（5） | **50/50 ×5**（+ 保真度套件） | **29/29 ×5** |

**Quarantine 审计**（`quarantine_audit.json`, 每算子记录 `variants()` /
`all_variants()` / `quarantined_variants()` 绑定）:
- `gemv_splitk4`: 不在 gemv 正常列表（`quarantine_leaked_into_normal` 空）,
  仍在 `all_variants`（受控历史审计入口保留, v0.5 语义不变）;
- `softmax_hsplit2`: 同上, 不在 softmax 正常列表;
- qgemv: `quarantined_variants()` = **空**（隔离机制保留, 当前无隔离变体）;
- 5 算子 `all_pass` 全 True, 无 quarantine 泄漏。

---

## 10. 独立 review（两路, 均为独立 subagent, 与本 session 主工作流隔离）

### 10.1 Benchmark methodology review —— **PASS WITH CAVEATS**

独立复算与审计结论: 6 个 pair 记录（QGEMV-0001..0004 + final_incumbent +
final_incumbent_reval）的中位数/9/9 faster 计数/CI95 用 `cudalab/evaluator/
stats.py` 的 bootstrap（n=10000, seed 20260919, median 统计量, 百分位索引
250/9749）**逐位精确复现**; 27 项报告数字抽查全过; harness 审计 PASS
（A/B 交替、同池、对称 spike guard、filter gate、`algorithmic_bytes`）;
cherry-pick 审计 PASS（11008 格 vec16_row 获胜已披露）; 不可变/隔离审计
PASS（main..HEAD 105 文件 +236,922/−5, 零删除, quarantine 完整）。
Findings: 2 MAJOR + 4 MINOR + 6 NIT, 逐条处置:

| Finding | 内容 | 处置 |
|---|---|---|
| M1 (MAJOR) | 「boost 实测 ~1545 MHz / 1545÷1350=1.144」归因无据, 且与 repo 内 gpu_state 负载采样（1815–1920 MHz）及 v0.5 受控对矛盾 | **已修（本稿; merge-review 轮进一步软化措辞）**: §1/§7/§11 + README + STATUS 全部改为两层分解 —— 锁频贡献 ~0.8%（v0.5 DRAM-bound 受控对 64.192/63.704 = +0.77%）+ observed NCU profiling perturbation ~4.4–4.7 µs（本 workload 观测差, 不声称跨 workload 固定常数; v0.5 ~3.8 µs 仅作量级参照）; 早期归因在文中明确标记为已更正。无决策影响（决策全在 API 口径内） |
| M2 (MAJOR) | 受审报告未提交、分支未 push, 与 §1 checklist 矛盾 | **已修（本稿后一步）**: §10 补全后提交 README/STATUS/报告并 push `v0.6-qgemv`, checklist 在提交态为真 |
| M3 (MINOR) | §5 NCU 表 vec16_scale 行 regs 40/barrier 7.3% 与 profile JSON（38 / 8.6%）不符 | **已修（自审轮）**: 更正为 38 / 8.6%（1.302 c/iss）; 7.3% 系 vec16_row 值误植。review 独立复核一致 |
| M4 (MINOR) | 「2.3–2.8× 且互差 ≤1.2%」不成立: (1024,4096) 格 2.22–2.23×; (4096,1024) streaming 变体间差 2.05× | **已修（本稿）**: 改为 winner-vs-baseline 2.22–2.92×, 长 K 格互差 ≤~2%, 短 1-dim 格 regime 级 2.05×（§9.1 + README + STATUS）。commit 5d80d4e 消息里的旧措辞按 append-only 不改, 以本报告为准 |
| M5 (MINOR) | (11008,4096) 「2.778×/2.772× vs baseline」—— 实际 2.821×/2.820× | **已核（本稿）**: 当前 §9.1 该行不含 vs-baseline 倍率（仅 µs + 对次优 CI）; §9.1 项 1 明确记录 2.821×/2.820×; 旧值仅存在于 commit 消息, 不改历史 |
| M6 (MINOR) | §4 baseline NCU 「sm ~20%」—— 实际 28.92% | **已修（本稿）**: 改为 28.92% |
| n1 | ccnone +13.7% | **已修**: +13.9%（35.88/31.513） |
| n2 | §2 比值 1.998 | **已修**: 1.997（33,570,816/16,809,984） |
| n3 | §4/§8 两个 baseline 数字混用 | **已修**: §4 84.96 µs = 197.9 GB/s; §8 84.885 µs = 198.0 GB/s |
| n4 | `v0.6_baseline.json` 带模板化 `filter_sensitive_reason`（"无 raw 数据"）但该记录是 v2.3 带 raw track | **记为已知差距, 不修**: 历史 artifact 不可变（v0.4/v0.5 先例）; 单变体记录, 无 raw/filtered 方向问题, 字段无害 |
| n5 | §11(7) INT4 投影 9.43 MB 不可由文档公式导出 | **已修（自审轮）**: 改为显式 G=128 推导 8.93 MB / 14.50 µs / 理论 1.88×（实测折损预计 ~1.7–1.8×） |
| n6 | §1 checklist「独立 review ✓」早于 §10 补全 | **已修（本稿）**: §10 于 commit 前补全, checklist 在提交态为真 |

review 无法独立验证项（如实记录）: ① 精确 DRAM 字节数 —— 本仓 NCU 采集
未含 `dram__bytes.sum`, 隐含字节核算（dram% × 时长 × 616 GB/s, 18.3–19.7 MB
vs 逻辑 16.81 MB）review 判定为合理非红旗, 精确审计需重采 profile;
② 9-round 窗口内连续 SM 时钟 trace（只有前后离散 gpu_state 采样）;
③ NCU 锁频保真度（JSON `clock_lock_warning: null`, raw 报告无 MHz 验证行）。

### 10.2 CUDA correctness review —— **PASS WITH CAVEATS**

（第一路独立 subagent 审计未产出交付物即被中断; 第二路以更窄 scope 完成
审计, 以下为其交付。审计方式: 纯代码/JSON 审计, 无 GPU 复跑。）

独立审计范围与结论: `kernels/qgemv/` 全部 7 文件（K 全覆盖——含 K%16≠0
无尾缺口、7 种 K 的 CPU 模拟覆盖核对; int8 符号扩展; vec16_scale 的
×scale 行末提升合法性——q·x 乘在 FP32 精确（7-bit × 11-bit ≤ 24-bit）,
合法重排; scale=0 行精确 0 且无 NaN/Inf 路径; 归约树步数/索引;
`__float2half_rn` 单一来源; host 侧 launch 前契约检查 + 单一来源标量
kernel 的 bit-identical 回退; grid/block 与 ilp4 四相位不相交完整性）;
正确性门（逐元素固定算术界, TOL_K=2 / 每 term 3 次舍入为模块常量, 无按
变体/shape/mode 的放宽路径; 双侧 finiteness 门; 保真度四量**不在判定
式**）; 负例 29 例 per-case 判定（归档 JSON 复核 reject_ok 23/23 +
pass_ok 6/6）; 量化器（torch.round = round-half-to-even; scale=0 双保险;
非有限输入显式拒绝; 无 RNG）; JSON 抽查（`quantization_fidelity.json`
5 个 summary 值与 25 例重算逐一精确相等; 5 变体 × 50 例共 250 个 per-case
pass 标志与存储量重算一致; max-ratio 例以存储量代入公式复现, 差 3e-13;
zeros mode 10 形状 × 5 变体 ratio==0 ∧ max_abs==0）。

Findings: 0 MAJOR, 1 MINOR, 2 NIT, 逐条处置:

| Finding | 内容 | 处置 |
|---|---|---|
| MINOR-1 | 负例 29 例全部是 host 元数据契约用例, 无值域类用例（scale=0 行 / NaN/Inf / 极端值） | **记为已知边界, 不修**: 仓库自身文档自洽（`qgemv_negative.py` 文件头自述 = 实际构造）; scale=0 行合同由 correctness 套件 zeros mode 钉死（全零 W → 全行 scale=0, 10 形状 × 5 变体 ratio==0 ∧ max_abs==0）, 极端值由 large mode（scale=10, fp16 有限性保证）间接覆盖; NaN/Inf 输入在 `qgemv_common.h` 文档化为**值域外**（garbage in garbage out, 与 v0.5 GEMV 语义一致）, kernel 路径无 D2H 值检查、该行为无测试钉死 —— 文档化设计选择, 非缺陷 |
| NIT-1 | 量化器 docstring「max_dequant_step_err 应 ≤ 0.5, 构造性保证」与实测 0.5000019 偏差 ~2e-6（FP32 的 scale 与 q·scale 两次舍入引入 O(2^-24·127) 量级） | **已修（本批提交）**: `qgemv_quantize.py` docstring 改为「精确算术意义下 ≤ 0.5 的构造性界; FP32 实现路径实测 ≈ 0.5000019; 本量为 report-only 自检量, 不参与判定」 |
| NIT-2 | arith 界 ulp 项为 0.5·ulp16(\|y\|) + 0.5·ulp16(\|exact\|)（y = impl 输出）而非 \|y_ref\| | **不修（不构成问题）**: 代码与自身 docstring 一致, 且数学成立（y = RN(v) 时 ulp16(\|y\|) ≥ ulp16(\|v\|), 第二项为额外余量）, 界对所有变体固定无放宽; 审计任务书中的「y_ref」为转述偏差 |

review 无法独立验证项（如实记录）: ① 内核未重新编译/运行（review-only,
无 GPU）—— bit-identical 回退以归档 JSON 记录断言 + 代码构造论证为据;
② correctness JSON 只存聚合量（无 y 向量/argmax 行号）, 「argmax 行
fp16 y 跨变体一致」为量级论证; ③ large mode 边界（10σ<65504）按生成器
注释核对, 未逐值重算; ④ 实验/profile JSON 超出本审计范围（已由 §10.1
benchmark 审计覆盖统计/CI/harness/不可变/隔离）。

**两路 review 总结论**: 均 **PASS WITH CAVEATS**; 无任何 finding 影响
kernel 正确性结论、统计决策或 incumbent / best-observed 选择; benchmark 路全部
MAJOR/MINOR 已在 §1/§2/§4/§5/§7/§9.1/§11 与 README/STATUS 修正（n4 记为
不可变 artifact 已知差距）; CUDA 路 MINOR-1 记为已知边界、NIT-1 已修
docstring（本批提交）。

---

## 11. 用户 7 问（逐条回答）

**(1) INT8 vs FP16 GEMV 加速多少?**
**1.90×**（best-observed 口径, warp_vec16 31.51 µs vs 59.93 µs, API-path,
主目标, 同 session 复测; 政策 incumbent 口径 vec16_row 31.808 µs → 1.88×,
结论不变）。理论 2.0×（逻辑 IO 16.81 vs 33.575 MB）的 **95%**。相对各自
baseline: QGEMV 2.69×（84.89 → 31.51）vs FP16 1.53×（91.93 → 59.93,
本 session）。

**(2) 权重流量减半后瓶颈移到哪了?**
**没有离开 DRAM** —— 只是从「喂不满 DRAM」(baseline dram 27.2%) 变成
「紧贴 DRAM 墙」(最终 dram 87.24%, long_scoreboard 84.1% 的 stall,
occ 90.43%, barrier 0)。INT8 减半的是**所需带宽**, 不是**可用带宽**:
kernel 现在只需用一半时间搬完数据。残余 ~13% 头部里 NCU 看不到 compute
瓶颈（sm 吞吐 ~36-39%, FP32 FMA 远低于峰值）—— 剩余是 DRAM 效率本身
的天花板（与 v0.5 FP16 91.4% 同 regime, QGEMV 86.6% 略低, 差 ~5% 来自
x 仍为 fp16 的混合流量与 scale 的额外 pass）。

**(3) 反量化开销是否抵消了带宽收益?**
**没有, 且被实验定量证明**: QGEMV-0002 把 ×scale 从内循环 hoist 到行尾
（16 乘/16 元素 → 1 乘/行）—— 若 ALU 是瓶颈, 应显著变快; 实测 0.9901×
（统计 SLOWER）, 因为计算削减换来了 occupancy 掉档（82.6% → 68.4%）。
×scale 完全被 DRAM 延迟隐藏; 逐元素反量化在这个 kernel 里**免费**。
带宽收益全额兑现: 84.89 → 31.51 µs（2.69×, 甚至略超 2×, 因为 baseline
本身只有 32% 效率, 向量化后到 86.6%）。

**(4) 哪个 QGEMV 变体最有效, 为什么?**
**best-observed 口径: `qgemv_warp_vec16`**（warp-per-row, ILP=2 双累加器,
纯 5 步 warp shuffle, 无 shared/barrier, 41 寄存器, occ 90.43%）。它比
政策 incumbent `qgemv_vec16_row` 统计显著快 ~1%（两次 fresh head-to-head
1.0094–1.0107×, CI95 均排除 1.00）, 但按统一政策 policy NEUTRAL（未达
5% 实质性阈值, 不替换 incumbent, 见 §6）。有效原因全部有 NCU 证据:
① 去掉 block 级 barrier（vec16_row barrier stall 7.3% → 0）;
② 占用率 82.6% → 90.43%（更多 warp 可掩盖延迟）;
③ dram 87.24% = 各变体最高。更深 ILP（0004, ILP=4）
在贴墙后不再有效（dram/occ 持平, 指令开销净负）—— 该结构的在途字节
数已经是这台 GPU 上该 kernel 形态的饱和点。

**(5) API / native / NCU 三口径一致吗?**
**一致, 无冲突**（§7 两层分解）: API 31.513 ≈ native w5000 32.103（+1.9%）;
NCU 36.16 高 +14.8% = observed NCU profiling perturbation（本 workload
观测差 ~4.4–4.7 µs, 不声称跨 workload 固定常数; v0.5 ~60 µs kernel 上
~3.8 µs 的观测仅作量级参照, 无专门 cross-workload 实验）+ base clock
锁定贡献 ~0.8%（v0.5 clkbase/clknone 受控对: DRAM-bound +0.77% /
latency-bound +15.9%, 本 kernel 属前者; 负载下实测 boost 1815–1920 MHz,
早期稿「1545 MHz / 1.144 比值」归因已按 review 更正, M1）。cc=all vs
cc=none 差 0.8%。所有口径排序一致, 任何口径选择
都不改变任何决策。

**(6) 量化误差有多大?**
由量化步长决定、与 kernel 无关（两层容差分离, 用户 §2）: 每元素反量化
误差 ≤ 0.5·量化步长 = 0.5·max|W_row|/127（实测 max_dequant_step_err
0.500002, 即 round-half-to-even 理论上限）。主目标 normal 数据: y 向量
cosine **0.9999617**, rmse 0.551（|y| 范数 4026.5, 即相对 ~0.014%）,
max_abs 2.57（个别近零行）。全 25 用例最坏 cosine 0.9998970（mixed_sign）。
large（scale=10）时绝对误差 ×10（max_abs 240–369）—— 绝对误差随权重
尺度线性增长, **相对精度不变**。zero-scale 行精确为 0。

**(7) 下一步值得上 INT4 / group-wise 吗?**
**值得, 但作为独立版本立项, 理由与风险如下。** 收益侧: INT4 再把权重流量
减半（0.5 B/元素 + group-wise 128 fp32 scale: 主目标逻辑 IO 16.81 →
8.93 MB, 算法下界 27.29 → 14.50 µs, 理论 1.88×）; 当前 kernel 已贴
DRAM 墙且 QGEMV 效率（86.6%）低于 FP16（90.9%）, 说明「搬得更快」的
杠杆还有 ~5% 结构性空间。风险/成本侧:
① INT4 每 16B 向量装 2 行（或 1 行 32 元素×2）的打包/解包 ALU 路径
  变复杂, 对齐契约从 16B/16 元素变成 16B/32 元素×2 的位级解包, 回退门
  与负例套件要重新设计（本轮 29/29 的架构可直接复用框架）;
② per-row scale 在 4-bit 下保真度不够（step ×4, cosine 预计掉到
  ~0.9998-0.9999 量级, 需实测）—— **group-wise（128/256 一组 scale）
  是配套需求而非可选项**, 会引入组边界 load 与 scale 表流量, 侵蚀部分
  带宽收益;
③ 当前 2080 Ti 的实测 DRAM 效率 86.6% 意味着下界本身有 ~13% 水分,
  INT4 的 1.88× 理论值在实测上预计折损到 ~1.7–1.8×（unpack ALU +
  组边界开销, 且假设效率不超过当前 86.6%）。**结论: 立项做,
  但按「INT4 + group-wise scale + 独立保真度门」整体设计, 不拆成两个
  半吊子版本; 本轮 v0.6 停止, 等待外部 review。**

---

## 12. 产物索引

- kernel: `kernels/qgemv/{qgemv_common.h, bindings.cpp, qgemv_baseline.cu,
  qgemv_vec16_row.cu, qgemv_vec16_scale.cu, qgemv_warp_vec16.cu,
  qgemv_warp_vec16_ilp4.cu}`
- 量化/算子/正确性/负例: `cudalab/{qgemv_quantize.py, qgemv_correctness.py,
  qgemv_negative.py, operators/qgemv.py}`
- 正确性/保真度: `experiments/qgemv/correctness/v0.6/`（50×5 变体 + 29×5
  负例 + quantization_fidelity.json）
- 实验记录: `experiments/qgemv/QGEMV-000{1,2,3,4}.json` +
  `final_incumbent*.json`（benchmarks/）+ `native_timing_*.json`
- profile: `profiles/qgemv/*`（baseline/vec16_row/vec16_scale/warp_vec16/
  warp_vec16_ilp4; best-observed 候选含 cc=all+cc=none）
- FP16 参考: `benchmarks/gemv/v0.6_fp16_ref.json`,
  `experiments/gemv/v0.6_ref/`, `profiles/gemv/v0.6_ref/`,
  `experiments/qgemv/v0.6_ref_pytorch_context_4096.json`
- 回归: `experiments/regression/v0.6/`（gemv/rmsnorm/softmax/rope/qgemv
  五算子 correctness+negative 全 PASS + quarantine_audit.json + README）
