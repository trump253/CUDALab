# CUDALab v0.7 Result — W4A16 Group-wise INT4 GEMV 优化

日期: 2026-09-22 · GPU 0 (2080 Ti, 616 GB/s 规格峰值, 30 SM, L2 5.5 MB,
sm_75, CUDA 11.8, PyTorch 2.4.1+cu118, NCU 2022.3, base 1350 MHz)
分支: `v0.7-int4-gemv`（基线 main = v0.6.1 = b0c5c7d = tag v0.6.1）。
**不 merge 回 main、不 force push**（用户指定; 本 commit 后 push 即 STOP
等外部评审）。

## 1. Status

**核心问题（用户规格）**: 权重从 INT8 1B → INT4 0.5B（W4A16, G=128
group-wise, fp16 scale）后, GEMV 能否拿到逻辑 IO 减半的 ~2× 收益?
INT4 的 nibble unpack / group scale 查找 / 指令开销代价多大?

**结果（4096×4096×4096, fp16, streaming, 本 session 同 GPU 同 harness）**:

- 政策 incumbent: **`int4gemv_rowtile4_hx`**（INT4GEMV-0004 KEEP 起）,
  API 路径 **20.50 µs**（矩阵口径; 0004 pair 19.894 µs）, vs baseline
  `int4gemv_baseline` 53.20 µs = **2.59×**（pair 链 51.7 → 19.894 =
  2.599×）。
- 三代对比（本 session fresh 测量, 不沿用历史数字）:
  **INT4 20.50 µs vs INT8 `qgemv_vec16_row` 31.93 µs = 1.557×**
  —— 理论 2.0× 的 **78%**; **vs FP16 `gemv_vec4_row` 59.98 µs =
  2.925×**。INT4 未接近 INT8 理论 2×: algorithmic 视图（逻辑 IO
  减半 1.940×, 理论流量削减）与 measured NCU 物理视图（物理流量削减
  1.796× × 物理带宽比 0.806× = NCU 1.447×）见 §8 两层分解;
  API 1.557× 单独口径单独报告, 不用 NCU 分解精确解释。
- 正确性: 层 A（kernel vs CPU 解包+反量化 FP32 参考, 固定算术界
  TOL_K=2）5 变体各 50/50; 层 B（CPU 量化+pack nibble 互逆）39/39;
  层 C（量化保真度）report-only; per-variant 负例 30/30 × 5
  （含 2 例对齐回退 bit-identical 钉死）。回归 `experiments/regression/
  v0.7/` 全绿（int4gemv 5 + gemv 4 + qgemv 5 变体）。
- 4 组预注册实验（INT4GEMV-0001..0004, 每轮 profile→假设→实现→
  正确性→负例→paired 9r→native→NCU→决策）: 3 KEEP + 1 NEUTRAL,
  失败/中性实验全部保留。
- 两个独立 subagent review（CUDA correctness + benchmark methodology）
  结论见 §10。

## 2. 算子定义与范围（用户 §1 合同）

- 输入: `W_packed uint8 (N, K/2)`（2×INT4/byte）+ `scale fp16
  (N, K/128)` + `x fp16 (K,)` → `out fp16 (N,)`, FP32 累加, 输出
  round-to-nearest-even。
- **nibble 契约**: packed row 的 byte b: 低 nibble = k=2b, 高 nibble =
  k=2b+1; 4-bit 二进制补码（q ∈ [-7,7]）; 1 byte = 2 连续 k 且**同
  group**（g = b>>6, G=128 → 1 group = 64 bytes = 32 uint4 向量）。
- 量化: symmetric per-group（G=128）, `scale = amax/7`（fp16 存储）,
  `q = clamp(round(W/scale), ±7)`; **量化 + packing 在计时区外**
  （`cudalab/int4gemv_quantize.py`, CPU 39/39 测试）。
- K%128==0 是**硬合同**（量化期拒绝, kernel 期 host 元数据拒绝）;
  对齐契约: W_packed 16B ∧ x 16B ∧ K%32==0 → 向量化; 不满足 → 回退
  `int4gemv_scalar_kernel`（与 baseline 同一代码源, bit-identical）,
  合法输入（含 1 字节 offset 连续视图）**绝不拒绝**。
- 5 变体: `int4gemv_baseline`（标量）/ `int4gemv_vec16_row`（0001,
  16B 向量化 one row/block, 每行重发 x）/ `int4gemv_rowtile4`（0002,
  R=4 行块 x 寄存器驻留）/ `int4gemv_rowtile8`（0003, R=8, 归档）/
  `int4gemv_rowtile4_hx`（0004, R=4 + x half 驻留, **incumbent**）。
- 主目标 (4096,4096); 5 形状 {(1024,4096),(4096,1024),(4096,4096),
  (11008,4096),(4096,11008)}; fp16 activation only。

## 3. 正确性（用户 §2 三层）

**层 A（硬门）**: kernel vs CPU 解包+反量化 FP32 参考, 固定算术误差界
`tol = 2·(3K·2⁻²²⁴·S_n + 0.5·ulp16(|y|) + 0.5·ulp16(|exact|))`,
TOL_K=2; 5 变体 × (5 输入模式 × 9 形状/边界) = **50/50 × 5 全绿**。
`max_arith_max_ratio = 0.245053042161`（5 变体一致 = 同一 term 序;
层 A 独立核验, 不依赖交叉等价假设）; max_abs 4.0–8.0（大数抵消场景
的 ulp 级, 预期内）。

**层 B（硬门, CPU 侧）**: 量化+pack 的 nibble 互逆测试 **39/39**
（低/高 nibble 位序、补码 round-trip、group 边界、K%128 拒绝、
scale=0 安全、fp16 scale 存储-读取同值）。

**层 C（report-only 不作门）**: 量化保真度 vs FP16 权重: 每元素误差
≤ 0.5·量化步长; normal 模式 cos ≥ 0.999（K=4096 深抵消场景）;
记录于 `quantization_fidelity.json`（append-only, 历史 artifact
时间戳不漂移 —— 本 session 修复的 `run_correctness` 读取已存文件
而非重跑）。

**负例**: 30/30 × 5 变体（25 reject + 5 pass, 含 2 例对齐回退
bit-identical 钉死: fallback_Wp_misaligned / fallback_x_misaligned）;
post_check_ok 验证 CUDA context 无污染。

## 4. Baseline（用户 §4: 先 baseline, 之后才能优化）

`int4gemv_baseline`: 标量 uint8 load, one row/block 256 线程, FP32
累加, 两级归约。4096²: API 53.20 µs; NCU (ccall clkbase) 66.43 µs,
DRAM **25.44%**, long_scoreboard **64.2%（9.795 cpi）** —— 标量 1B
load 延迟受限（每线程在途仅 1–2 B）, 与 v0.5 FP16 / v0.6 INT8
baseline 同形态（DRAM 25–49% + 高 long_scoreboard）。

## 5. 四个证据驱动实验（用户 §5: 每轮 NCU 选下一实验, 不预机制化）

paired-streaming-v2.3, 9 轮, 主目标 4096², 链式 parent; 决策 = 固定
政策（statistical_relation 只看 CI95; policy_decision 5% 带:
KEEP ⇔ median ≥ 1.05 ∧ faster_frac ≥ 0.70 ∧ CI_lo > 1.00, filter
gate 之后; 5% 带内 = NEUTRAL **即使 CI 排除 1.00**）。

| ID | 候选 | parent | parent→cand (µs) | 加速 | CI95 | faster | stat / policy |
|---|---|---|---|---|---|---|---|
| 0001 | vec16_row | baseline | 51.7 → 25.721 | 2.000× | [1.9975, 2.0105] | 9/9 | FASTER / **KEEP** |
| 0002 | rowtile4 | vec16_row | 25.667 → 21.600 | 1.188× | [1.1827, 1.1909] | 9/9 | FASTER / **KEEP** |
| 0003 | rowtile8 | rowtile4 | 21.677 → 22.137 | 0.978× | [0.9748, 0.9811] | 0/9 | SLOWER / **NEUTRAL** |
| 0004 | rowtile4_hx | rowtile4 | 21.570 → 19.894 | 1.087× | [1.0822, 1.0887] | 9/9 | FASTER / **KEEP** |

**0001 vec16_row（KEEP 2.000×）**: baseline NCU 的 long_scoreboard
64.2% → 假设「16B 向量化 + 每线程多 outstanding load」。DRAM 25.44% →
48.98% 命中; 但新第一停顿 **lg_throttle 43.9%（6.555 cpi, LSU 发射
队列满）**—— x (K,) 对所有行共享, 每行被完整重发一次（768 内存
指令/行, 其中 x 侧 512 条 LDG.128/行）。

**0002 rowtile4（KEEP 1.188×）**: 假设「R=4 行块, x 片段寄存器驻留
跨 4 行复用」→ 内存指令 768→384/行（x 512→128）, 4 条独立 FMA 链
（ILP 4）。lg_throttle 43.9% → **2.3%** 命中; 但 long_scoreboard
回升 18.4%→44.3%（4.31 cpi）成为第一停顿, DRAM 48.98%→62.34% ——
剩余瓶颈判定为 W 侧 DRAM 延迟。代价: 80 寄存器, 占用率 84.85%→
62.83%。

**0003 rowtile8（NEUTRAL 0.978×, 统计 SLOWER）**: 假设「R=8 把每线程
outstanding W load 加倍（MLP 4→8, 在飞 W 字节 2×4096B×4→8）」→
DRAM 应继续上行。结果: long_scoreboard cpi 4.31→3.343（MLP 确实
生效, 每 warp 停顿缩短）但 **DRAM 62.34%→62.55% 几乎不动**—— 117
寄存器, 占用率 **62.83%→43.09% 塌方**（占用率与寄存器数严格成反比:
80/117 = 0.684 vs 43.09/62.83 = 0.686）。「更多 MLP per thread」与
「更多 resident thread」的边际贡献互相抵消。median 0.9781 落在 5%
带内 → policy NEUTRAL（CI 排除 1.00 不升级 —— 固定政策的活例）。
rowtile8 归档为失败结构（正确但慢, 不隔离）。

**0004 rowtile4_hx（KEEP 1.087×, 当前 incumbent）**: 0003 的对照
实验 —— 保持 R=4 结构（MLP 4, 384 指令/行）但把 x 片段驻留从
`float xw[32]`（32 寄存器）降到 `__half2 xh[16]`（16 寄存器, 逐行
`__half22float2` 转换, 每 v +48 次转换）。**ptxas 实测 80→64 寄存器
（无 spill）, 编译器未把转换 hoist 出 r 循环** → 占用率 62.83%→
**82.78%**, 在飞 W 字节 +24%。结果三口径同向: API 1.0869× / native
w5000 24.238→22.208 = 1.0915× / NCU 27.56→25.176 = 1.0947×;
DRAM 62.34%→**68.84%**（+6.5 点）; long_scoreboard 44.3%→39.4%;
not_selected 7.4%→14.0%（高占用率下 scheduler 争抢, 预期代价）;
math_pipe_throttle 4.0%→6.3%（逐行转换增量, 如预期但未成瓶颈）。
**结论: 占用率是比 MLP 更强的 DRAM 杠杆** —— 0003 与 0004 的受控
对照（同在飞 W 字节增量, 一个走 MLP 失败, 一个走占用率成功）是
本版本最重要的机制证据。

**政策链**: baseline → vec16_row（0001 KEEP）→ rowtile4（0002 KEEP）
→ **rowtile4_hx（0004 KEEP, 当前 incumbent）**。

## 6. Incumbent 决策与主目标复核

政策 incumbent = `int4gemv_rowtile4_hx`（自 INT4GEMV-0004 KEEP 起）。
全矩阵中该变体 6/10 格 SIGNIFICANT_WINNER（(4096,1024)/(4096,4096)/
(4096,11008) 双 mode, 1.05–1.17×）, 其余 4 格 NO_UNIQUE_WINNER:
(1024,4096) 双 mode 与 rowtile4 打平（±0.3%, CI 含 1.0）,
(11008,4096) 双 mode FASTER +4.7% < 5% 政策带 → NEUTRAL —— 均不触发
incumbent 变更条件（变更只由 5% KEEP 决策驱动）。主目标三口径终值（4096², streaming, fp16）:

| 口径 | baseline | vec16_row | rowtile4 | rowtile8 | **rowtile4_hx** |
|---|---|---|---|---|---|
| API 路径 (µs) | 53.197 | 26.766 | 22.232 | 22.693 | **20.503** |
| native w5000 (µs) | — | 25.000 | 24.238 | 24.768 | **22.208** |
| NCU ccall clkbase (µs) | 66.432 | 33.352 | 27.56 | 28.536 | **25.176** |

## 7. 三种计时口径与一致性（用户 §7: 冲突必查, 不许挑好看的）

三口径定义（v0.5/v0.6 沿用）: **API 路径** = bench 引擎经
Python↔C++ 边界, 每样本 32 连发, streaming 池（W L2-cold 轮换）;
**native kernel-loop** = C++ 内连续 raw launch, CUDA events / 64,
10 windows（无 Python 边界, W 部分 L2-warm）; **NCU** = profiler
replay 纯 kernel 时间（cc=all 每 launch 前 flush, clkbase 锁 1350
MHz）。三者不混用, 冲突时记录并调查。

主目标（rowtile4_hx 及其前代）三口径一致性:

- **API vs native**: 本 session 的 regime 差已知且方向一致 —— API
  streaming 池保持 W L2-cold, native 连续循环部分 L2-warm; 差距
  0001 2.0×→1.087× 各代 3–19% 不等（rowtile4: API 22.232 vs native
  24.238 —— native 反而略慢, 因 native 循环 W 工作集 8.4MB > L2
  5.5MB 持续 thrash; rowtile4_hx: API 20.503 vs native 22.208 =
  -8.3%）。**两口径排序在所有变体间完全一致**（hx < rowtile4 <
  rowtile8 < vec16_row < baseline, 三口径全同序）, 决策不受口径
  选择影响。
- **NCU vs native**: rowtile4_hx 25.176 vs 22.208 µs,
  observed NCU profiling perturbation ≈ **+2.97 µs / +13.4%**
  （本 workload 观测值, 不声称
  跨 workload 固定常数 —— v0.5 ~3.8 µs / v0.6 ~4.4 µs 仅量级参照）。
- 所有实验记录同时保存三口径原始数据; 无任何「挑最好看数字」路径
  （decision 只用 API pair 口径, 固定）。

## 8. 性能对比总表（用户 §8: 三代同 session, fresh 测量）

@4096×4096 fp16 streaming, 2026-09-22 同 GPU 同 harness。
**逻辑 IO（每 launch 一次, 三代统一口径 = W(+scale) + x K·2B +
out N·2B）**: FP16 33,570,816 B（W N·K·2B + x + out）/ INT8
16,809,984 B（W N·K + scale N·4B + x + out）/ INT4 8,667,136 B
（W_packed N·K/2 + scale N·(K/128)·2B + x + out）。**W 字节单独**:
FP16 33,554,432 / INT8 16,777,216 / INT4 8,388,608 → **W 字节
INT4/INT8 = 恰好 2.0×, INT8/FP16 = 恰好 2.0×**（"理论 2×" 指此
W 字节减半）; 含 scale/x/out 的全逻辑 IO 理想比则因 INT4 的
group-scale 开销（N·(K/128)·2B = 262,144 B ≫ INT8 行 scale
N·4B = 16,384 B）而 INT4/INT8 = **1.940×**、INT8/FP16 = 1.997×。
**算法 BW（下表）= 全逻辑 IO / 时间, ≠ 实测 DRAM BW**（逻辑值 =
理想无冗余流量; NCU dram% 是实测 DRAM 吞吐, 后者才是物理证据）。

| 实现 | API-path (µs) | native w5000 (µs) | NCU @base cc=all (µs, dram%) | 逻辑 BW (API) | 效率 @616 GB/s |
|---|---|---|---|---|---|
| FP16 `gemv_vec4_row`（v0.5 incumbent） | 59.977 | 59.673 | 64.24 (89.3%) | 559.7 GB/s | 90.9% |
| INT8 `qgemv_vec16_row`（v0.6 incumbent） | 31.931 | 31.486 | 36.432 (86.55%) | 526.4 GB/s | 85.5% |
| **INT4 `int4gemv_rowtile4_hx`（v0.7 incumbent）** | **20.503** | **22.208** | **25.176 (68.84%)** | **422.7 GB/s** | **68.6%** |

表中三代 NCU 值均为 **v0.7 fresh pass**（`profiles/gen3_v0.7/`,
2026-09-22）。v0.7.1 merge 修复后 `profiles/gemv|qgemv/` 历史路径已恢复
v0.5/v0.6 发布原值（gemv ccall 64.192 µs / 87.89%、ccnone 63.808 µs /
87.91%; qgemv ccall 36.368 µs / 86.02%）—— 见
`profiles/gen3_v0.7/README.md`。

**主结果**:
- **INT4 vs INT8 = 1.557×**（API）/ 1.418×（native）/ 1.447×（NCU）
  —— **理论 2.0× 的 78%, 未接近 2×**。
- INT4 vs FP16 = 2.925×（API）/ 2.687× / 2.552×。
- INT8 vs FP16 = 1.878×（API; 理论 2.0× 的 94% —— v0.6 结论
  1.90× 的 fresh 复测确认）。

**INT4 距理论 2× 的缺口分解（两层视图, 全部可从 raw 归档复算）**:

*层 1 — Algorithmic 视图（理论流量削减, 不等同于任何实测 speedup）*:
全逻辑 IO INT8/INT4 比 = 16,809,984/8,667,136 = **1.940×**
（W 字节单独 = 恰好 2.0×; 差 0.06× 来自 INT4 group-scale 262,144 B
≫ INT8 行 scale 16,384 B）。

*层 2 — Measured NCU 物理视图（同口径分解, NCU kernel duration）*:
指令构成 pass raw CSV（`profiles/int4gemv/gen3_pipe_{qgemv,int4}.csv`,
12 指标 × 4 launch, ncu CSV export, 以下均为 4-launch 均值）:
- 物理 DRAM 流量 `dram__bytes.sum`: INT8 = **18,435,144 B**
  （逻辑 16,809,984 B, +9.7%）; INT4 = **10,262,584 B**
  （逻辑 8,667,136 B, +18.4%）;
- **物理流量削减 = 18,435,144/10,262,584 = 1.7963×**;
- 实际物理带宽（物理流量 ÷ NCU 主 pass duration,
  `profiles/gen3_v0.7/`）: INT8 = 18,435,144 B/36.432 µs =
  **506.0 GB/s**; INT4 = 10,262,584 B/25.176 µs = **407.6 GB/s**;
  物理带宽比（INT8/INT4）= **1.2413×**（即 INT4/INT8 = 0.8056×）;
- **物理流量削减 × 物理带宽比 = 1.7963 × 0.8056 = 1.4471× =
  NCU duration speedup 36.432/25.176 = 1.4471×**（同口径恒等式,
  可从 raw 数据直接复算）。若改用 CSV pass 自身 duration（4-launch
  均值 36.448/25.496 µs）: 物理带宽 505.8/402.5 GB/s, NCU speedup
  1.4296× —— 方向与结论不变。

*API 路径（单独口径, 不做跨口径精确分解）*: INT4 vs INT8 API =
20.503/31.931 = **1.557×**（native 1.418×）。API 口径含 Python↔C++
边界与 streaming 池（W L2-cold）regime, 与 NCU kernel duration 不同
（§7 已给出 API/native/NCU 三口径差及解释）。上节 NCU 物理分解解释
**NCU 口径** speedup,
**不用于**精确解释 API latency。

物理带宽差距（INT4 407.6 vs INT8 506.0 GB/s）的候选组成（标注事实
与假设）:
1. **实测 DRAM 流量超逻辑 18.4%（实测事实）**: NCU
   `dram__bytes.sum` INT4 10.26 MB vs 逻辑 8.67 MB（+1.59 MB; FP16
   +4.8% / INT8 +9.7%, 同 pass 对照; pass 原始 CSV 归档
   `profiles/int4gemv/gen3_pipe_*.csv`, 12 指标 × 4 launch, 百分比
   为 4-launch 均值, 可复算）。**归因是假设而非结论**: 与 x (8 KB)
   被 1024 个 block 各读一次、W 流 (8.4 MB > L2 5.5 MB) streaming
   逐出 x 导致 DRAM 再取（~196 block 量级）的假设一致; 也可能包含
   DRAM transaction / cache-line 粒度等额外流量, 需控制实验确认
   （v0.8 调查项见 §11 Q7）。
2. **指令开销（指令数为实测事实, 机制为假设）**: 整数线程指令
   INT4 46.33M vs INT8 38.93M = **+19%**（每 product 2.76 vs 2.32
   条; 增量 = 32 nibble 解包/向量（2 整数指令/nibble）+ group scale
   索引）; ALU pipe 36.58%（INT8 相当管线上移但未饱和）。更长的整数
   指令流可能降低每 SM 单位时间可维持的在飞 W load（未独立验证）,
   是物理带宽差距的候选组成因素。

## 9. 全 shape 矩阵与回归

### 9.1 矩阵（5 形状 × {hot, streaming} × 5 变体 × 9 轮, tag `v0.7`）

中位数 µs（API 路径, streaming; `benchmarks/int4gemv/v0.7_*.json`
+ `v0.7_shape_winners.json`, classify_cell 派生）:

| (N, K) | baseline | vec16_row | rowtile4 | rowtile8 | **hx** | 最低中位数 |
|---|---|---|---|---|---|---|
| (1024,4096) | 11.251 | 8.404 | 6.936 | 7.360 | **6.906** | hx（+0.2% vs rowtile4, CI 含 1.0） |
| (4096,1024) | 16.377 | 10.496 | 10.514 | 10.944 | **9.027** | hx (+16.4% vs vec16_row) |
| (4096,4096) | 53.197 | 26.766 | 22.232 | 22.693 | **20.503** | hx |
| (11008,4096) | 136.997 | 67.130 | 48.685 | 53.554 | **46.479** | hx (+4.8% vs rowtile4) |
| (4096,11008) | 131.963 | 63.651 | 52.203 | 53.376 | **48.772** | hx (+6.9% vs rowtile4) |

（上表为 streaming 口径; (1024,4096) hot 行: 11.026 / 8.238 /
**6.762** / 7.248 / 6.784 —— 全矩阵 10 格中唯一 hx 非最低中位数的
格, hot 最低为 rowtile4, winners JSON ratio 1.0031×。）

classify_cell: hx **SIGNIFICANT_WINNER 6/10 格**（(4096,1024)/
(4096,4096)/(4096,11008) 双 mode, 1.054–1.166×, FASTER+KEEP）;
(11008,4096) 双 mode FASTER 1.047–1.048× < 5% 政策带 → policy
NEUTRAL → NO_UNIQUE_WINNER; (1024,4096) 双 mode rowtile4≈hx 打平
（hot 1.0031× / streaming 1.0022×, CI 含 1.0, 交替最低）→
NO_UNIQUE_WINNER。hx 为 9/10 格最低中位数（唯一例外 (1024,4096)
hot: rowtile4 低 0.31%）。hot/streaming 排序一致（hx 领先幅度
streaming 略大, 与 W L2-cold 假设同向）。失败结构 rowtile8 /
vec16_row 10/10 格非获胜者（与 0002/0003 一致）。

**K=1024 打平的解释**（§11 Q6 详述）: nvec = 32 < 128 线程, x 片段
只占 32 个线程, 省下的 16 寄存器不再改变占用率结构; hx 的逐行转换
增量相对占比上升 → 两结构互有 ±0.3% 内微弱优势。

### 9.2 回归 smoke 与 quarantine 审计（`experiments/regression/v0.7/`）

`$PYTHON scripts/regression_v07.py`（可复现, append-only）:
- **int4gemv**: 5 变体 correctness 50/50 + negative 30/30 全绿
  （quarantine 集为空, 机制保留）;
- **gemv**: 4 正常变体 100/100 + 24/24 全绿; `gemv_splitk4` 保持
  隔离（正常列表缺席, all_variants 保留, 未运行）;
- **qgemv**: 5 变体 50/50 + 29/29 全绿;
- `quarantine_audit.json`: 3 算子绑定列表审计, 泄漏检查全空;
  `softmax_hsplit2`（v0.4 起隔离）与本分支算子集无关, 状态以 main
  上 v0.5/v0.6 审计为准。

## 10. 独立 review（两路, 均为独立 subagent, 与本 session 主工作流隔离）

两路 review 均在本版本全部 GPU 测量与文档定稿**之后**发起, 独立
subagent 执行, 纯代码/JSON 审计（不运行 GPU 命令, 不触碰 CUDA 脚本）,
与本 session 主工作流隔离。

### 10.1 CUDA correctness 独立评审

**范围**: `kernels/int4gemv/` 全部 7 文件（common.h + bindings.cpp +
5 .cu）+ `cudalab/int4gemv_quantize.py` / `int4gemv_correctness.py` /
`int4gemv_negative.py` + `cudalab/operators/int4gemv.py` +
`tests/test_int4gemv_cpu.py` + `cudalab/build.py`, 并对照 v0.5
`gemv_common.h` / v0.6 `qgemv_common.h` 先例。

**结论: 无阻断项、无建议级正确性问题 → 可进入外部评审。** 分节核验
（A–F）全通过:

| 节 | 核验对象 | 结论 |
|---|---|---|
| A | nibble unpack 映射 | 全部 OK。common.h 解包公式与用户契约逐字相同（lo=(int8_t)((b&0x0F)<<4)>>4, hi=(int8_t)(b)>>4）, 256 全域推演符号扩展正确无分支；四变体 q↔x 配对逐一推演无误；**单 group 引理 g=v>>2 边界证明成立**（32v mod 128 ∈ {0,32,64,96} 最大 96 < 97, 恒单 group, 与对齐无关的纯算术）；行距 K/2 是 64 倍数 → 行基址保持 16B 对齐 |
| B | rowtile4_hx vs rowtile4 数值等价 | OK（预期 bit-identical）。逐 term 顺序完全相同；`__half22float2` 为精确拓宽（fp16→fp32 无舍入）→ 逐行重转与一次性转换逐位相同；两级归约（5 步 shfl + shared[4][4] + 2 步 shfl + `__float2half_rn`）逐位同构, FMA 收缩决策一致。层 A 独立核验（max_arith_max_ratio 5 变体一致 0.245053042161）与该推演互证, docstring 明确不做交叉等价假设 |
| C | 边界 | 全部 OK。N%4≠0 末 block 双 guard（累加循环 + 归约写路径, row>=N 在任何解引用前 continue）; K=128 (nvec=4<128) 空线程 acc≡0 无死锁; K%128≠0 三入口（forward/forward_into/native_timing）硬拒绝先于其余检查; 对齐回退四变体 fallback 块逐字同一 `launch_int4gemv_scalar` 单一来源 → bit-identical 由构造保证; 1 字节 offset 连续视图 → 回退非拒绝（negative 钉死）; 契约检查先于 cast/launch |
| D | bindings.cpp host 校验 | 全部 OK。Wp/scale/x/out 元数据校验完整（dim/contig/dtype/device/同设备）; **无 D2H 泄漏**（validate 全元数据, 无 .item()/.cpu()/sync; 量化器内同步仅池构造期, 合同明确计时区外）; native_timing 内 synchronize 是 event 测量协议必要步骤非数据验证 |
| E | UB / 数据竞争 | OK。warp_sums 单写者 → syncthreads → 仅 warp0 读, 无竞争; `__shfl_down_sync 0xffffffff` 全调用在统一控制路径（warp0 二次归约全 lane 同进, 无 warp 内分支）→ 全掩码安全; reinterpret_cast 对齐仅在契约通过后; 无未初始化读（acc 显式清零, q/xw/xh 全量写入后才读, 全部索引经循环界 + guard 验证在界内） |
| F | quantize.py 互逆性 | OK。pack/unpack 全域 [-8,7] 互逆推演成立; 设备端 unpack 与 CPU 镜像为同一符号扩展函数; group 索引 CPU k/128 = 设备 b>>6; 量化 fp32 域 + round-half-even + clamp ±7, 零 group 双保险无 NaN; **fp16 scale 同值**贯穿三方（kernel `__half2float` / 层 A `scale.float()` / arith `scale.double()` 精确读同一存储值） |

**备注 2 条（非问题, 无需处置）**:
1. `int4gemv_vec_acc_unpack` 的 union 写 v 成员后读 b/h 成员惯用法属
   严格 C++ active-member 灰区, 但为**标准 CUDA idiom**, 与 v0.5
   `gemv_common.h` U16 / v0.6 `qgemv_common.h` U16Q 完全同模式, 三代
   算子 GPU 实证 —— 不处置（与 v0.5/v0.6 处置一致）。
2. `int4gemv_rowtile8.cu:69` 注释「转 float[32] 一次, 跨 R=4 行复用」
   为 rowtile4 复制残留（本变体 R=8）—— 纯注释笔误, 无代码影响,
   不处置（rowtile8 为归档失败结构, 不改动保留历史原样）。

**决策参考注（非本审计范围, 非阻断）**: rowtile8（NEUTRAL 归档）保留
在正常 dispatch 列表且 int4gemv quarantined_set 为空 —— 与 v0.6 先例
一致（quarantine 机制仅用于 REJECTED/UNSAFE, v0.6 qgemv 隔离集亦为空;
机制在 `bindings.cpp` quarantined_set + 头注释中保留可用）。属
dispatch 政策选择, 不影响正确性结论, 不处置。

### 10.2 Benchmark methodology 独立审计

**范围**: INT4GEMV-0001..0004 证据链 + 5 形状 × {hot,streaming} 全
矩阵 + 三代（FP16/INT8/INT4）对比一致性; 纯 JSON/代码审计, 零 GPU
命令。方法: 所有 pair/矩阵/native/NCU 中位数由原始轮次数据重算
（per-round ratio 的 median, 非记录副本）; bootstrap CI95 用本仓库
`cudalab.evaluator.stats.bootstrap_ci`（seed 20260919）对 raw rounds
重跑逐位比对; `classify_cell`（min_valid_rounds=5）对 10 个矩阵记录
全量重跑并与 winners JSON 逐字段 deep-diff; 三种计时口径全程分开
核对。

**结论: PASS WITH CAVEATS —— 无阻断项**, 可支撑最终报告结论。

| 检查点 | 结果 |
|---|---|
| 1. 4 × pair decision vs raw | **一致**。9/9 轮 median/faster_frac 重算精确一致; bootstrap CI95 逐位一致; 0001 2.000× KEEP / 0002 1.188× KEEP / 0003 0.978× SLOWER→policy NEUTRAL（固定 5% 带政策既定行为, 非 bug）/ 0004 1.087× KEEP; parent/candidate 中位数与同源 pair JSON 逐轮一致（无挑数） |
| 2. winners JSON vs 矩阵 raw | **一致**。`classify_cell` 重跑 10/10 格逐字段相等（winner/runner_up/ratio/CI95/status/policy/statistical_relation）; 分布 6/10 SIGNIFICANT_WINNER + 4/10 NO_UNIQUE_WINNER; 必查 4 格（(4096,4096)×2, (1024,4096) hot, (11008,4096) streaming）独立复核通过 |
| 3. 三代口径一致性 | **口径全过**。API（harness/seed/warmup/iters/batch/clock_policy 逐字相同, 时间戳相邻）/ native（windows=10, w200+w5000 均在档）/ NCU（cc=all clkbase, 4 launch, raw 行均值 = JSON）逐数重算精确相等; §8 主结果 1.557×/2.925×/1.878×、78%/94% 复算全中; 发现 MAJOR-1 + MINOR-3（见处置表） |
| 4. 挑数迹象 + 政策语义 | **一致（无挑数）**。不利数据全在档（int4 baseline w200 61.888 vs w5000 50.579 warmup gap; rowtile8 全面慢仍入档入矩阵）; 政策语义双活例: rowtile8 SLOWER→NEUTRAL、(11008,4096) FASTER→NEUTRAL, stat/policy 严格分离, 无「CI 显著即覆盖 5% 带」的事后规则; 报告 §1 同时呈现矩阵口径 20.50 与 pair 口径 19.894 |
| 5. 实验顺序合规 | **一致**。parent 链 0001→0002→0003→0004 正确, 0003 NEUTRAL 后 incumbent 未变; 各轮 hypothesis 引用的前轮 profile 数字（long_scoreboard 64.2% / lg_throttle 43.9% 6.555 cpi / 18.4% / 44.3% 4.31 cpi / 2.3% / DRAM 62.34% / SM 46.95% / 80 寄存器 / 占用率 62.83%→43.09% / 62.55%）与归档 NCU 全等; 唯一例外 0004 的「issue 45.80%」（NIT-1） |

**发现处置表**:

| 编号 | 级别 | 发现 | 处置 |
|---|---|---|---|
| MAJOR-1 | Major（不阻断） | §8 item 2/3 引用的 NCU 补充 pass 指标（`dram__bytes.sum` / 整数指令 / ALU pipe）在分支上无归档原始输出; 反推 peak 不一致（593.7/582.0/610.0 GB/s —— 系 duration×dram%×peak 近似反推, 非计数值） | **已修复（本 commit）**: pass 原始 CSV 归档 `profiles/int4gemv/gen3_pipe_{gemv,qgemv,int4}.csv`（12 指标 × 4 launch, ncu CSV export）; §8/Q3/§12 已加归档指针; 全部百分比 = 归档 CSV 的 4-launch 均值, 可独立复算。决策链不受影响（所有 KEEP/NEUTRAL 只依赖已验证 paired bench 与已归档 dram%） |
| MINOR-1 | Minor | 报告 §9.1 曾写「hx SIGNIFICANT_WINNER 8/10 格」（statistical FASTER 格数与 policy status 混标, 恰为 v0.6 审计修正过的同类） | **已修复**（审计进行中发现, 全部 v0.7 文档改 6/10 SIGNIFICANT_WINNER + 4/10 NO_UNIQUE_WINNER; 审计独立重算确认该分布; v0.6 历史段落「8/10」为 v0.6 自身数字, 保留） |
| MINOR-2 | Minor | §9.1 表格标题 streaming 但 (1024,4096) 行误填 hot 中位数（11.026/8.238/6.762/7.248/6.784; streaming 实为 11.251/8.404/6.936/7.360/6.906, 其余 4 行 20/20 值正确） | **已修复（本 commit）**: 该行改 streaming 值, 最低中位数 hx 6.906（rowtile4 6.936 略高, winners ratio 1.0022× CI 含 1.0 → UNRESOLVED）; 补 hot 行脚注; 政策结论不变（双 mode NO_UNIQUE_WINNER）, 「hx 9/10 格最低中位数, 唯一例外 (1024,4096) hot」表述经核验保持正确 |
| MINOR-3 | Minor | INT8 逻辑字节 16,783,872 B 与自述公式（16,793,600）及 harness 口径（16,809,984）均不符, 疑似 16.794→16.784 数字换位（影响 ≤0.1%） | **已修复**: 本报告 §8 统一用 harness 口径 16,809,984（= W 16,777,216 + scale 16,384 + x + out）并声明口径; b2580d6 commit message 中的历史数字不重写（no force push）, 以本报告为准 |
| NIT-1 | NIT | 0004 hypothesis「rowtile4 … issue 45.80%」不见于任何归档 profile（归档 rowtile4 sm_throughput = 46.95%）, 推测来自某次未归档 NCU pass 的 SM Issue 读数 | **不修复**（证据文件不可变）: 该数为 hypothesis 中的次要引用, 不参与决策（0004 决策依据 = 80→64 寄存器 / 占用率 62.83%→82.78% / DRAM 62.34%→68.84%, 全部归档且经本审计核验）; INT4GEMV-0004.json 作为预注册证据保留原样 |
| NIT-2 | NIT | §8 「逻辑 BW」列与正文字节行口径漂移 ≤0.1% | **已修复**: §8 口径声明统一为「算法 BW = 全逻辑 IO（W(+scale)+x+out）/ 时间」, 表列与正文一致 |
| 过程 a | 过程 | 报告曾写「分支已 push」, 早于实际 push | **已修复**（本 commit 修正 §1/§12 措辞; 本 commit 即首次 push） |
| 过程 b | 过程 | b2580d6 就地重写了 3 个 v0.5/v0.6 时代 profile JSON（gemv_vec4_row ccall 64.192→64.24, ccnone 同步; qgemv_vec16_row ccall→36.432） | 就地覆盖违反 historical-artifact 不可变约定; **v0.7.1 merge 修复已处置**: 3 个历史文件恢复为 main 原版本, fresh 值迁移至 `profiles/gen3_v0.7/`（append-only, 见该目录 README）; 本报告三代 NCU 数字一律以 `profiles/gen3_v0.7/` fresh 值为口径 |

**审计确认的关键不变量**: 决策链证据完整且可复现（4/4 pair
decision 含 seed 20260919 bootstrap 逐位复算、10/10 winners 格
逐字段相等、三代三口径逐数可复算）; 无挑数; parent 链合规; 政策
语义正确。本 commit 对 §9.1 表格与 §8 的修正均以审计独立重算值为
准。

## 11. 用户 7 问（逐条回答）

**Q1: INT4 vs INT8 实际加速比?**
**1.557×**（API, 4096² streaming, 本 session fresh: 20.503 vs
31.931 µs; native 1.418×; NCU 1.447×）。**未达到理论 2×（78%）**。
两层视图（§8）: (1) algorithmic 视图 —— 逻辑 IO 减半 **1.940×**
（W-only 2.0×）是理论流量削减; (2) measured NCU 物理视图 ——
物理流量削减 **1.7963×**（18,435,144 B vs 10,262,584 B, raw CSV
4-launch 均值）× 物理带宽比 **0.8056×**（506.0 vs 407.6 GB/s）=
NCU **1.4471×**。API 1.557× 单独报告（与 NCU 不同口径, 不用 NCU
分解精确解释）。物理带宽差距的候选组成: 实测流量超逻辑 18.4%
（+1.59 MB; 归因为假设: x 跨 block 重读 / W streaming 的 L2 再取,
亦可能含 transaction 级流量, 需控制实验确认, 见 §11 Q7）+ 指令开销
（+19% 整数线程指令; 机制为假设）。

**Q2: INT4 vs FP16 实际加速比?**
**2.925×**（API: 20.503 vs 59.977 µs; native 2.687×; NCU 2.552×）。
三代链: FP16 59.98 → INT8 31.93（1.878×）→ INT4 20.50（2.925×）。
逻辑 IO 33.55 → 16.78 → 8.67 MB; 实测效率 90.9% → 85.5% → 68.6%
（INT4 的效率折损吃掉了约 1/4 的理论收益）。

**Q3: nibble 解包代价多大?**
量化（NCU 指令构成 pass, cc=all clkbase, 4 launch 均值, 4096²; 原始
CSV `profiles/int4gemv/gen3_pipe_*.csv`）:
整数线程指令 FP16 48.50M / INT8 38.93M / **INT4 46.33M** —— INT4 比
INT8 **+19%（+7.4M）**, 每 product 2.76 vs 2.32 条整数指令; 增量
来源 = 每 16B 向量 32 次 nibble 提取（低: AND+SHL, 高: SHR, ~2 整数
指令/nibble）+ group scale 索引。ALU pipe 占用 36.58%（未饱和;
INT8 侧同族结构更低）。**结论: 解包不是第一瓶颈**（第一瓶颈是
DRAM 效率/long_scoreboard 39.4%）, 但它抬高了每线程指令流长度,
间接压低每 SM 的在飞字节数, 是 INT4 DRAM 效率低于 INT8 的组成
因素之一。另: 0004 的 half 驻留 x 引入的逐行转换（每 v +48 次
`__half22float2`）把 math_pipe_throttle 4.0%→6.3% —— 占用率收益
（+20 点）远大于该代价（+2.3 点停顿）。

**Q4: group scale 查找是瓶颈吗?**
**不是**（G=128 下）。单 group 引理: 16B W 向量覆盖 32 连续 k,
恒在单个 128-group 内（g = v>>2; 边界 56+7<64 已证）→ 每 32 个
product 只 1 条 2B scale load, 且 64 B/行 的 scale 行 L1 常驻。
NCU 证据: short_scoreboard（L1/shared 依赖）在 hx 上仅 4.5%
（rowtile4 6.9%, 下降）, 无 scale 专属停顿; scale 总流量 0.26 MB
（逻辑的 3%）。G=128 是保真度与查找开销的合理折中（G=256 会把
scale 流量再减半但保真度下降, 不在本版本范围）。

**Q5: DRAM 吞吐 / long scoreboard / 指令停顿如何变化?**
NCU 轨迹（4096², ccall clkbase, 本 session）:

| 变体 | duration (µs) | DRAM% | 第一停顿 | 寄存器 | 占用率 |
|---|---|---|---|---|---|
| baseline | 66.432 | 25.44% | long_scoreboard 64.2% (9.795 cpi) | 26 | 90.9% |
| vec16_row | 33.352 | 48.98% | **lg_throttle 43.9% (6.555 cpi)** | 48 | 84.85% |
| rowtile4 | 27.56 | 62.34% | long_scoreboard 44.3% (4.31 cpi) | 80 | 62.83% |
| rowtile8 | 28.536 | 62.55% | long_scoreboard 47.8% (3.343 cpi) | 117 | 43.09% |
| **rowtile4_hx** | **25.176** | **68.84%** | long_scoreboard 39.4% (4.178 cpi) + not_selected 14.0% | 64 | **82.78%** |

两次 lever 轮换: （1）lg_throttle（x 重发的 LSU 发射压力）→ x 寄存器
驻留（0002）; （2）long_scoreboard（W DRAM 延迟）→ 占用率（0004,
0003 证明同方向走 MLP 无效）。DRAM 吞吐单调 25.4→49.0→62.3→62.6→
68.8%。**当前墙 = DRAM ~69%**, 距 INT8 的 86.55% 仍有 18 点空间,
候选组成两项（§8: +18.4% 实测流量冗余[归因为假设] + +19% 整数指令
开销[机制为假设]）。

**Q6: K=1024/4096/11008 各适合哪种结构?**
- **K=4096（nvec=128, 128 线程全活）**: `rowtile4_hx` 最优
  （vs rowtile4 +7.9–8.7%（矩阵 1.079–1.083× / 0004 pair 1.0869×）,
  vs vec16_row +16.4–30.5%）—— R=4 行块 + half x 驻留是占用率
  lever 的甜点。
- **K=1024（nvec=32, 仅 32/128 线程持片段）**: **平地区域** ——
  rowtile4 与 hx 打平（±0.3%, 双 mode NO_UNIQUE_WINNER）。寄存器
  驻留节省不再改变占用率（x 片段只占 32 线程）, 而 hx 的逐行转换
  相对成本上升。结构上真正缺的是「128 线程中 96 个空转」的
  线程利用率（如 32 线程/block 或 split-K）—— 超出本版本 4 组
  实验范围, 记录为 v0.8 候选。
- **K=11008（nvec=344 = 2×128 + 88, 末段 88/128 线程活）**: hx 仍
  领先 rowtile4 +4.8–7.0%（(11008,4096) +4.8% 差 0.2 点即达 5%
  政策带, 如实报告 NEUTRAL）; 末段线程不均衡（~14%）是唯一可见
  结构瑕疵, 未值得单独实验。
- N 维度: R=4 行块 + 末 block guard 在所有 N 上无瑕疵。
- **部署建议**: 单一 `rowtile4_hx` 覆盖全部 5 形状（9/10 格最低
  中位数, 唯一例外 (1024,4096) hot 与 rowtile4 差 0.31% 打平）。

**Q7: 继续 INT4 优化还是启动 CUDALM 集成?**
**建议启动 CUDALM 集成, 以 `int4gemv_rowtile4_hx` 为交付内核**;
INT4 结构性优化在 v0.7 收尾。依据:
1. 高 ROI 结构 lever 已耗尽 —— 4 组实验收益递减 2.0× → 1.188× →
   1.087×, 且 0003 证明 MLP 方向已死;
2. 剩余已量化 lever 的期望收益有限或需新机制: （a）+18.4% DRAM
   流量冗余 —— 归因尚未证实（假设: x 跨 block 重读 / W streaming
   的 L2 再取; 亦可能含 transaction/cache-line 额外流量）。**当前
   硬件 Turing sm_75 不支持 Ampere+ 的 cudaAccessPolicyWindow /
   persisting-L2, 不可直接执行** —— v0.8 该项 = Turing 兼容的
   cache 行为调查: 区分 x refetch vs transaction 开销、NCU 检查
   load/cache 行为、如有依据再考虑 PTX/cache-policy hint（期望
   5–10%, 不保证）; （b）指令开销（解包 ALU 融合, 需编译器级
   控制, 期望 <5%）; （c）K=1024 线程利用率（需新结构, 仅影响
   1/5 形状）;
3. incumbent 稳定（6/10 格 SIGNIFICANT_WINNER, 9/10 格最低中位数,
   余 1 格 rowtile4 低 0.31% 打平、(11008,4096) 双格 +4.7% 子 5%
   带——全部无劣化）, 正确性
   三层 + 负例 + 回归全绿, 满足集成准入。
(a)–(c) 作为 v0.8 backlog 记录（本报告 §8/§11 已给出量化基线）,
CUDALM 集成后若 W4A16 层在端到端负载中暴露新的瓶颈形状, 再按
profile 驱动原则重开实验。

## 12. 产物索引

- 内核: `kernels/int4gemv/`（common.h + bindings.cpp + 5 .cu）
- 量化/正确性/负例: `cudalab/int4gemv_{quantize,correctness,negative}.py`
  + `cudalab/operators/int4gemv.py` + `tests/test_int4gemv_cpu.py`（39/39）
- 实验记录: `experiments/int4gemv/INT4GEMV-000{1..4}.json`
  （decision 字典含 statistical_relation/policy_decision/ci95/
  speedups/rule）+ `correctness/v0.7/` + `native_timing_*.json`
- 基准: `benchmarks/int4gemv/`（4 × pair JSON + 10 × 矩阵 JSON +
  `v0.7_shape_winners.json`）; 三代参照 `benchmarks/gemv/v0.7.json` /
  `benchmarks/qgemv/v0.7.json` + 两侧 native timing JSON
- NCU: `profiles/int4gemv/*_M4096_H4096_ccall_clkbase.json`（5 变体;
  raw 目录 gitignored）+ **三代 v0.7 fresh 参照
  `profiles/gen3_v0.7/`（5 JSON + README, append-only; §8/§11 三代
  NCU 数字一律以该目录为口径）** + 指令构成补充 pass 原始 CSV
  `profiles/int4gemv/gen3_pipe_{gemv,qgemv,int4}.csv`（12 指标 ×
  4 launch, ncu CSV export; ncu 命令行 + driver 见 commit b2580d6
  消息）。`profiles/gemv|qgemv/` 历史路径保留 v0.5/v0.6 发布原值
  （v0.7.1 恢复）
- 回归: `experiments/regression/v0.7/`（README + summary.json +
  quarantine_audit.json + 3 算子子目录）+ `scripts/regression_v07.py`
- 三代测量: `scripts/gen3_native_timing.py`
- git: 分支 `v0.7-int4-gemv`（未 merge main; 本 commit 后 push 即
  STOP 等外部评审）
- 溯源注（v0.7.1 更新）: commit b2580d6（三代对比）当时按「fresh
  测量不沿用历史」协议就地重写了 3 个 v0.5/v0.6 时代 profile JSON
  （gemv_vec4_row ccall 64.192→64.24 µs, ccnone 同步; qgemv_vec16_row
  ccall→36.432 µs）; **v0.7.1 merge 修复已把这 3 个文件恢复为 main
  原版本**（gemv ccall 64.192 / ccnone 63.808 / qgemv ccall 36.368
  µs 等）, v0.7 fresh 值迁移至 `profiles/gen3_v0.7/`（append-only,
  详见该目录 README）。两个版本在 git 历史中均可追溯（fresh 值见
  b2580d6, 历史原值见 main）。**约定: historical artifacts =
  immutable; fresh cross-generation measurements = append-only。**
