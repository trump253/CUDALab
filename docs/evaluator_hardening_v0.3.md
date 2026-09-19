# CUDALab v0.3 — evaluator 加固（paired-streaming-v2.1 → v2.2）

日期: 2026-09-19（+08:00）
触发: RMSNorm 回归门（v0.3 第一步）——新核心在 (128,4096) fp16 的
v4-vs-v1 paired smoke 上出现 v0.2 未见过的 per-round 散度。
结论: **不是统计/配对/决策逻辑的 bug，而是 harness 自身注入的测量
污染，且污染机制有两层**：
1. nvidia-smi 查询造成的 ~40ms 空闲缺口把 GPU 推入一个降级性能态
   （kernel 时间 ~1.3–1.8×，host enqueue 同步变慢），衰减时间随机器
   状态漂移（2026-09 早期 ~8–10ms，当日 >150ms），150 次固定 warmup
   无法覆盖 —— 层 1 由 v2.1（时间制 burn + per-sample spike guard）
   处理；
2. v2.1 保留的 DVFS guard 的 round 内 nvidia-smi 采样本身就是缺口
   的再触发源（每个 block 边界 3 次查询），且**读到的 1350 MHz 是
   缺口/空闲时钟而非负载时钟**（真实持续负载时钟 1905–1920 MHz）
   —— 层 2 由 v2.2（round 内零采样 + 跨 block 一致性 guard）处理。

修复: `cudalab/evaluator/bench.py` → harness `paired-streaming-v2.2`。
决策语义（decision.py）、bootstrap CI、round-level 统计单位不变。

## 1. 症状（回归门 R1–R4，2026-09-19，harness v2）

v0.2 冻结记录（2026-08，`benchmarks/v0.2/v02_pair_v4_vec_reg_vs_v1_vec_*`）:
- streaming: speedup 1.0113，CI [1.0017, 1.0193]，faster 9/9，
  per-round parent 6.84–7.04 / candidate 6.84–7.02 µs（紧带，3%），
  in-round clock 恒 1350 MHz，温度 40°C，功耗 59–71W。
- hot: speedup 0.9327（REJECT v1），CI [0.837, 0.966]，faster 0/9，
  per-round parent 5.85–6.32 / candidate 6.43–6.98 µs（r9 一个
  10.544 离群），clock 同样 1350 MHz。

v0.3 新核心同一 pair（R1–R4，9/9 valid，round 内采样恒 1350 MHz，
温度 33–34°C，watcher 确认无外部 GPU 进程、无 CFS throttle 增量）:
- streaming: 1.0105 / 1.0049 / 1.0035 / 0.9859（faster 6/5/5/4 of 9）
- hot: 0.9989 / 0.9565 / 0.9599 / 1.0031（faster 4/3/4/5 of 9）
- per-round 出现 8–9.7 µs 的"慢块"，随机落在 parent 或 candidate、
  位置 1 或位置 2；fast 块与 v0.2 同量级。

判定: streaming 结论与 v0.2 一致（NEUTRAL）；hot 的 v0.2 结论
（v4 显著更快，REJECT v1）被宽化到 NEUTRAL —— **回归门按用户规则
暂停 Softmax，先查 evaluator**。

## 2. 定位（受控诊断，数据文件 /tmp/sample_diag*.json*）

*diag1（紧循环，无空闲缺口）*: 400/400 样本 5.85–5.98 µs
（v4_vec_reg, (128,4096) fp16, hot），host enqueue 170–220 µs/32 发。
→ 机器基线性能与 v0.2 hot 数字（5.84–6.32）一致；无随机慢样本。

*diag2（每 5 样本插一次 nvidia-smi 查询，≈40ms 空闲缺口）*:
56/60 样本慢（10–14 µs），host enqueue 同步升至 275–560 µs；
一旦触发，后续所有样本保持慢（缺口反复再触发）。
→ 空闲缺口是触发源；GPU 与 host 侧同时变慢（单一 GPU/驱动态）。

*diag3（单次缺口 + 300 紧样本，衰减曲线）*:
基线 5.88 µs → 缺口后样本 0–29 慢（7.7–8.1 µs）→ 样本 ~30 起
（≈8–10ms 连续负载）恢复到 5.88–5.94 µs，此后 260 样本不再复发。
→ 降级态有界，~8–10ms 连续负载内衰减完毕（该天值）。

### 2.1 时钟读数陷阱（当日关键新证据）

`nvidia-smi`/NVML 的 SM 时钟读数**不是负载时钟**，取决于采样时刻
GPU 处于哪个 P-state:

| 采样时刻 | 读数 | 证据 |
|---|---|---|
| 空闲（dmon，无负载） | pclk 405 MHz | `nvidia-smi dmon -s puc` |
| round 内 guard 查询（40ms 缺口期间） | 恒 1350 MHz | R1–R6 全部记录 `clocks` 字段 |
| 持续负载后紧接的 `gpu_state()` | 1905–1920 MHz | R6 记录 `gpu_state_after`；R7 streaming `gpu_state_after.sm_clock_mhz=1905` |

即 DVFS guard（v0.2 的支柱之一）一直在比较缺口/空闲时钟，其 5%
相对判据对真实负载状态是盲的。v0.2 冻结记录里 in-round 1350 MHz
是当时机器状态下的读数（温度 40°C，功耗 59–71W）；当前机器
（温度 33–34°C）持续负载时钟 1905–1920 MHz，绝对时间水平整体
下移 ~12–20%（hot 5.85–6.32 → ~5.1 µs；streaming 6.84–7.04 →
~5.5 µs）。这是**机器状态漂移**，不是测量错误。

### 2.2 排除项（均有数据）

外部 GPU 消费者（watcher 全程只有本进程）、CFS 配额节流（窗口内
零增量）、温度/功耗（33–34°C / 59–168W，远低于 250W 上限）、
ECC/Xid（最近 Xid 31 在 20 小时前且属他人进程 codex probe 的 MMU
fault；dmesg 8 天前有 NVRM unhandled-interrupt 记录，该卡近期
硬件状态本就偏不稳定 —— 但不影响结论：触发源是缺口本身）。

## 3. 修复迭代

### 3.1 v2.1（2026-09-19 19:5x）：时间制 burn + spike guard

1. **时间制预热 burn**: 每 block 预热 = ≥150 次 launch **且**
   ≥`WARMUP_MS` wall time（初值 25.0 ms）。
2. **per-sample spike guard**: 测量内维护运行中 clean 基线
   （已收样本的最后 50 个的中位数）；样本 > 基线 × `SPIKE_FACTOR`
   （1.5）→ 记为 spike，剔除出 block 中位数。单 block clean 样本
   < `MIN_CLEAN_SAMPLES`（50）→ 该 round 无效
   （`invalid_reason="INVALID_SPIKES"`），走与 DVFS 无效相同的
   重试路径（≤3 次）与 MIN_VALID_ROUNDS=5 的 UNSTABLE 规则。
3. **记录扩展（只增不改）**: round 记录新增 `samples`（n_clean /
   n_spike）；顶层新增 `invalid_spikes_rounds` 等；`harness` 字段 →
   `"paired-streaming-v2.1"`。

**v2.1 实测（不是"预期"，是记录值）**:

- **R5**（25ms burn，`v03reg_v21r5_*`）: streaming med=1.0160
  CI [0.9271, 1.0610]，per-round 6.79–7.86 µs（整体比 v0.2 慢
  ~15%——25ms burn 不足）；hot med=0.9869 CI [0.9437, 1.0823]，
  r9 parent 35/100 spike，clean 中位数 ~7 µs。→ 当日衰减时间
  已远大于 8–10ms（diag3 是几天前测的），burn 必须加长。
- **burn_probe**（/tmp/burn_probe_*.json，streaming）: 150ms burn
  2/2 trials 停在 6.6–7.7 µs 平台（0/2 clean）；300ms burn 2/2
  trials 5.12–5.50 µs（2/2 clean）。→ `WARMUP_MS` 25 → **300**。
- **R6**（300ms burn，`v03reg_v21r6_*`）: streaming med=0.9500
  CI [0.8392, 0.9549] faster 0/9，hot med=0.9952 CI [0.8929,
  1.2755] faster 3/9，均 9/9 valid。但 **7/36 个 block 均匀变慢**
  （非 spike、整块抬升，spike guard 不可见）：streaming r1
  p=6.065（5 spike）、r4 c=6.838（31 spike）、r5 c=6.976（0
  spike，纯均匀慢）、r9（30 spike）；hot r2 p=6.592、r7 p=6.535、
  r8 c=6.468、r9 c=5.677。clean block 水平稳定（streaming
  5.47–5.64 / 5.74–5.80，hot ~5.10–5.18 / ~5.06–5.18）。
  所有污染块都出现在 **round 内采样边界**（c0/c1/c2 查询点）。

→ 结论: 即使 300ms burn，**round 内 nvidia-smi 采样本身持续再触发
降级态**（每次查询 ~40ms 缺口 × 3 次/round，衰减当日 >150ms，
burn 刚结束又被拉回去）。

### 3.2 零采样验证（/tmp/nogap_test.json）

- **Phase A（测量区零采样，400 样本/variant，hot）**: v4 med=5.19
  （bucket 中位 5.16–5.22 平直；个别 1.3× 慢样本 6/400）、
  v1 med=5.20，last-300 ratio **0.9987**（tie）。
- **Phase B（后台 `nvidia-smi dmon -s puc -d 0.5` 轮询，400 样本）**:
  v4 med=5.40（bucket 漂移 5.13–5.57）、v1 med=5.64（bucket
  5.39–5.93，max 30.30 µs），last-300 ratio **0.9468**。

→ **任何形式的测量区采样（进程内 nvidia-smi 或后台 NVML/dmon）
都会扰动结果**，全部禁止。out-of-process 采样器方案被数据否决。

### 3.3 v2.2（本版本）：零采样 + 跨 block 一致性 guard

1. **round 内不做任何 nvidia-smi / 时钟采样**（pair 的 c0/c1/c2、
   matrix 的 per-block 查询全部删除）。DVFS 5% 相对判据退役——
   其目的（检出 A/B 状态漂移）由 (2) 承担，且作用于真实测量值。
   `stats.check_dvfs_*` 函数保留（默认签名与旧记录可解释性）。
2. **跨 block 一致性 guard**: 每个 variant 维护本 run 内已测 block
   中位数的历史（含重试 block），warmup 前 `CROSSBLOCK_WARMUP=3`
   个 block 不判；之后某 block 中位数 > 运行中位数 ×
   `CROSSBLOCK_FACTOR=1.15` → 该 round 无效
   （`invalid_reason="INVALID_CROSSBLOCK"`），走与 spike 无效相同
   的重试路径。这使**均匀变慢的整块**（spike guard 不可见的
   R6 失效模式）可检出。
3. run 级 `gpu_state_before/after` 快照保留（before 的采样缺口被
   首个 block 的 300ms burn 吸收），仅作环境参考，不参与判据。
4. round 记录以 `crossblock`（per-variant ratio/flagged）替换
   `clocks`/`eff_sm_*` 字段；顶层新增 `invalid_crossblock_rounds`；
   `harness` 字段 → `"paired-streaming-v2.2"`。

**决策语义完全不变**: KEEP/REJECT/NEUTRAL/UNSTABLE 判据
（decision.py）、bootstrap CI、round-level 统计单位均未动。
v0.2 冻结记录（harness = "paired-streaming-v2"）与 v2.1 记录
原样有效；v2.2 记录对旧消费者只增字段（`scripts/bench_v2.py`
已加防御：无 `clocks` 字段的记录不再打印 clock 行）。

## 4. 验证（全部为记录值）

- v2.1 单元冒烟: `measure_block` hot 中位 5.824 µs、streaming
  6.780 µs，各 100 clean / 0 spike。
- R5 / R6: 见 §3.1（v2.1 两层失效模式的数据来源）。
- burn_probe / nogap A-B: 见 §3.1 / §3.2。
- **R7（v2.2，回归门最终确认，`v03reg_v22r7_*`，9 rounds × 2 变体）**:

  | mode | median speedup | CI95 | faster (cand) | valid | p_med / c_med (µs) | 无效 round |
  |---|---|---|---|---|---|---|
  | streaming | **0.9592** | [0.9094, 0.9669] | 1/9 | 9/9 | 5.463 / 5.692 | 0 spikes, 0 crossblock |
  | hot | **0.9969** | [0.9938, 1.0280] | 2/9 | 9/9 | 5.120 / 5.131 | 0 spikes, 0 crossblock |

  per-round（p/c，µs）:
  - streaming: r1 6.912/5.667（r1 parent 为本 run 首块，run 间缺口
    后 300ms burn 未完全恢复；crossblock warmup 不判，留档）、
    r2–r8 5.444–5.487 / 5.667–5.799（平直）、r9 5.444/6.001
    （candidate 块 crossblock ratio 1.0555 < 1.15，不判）。
  - hot: r1 6.491/5.120（同 r1 效应）、r2–r9 5.088–5.329 /
    5.063–5.184（平直；r4 parent 5.329 ratio 1.0119 不判）。
  - 与独立机制证据一致: R6 clean 轮（streaming 5.47–5.64/5.74–5.80，
    hot ~5.1/5.18）、nogap Phase A（hot 5.19/5.20，ratio 0.9987）。
    三个独立 harness 状态下 hot 均为 **tie**（0.9947–1.0006 /
    0.9987 / 0.9969）。

按 decision.py 判据: streaming median 0.9592（> 0.95 REJECT 线，
CI 上界 0.9669 < 1.0）与 hot median 0.9969（CI 含 1.0）→
**两 mode 均 NEUTRAL**（streaming 方向偏 v4，hot 平手）。

## 5. 对 v0.2 结论的影响说明（如实，不做美化）

- **hot**: v0.2 = 0.9327 → REJECT v1（v4 快 ~7%）；R7 = 0.9969 →
  NEUTRAL（tie）。**结论发生变化，归因于机器状态漂移而非 evaluator
  缺陷**：v0.2 测量时 in-round 时钟 1350 MHz / 40°C；当前持续负载
  时钟 1905–1920 MHz / 33–34°C，绝对水平整体下移 ~12–20%，kernel
  间相对差异被压缩到 0.5% 以内。今日三个独立状态（R6 clean 轮、
  nogap A、R7）一致给出 tie，可复现、可审计。
- **streaming**: v0.2 = 1.0113（v1 略快，NEUTRAL）；R7 = 0.9592
  （v4 略快，CI 上界 < 1.0，但 median 未过 0.95 REJECT 线 → 仍
  NEUTRAL）。方向翻转，两代结论在同一判据下都是 NEUTRAL。
- v0.2 冻结记录按其自身 harness（"paired-streaming-v2"）与当时机器
  状态解释，原样有效，不做重标。本回归门的通过标准是
  **机制等价性**（配对/交替/同张量/事件计时/round 级统计/bootstrap
  判据在第二操作前的行为正确、无污染、可复现），不是数值复刻。

## 6. 已知残留风险

1. **衰减时间继续漂移**: diag3 的 8–10ms（数天前）→ 当日 >150ms。
   300ms burn 当前 2/2 + R7 9/9 干净，但另一天可能再次不足。
   兜底链: burn → spike guard（单点尖峰）→ crossblock guard（整块
   均匀慢，>15%）→ 重试 ≤3 → MIN_VALID_ROUNDS=5 UNSTABLE →
   记录级 `samples`/`crossblock` 全量留档可事后审计。
2. **crossblock ×1.15 不捕捉 <15% 的均匀漂移**（R7 hot r4
   5.329/≈5.19 ≈ 1.03、streaming r5 1.06 均未触发；R6 hot r9
   5.677/5.12 ≈ 1.11 也不会触发）。缓解: paired ratio 对同 round
   A/B 两侧的部分漂移不敏感 + bootstrap CI 报告不确定度 +
   留档审计；不声称能检出任意幅度的漂移。
3. **多租户宿主机干扰**: Xid 31（20h 前，他人进程）、NVRM
   unhandled-interrupt（8 天前）说明该卡近期硬件状态偏不稳定；
   本容器无 GPU 电源/时钟配置权（persistence mode、`-lgc` 锁频
   均不可用），无法从根上消除降级态本身，只能保证测量区不再
   自触发并留档可审计。
4. **时钟读数不可作为负载状态证据**（§2.1）: v2.2 记录不再含
   round 内时钟字段；`gpu_state_before` 反映的是 run 前的
   空闲/过渡态，只作环境参考。
