"""CUDALab 通用 paired benchmark 引擎（paired-streaming-v2.3，算子无关）。

v0.2 方法论（paired-streaming-v2）原样保留，仅把算子差异（输入生成 /
launch / 算法 IO / 记录目录）隔离到 cudalab/operators 的 adapter：

1. **paired 测量**: 每个 round 内 parent/candidate 时间相邻测量，且使用
   **同一组预分配张量**（同一 round 内绝不重新生成输入、绝不隔着其他
   shape 比较）。
2. **顺序去偏**: paired 模式每轮 A→B / B→A 交替（顺序记录在案）；
   矩阵模式每个 shape 内 variant 按 round-robin 轮换，使 variant 的
   测量位置与时间漂移（热/DVFS）解耦。
3. **DVFS guard**: 每个 round 在各 variant 测量区间前后采样 nvidia-smi
   （SM clock / mem clock / 温度 / 功耗）。每个 variant 的"有效时钟"
   = 其测量区间前后两次采样的均值。有效时钟相对差 > 5% → 该 round
   `INVALID_DVFS`，不进入统计；每 round 最多重试 3 次；最终 valid
   round < 5 → 决策 UNSTABLE（不强行 KEEP/REJECT）。不修改 power
   limit、不锁时钟（容器不允许）——只记录 + 判无效。
4. **cache mode**:
   - `hot`: 单 x/out 缓冲、连续启动（cache 友好稳态）；
   - `streaming`: 预分配 pool_size 个 x/out 缓冲，计时区域内 kernel
     轮换 buffer（无 malloc / 随机数 / copy）；working set 记录在案，
     主目标 shape 下 >> L2 (5.5MB)。不声称"完全 cold cache"
     （rotating-buffer / cache-cold-ish）。
5. **带宽指标**: `algorithmic_bw_gbps` = 算子最小有用 IO（由 adapter
   的 `algorithmic_bytes(M, H, element_size)` 给出）/ 时间。这是逻辑
   算法流量，不是实测 DRAM 吞吐（真实 DRAM 行为以 NCU 为准）。
6. **统计单位**: 独立 round（round 内 sample 仅用于稳健中位数），
   round-level paired speedup → median/mean/min/max + faster_frac +
   95% bootstrap CI（见 stats.py / decision.py）。

v2.1 加固（2026-09-19，RMSNorm 回归门发现；详见
`docs/evaluator_hardening_v0.3.md`）:
7. **idle-gap 降级态**: 实测（2026-09-19，RTX 2080 Ti / 本容器环境）
   发现：一次 nvidia-smi 查询造成的 ~40ms GPU 空闲缺口会把 GPU 推入
   一个**降级性能态**（kernel 时间 ~1.3–1.8x，host 侧 launch 同步变
   慢）；期间 `nvidia-smi clocks.sm` 读到的不是负载时钟（缺口/空闲
   采样值，当日实测 1350 MHz，而持续负载下 gpu_state 读 1905–1920
   MHz，dmon 空闲读 405 MHz）——DVFS guard 的轮询采样对真实负载
   时钟是盲的。v0.2 harness 的 warmup 仅 150 次 launch（~1ms），
   不足以覆盖衰减（衰减时间实测逐日漂移: 早期 ~8–10ms，当日
   150ms burn 后仍停在 6.6–7.7µs 平台、300ms burn 后 2/2 试次回到
   5.1–5.7µs 紧带）。修复:
   - warmup 从固定 150 次改为 **≥150 次且 ≥WARMUP_MS(300ms) wall
     time**（时间制 burn，保证最后一个 idle 缺口后的降级态衰减完毕）;
   - 测量内 **per-sample spike guard**: 相对运行中 clean 基线中位数
     > SPIKE_FACTOR(1.5x) 的样本标记为 spike 并剔除出 block 中位数;
     单 block clean 样本 < MIN_CLEAN_SAMPLES(50) → 该 round 无效
     （`invalid_reason="INVALID_SPIKES"`），走与 DVFS 无效相同的
     重试路径。

v2.2 加固（2026-09-19 同日，R6 之后）:
8. **guard 采样本身是触发源**: R6（v2.1 + 300ms burn）仍有 7/36
   block 均匀变慢（无 spike、中位数整体抬高，spike guard 不可见）；
   受控实验证明：测量路径零采样时 400 样本全程平坦（v4/v1 hot
   比值 0.9987 紧带），而 500ms 周期的后台 dmon 轮询也会扰动结果
   （比值漂移到 0.9468、样本整体漂移 + 单发 30µs spike）。结论:
   round 内任何时钟采样（进程内 nvidia-smi 或后台 NVML 轮询）都
   不可接受。v2.2 因此:
   - round 循环内**不再做任何 nvidia-smi 采样**（v0.2 DVFS guard
     的"5% 时钟相对差"判据退役；其目的——检出 A/B 测量区间间的
     状态漂移——由下述跨 block 一致性 guard 承接，且不再受"采样值
     是缺口/空闲时钟"的盲区影响）;
   - **cross-block consistency guard**: 每个 variant 维护本 run 内
     已测 block 中位数的运行中位数；warmup（前 CROSSBLOCK_WARMUP=3
     个 block）后，某 block 中位数 > 运行中位数 × CROSSBLOCK_FACTOR
     (1.15) → 该 round 无效（`invalid_reason="INVALID_CROSSBLOCK"`），
     走相同的重试路径（≤3 次）。均匀慢块（spike guard 不可见的
     失效模式）由此变为可检出、可重试；
   - run 级环境快照保留: `gpu_state_before/after`（各一次 nvidia-smi；
     before 的空隙由首 block 的 300ms burn 吸收）。
   决策语义（KEEP/REJECT/NEUTRAL/UNSTABLE 判据）不变；v0.2/v2.1
   冻结记录不受影响（按其各自 harness 版本解释）。

v0.3.1 语义澄清（2026-09-19 合并 review 之后，无重跑）:
9. **变体隔离（quarantine）**: 正常基准路径只接受 `op.variants(ext)`
   （正常可 dispatch 列表）中的变体。被隔离变体（NOT_FOR_NORMAL_
   DISPATCH，如 softmax 的 `softmax_hsplit2`）若被显式请求，引擎以
   隔离原因明确拒绝，而不是静默跳过；显式 `ext.forward(name, ...)`
   调用是受控的历史审计入口，不属于正常 dispatch（协议见
   `Operator.unsafe_variants`）。
10. **无效轮计数字段更名**: v2.2 起 round 内不再做 DVFS 采样，
    旧字段名 `invalid_dvfs_rounds` / `invalid_dvfs_only_rounds`
    已成陈旧命名（它们统计的是 spike / cross-block 等**环境**无效
    轮）。新记录写 `invalid_environment_rounds`（全部无效轮）与
    `invalid_environment_only_rounds`（非 spike 类无效轮，pair 记录）；
    旧字段名以原值保留为 legacy alias（旧 JSON 兼容；旧记录仍按旧名
    解释）。
11. **KNOWN LIMITATION（guard 不对称性，v2.3 已修复）**: spike guard
    （样本 > 1.5× 运行中位数）与 cross-block guard（block 中位数 >
    运行中位数 ×1.15）曾只拒绝**异常慢**的状态（ratio > 阈值），不
    拒绝异常快的状态——理论上可能偏好性剔除慢 excursion，构成选择
    偏差。已登记于 docs/evaluator_hardening_v0.3.md（Evaluator v2.3
    TODO: 对称阈值或 log-latency 稳健偏差）；v2.3（下一条）实现了对称
    判据并双轨记录 raw/filtered，本 limitation 对 v2.3 及以后记录不
    再成立。SFM-0001 primary streaming 记录
    invalid_spikes_rounds=0、invalid_crossblock_rounds=0，其 1.68×
    结论不依赖这些过滤。
12. **v2.3 对称 guard + raw/filtered 双轨 + FILTER_SENSITIVE**
    （2026-09-19，v0.4；设计文档 `docs/evaluator_v2_3.md`）:
    - **对称 guard**（修复第 11 条 limitation，判据明确/可解释/可单测/
      固定规则，parent 与 candidate 完全同一规则、同一阈值）:
      per-sample guard 与 cross-block guard 都改为对称判据——等价 log
      空间 `|log(t/ref)| > log(F)`:
      * spike: F=SPIKE_FACTOR(1.5)，ref = 最近 SPIKE_WINDOW(50) 个
        accepted 样本中位数；t > ref×1.5 → 拒（slow），
        t < ref/1.5 → 拒（fast，v2.3 新增）；
      * cross-block: F=CROSSBLOCK_FACTOR(1.15)，ref = 该 variant 本
        run 已测 block filtered 中位数的运行中位数（warmup 前
        CROSSBLOCK_WARMUP(3) 个 block 不判）；ratio > 1.15 → 拒
        （均匀慢块），ratio < 1/1.15 → 拒（均匀快块，v2.3 新增）。
      guard 逻辑抽成纯 CPU 函数（apply_spike_guard / block_stats /
      crossblock_flag），由 tests/test_evaluator_v23_cpu.py 确定性
      单测（对称 spike / 对称 block / 无偏 swap / filter-sensitive
      构造案例）。
    - **raw + filtered 双轨记录**: 每个 block 同时记录 guard 前
      （raw_median_us, n_raw）与 guard 后（median_us, n_accepted,
      n_rejected_fast, n_rejected_slow）统计；round 级
      parent_raw_us / candidate_raw_us / raw_speedup 落盘；pair 级
      增加 raw_speedup 与 filtered_speedup（median of paired per-round
      ratios）+ 各自 bootstrap CI95；矩阵记录增加每 round us_raw 与
      per-variant raw_median_us。旧字段（invalid_environment_rounds
      等）原样保留；历史 v2/v2.1/v2.2 记录永不被修改（按其各自
      harness 版本解释）。
    - **FILTER_SENSITIVE 判定**: 若 raw 与 filtered 的 speedup 方向
      翻转（raw<1<filtered 或 filtered<1<raw），或
      |log(filtered/raw)| > FILTER_LOG_DELTA = log(1.10)，记录标记
      `filter_sensitive=true` 并写明原因；决策层
      （decision.py::apply_filter_gate）随后把最终 policy_decision
      一律降级为 UNSTABLE（v0.4.1 起 KEEP/REJECT/NEUTRAL 均降级，
      原决策记入 original_decision）——不强行给出任何 policy 判定。
    - 记录 schema 新增: `environment_guard{method, symmetric, ...}`、
      `raw{parent_median_us, candidate_median_us, speedup,
      bootstrap_ci_95}`、`filtered{...}`、
      `rejected_samples{fast, slow}`、`filter_sensitive`
      （+ `filter_sensitive_reason`）。

adapter 协议（cudalab/operators/base.py::Operator）:
- `op.name` / `op.bench_shapes` / `op.primary_target`
- `op.make_bench_pool(M, H, dtype, mode, seed, pool_size) -> BenchPool`
- `op.algorithmic_bytes(M, H, element_size) -> int`
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch

from . import stats as _stats
from .gpu import now_iso, gpu_state
from .experiment import classify_cell  # v0.6 merge-review: matrix winner 单一来源

HARNESS_VERSION = "paired-streaming-v2.3"

WARMUP = 150        # 每 variant 每 round 不计时预热启动（v2.1: 最小次数）
WARMUP_MS = 300.0   # v2.1: 预热还需满足的 wall time（ms）。idle-gap 降级态
                    # 衰减时间实测逐日漂移: 2026-09-19 早期 ~8–10ms, 当日
                    # 150ms burn 后仍停在 6.6–7.7µs 平台（2/2 试次），
                    # 300ms burn 后 2/2 试次回到 5.1–5.7µs 紧带
                    # （burn_probe 数据, 见 docs/evaluator_hardening_v0.3.md）
ITERS = 100         # 每 round 计时样本数
BATCH = 32          # 每样本连续启动数
ROUNDS = 9          # 独立 round 数（>=7 推荐值）
MAX_RETRIES = 3     # 无效 round（SPIKES / CROSSBLOCK）的最大重试次数
MIN_VALID_ROUNDS = 5
DVFS_TOL = 0.05     # v0.2 判据（v2.2 起 round 内不再采样；常量保留以
                    # 维持 stats.check_dvfs_* 的默认签名与旧记录可解释性）
MIN_CLEAN_SAMPLES = 50  # v2.1: 单 block accepted 样本下限，低于则 round 无效
# v2.3 对称 guard 参数与纯函数在 stats.py（纯 CPU、无 torch 依赖，可单测）:
# - SPIKE_FACTOR(1.5) / SPIKE_WINDOW(50): per-sample guard, 样本 >
#   运行中 accepted 基线中位数 × 1.5 → 拒（慢）；< /1.5 → 拒（快）。
#   等价 log 空间 |log(t/ref)| > log(1.5) 的对称偏差判据。
# - CROSSBLOCK_FACTOR(1.15) / CROSSBLOCK_WARMUP(3): cross-block guard,
#   ratio > 1.15 → 拒（均匀慢块）；ratio < 1/1.15 → 拒（均匀快块）；
#   每 variant 前 3 个 block 不判（吸收 run 前空隙 / first-touch 态）。
from .stats import (  # noqa: E402,F401  (re-export 兼容 from bench import ...)
    SPIKE_FACTOR, SPIKE_WINDOW,
    CROSSBLOCK_FACTOR, CROSSBLOCK_WARMUP,
    apply_spike_guard, block_stats, crossblock_flag,
)
POOL_SIZE = 16      # streaming 模式缓冲池大小
SEED = 1234         # 输入张量生成 seed（固定，可复现）
L2_BYTES = 5.5 * 1024 * 1024  # RTX 2080 Ti L2（记录用；小 shape 无法
                              # 超过 L2，结果中如实标注）


@dataclass
class BenchPool:
    """预分配的计时张量池（计时区域内永不 malloc / 随机数 / copy）。

    `launch(ext, variant, i)` 是 adapter 提供的一次计时启动：
    hot 模式恒用 index 0；streaming 模式由引擎按 index 轮换。

    v2.3 / RoPE 扩展（全部可选，旧算子 adapter 不填）:
    - `positions`: 每池位置张量（与 xs 同长；RoPE 用；None = 无）。
      当前引擎对 positions 不做轮换（真实 inference 中同一 batch 的
      位置序列不随 activation 缓冲轮换而变化）。
    - `shared`: 共享（不轮换、长期驻留）张量，如 RoPE 的 cos/sin
      表；不进入 working_set_bytes（那是轮换池的 x/out 工作集）。
    - `shared_bytes`: 共享张量总字节数（记录用）。
    - `pool_extra`: 算子自定义的池信息，原样写入记录的
      `pool.pool_extra`（如 RoPE 的 cos/sin 表字节数、总逻辑工作集、
      positions 模式等）。
    """
    xs: list
    outs: list
    pool_size: int
    working_set_bytes: int
    element_size: int
    mode: str
    launch: Callable = field(repr=False)
    positions: Optional[list] = None
    shared: dict = field(default_factory=dict, repr=False)
    shared_bytes: int = 0
    pool_extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in ("hot", "streaming"):
            raise ValueError(f"未知 cache mode: {self.mode}")
        if self.mode == "hot":
            if len(self.xs) != 1 or len(self.outs) != 1:
                raise ValueError("hot 模式必须有且仅有一个 x/out 缓冲")
        elif len(self.xs) != self.pool_size or len(self.outs) != self.pool_size:
            raise ValueError("streaming 模式缓冲数必须等于 pool_size")
        if self.positions is not None:
            n = 1 if self.mode == "hot" else self.pool_size
            if len(self.positions) != n:
                raise ValueError(
                    f"positions 长度 {len(self.positions)} 与模式 "
                    f"{self.mode} 的缓冲数 {n} 不一致")


def _environment_guard() -> dict:
    """v2.3 记录 schema 要求的 environment_guard 块（自描述 + 参数）。"""
    return {
        "method": ("symmetric deviation guard: per-sample reject iff "
                   "|log(t/ref)| > log(spike_factor), ref = median of the "
                   "last spike_window ACCEPTED samples; per-block reject "
                   "iff |log(med/running_med)| > log(crossblock_factor), "
                   "running_med = median of that variant's block filtered "
                   "medians so far in this run (after crossblock_warmup "
                   "blocks). Fast and slow excursions are rejected "
                   "symmetrically; parent and candidate use the exact "
                   "same rule and thresholds."),
        "symmetric": True,
        "spike_factor": SPIKE_FACTOR,
        "spike_window": SPIKE_WINDOW,
        "crossblock_factor": CROSSBLOCK_FACTOR,
        "crossblock_warmup": CROSSBLOCK_WARMUP,
        "min_accepted_samples": MIN_CLEAN_SAMPLES,
        "filter_sensitive_log_delta": round(_stats.FILTER_LOG_DELTA, 6),
        "raw_and_filtered_recorded": True,
    }


def measure_block(pool: BenchPool, ext, variant: str,
                  warmup: int = WARMUP, iters: int = ITERS,
                  batch: int = BATCH, warmup_ms: float = WARMUP_MS,
                  spike_factor: float = SPIKE_FACTOR) -> dict:
    """测量一个 variant 在一个 round 内的中位单发时间（us）。

    hot: 固定 buffer 连续启动；streaming: 轮换 buffer（每次启动换 buffer）。

    v2.1:
    - 预热 = ≥warmup 次 launch **且** ≥warmup_ms wall time（时间制
      burn）。block 之前总有一次 nvidia-smi 空闲缺口，缺口后的 GPU
      降级态衰减时间逐日漂移（实测 8–10ms 至 >150ms）；固定 150 次
      （~1ms）不够，300ms burn 实测 2/2 试次干净。
    - per-sample spike guard: 相对运行中 accepted 基线中位数超出的
      样本标记为 spike 并剔除出 block 中位数。
    v2.3:
    - guard 对称化（快/慢双向拒绝，等价 log 空间对称偏差）；
    - 返回 raw + filtered 双套统计（见 block_stats）:
      {"median_us"(filtered), "raw_median_us", "n_raw", "n_accepted",
       "n_rejected_fast", "n_rejected_slow", legacy n_clean/n_spike,
       raw_samples_us, accepted_samples_us}。
    """
    if pool.mode == "hot":
        def one():
            pool.launch(ext, variant, 0)
    else:
        xs_n = pool.pool_size
        state = {"i": 0}

        def one():
            i = state["i"]
            pool.launch(ext, variant, i)
            state["i"] = (i + 1) % xs_n

    # 时间制预热 burn: 覆盖最后一个 idle 缺口后的降级态衰减
    t_burn = time.perf_counter()
    n_warm = 0
    while n_warm < warmup or (time.perf_counter() - t_burn) * 1e3 < warmup_ms:
        one()
        n_warm += 1
    torch.cuda.synchronize()
    raw: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        for _ in range(batch):
            one()
        stop.record()
        torch.cuda.synchronize()
        raw.append(start.elapsed_time(stop) * 1e3 / batch)
    # guard / 统计全部走纯函数 block_stats（v2.3，CPU 可单测）
    return block_stats(raw, spike_factor=spike_factor, window=SPIKE_WINDOW)


def _require_normal_variant(op, ext, v: str) -> None:
    """正常基准路径的变体门禁（v0.3.1 quarantine 语义，见模块 docstring
    第 9 条）: 变体必须在 `op.variants(ext)` 正常列表中；被隔离变体
    （NOT_FOR_NORMAL_DISPATCH）明确拒绝并说明隔离原因。"""
    if v in op.variants(ext):
        return
    if v in op.unsafe_variants(ext):
        raise ValueError(
            f"变体 {v!r} 已被隔离（UNSAFE_HISTORICAL_EXPERIMENT / REJECTED / "
            f"NOT_FOR_NORMAL_DISPATCH），不能进入正常基准路径；隔离理由见"
            f"该变体的实验记录。显式 ext.forward({v!r}, ...) 调用是受控的"
            f"历史审计入口，不属于正常 dispatch")
    raise ValueError(f"未知变体 {v}")


def bench_pair(op, ext, parent: str, candidate: str, M: int, H: int,
               dtype: torch.dtype = torch.float16, mode: str = "streaming",
               rounds: int = ROUNDS, warmup: int = WARMUP,
               iters: int = ITERS, batch: int = BATCH,
               max_retries: int = MAX_RETRIES, seed: int = SEED) -> dict:
    """真正的 paired benchmark: 每 round 内 A/B 相邻、同张量、顺序交替。"""
    if mode not in ("hot", "streaming"):
        raise ValueError(f"未知 cache mode: {mode}")
    for v in (parent, candidate):
        _require_normal_variant(op, ext, v)

    pool = op.make_bench_pool(M, H, dtype, mode, seed=seed, pool_size=POOL_SIZE)
    algo_bytes = op.algorithmic_bytes(M, H, pool.element_size)

    record: dict = {
        "harness": HARNESS_VERSION,
        "operator": op.name,
        "parent": parent,
        "candidate": candidate,
        "shape": [M, H],
        "dtype": str(dtype).split(".")[-1],
        "cache_mode": mode,
        "seed": seed,
        "warmup": warmup, "iters": iters, "batch": batch,
        "pool": {"pool_size": pool.pool_size,
                 "working_set_bytes": pool.working_set_bytes,
                 "working_set_gt_l2": pool.working_set_bytes > L2_BYTES,
                 # v2.3/RoPE: 共享（不轮换）张量字节 + 算子自定义池信息
                 "shared_bytes": pool.shared_bytes,
                 "pool_extra": pool.pool_extra},
        "clock_policy": {"sm_clock_rel_tolerance": DVFS_TOL,
                         "min_valid_rounds": MIN_VALID_ROUNDS,
                         "max_retries": max_retries,
                         "warmup_ms": WARMUP_MS,
                         "spike_factor": SPIKE_FACTOR,
                         "spike_window": SPIKE_WINDOW,
                         "min_clean_samples": MIN_CLEAN_SAMPLES,
                         "crossblock_factor": CROSSBLOCK_FACTOR,
                         "crossblock_warmup": CROSSBLOCK_WARMUP,
                         "note": "v2.3: round 内不做 nvidia-smi 采样"
                                 "（同 v2.2: 采样本身是 idle-gap 触发源,"
                                 " 且采样值为缺口/空闲时钟而非负载时钟）;"
                                 " A/B 状态漂移由对称跨 block 一致性 "
                                 "guard（block filtered 中位数 vs 该 "
                                 "variant 本 run 运行中位数, ratio > "
                                 "1.15 或 < 1/1.15 均判无效, 快慢对称）"
                                 "检出; per-sample guard 同样对称"
                                 "（|log(t/ref)| > log(1.5) 拒绝, 快/慢"
                                 "双向）; run 级 gpu_state before/after "
                                 "快照保留"},
        "generated": now_iso(),
    }
    # v2.3: 环境 guard 的自描述块（判据 + 对称性 + 全部阈值参数）
    record["environment_guard"] = _environment_guard()
    record["gpu_state_before"] = gpu_state()

    # v2.2 跨 block 一致性 guard 的 per-variant 历史（含重试 block）
    block_hist: dict[str, list[float]] = {parent: [], candidate: []}

    def _crossblock_check(variant: str, block: dict) -> dict:
        # v2.3: 判据抽成纯函数 crossblock_flag（对称, CPU 可单测）;
        # 此处只负责 per-variant 历史的持有
        return crossblock_flag(block_hist[variant], block["median_us"])

    rounds_out: list[dict] = []
    for slot in range(rounds):
        # 顺序: slot 偶数 parent 先, 奇数 candidate 先（确定性交替）
        order = [parent, candidate] if slot % 2 == 0 else [candidate, parent]
        res = None
        for attempt in range(max_retries + 1):
            # v2.2: 此处无 nvidia-smi 采样（见模块 docstring 第 8 条）
            b_first = measure_block(pool, ext, order[0], warmup, iters, batch)
            b_second = measure_block(pool, ext, order[1], warmup, iters, batch)
            first_is_parent = order[0] == parent
            t_parent = b_first["median_us"] if first_is_parent else b_second["median_us"]
            t_cand = b_second["median_us"] if first_is_parent else b_first["median_us"]
            t_parent_raw = b_first["raw_median_us"] if first_is_parent else b_second["raw_median_us"]
            t_cand_raw = b_second["raw_median_us"] if first_is_parent else b_first["raw_median_us"]
            b_parent = b_first if first_is_parent else b_second
            b_cand = b_second if first_is_parent else b_first
            crossblock_info = {
                parent: _crossblock_check(parent, b_parent),
                candidate: _crossblock_check(candidate, b_cand),
            }
            spikes_ok = (b_parent["n_clean"] >= MIN_CLEAN_SAMPLES
                         and b_cand["n_clean"] >= MIN_CLEAN_SAMPLES)
            cross_ok = not (crossblock_info[parent]["flagged"]
                            or crossblock_info[candidate]["flagged"])
            if not spikes_ok:
                invalid_reason = "INVALID_SPIKES"
            elif not cross_ok:
                invalid_reason = "INVALID_CROSSBLOCK"
            else:
                invalid_reason = None
            res = {
                "round": slot + 1,
                "order": order,
                "retries": attempt,
                # filtered（guard 后）与 raw（guard 前）双轨落盘（v2.3）
                "parent_us": round(t_parent, 3),
                "candidate_us": round(t_cand, 3),
                "speedup": round(t_parent / t_cand, 6),
                "parent_raw_us":
                    round(t_parent_raw, 3) if t_parent_raw else None,
                "candidate_raw_us":
                    round(t_cand_raw, 3) if t_cand_raw else None,
                "raw_speedup":
                    round(t_parent_raw / t_cand_raw, 6)
                    if t_parent_raw and t_cand_raw else None,
                "valid": spikes_ok and cross_ok,
                "invalid_reason": invalid_reason,
                # v2.3: 完整 block_stats（raw/filtered 双套统计 +
                # 全量样本审计）；n_clean/n_spike 作为 legacy alias
                # 仍包含在内
                "samples": {
                    "parent": b_parent,
                    "candidate": b_cand,
                },
                "crossblock": crossblock_info,
            }
            if spikes_ok and cross_ok:
                break
        rounds_out.append(res)

    valid = [r for r in rounds_out if r["valid"]]
    # filtered（guard 后，v2.2 主统计路径）
    speedups = _stats.paired_speedups([r["parent_us"] for r in valid],
                                      [r["candidate_us"] for r in valid])
    s = _stats.summarize(speedups)
    ci95 = _stats.bootstrap_ci(speedups)
    parent_med = statistics.median([r["parent_us"] for r in valid]) if valid else None
    cand_med = statistics.median([r["candidate_us"] for r in valid]) if valid else None
    # v2.3: raw（guard 前）同口径 paired 统计 + filter-sensitivity 判定
    raw_parent_meds = [r["parent_raw_us"] for r in valid
                       if r["parent_raw_us"] is not None]
    raw_cand_meds = [r["candidate_raw_us"] for r in valid
                     if r["candidate_raw_us"] is not None]
    if len(raw_parent_meds) == len(raw_cand_meds) and raw_parent_meds:
        raw_speedups = _stats.paired_speedups(raw_parent_meds, raw_cand_meds)
        s_raw = _stats.summarize(raw_speedups)
        ci95_raw = _stats.bootstrap_ci(raw_speedups)
        raw_speedup = s_raw["median"]
    else:
        raw_speedups = s_raw = None
        ci95_raw = None
        raw_speedup = None
    raw_parent_med = (statistics.median(raw_parent_meds)
                      if raw_parent_meds else None)
    raw_cand_med = statistics.median(raw_cand_meds) if raw_cand_meds else None
    filtered_speedup = s["median"]
    filter_sensitive, filter_sensitive_reason = _stats.filter_sensitive(
        raw_speedup, filtered_speedup)
    # v2.3: 全部 round（含无效重试）的 rejected 样本计数合计
    n_rejected_fast = sum(r["samples"][v]["n_rejected_fast"]
                          for r in rounds_out
                          for v in ("parent", "candidate"))
    n_rejected_slow = sum(r["samples"][v]["n_rejected_slow"]
                          for r in rounds_out
                          for v in ("parent", "candidate"))

    # v0.3.1: 无效轮计数的规范字段是 invalid_environment_*（round 内
    # 已无 DVFS 采样；无效原因只有 SPIKES / CROSSBLOCK 两类环境因素）。
    # 旧字段名 invalid_dvfs_* 以原值保留为 legacy alias（旧 JSON 兼容）。
    n_invalid = len(rounds_out) - len(valid)
    n_invalid_only = sum(1 for r in rounds_out
                         if r["invalid_reason"] != "INVALID_SPIKES"
                         and not r["valid"])
    record.update({
        "n_rounds": len(rounds_out),
        "valid_rounds": len(valid),
        "invalid_environment_rounds": n_invalid,
        "invalid_environment_only_rounds": n_invalid_only,
        # legacy alias（v0.2–v0.3 字段名，值不变）
        "invalid_dvfs_rounds": n_invalid,
        "invalid_dvfs_only_rounds": n_invalid_only,
        "invalid_spikes_rounds":
            sum(1 for r in rounds_out if r["invalid_reason"] == "INVALID_SPIKES"),
        "invalid_crossblock_rounds":
            sum(1 for r in rounds_out
                if r["invalid_reason"] == "INVALID_CROSSBLOCK"),
        "rounds": rounds_out,
        "speedups": speedups,
        "median_speedup": s["median"],
        "mean_speedup": s["mean"],
        "min_speedup": s["min"],
        "max_speedup": s["max"],
        "faster_rounds": f"{s['faster_count']}/{s['n']}",
        "bootstrap_ci_95": ci95,
        "bootstrap": {"n_boot": 10000, "seed": 20260919, "statistic": "median"},
        "parent_median_us": round(parent_med, 3) if parent_med else None,
        "candidate_median_us": round(cand_med, 3) if cand_med else None,
        # v2.3: raw / filtered 双轨 pair 级统计 + filter-sensitivity
        "raw_speedup": raw_speedup,
        "filtered_speedup": filtered_speedup,
        "filter_sensitive": filter_sensitive,
        "filter_sensitive_reason": filter_sensitive_reason,
        "raw": {
            "parent_median_us": round(raw_parent_med, 3)
                if raw_parent_med else None,
            "candidate_median_us": round(raw_cand_med, 3)
                if raw_cand_med else None,
            "speedup": raw_speedup,
            "bootstrap_ci_95": ci95_raw,
        },
        "filtered": {
            "parent_median_us": round(parent_med, 3) if parent_med else None,
            "candidate_median_us": round(cand_med, 3) if cand_med else None,
            "speedup": filtered_speedup,
            "bootstrap_ci_95": ci95,
        },
        "rejected_samples": {"fast": n_rejected_fast,
                             "slow": n_rejected_slow},
        "algorithmic_bw_gbps_parent":
            round(algo_bytes / (parent_med * 1e-6) / 1e9, 1) if parent_med else None,
        "algorithmic_bw_gbps_candidate":
            round(algo_bytes / (cand_med * 1e-6) / 1e9, 1) if cand_med else None,
    })
    record["gpu_state_after"] = gpu_state()
    del pool
    torch.cuda.empty_cache()
    return record


def bench_matrix(op, ext, variants: list[str], M: int, H: int,
                 dtype: torch.dtype = torch.float16, mode: str = "streaming",
                 rounds: int = ROUNDS, warmup: int = WARMUP,
                 iters: int = ITERS, batch: int = BATCH,
                 max_retries: int = MAX_RETRIES, seed: int = SEED) -> dict:
    """全矩阵 paired round-robin: 每个 round 内所有 variant 按轮换顺序
    相邻测量（同张量池），variant 位置与时间漂移解耦。"""
    for v in variants:
        _require_normal_variant(op, ext, v)
    pool = op.make_bench_pool(M, H, dtype, mode, seed=seed, pool_size=POOL_SIZE)
    algo_bytes = op.algorithmic_bytes(M, H, pool.element_size)
    n = len(variants)

    record: dict = {
        "harness": HARNESS_VERSION,
        "operator": op.name,
        "shape": [M, H],
        "dtype": str(dtype).split(".")[-1],
        "cache_mode": mode,
        "seed": seed,
        "warmup": warmup, "iters": iters, "batch": batch,
        "pool": {"pool_size": pool.pool_size,
                 "working_set_bytes": pool.working_set_bytes,
                 "working_set_gt_l2": pool.working_set_bytes > L2_BYTES,
                 # v2.3/RoPE: 共享（不轮换）张量字节 + 算子自定义池信息
                 "shared_bytes": pool.shared_bytes,
                 "pool_extra": pool.pool_extra},
        "clock_policy": {"sm_clock_rel_tolerance": DVFS_TOL,
                         "min_valid_rounds": MIN_VALID_ROUNDS,
                         "max_retries": max_retries,
                         "warmup_ms": WARMUP_MS,
                         "spike_factor": SPIKE_FACTOR,
                         "spike_window": SPIKE_WINDOW,
                         "min_clean_samples": MIN_CLEAN_SAMPLES,
                         "crossblock_factor": CROSSBLOCK_FACTOR,
                         "crossblock_warmup": CROSSBLOCK_WARMUP,
                         "note": "v2.3: round 内不做 nvidia-smi 采样"
                                 "（同 v2.2: 采样本身是 idle-gap 触发源,"
                                 " 且采样值为缺口/空闲时钟而非负载时钟）;"
                                 " A/B 状态漂移由对称跨 block 一致性 "
                                 "guard（block filtered 中位数 vs 该 "
                                 "variant 本 run 运行中位数, ratio > "
                                 "1.15 或 < 1/1.15 均判无效, 快慢对称）"
                                 "检出; per-sample guard 同样对称"
                                 "（|log(t/ref)| > log(1.5) 拒绝, 快/慢"
                                 "双向）; run 级 gpu_state before/after "
                                 "快照保留"},
        "generated": now_iso(),
    }
    # v2.3: 环境 guard 的自描述块（判据 + 对称性 + 全部阈值参数）
    record["environment_guard"] = _environment_guard()
    record["gpu_state_before"] = gpu_state()

    # v2.2 跨 block 一致性 guard 的 per-variant 历史（含重试 block）
    block_hist: dict[str, list[float]] = {v: [] for v in variants}

    def _crossblock_check(variant: str, block: dict) -> dict:
        # v2.3: 判据抽成纯函数 crossblock_flag（对称, CPU 可单测）;
        # 此处只负责 per-variant 历史的持有
        return crossblock_flag(block_hist[variant], block["median_us"])

    rounds_out: list[dict] = []
    for slot in range(rounds):
        order = variants[slot % n:] + variants[:slot % n]  # round-robin
        res = None
        for attempt in range(max_retries + 1):
            per: dict[str, dict] = {}
            for v in order:
                per[v] = measure_block(pool, ext, v, warmup, iters, batch)
            crossblock_info = {v: _crossblock_check(v, per[v])
                               for v in order}
            spikes_ok = all(per[v]["n_clean"] >= MIN_CLEAN_SAMPLES
                            for v in order)
            cross_ok = all(not info["flagged"]
                            for info in crossblock_info.values())
            if not spikes_ok:
                invalid_reason = "INVALID_SPIKES"
            elif not cross_ok:
                invalid_reason = "INVALID_CROSSBLOCK"
            else:
                invalid_reason = None
            res = {
                "round": slot + 1,
                "order": order,
                "retries": attempt,
                "valid": spikes_ok and cross_ok,
                "invalid_reason": invalid_reason,
                # v2.3: filtered（us）+ raw（us_raw）双轨
                "us": {v: round(per[v]["median_us"], 3) for v in order},
                "us_raw": {v: round(per[v]["raw_median_us"], 3)
                           if per[v]["raw_median_us"] else None
                           for v in order},
                # v2.3: 完整 block_stats（含 legacy n_clean/n_spike）
                "samples": {v: per[v] for v in order},
                "crossblock": crossblock_info,
            }
            if spikes_ok and cross_ok:
                break
        rounds_out.append(res)

    valid = [r for r in rounds_out if r["valid"]]
    per_variant: dict = {}
    for v in variants:
        meds = [r["us"][v] for r in valid]
        med = statistics.median(meds) if meds else None
        # v2.3: raw 中位数同口径（guard 前样本的 per-round 中位数再取中位）
        raw_meds = [r["us_raw"][v] for r in valid
                    if r["us_raw"].get(v) is not None]
        raw_med = statistics.median(raw_meds) if raw_meds else None
        per_variant[v] = {
            "round_medians_us": [r["us"][v] for r in valid],
            "median_us": round(med, 3) if med else None,
            "round_raw_medians_us": [r["us_raw"][v] for r in valid],
            "raw_median_us": round(raw_med, 3) if raw_med else None,
            "algorithmic_bw_gbps":
                round(algo_bytes / (med * 1e-6) / 1e9, 1) if med else None,
            "n_valid_rounds": len(meds),
        }
    # v2.3: 记录级 raw vs filtered。矩阵上下文中"runner/winner 比值"
    # 在各自排序内恒 >1，方向翻转表现为 **winner 不同**；显式判
    # （1）raw winner ≠ filtered winner（winner flip，即 guard 改变了
    # 结论）；（2）否则用共同 top-2（filtered winner W / runner R）的
    # raw 比值 vs filtered 比值走 stats.filter_sensitive（10% 差判据）。
    # v2.2 及更早记录没有 raw 数据 → 未评估（不假装可信）。
    has_raw = all(pv.get("raw_median_us") is not None
                  for pv in per_variant.values())
    filter_sensitive = False
    filter_sensitive_reason = "无 raw 数据（v2.3 之前 harness）"
    winner_raw = None
    if has_raw and valid:
        ranked_f = sorted((v for v in variants
                           if per_variant[v]["median_us"] is not None),
                          key=lambda v: per_variant[v]["median_us"])
        ranked_r = sorted((v for v in variants
                           if per_variant[v]["raw_median_us"] is not None),
                          key=lambda v: per_variant[v]["raw_median_us"])
        if len(ranked_f) >= 2 and len(ranked_r) >= 2:
            winner_raw = ranked_r[0]
            wf, rf = ranked_f[0], ranked_f[1]
            if winner_raw != wf:
                filter_sensitive = True
                filter_sensitive_reason = (
                    f"winner flip: raw winner {winner_raw!r} != filtered "
                    f"winner {wf!r}（guard 改变了结论）")
            else:
                f_speedup = (per_variant[rf]["median_us"]
                             / per_variant[wf]["median_us"])
                r_speedup = (per_variant[rf]["raw_median_us"]
                             / per_variant[wf]["raw_median_us"])
                filter_sensitive, filter_sensitive_reason = \
                    _stats.filter_sensitive(r_speedup, f_speedup)
    n_rejected_fast = sum(r["samples"][v]["n_rejected_fast"]
                          for r in rounds_out for v in r["samples"])
    n_rejected_slow = sum(r["samples"][v]["n_rejected_slow"]
                          for r in rounds_out for v in r["samples"])
    record.update({
        "variants": variants,
        "n_rounds": len(rounds_out),
        "valid_rounds": len(valid),
        "invalid_environment_rounds": len(rounds_out) - len(valid),
        # legacy alias（v0.2–v0.3 字段名，值不变）
        "invalid_dvfs_rounds": len(rounds_out) - len(valid),
        "invalid_spikes_rounds":
            sum(1 for r in rounds_out if r["invalid_reason"] == "INVALID_SPIKES"),
        "invalid_crossblock_rounds":
            sum(1 for r in rounds_out
                if r["invalid_reason"] == "INVALID_CROSSBLOCK"),
        # v2.3: raw 轨 winner + filter-sensitivity + rejected 合计
        "winner_raw": winner_raw,
        "filter_sensitive": filter_sensitive,
        "filter_sensitive_reason": filter_sensitive_reason,
        "rejected_samples": {"fast": n_rejected_fast,
                             "slow": n_rejected_slow},
        "rounds": rounds_out,
        "per_variant": per_variant,
    })
    record["gpu_state_after"] = gpu_state()
    del pool
    torch.cuda.empty_cache()
    return record


def analyze_shape_winners(records: list[dict]) -> list[dict]:
    """从矩阵记录生成 shape-specific winner（v0.2 要求: 不只报 global best）。

    v0.6 merge-review 更正: 逐格分类**统一委托**
    `cudalab.evaluator.experiment.classify_cell`（单一来源, v0.3/v0.4.1
    语义），不再在本函数内自维护一套统计:
    - winner = valid round 跨轮中位数最小的 variant（**观测中位数排名**,
      不等于统计/政策意义上的"胜者"）;
    - winner vs runner-up 用 round-level paired 比值（per-round
      runner/winner 比值的中位数）+ bootstrap CI;
    - `statistical_relation`（只看 CI95 是否排除 1.00）与
      `policy_decision`（5% 政策带 + filter gate）分列输出 —— 差距 <5%
      时 policy 一律 NEUTRAL / NO_UNIQUE_WINNER, 即使 CI 排除 1.00;
    - raw 轨与 filter-sensitivity 判据与 classify_cell 一致（raw 侧与
      filtered 侧同一 paired 聚合约定, v0.4 review 更正; 旧版在本函数
      内对 raw 侧误用"跨 round 中位数之比", 与 filtered 侧的
      "per-round 比值中位数"约定混用, 已废止）。
    """
    out = []
    for rec in records:
        cell = classify_cell(rec["variants"], rec["rounds"])
        out.append({
            "shape": rec["shape"],
            "dtype": rec["dtype"],
            "cache_mode": rec["cache_mode"],
            **cell,
        })
    return out


def save_record(record: dict, out_dir: Path, tag: str = "") -> Path:
    import json
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{tag}.json"
    p.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return p


def time_call(fn: Callable, warmup: int = 100, iters: int = 200,
              batch: int = 32) -> dict:
    """通用计时 helper（PyTorch context 等"函数参照"用）。

    warmup 次不计时启动；随后 iters 个样本，每样本 batch 次连续启动，
    CUDA event 计时取样本中位数。与 measure_block 相同的批量方案。
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        for _ in range(batch):
            fn()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop) * 1e3 / batch)
    return {"median_us": round(statistics.median(times), 3),
            "min_us": round(min(times), 3),
            "n_samples": len(times)}
