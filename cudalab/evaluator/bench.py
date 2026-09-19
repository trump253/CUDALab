"""CUDALab v0.3 — 通用 paired benchmark 引擎（paired-streaming-v2，算子无关）。

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

HARNESS_VERSION = "paired-streaming-v2.2"

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
SPIKE_FACTOR = 1.5  # v2.1: 样本 > 运行中 clean 基线中位数 * 1.5 -> spike
MIN_CLEAN_SAMPLES = 50  # v2.1: 单 block clean 样本下限，低于则 round 无效
CROSSBLOCK_FACTOR = 1.15   # v2.2: block 中位数 > variant 运行中位数 * 1.15
                           # -> round 无效（均匀慢块检出）
CROSSBLOCK_WARMUP = 3      # v2.2: 每 variant 前 3 个 block 不判（吸收
                           # 首 block 的 run 前空隙 / first-touch 态）
POOL_SIZE = 16      # streaming 模式缓冲池大小
SEED = 1234         # 输入张量生成 seed（固定，可复现）
L2_BYTES = 5.5 * 1024 * 1024  # RTX 2080 Ti L2（记录用；小 shape 无法
                              # 超过 L2，结果中如实标注）


@dataclass
class BenchPool:
    """预分配的计时张量池（计时区域内永不 malloc / 随机数 / copy）。

    `launch(ext, variant, i)` 是 adapter 提供的一次计时启动：
    hot 模式恒用 index 0；streaming 模式由引擎按 index 轮换。
    """
    xs: list
    outs: list
    pool_size: int
    working_set_bytes: int
    element_size: int
    mode: str
    launch: Callable = field(repr=False)

    def __post_init__(self):
        if self.mode not in ("hot", "streaming"):
            raise ValueError(f"未知 cache mode: {self.mode}")
        if self.mode == "hot":
            if len(self.xs) != 1 or len(self.outs) != 1:
                raise ValueError("hot 模式必须有且仅有一个 x/out 缓冲")
        elif len(self.xs) != self.pool_size or len(self.outs) != self.pool_size:
            raise ValueError("streaming 模式缓冲数必须等于 pool_size")


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
    - per-sample spike guard: 样本 gpu_us > 运行中 clean 基线中位数
      × spike_factor → 标记 spike，不进入 block 中位数。
    - 返回 {"median_us", "n_clean", "n_spike"}。
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
    clean: list[float] = []
    n_spike = 0
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        for _ in range(batch):
            one()
        stop.record()
        torch.cuda.synchronize()
        t_us = start.elapsed_time(stop) * 1e3 / batch
        if clean and t_us > spike_factor * statistics.median(clean[-50:]):
            n_spike += 1
        else:
            clean.append(t_us)
    return {"median_us": statistics.median(clean) if clean else None,
            "n_clean": len(clean), "n_spike": n_spike}


def bench_pair(op, ext, parent: str, candidate: str, M: int, H: int,
               dtype: torch.dtype = torch.float16, mode: str = "streaming",
               rounds: int = ROUNDS, warmup: int = WARMUP,
               iters: int = ITERS, batch: int = BATCH,
               max_retries: int = MAX_RETRIES, seed: int = SEED) -> dict:
    """真正的 paired benchmark: 每 round 内 A/B 相邻、同张量、顺序交替。"""
    if mode not in ("hot", "streaming"):
        raise ValueError(f"未知 cache mode: {mode}")
    for v in (parent, candidate):
        if v not in op.variants(ext):
            raise ValueError(f"未知变体 {v}")

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
                 "working_set_gt_l2": pool.working_set_bytes > L2_BYTES},
        "clock_policy": {"sm_clock_rel_tolerance": DVFS_TOL,
                         "min_valid_rounds": MIN_VALID_ROUNDS,
                         "max_retries": max_retries,
                         "warmup_ms": WARMUP_MS,
                         "spike_factor": SPIKE_FACTOR,
                         "min_clean_samples": MIN_CLEAN_SAMPLES,
                          "crossblock_factor": CROSSBLOCK_FACTOR,
                          "crossblock_warmup": CROSSBLOCK_WARMUP,
                         "note": "v2.2: round 内不做 nvidia-smi 采样"
                                 "（采样本身是 idle-gap 触发源, 且采样值"
                                 " 为缺口/空闲时钟而非负载时钟）; "
                                 "A/B 状态漂移由跨 block 一致性 guard "
                                 "（block 中位数 vs 该 variant 本 run "
                                 "运行中位数）检出; run 级 gpu_state "
                                 "before/after 快照保留"},
        "generated": now_iso(),
    }
    record["gpu_state_before"] = gpu_state()

    # v2.2 跨 block 一致性 guard 的 per-variant 历史（含重试 block）
    block_hist: dict[str, list[float]] = {parent: [], candidate: []}

    def _crossblock_check(variant: str, block: dict) -> dict:
        hist = block_hist[variant]
        med = block["median_us"]
        info: dict = {"running_med_us": None, "ratio": None,
                      "flagged": False}
        if med is not None and len(hist) >= CROSSBLOCK_WARMUP:
            running_med = statistics.median(hist)
            ratio = med / running_med
            info["running_med_us"] = round(running_med, 3)
            info["ratio"] = round(ratio, 4)
            info["flagged"] = ratio > CROSSBLOCK_FACTOR
        if med is not None:
            hist.append(med)
        return info

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
                "parent_us": round(t_parent, 3),
                "candidate_us": round(t_cand, 3),
                "speedup": round(t_parent / t_cand, 6),
                "valid": spikes_ok and cross_ok,
                "invalid_reason": invalid_reason,
                "samples": {
                    "parent": {"n_clean": b_parent["n_clean"],
                               "n_spike": b_parent["n_spike"]},
                    "candidate": {"n_clean": b_cand["n_clean"],
                                  "n_spike": b_cand["n_spike"]},
                },
                "crossblock": crossblock_info,
            }
            if spikes_ok and cross_ok:
                break
        rounds_out.append(res)

    valid = [r for r in rounds_out if r["valid"]]
    speedups = _stats.paired_speedups([r["parent_us"] for r in valid],
                                      [r["candidate_us"] for r in valid])
    s = _stats.summarize(speedups)
    ci95 = _stats.bootstrap_ci(speedups)
    parent_med = statistics.median([r["parent_us"] for r in valid]) if valid else None
    cand_med = statistics.median([r["candidate_us"] for r in valid]) if valid else None

    record.update({
        "n_rounds": len(rounds_out),
        "valid_rounds": len(valid),
        "invalid_dvfs_rounds": len(rounds_out) - len(valid),
        "invalid_dvfs_only_rounds":
            sum(1 for r in rounds_out if r["invalid_reason"] != "INVALID_SPIKES"
                and not r["valid"]),
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
        if v not in op.variants(ext):
            raise ValueError(f"未知变体 {v}")
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
                 "working_set_gt_l2": pool.working_set_bytes > L2_BYTES},
        "clock_policy": {"sm_clock_rel_tolerance": DVFS_TOL,
                         "min_valid_rounds": MIN_VALID_ROUNDS,
                         "max_retries": max_retries,
                         "warmup_ms": WARMUP_MS,
                         "spike_factor": SPIKE_FACTOR,
                         "min_clean_samples": MIN_CLEAN_SAMPLES,
                          "crossblock_factor": CROSSBLOCK_FACTOR,
                          "crossblock_warmup": CROSSBLOCK_WARMUP,
                         "note": "v2.2: round 内不做 nvidia-smi 采样"
                                 "（采样本身是 idle-gap 触发源, 且采样值"
                                 " 为缺口/空闲时钟而非负载时钟）; "
                                 "A/B 状态漂移由跨 block 一致性 guard "
                                 "（block 中位数 vs 该 variant 本 run "
                                 "运行中位数）检出; run 级 gpu_state "
                                 "before/after 快照保留"},
        "generated": now_iso(),
    }
    record["gpu_state_before"] = gpu_state()

    # v2.2 跨 block 一致性 guard 的 per-variant 历史（含重试 block）
    block_hist: dict[str, list[float]] = {v: [] for v in variants}

    def _crossblock_check(variant: str, block: dict) -> dict:
        hist = block_hist[variant]
        med = block["median_us"]
        info: dict = {"running_med_us": None, "ratio": None,
                      "flagged": False}
        if med is not None and len(hist) >= CROSSBLOCK_WARMUP:
            running_med = statistics.median(hist)
            ratio = med / running_med
            info["running_med_us"] = round(running_med, 3)
            info["ratio"] = round(ratio, 4)
            info["flagged"] = ratio > CROSSBLOCK_FACTOR
        if med is not None:
            hist.append(med)
        return info

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
                "us": {v: round(per[v]["median_us"], 3) for v in order},
                "samples": {v: {"n_clean": per[v]["n_clean"],
                                "n_spike": per[v]["n_spike"]} for v in order},
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
        per_variant[v] = {
            "round_medians_us": [r["us"][v] for r in valid],
            "median_us": round(med, 3) if med else None,
            "algorithmic_bw_gbps":
                round(algo_bytes / (med * 1e-6) / 1e9, 1) if med else None,
            "n_valid_rounds": len(meds),
        }
    record.update({
        "variants": variants,
        "n_rounds": len(rounds_out),
        "valid_rounds": len(valid),
        "invalid_dvfs_rounds": len(rounds_out) - len(valid),
        "invalid_spikes_rounds":
            sum(1 for r in rounds_out if r["invalid_reason"] == "INVALID_SPIKES"),
        "invalid_crossblock_rounds":
            sum(1 for r in rounds_out
                if r["invalid_reason"] == "INVALID_CROSSBLOCK"),
        "rounds": rounds_out,
        "per_variant": per_variant,
    })
    record["gpu_state_after"] = gpu_state()
    del pool
    torch.cuda.empty_cache()
    return record


def analyze_shape_winners(records: list[dict]) -> list[dict]:
    """从矩阵记录生成 shape-specific winner（v0.2 要求: 不只报 global best）。

    winner = valid round 跨轮中位数最小的 variant；paired 对比用
    runner-up 与 winner 的 per-round 中位数之比（round-level paired），
    附 bootstrap CI。
    """
    out = []
    for rec in records:
        pv = rec["per_variant"]
        ranked = sorted(
            (v for v in pv if pv[v]["median_us"] is not None),
            key=lambda v: pv[v]["median_us"])
        if len(ranked) < 2:
            continue
        winner, runner = ranked[0], ranked[1]
        # round-level paired: 只取两个 variant 都有效的 round（矩阵模式下
        # round 有效即全部 variant 有效）
        ratios = []
        for r in rec["rounds"]:
            if r["valid"] and winner in r["us"] and runner in r["us"]:
                ratios.append(r["us"][runner] / r["us"][winner])
        s = _stats.summarize(ratios)
        out.append({
            "shape": rec["shape"],
            "dtype": rec["dtype"],
            "cache_mode": rec["cache_mode"],
            "winner": winner,
            "runner_up": runner,
            "winner_median_us": pv[winner]["median_us"],
            "runner_up_median_us": pv[runner]["median_us"],
            "median_ratio_runner_over_winner": s["median"],
            "bootstrap_ci_95": _stats.bootstrap_ci(ratios),
            "valid_rounds": rec["valid_rounds"],
            "all_variants_median_us": {v: pv[v]["median_us"]
                                       for v in rec["variants"]},
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
