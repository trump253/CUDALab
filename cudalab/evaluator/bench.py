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

adapter 协议（cudalab/operators/base.py::Operator）:
- `op.name` / `op.bench_shapes` / `op.primary_target`
- `op.make_bench_pool(M, H, dtype, mode, seed, pool_size) -> BenchPool`
- `op.algorithmic_bytes(M, H, element_size) -> int`
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch

from . import stats as _stats
from .gpu import now_iso, gpu_state, gpu_clocks, condense_clocks

HARNESS_VERSION = "paired-streaming-v2"

WARMUP = 150        # 每 variant 每 round 不计时预热启动
ITERS = 100         # 每 round 计时样本数
BATCH = 32          # 每样本连续启动数
ROUNDS = 9          # 独立 round 数（>=7 推荐值）
MAX_RETRIES = 3     # DVFS 无效 round 的最大重试次数
MIN_VALID_ROUNDS = 5
DVFS_TOL = 0.05     # A/B 有效 SM clock 相对差 > 5% -> round 无效
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
                  batch: int = BATCH) -> float:
    """测量一个 variant 在一个 round 内的中位单发时间（us）。

    hot: 固定 buffer 连续启动；streaming: 轮换 buffer（每次启动换 buffer）。
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

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        for _ in range(batch):
            one()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop) * 1e3 / batch)
    return statistics.median(times)


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
                         "note": "有效时钟 = variant 测量区间前后两次 "
                                 "nvidia-smi 采样的均值（轮询采样近似）"},
        "generated": now_iso(),
    }
    record["gpu_state_before"] = gpu_state()

    rounds_out: list[dict] = []
    for slot in range(rounds):
        # 顺序: slot 偶数 parent 先, 奇数 candidate 先（确定性交替）
        order = [parent, candidate] if slot % 2 == 0 else [candidate, parent]
        res = None
        for attempt in range(max_retries + 1):
            c0 = gpu_clocks()
            t_first = measure_block(pool, ext, order[0], warmup, iters, batch)
            c1 = gpu_clocks()
            t_second = measure_block(pool, ext, order[1], warmup, iters, batch)
            c2 = gpu_clocks()
            t_parent = t_first if order[0] == parent else t_second
            t_cand = t_second if order[0] == parent else t_first
            dvfs_ok, reason, eff_parent, eff_cand = _stats.check_dvfs_pair(
                c0.get("sm_clock_mhz"), c1.get("sm_clock_mhz"),
                c2.get("sm_clock_mhz"), order[0] == parent, tol=DVFS_TOL)
            res = {
                "round": slot + 1,
                "order": order,
                "retries": attempt,
                "parent_us": round(t_parent, 3),
                "candidate_us": round(t_cand, 3),
                "speedup": round(t_parent / t_cand, 6),
                "valid": dvfs_ok,
                "invalid_reason": reason if not dvfs_ok else None,
                "clocks": {
                    "before_pair": condense_clocks(c0),
                    "after_first": condense_clocks(c1),
                    "after_second": condense_clocks(c2),
                    "eff_sm_parent_mhz": eff_parent,
                    "eff_sm_candidate_mhz": eff_cand,
                },
            }
            if dvfs_ok:
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
                         "note": "round 有效 = 该 round 内所有 variant 有效 "
                                 "SM clock 的极差/均值 <= 5%"},
        "generated": now_iso(),
    }
    record["gpu_state_before"] = gpu_state()

    rounds_out: list[dict] = []
    for slot in range(rounds):
        order = variants[slot % n:] + variants[:slot % n]  # round-robin
        res = None
        for attempt in range(max_retries + 1):
            clocks: list[dict] = [gpu_clocks()]
            per: dict[str, float] = {}
            for v in order:
                per[v] = measure_block(pool, ext, v, warmup, iters, batch)
                clocks.append(gpu_clocks())
            sms = [c.get("sm_clock_mhz") for c in clocks]
            dvfs_ok, reason, effs = _stats.check_dvfs_matrix(sms, tol=DVFS_TOL)
            eff = {v: e for v, e in zip(order, effs)}
            res = {
                "round": slot + 1,
                "order": order,
                "retries": attempt,
                "valid": dvfs_ok,
                "invalid_reason": reason if not dvfs_ok else None,
                "us": {v: round(per[v], 3) for v in order},
                "eff_sm_mhz": {v: eff[v] for v in order},
                "clocks": [condense_clocks(c) for c in clocks],
            }
            if dvfs_ok:
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
