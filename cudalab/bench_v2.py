"""CUDALab v0.2 — paired benchmark harness（paired-streaming-v2）。

修复 v0.1 框架的方法学问题:

1. **paired 测量**（v0.1: variant A 全部 shape 跑完再跑 B，A 的 round 1
   与很久之后的 B 的 round 1 被当作 paired —— 系统性 DVFS/热漂移偏差，
   真实案例 EXP-0007: v1 @ ~1350 MHz vs v4 @ ~1905 MHz 的 1.231×）。
   v0.2: 每个 round 内 parent/candidate 时间相邻测量，且使用**同一组
   预分配张量**（同一 round 内绝不重新生成输入、绝不隔着其他 shape 比较）。

2. **顺序去偏**: paired 模式每轮 A→B / B→A 交替（顺序记录在案）；
   矩阵模式每个 shape 内 variant 按 round-robin 轮换，使 variant 的
   测量位置与时间漂移（热/DVFS）解耦。

3. **DVFS guard**: 每个 round 在 pair 前、A 后、B 后采样 nvidia-smi
   （SM clock / mem clock / 温度 / 功耗）。每个 variant 的"有效时钟"
   = 其测量区间前后两次采样的均值。两 variant 有效时钟相对差 > 5%
   → 该 round `INVALID_DVFS`，不进入统计；每 round 最多重试 3 次；
   最终 valid round < 5 → 决策 UNSTABLE（不强行 KEEP/REJECT）。
   不修改 power limit、不锁时钟（容器不允许）——只记录 + 判无效。
   已知近似: nvidia-smi 是轮询采样而非逐 kernel 时钟，是锁频不可用
   情况下的最佳代理（记录在案）。

4. **cache mode**:
   - `hot`: 单 x/w/out 缓冲、连续启动（cache 友好稳态；不再表述为
     "唯一真实推理场景"）；
   - `streaming`: 预分配 pool_size 个 x/out 缓冲，计时区域内 kernel
     轮换 buffer（无 malloc / 随机数 / copy）；working set 记录在案；
     主目标 (128,4096) fp16 下 >> L2 (5.5MB)。不声称"完全 cold
     cache"（rotating-buffer / cache-cold-ish）。

5. **带宽指标**: `algorithmic_bw_gbps` = RMSNorm 最小有用 IO
   （读 x 一次 + 读 w 一次 + 写 y 一次）/ 时间，按真实 element size
   计算（修复 v0.1 `effective_bw_gbps` 的两个问题: fp32 按 2B 计、
   忽略 baseline/v1/v3 实际读 x 两次）。这是逻辑算法流量，不是实测
   DRAM 吞吐（真实 DRAM 行为以 NCU 为准）。

6. **统计单位**: 独立 round（round 内 sample 仅用于稳健中位数），
   round-level paired speedup → median/mean/min/max + 70% faster +
   95% bootstrap CI（见 stats.py / decision.py）。
"""
from __future__ import annotations

import statistics
import subprocess
import time
from pathlib import Path

import torch

from .build import build
from .stats import paired_speedups, summarize, bootstrap_ci

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks" / "v0.2"

HARNESS_VERSION = "paired-streaming-v2"

# v0.2 基准矩阵（与 v0.1 相同的 7 个 shape；主目标 (128, 4096)）
BENCH_MATRIX_V2 = [
    (1, 4096),
    (16, 4096),
    (128, 4096),
    (1024, 4096),
    (128, 8192),
    (1, 1024),
    (128, 1024),
]
PRIMARY_TARGET = (128, 4096)

WARMUP = 150        # 每 variant 每 round 不计时预热启动
ITERS = 100         # 每 round 计时样本数
BATCH = 32          # 每样本连续启动数（沿用 v0.1 已验证的批量方案）
ROUNDS = 9          # 独立 round 数（>=7 推荐值）
MAX_RETRIES = 3     # DVFS 无效 round 的最大重试次数
MIN_VALID_ROUNDS = 5
DVFS_TOL = 0.05     # A/B 有效 SM clock 相对差 > 5% -> round 无效
POOL_SIZE = 16      # streaming 模式缓冲池大小
SEED = 1234         # 输入张量生成 seed（固定，可复现）
L2_BYTES = 5.5 * 1024 * 1024  # RTX 2080 Ti L2（记录用；小 shape 无法
                              # 超过 L2，结果中如实标注）


def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def gpu_state() -> dict:
    """完整 nvidia-smi 快照（round 前/后，尽力而为）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,temperature.gpu,clocks.sm,clocks.mem,"
             "power.draw,utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout
        fields = out.strip().splitlines()[0].split(",")
        keys = ["gpu_index", "name", "temp_c", "sm_clock_mhz", "mem_clock_mhz",
                "power_w", "gpu_util_pct", "mem_util_pct"]
        d = {}
        for k, f in zip(keys, fields):
            try:
                d[k] = float(f) if "." in f else int(f)
            except ValueError:
                d[k] = f
        d["ts"] = time.time()
        return d
    except Exception as e:  # 尽力而为
        return {"error": str(e)}


def gpu_clocks() -> dict:
    """轻量 nvidia-smi 采样（round 内高频使用，只取时钟/温度/功耗）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        f = out.strip().splitlines()[0].split(",")
        return {"sm_clock_mhz": int(f[0]), "mem_clock_mhz": int(f[1]),
                "temp_c": int(f[2]), "power_w": float(f[3])}
    except Exception:
        return {}


def _rel_diff(a: float, b: float) -> float:
    m = (a + b) / 2.0
    return abs(a - b) / m if m > 0 else 0.0


# check_dvfs_pair / check_dvfs_matrix 位于 stats.py（纯 stdlib，
# 可无 GPU 单元测试）；此处 re-export 方便 bench_v2 使用者。
from .stats import check_dvfs_pair, check_dvfs_matrix  # noqa: E402,F401


def _make_pool(M: int, H: int, dtype: torch.dtype, mode: str,
               seed: int = SEED) -> dict:
    """预分配全部计时张量（计时区域内永不 malloc / 随机数 / copy）。"""
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    xs = [(torch.randn(M, H, generator=g, dtype=torch.float32, device=dev)
           .to(dtype).contiguous()) for _ in range(POOL_SIZE if mode == "streaming" else 1)]
    w = (torch.randn(H, generator=g, dtype=torch.float32, device=dev) * 0.5 + 1.0
         ).to(dtype).contiguous()
    outs = [torch.empty_like(xs[0]) for _ in xs]
    es = 2 if dtype == torch.float16 else 4
    per = M * H * es
    working_set = len(xs) * 2 * per + H * es  # x pool + out pool + w
    return {"xs": xs, "w": w, "outs": outs, "mode": mode,
            "pool_size": len(xs), "working_set_bytes": working_set,
            "element_size": es}


def _measure_block(ext, variant: str, pool: dict,
                   warmup: int = WARMUP, iters: int = ITERS,
                   batch: int = BATCH) -> float:
    """测量一个 variant 在一个 round 内的中位单发时间（us）。

    hot: 固定 buffer 连续启动；streaming: 轮换 buffer（每次启动换 buffer）。
    """
    if pool["mode"] == "hot":
        x, w, out = pool["xs"][0], pool["w"], pool["outs"][0]

        def one():
            ext.forward_into(variant, x, w, out, 1e-5)
    else:
        xs, w, outs, n = pool["xs"], pool["w"], pool["outs"], pool["pool_size"]
        state = {"i": 0}

        def one():
            i = state["i"]
            ext.forward_into(variant, xs[i], w, outs[i], 1e-5)
            state["i"] = (i + 1) % n

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


def _condense_clocks(c: dict) -> dict:
    return {k: c.get(k) for k in ("sm_clock_mhz", "temp_c", "power_w")}


def bench_pair(parent: str, candidate: str, M: int, H: int,
               dtype: torch.dtype = torch.float16, mode: str = "streaming",
               rounds: int = ROUNDS, warmup: int = WARMUP,
               iters: int = ITERS, batch: int = BATCH,
               max_retries: int = MAX_RETRIES, seed: int = SEED,
               ext=None) -> dict:
    """真正的 paired benchmark: 每 round 内 A/B 相邻、同张量、顺序交替。"""
    if mode not in ("hot", "streaming"):
        raise ValueError(f"未知 cache mode: {mode}")
    if ext is None:
        ext = build()
    for v in (parent, candidate):
        if v not in ext.variants():
            raise ValueError(f"未知变体 {v}")

    pool = _make_pool(M, H, dtype, mode, seed)
    es = pool["element_size"]
    algo_bytes = (M * H + H + M * H) * es

    record: dict = {
        "harness": HARNESS_VERSION,
        "parent": parent,
        "candidate": candidate,
        "shape": [M, H],
        "dtype": str(dtype).split(".")[-1],
        "cache_mode": mode,
        "seed": seed,
        "warmup": warmup, "iters": iters, "batch": batch,
        "pool": {"pool_size": pool["pool_size"],
                 "working_set_bytes": pool["working_set_bytes"],
                 "working_set_gt_l2": pool["working_set_bytes"] > L2_BYTES},
        "clock_policy": {"sm_clock_rel_tolerance": DVFS_TOL,
                         "min_valid_rounds": MIN_VALID_ROUNDS,
                         "max_retries": max_retries,
                         "note": "有效时钟 = variant 测量区间前后两次 "
                                 "nvidia-smi 采样的均值（轮询采样近似）"},
        "generated": _now_iso(),
    }
    record["gpu_state_before"] = gpu_state()

    rounds_out: list[dict] = []
    for slot in range(rounds):
        # 顺序: slot 偶数 parent 先, 奇数 candidate 先（确定性交替）
        order = [parent, candidate] if slot % 2 == 0 else [candidate, parent]
        res = None
        for attempt in range(max_retries + 1):
            c0 = gpu_clocks()
            t_first = _measure_block(ext, order[0], pool, warmup, iters, batch)
            c1 = gpu_clocks()
            t_second = _measure_block(ext, order[1], pool, warmup, iters, batch)
            c2 = gpu_clocks()
            t_parent = t_first if order[0] == parent else t_second
            t_cand = t_second if order[0] == parent else t_first
            dvfs_ok, reason, eff_parent, eff_cand = check_dvfs_pair(
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
                    "before_pair": _condense_clocks(c0),
                    "after_first": _condense_clocks(c1),
                    "after_second": _condense_clocks(c2),
                    "eff_sm_parent_mhz": eff_parent,
                    "eff_sm_candidate_mhz": eff_cand,
                },
            }
            if dvfs_ok:
                break
        rounds_out.append(res)

    valid = [r for r in rounds_out if r["valid"]]
    speedups = paired_speedups([r["parent_us"] for r in valid],
                               [r["candidate_us"] for r in valid])
    s = summarize(speedups)
    ci95 = bootstrap_ci(speedups)
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


def bench_matrix(variants: list[str], M: int, H: int,
                 dtype: torch.dtype = torch.float16, mode: str = "streaming",
                 rounds: int = ROUNDS, warmup: int = WARMUP,
                 iters: int = ITERS, batch: int = BATCH,
                 max_retries: int = MAX_RETRIES, seed: int = SEED,
                 ext=None) -> dict:
    """全矩阵 paired round-robin: 每个 round 内所有 variant 按轮换顺序
    相邻测量（同张量池），variant 位置与时间漂移解耦。"""
    if ext is None:
        ext = build()
    pool = _make_pool(M, H, dtype, mode, seed)
    es = pool["element_size"]
    algo_bytes = (M * H + H + M * H) * es
    n = len(variants)

    record: dict = {
        "harness": HARNESS_VERSION,
        "shape": [M, H],
        "dtype": str(dtype).split(".")[-1],
        "cache_mode": mode,
        "seed": seed,
        "warmup": warmup, "iters": iters, "batch": batch,
        "pool": {"pool_size": pool["pool_size"],
                 "working_set_bytes": pool["working_set_bytes"],
                 "working_set_gt_l2": pool["working_set_bytes"] > L2_BYTES},
        "clock_policy": {"sm_clock_rel_tolerance": DVFS_TOL,
                         "min_valid_rounds": MIN_VALID_ROUNDS,
                         "max_retries": max_retries,
                         "note": "round 有效 = 该 round 内所有 variant 有效 "
                                 "SM clock 的极差/均值 <= 5%"},
        "generated": _now_iso(),
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
                per[v] = _measure_block(ext, v, pool, warmup, iters, batch)
                clocks.append(gpu_clocks())
            sms = [c.get("sm_clock_mhz") for c in clocks]
            dvfs_ok, reason, effs = check_dvfs_matrix(sms, tol=DVFS_TOL)
            eff = {v: e for v, e in zip(order, effs)}
            res = {
                "round": slot + 1,
                "order": order,
                "retries": attempt,
                "valid": dvfs_ok,
                "invalid_reason": reason if not dvfs_ok else None,
                "us": {v: round(per[v], 3) for v in order},
                "eff_sm_mhz": {v: eff[v] for v in order},
                "clocks": [{k: c.get(k) for k in
                            ("sm_clock_mhz", "temp_c", "power_w")}
                           for c in clocks],
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


def pytorch_ref_latency(M: int, H: int, dtype: torch.dtype = torch.float16,
                        iters: int = 200, batch: int = 32) -> dict:
    """PyTorch 官方实现的延迟参照（torch 2.4.1 有 F.rms_norm）。

    注意: 这是 "PyTorch implementation context"，不是公平 fused-kernel
    baseline 对比（PyTorch 路径可能含额外 kernel/内存操作）。
    """
    try:
        from torch.nn.functional import rms_norm
    except ImportError:
        return {"available": False, "note": "torch.nn.functional.rms_norm 不存在"}
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)
    x = (torch.randn(M, H, generator=g, dtype=torch.float32, device=dev)
         .to(dtype).contiguous())
    w = (torch.randn(H, generator=g, dtype=torch.float32, device=dev) * 0.5 + 1.0
         ).to(dtype).contiguous()
    try:
        y = rms_norm(x, w)
        torch.cuda.synchronize()
    except Exception as e:
        return {"available": True, "error": f"{type(e).__name__}: {e}",
                "note": "F.rms_norm 在该 dtype/device 上失败；不伪造数字"}
    for _ in range(100):
        rms_norm(x, w)
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        for _ in range(batch):
            rms_norm(x, w)
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop) * 1e3 / batch)
    return {"available": True,
            "median_us": round(statistics.median(times), 3),
            "min_us": round(min(times), 3),
            "n_samples": len(times),
            "note": "PyTorch implementation context（非公平 fused-kernel 对比）"}


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
        s = summarize(ratios)
        out.append({
            "shape": rec["shape"],
            "dtype": rec["dtype"],
            "cache_mode": rec["cache_mode"],
            "winner": winner,
            "runner_up": runner,
            "winner_median_us": pv[winner]["median_us"],
            "runner_up_median_us": pv[runner]["median_us"],
            "median_ratio_runner_over_winner": s["median"],
            "bootstrap_ci_95": bootstrap_ci(ratios),
            "valid_rounds": rec["valid_rounds"],
            "all_variants_median_us": {v: pv[v]["median_us"]
                                       for v in rec["variants"]},
        })
    return out


def save_record(record: dict, out_dir: Path = BENCH_DIR, tag: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    p = out_dir / f"{tag}.json"
    p.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return p
