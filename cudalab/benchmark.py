"""CUDALab benchmark harness.

Methodology (fixed for all variants — no per-candidate tuning of the
measurement):
- torch.cuda.Event timing with explicit synchronization; NEVER wall-clock
  Python time on the GPU path.
- Compilation must finish BEFORE any timing (callers: build first).
- Fixed input tensors: the SAME x/w tensors (same values) are reused for
  every variant of a given (shape, dtype). Inputs are generated once.
- Per (variant, shape, dtype):
    warmup iterations (>=100) untimed,
    then `iters` (>=200) timed iterations per round,
    `rounds` (>=5) independent rounds, each round = fresh warmup + timed
    batch. Per-iteration event pairs.
- Primary metric: MEDIAN of all per-iteration samples (across rounds).
  Also report p95, min, max, and per-round medians (for the decision rule
  that requires "most rounds consistently faster").
- Effective DRAM bandwidth: (M*H*2 + H*2 + M*H*2) bytes / median time
  (fp16). This is the memory traffic the kernel must move; reported as a
  derived number, never fabricated.

Fairness:
- Same GPU (CUDA_VISIBLE_DEVICES=0), same inputs, same shape, same dtype.
- GPU state (nvidia-smi) snapshot before/after each variant's suite run.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

WARMUP = 150
ITERS = 300
ROUNDS = 5

# Benchmark matrix: (M, H). Primary optimization target: (128, 4096).
BENCH_MATRIX = [
    (1, 4096),
    (16, 4096),
    (128, 4096),
    (1024, 4096),
    (128, 8192),
    (1, 1024),
    (128, 1024),
]
PRIMARY_TARGET = (128, 4096)


def gpu_state() -> dict:
    """Snapshot key GPU counters via nvidia-smi (best effort)."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout
        fields = out.strip().splitlines()[0].split(",")
        keys = ["gpu_index", "name", "temp_c", "sm_clock_mhz", "mem_clock_mhz",
                "power_w", "gpu_util_pct", "mem_util_pct"]
        d = {}
        for k, f in zip(keys, fields):
            k2 = k.replace(" ", "_")
            try:
                d[k2] = float(f) if "." in f else int(f)
            except ValueError:
                d[k2] = f
        d["name"] = fields[1]
        d["ts"] = time.time()
        return d
    except Exception as e:  # best effort, never fatal
        return {"error": str(e)}


def _time_one_round(fn, warmup: int, iters: int) -> list[float]:
    """One independent round: untimed warmup, then `iters` event-timed calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        stops[i].record()
    torch.cuda.synchronize()
    for i in range(iters):
        times.append(starts[i].elapsed_time(stops[i]) * 1e3)  # ms -> us
    return times


def bench_variant(variant: str, ext, M: int, H: int,
                  dtype: torch.dtype = torch.float16,
                  warmup: int = WARMUP, iters: int = ITERS,
                  rounds: int = ROUNDS) -> dict:
    """Benchmark one (variant, shape, dtype). Returns a full record."""
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(1234)
    x = (torch.randn(M, H, generator=g, dtype=torch.float32, device=dev)
         .to(dtype).contiguous())
    w = (torch.randn(H, generator=g, dtype=torch.float32, device=dev) * 0.5 + 1.0
         ).to(dtype).contiguous()

    out = torch.empty_like(x)

    def fn():
        # pre-allocated output: timed region = kernel launch only,
        # no allocation, no copy.
        ext.forward_into(variant, x, w, out, 1e-5)

    state_before = gpu_state()
    all_times: list[float] = []
    round_medians: list[float] = []
    for _ in range(rounds):
        t = _time_one_round(fn, warmup, iters)
        all_times.extend(t)
        round_medians.append(statistics.median(t))
    state_after = gpu_state()

    n = M * H
    bytes_moved = (n * 2) + (H * 2) + (n * 2)  # read x + read w + write y (fp16)
    med_us = statistics.median(all_times)
    p95_us = float(sorted(all_times)[int(0.95 * len(all_times)) - 1])

    rec = {
        "variant": variant,
        "shape": [M, H],
        "dtype": str(dtype).split(".")[-1],
        "median_us": round(med_us, 3),
        "p95_us": round(p95_us, 3),
        "min_us": round(min(all_times), 3),
        "max_us": round(max(all_times), 3),
        "round_medians_us": [round(r, 3) for r in round_medians],
        "n_samples": len(all_times),
        "warmup": warmup, "iters": iters, "rounds": rounds,
        "effective_bw_gbps": round(bytes_moved / (med_us * 1e-6) / 1e9, 1),
        "gpu_state_before": state_before,
        "gpu_state_after": state_after,
    }
    del x, w, out
    torch.cuda.empty_cache()
    return rec


def bench_matrix(variants: list[str], ext, shapes: list = BENCH_MATRIX,
                 dtype: torch.dtype = torch.float16) -> list[dict]:
    recs = []
    for v in variants:
        for (M, H) in shapes:
            recs.append(bench_variant(v, ext, M, H, dtype))
    return recs


def annotate_speedups(recs: list[dict],
                      baseline: str = "baseline",
                      pytorch_ref_fn=None) -> list[dict]:
    """Add speedup_vs_cuda_baseline / speedup_vs_pytorch_reference columns."""
    by_shape = {}
    for r in recs:
        by_shape.setdefault((tuple(r["shape"]), r["dtype"]), {})[r["variant"]] = r
    for r in recs:
        base = by_shape.get((tuple(r["shape"]), r["dtype"]), {}).get(baseline)
        r["speedup_vs_cuda_baseline"] = (
            round(base["median_us"] / r["median_us"], 4) if base else None)
        r["speedup_vs_pytorch_reference"] = None
    return recs


def save_bench(recs: list[dict], out_dir: Path, tag: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"bench_{tag}.json"
    jp.write_text(json.dumps(recs, indent=2))
    cp = out_dir / f"bench_{tag}.csv"
    if recs:
        cols = ["variant", "M", "H", "dtype", "median_us", "p95_us", "min_us",
                "max_us", "effective_bw_gbps", "speedup_vs_cuda_baseline",
                "speedup_vs_pytorch_reference"]
        lines = [",".join(cols)]
        for r in recs:
            lines.append(",".join(str({
                "M": r["shape"][0], "H": r["shape"][1],
            }.get(c, r.get(c, ""))) for c in cols))
        cp.write_text("\n".join(lines) + "\n")
    return jp, cp
