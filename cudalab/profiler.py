"""CUDALab GPU profiler integration (Nsight Compute 2022.3, sm_75).

Runs `ncu --csv` over a tiny profile-only driver and converts the report
into a structured JSON summary. Null fields when a metric is unavailable —
never fabricated numbers. Raw ncu output (stdout+stderr) is always kept
under profiles/rmsnorm/raw/ for audit.

Metric names verified via `ncu --query-metrics` on this GPU/ncu version.

Interpretation notes (sm_75, NCU 2022.3):
- `gpu__time_duration.sum` is reported in **nsecond** by --csv; converted
  to microseconds here.
- stall reasons use `smsp__average_warps_issue_stalled_<r>_per_issue_active.ratio`
  = stalled warp-cycles per issue-active cycle (unit `inst`). The derived
  `.pct` field for these metrics is not meaningful (>100%), so percentages
  are computed here as share of the sum of all stall reasons.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROF_DIR = ROOT / "profiles" / "rmsnorm"

NCU = "/usr/local/bin/ncu"
# ncu in this container cannot exec a symlinked interpreter; use the real binary.
PYTHON_REAL = "/root/miniconda3/envs/pytorch/bin/python3.10"

# Verified-available metrics (NCU 2022.3, sm_75).
METRICS = [
    "gpu__time_duration.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active",
    "smsp__average_warps_issue_stalled_wait_per_issue_active",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active",
    "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active",
    "smsp__average_warps_issue_stalled_drain_per_issue_active",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active",
    "smsp__average_warps_issue_stalled_selected_per_issue_active",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active",
    "smsp__average_warps_issue_stalled_sleeping_per_issue_active",
    "smsp__average_warps_issue_stalled_membar_per_issue_active",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active",
]

STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active"


def _driver_source(variant: str, M: int, H: int) -> str:
    return f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import torch
from cudalab.build import build

ext = build()
assert "{variant}" in ext.variants(), ext.variants()
g = torch.Generator(device="cuda"); g.manual_seed(1234)
x = torch.randn({M}, {H}, generator=g, dtype=torch.float32, device="cuda").half().contiguous()
w = (torch.randn({H}, generator=g, dtype=torch.float32, device="cuda") * 0.5 + 1.0).half().contiguous()
out = torch.empty_like(x)
# 2 untimed warmup launches (ncu --launch-skip 2), then 4 profiled
for _ in range(6):
    ext.forward_into("{variant}", x, w, out, 1e-5)
torch.cuda.synchronize()
print("profile driver done")
"""


def _parse_csv_launches(text: str) -> tuple[list[dict], str | None]:
    """Parse ncu --csv output. Returns (list of per-launch metric dicts, kernel name)."""
    # strip non-CSV progress lines before the header
    lines = text.splitlines()
    start = 0
    for i, ln in enumerate(lines):
        if ln.startswith('"ID"'):
            start = i
            break
    if start == 0 and not text.lstrip().startswith('"ID"'):
        return [], None
    rows = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
    header = rows[0]
    idx = {name: i for i, name in enumerate(header)}
    launches: dict[str, dict] = {}
    kernel_name = None
    for r in rows[1:]:
        if len(r) < len(header):
            continue
        lid = r[idx["ID"]]
        m = launches.setdefault(lid, {})
        metric = r[idx["Metric Name"]]
        unit = r[idx["Metric Unit"]]
        try:
            val = float(r[idx["Metric Value"]])
        except ValueError:
            continue
        kernel_name = r[idx["Kernel Name"]]
        m[(metric, unit)] = val
    return list(launches.values()), kernel_name


def _avg(launches: list[dict], metric: str, unit: str = None):
    vals = []
    for d in launches:
        for (m, u), v in d.items():
            if m == metric and (unit is None or u == unit):
                vals.append(v)
    if not vals:
        return None
    return sum(vals) / len(vals)


def profile_variant(variant: str, M: int = 128, H: int = 4096,
                    out_path: Path | None = None, ncu: str = NCU,
                    launch_skip: int = 2, launch_count: int = 4) -> dict:
    """Profile one variant at one shape; returns (and saves) the summary."""
    PROF_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = PROF_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)
    if out_path is None:
        out_path = PROF_DIR / f"{variant}_M{M}_H{H}.json"

    drv = raw_dir / f"{variant}_M{M}_H{H}_drv.py"
    drv.write_text(_driver_source(variant, M, H))
    raw_txt = raw_dir / f"{variant}_M{M}_H{H}.ncu.txt"

    cmd = [
        ncu, "--csv",
        "-k", "regex:rmsnorm",
        "--launch-skip", str(launch_skip),
        "--launch-count", str(launch_count),
        "--metrics", ",".join(METRICS),
        PYTHON_REAL, str(drv),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    except FileNotFoundError:
        raise RuntimeError(f"ncu not found at {ncu}")
    raw_txt.write_text("$ ncu cmd\n" + " ".join(cmd) + "\n\n=== stdout ===\n"
                       + proc.stdout + "\n=== stderr ===\n" + proc.stderr)

    summary: dict = {
        "variant": variant,
        "shape": [M, H],
        "ncu_version": _ncu_version(ncu),
        "kernel_name": None,
        "kernel_duration_us": None,
        "dram_throughput_pct": None,
        "sm_throughput_pct": None,
        "achieved_occupancy_pct": None,
        "registers_per_thread": None,
        "shared_memory_bytes": None,
        "warp_stalls": {},
        "n_launches_profiled": 0,
        "raw_report": str(raw_txt),
    }

    if proc.returncode != 0:
        summary["error"] = (proc.stderr + proc.stdout).strip()[-2000:]
        out_path.write_text(json.dumps(summary, indent=2))
        return summary

    launches, kernel_name = _parse_csv_launches(proc.stdout)
    if not launches:
        summary["error"] = "ncu exited 0 but no metric rows parsed; see raw report"
        out_path.write_text(json.dumps(summary, indent=2))
        return summary

    summary["n_launches_profiled"] = len(launches)
    summary["kernel_name"] = kernel_name

    def pct(metric: str):
        v = _avg(launches, metric, "%")
        return round(v, 2) if v is not None else None

    dur_ns = _avg(launches, "gpu__time_duration.sum", "nsecond")
    summary["kernel_duration_us"] = round(dur_ns / 1e3, 3) if dur_ns is not None else None
    summary["dram_throughput_pct"] = pct(
        "dram__throughput.avg.pct_of_peak_sustained_elapsed")
    summary["sm_throughput_pct"] = pct(
        "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    summary["achieved_occupancy_pct"] = pct(
        "sm__warps_active.avg.pct_of_peak_sustained_active")
    rgt = _avg(launches, "launch__registers_per_thread", "register/thread")
    summary["registers_per_thread"] = int(rgt) if rgt is not None else None
    sm_s = _avg(launches, "launch__shared_mem_per_block_static", "byte/block")
    sm_d = _avg(launches, "launch__shared_mem_per_block_dynamic", "byte/block")
    if sm_s is not None or sm_d is not None:
        summary["shared_memory_bytes"] = int((sm_s or 0) + (sm_d or 0))

    # stall reasons: .ratio = stalled warp-cycles per issue-active cycle
    stalls = {}
    for m in METRICS:
        if m.startswith(STALL_PREFIX) and m.endswith(STALL_SUFFIX):
            reason = m[len(STALL_PREFIX):-len(STALL_SUFFIX)]
            v = _avg(launches, m + ".ratio", "inst")
            if v is not None:
                stalls[reason] = round(v, 3)
    if stalls:
        total = sum(stalls.values())
        summary["warp_stalls"] = {
            k: {"stalled_cycles_per_issue": v,
                "pct_of_stalls": round(100.0 * v / total, 1) if total > 0 else 0.0}
            for k, v in sorted(stalls.items(), key=lambda kv: -kv[1])
        }
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def _ncu_version(ncu: str) -> str:
    try:
        out = subprocess.run([ncu, "--version"], capture_output=True, text=True,
                             timeout=30).stdout
        m = re.search(r"Version\s+([\d.]+)", out)
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    import sys
    from .build import build
    v = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    s = profile_variant(v, 128, 4096)
    print(json.dumps(s, indent=2))
