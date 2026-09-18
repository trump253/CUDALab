#!/usr/bin/env python
"""CUDALab optimization entry point (v0.1).

The Harness agent is the outer optimizer: it analyzes profiles, edits the
CUDA sources (adding a new kernels/rmsnorm/rmsnorm_*.cu), then calls this
script, which runs the objective pipeline and records the experiment.

Subcommands:
    baseline          establish the current-best baseline:
                      build -> correctness -> benchmark -> profile -> best.json
    evaluate          run one experiment against the current best:
                      build -> correctness(candidate) -> benchmark matrix
                      (current best + candidate) -> profile(candidate)
                      -> acceptance rules -> experiment record -> update best

Examples:
    $PYTHON scripts/optimize_rmsnorm.py baseline
    $PYTHON scripts/optimize_rmsnorm.py evaluate --variant v1 \
        --hypothesis "vectorized 16B loads reduce instruction count" \
        --changes "float4 loads in both passes" --parent baseline
"""
import argparse
import json
import sys
import time
from pathlib import Path

from _common import ROOT, get_ext
import torch

from cudalab.benchmark import bench_matrix, annotate_speedups, save_bench, BENCH_MATRIX
from cudalab.correctness import run_suite, summarize, save_results
from cudalab.experiment import (evaluate, save_experiment, current_best,
                                set_best, KEEP)
from cudalab.profiler import profile_variant


def _correctness(ext, variant: str):
    results = run_suite(variant, ext)
    summary = summarize(results)
    out = ROOT / "experiments" / "rmsnorm" / "correctness" / f"{variant}.json"
    save_results(results, out)
    rows = [[f"{r.shape} {r.dtype} {r.mode}s{r.seed}",
             "PASS" if r.passed else "FAIL",
             f"{r.max_abs_error:.2e}", f"{r.max_rel_error:.2e}"] for r in results]
    print(f"--- correctness [{variant}]: {summary['n_pass']}/{summary['n_total']} pass")
    for r in summary["failed"][:10]:
        print("   FAIL:", json.dumps(r))
    return summary


def _bench(ext, variants: list[str], tag: str):
    recs = bench_matrix(variants, ext)
    recs = annotate_speedups(recs)
    jp, cp = save_bench(recs, ROOT / "benchmarks", tag)
    print(f"--- benchmark matrix [{tag}]: {len(recs)} records")
    for r in recs:
        print(f"   {r['variant']:>10}  {r['shape']}  med={r['median_us']:>8.2f}us "
              f"p95={r['p95_us']:>8.2f}us  BW={r['effective_bw_gbps']:>6.1f} GB/s")
    return recs


def _profile(variant: str, M=128, H=4096):
    s = profile_variant(variant, M, H)
    if "error" in s:
        print(f"--- profile [{variant}]: ERROR {s['error'][:300]}")
        return s
    print(f"--- profile [{variant}] {s['kernel_duration_us']}us "
          f"DRAM={s['dram_throughput_pct']}% SM={s['sm_throughput_pct']}% "
          f"occ={s['achieved_occupancy_pct']}% regs={s['registers_per_thread']} "
          f"smem={s['shared_memory_bytes']}B")
    top = list(s["warp_stalls"].items())[:3]
    for k, v in top:
        print(f"   stall {k}: {v['stalled_cycles_per_issue']} cyc "
              f"({v['pct_of_stalls']}%)")
    return s


def cmd_baseline(args):
    ext = get_ext(verbose=args.verbose)
    if "baseline" not in ext.variants():
        sys.exit("baseline variant not built")
    cs = _correctness(ext, "baseline")
    recs = _bench(ext, ["baseline"], tag="baseline")
    prof = _profile("baseline")
    rec = next(r for r in recs if r["shape"] == [128, 4096] and r["variant"] == "baseline")
    set_best("baseline", "initial baseline", median_us_target=rec["median_us"])
    save_experiment({
        "id": None,  # assigned
        "kind": "baseline",
        "parent": None,
        "hypothesis": "reference implementation: one block per row, scalar loads, "
                      "two-pass FP32-accumulated reduction",
        "changes": ["new baseline kernel (rmsnorm_baseline.cu)"],
        "correctness": cs,
        "benchmark": {"baseline_median_us": rec["median_us"],
                      "candidate_median_us": rec["median_us"], "speedup": 1.0},
        "profile_observation": json.dumps(prof),
        "decision": "KEEP",
        "notes": "baseline accepted as current best",
    })
    print("[baseline] current best = baseline")


def cmd_evaluate(args):
    ext = get_ext(verbose=args.verbose)
    best = current_best()
    parent = args.parent if args.parent else best["variant"]
    if args.variant not in ext.variants():
        sys.exit(f"variant {args.variant!r} not built; available: {ext.variants()}")

    cs = _correctness(ext, args.variant)
    variants = [parent, args.variant] if parent != args.variant else [args.variant]
    tag = f"exp_{args.variant}_vs_{parent}"
    recs = _bench(ext, variants, tag=tag)
    prof = _profile(args.variant, args.M, args.H)

    verdict = evaluate(recs, parent, cs["all_pass"], cs,
                       primary_shape=(args.M, args.H), dtype="float16")
    decision = verdict["decision"]

    # full-matrix delta vs parent (never cherry-picked)
    matrix = []
    for r in recs:
        if r["variant"] == args.variant:
            p = next((q for q in recs
                      if q["variant"] == parent and q["shape"] == r["shape"]), None)
            matrix.append({
                "shape": r["shape"], "dtype": r["dtype"],
                "parent_median_us": p["median_us"] if p else None,
                "candidate_median_us": r["median_us"],
                "speedup": (round(p["median_us"] / r["median_us"], 4)
                            if p and r["median_us"] else None),
            })
        matrix_all = {
            "parent": [{ "shape": r["shape"], "median_us": r["median_us"]} for r in recs if r["variant"] == parent],
            "candidate": [{"shape": r["shape"], "median_us": r["median_us"]} for r in recs if r["variant"] == args.variant],
        }

    exp = {
        "id": None,
        "parent": parent,
        "variant": args.variant,
        "hypothesis": args.hypothesis,
        "changes": args.changes.split(";") if args.changes else [],
        "correctness": cs,
        "benchmark": {
            "baseline_median_us": verdict["detail"].get("current_best_median_us"),
            "candidate_median_us": verdict["detail"].get("candidate_median_us"),
            "speedup": verdict["detail"].get("speedup_vs_current_best"),
            "primary_shape": verdict["detail"].get("primary_shape"),
            "full_matrix_vs_parent": matrix_all,
        },
        "profile_observation": (json.dumps(prof) if prof else None),
        "decision": decision,
        "decision_rule": verdict["detail"].get("rule"),
        "timestamp": time.time(),
    }
    p = save_experiment(exp)
    print(f"[experiment] {p.name}: {decision} — {verdict['detail'].get('rule')}")

    if decision == KEEP:
        rec = next(r for r in recs
                   if r["shape"] == [args.M, args.H] and r["variant"] == args.variant)
        set_best(args.variant,
                 f"{exp['id']} kept: {verdict['detail'].get('rule')}",
                 median_us_target=rec["median_us"])
        print(f"[experiment] current best -> {args.variant}")
    elif decision == "NEUTRAL":
        print(f"[experiment] current best remains {parent}")
    else:
        print(f"[experiment] current best remains {parent} (REJECT)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("baseline")
    p1.add_argument("--verbose", action="store_true")
    p2 = sub.add_parser("evaluate")
    p2.add_argument("--variant", required=True)
    p2.add_argument("--parent", default=None,
                    help="compare against this variant (default: current best)")
    p2.add_argument("--hypothesis", default="")
    p2.add_argument("--changes", default="")
    p2.add_argument("--M", type=int, default=128)
    p2.add_argument("--H", type=int, default=4096)
    p2.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.cmd == "baseline":
        cmd_baseline(args)
    else:
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
