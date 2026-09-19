#!/usr/bin/env python
"""CUDALab 优化入口（v0.1）。

外层 Harness 智能体是优化者：它分析剖析数据、编辑 CUDA 源码
（新增 kernels/rmsnorm/rmsnorm_*.cu），然后调用本脚本；脚本运行
客观流水线并记录实验。

子命令:
    baseline          建立当前最佳基线:
                      构建 -> 正确性 -> 基准 -> 剖析 -> best.json
    evaluate          针对当前最佳运行一个实验:
                      构建 -> 正确性(候选) -> 基准矩阵
                      (当前最佳 + 候选) -> 剖析(候选)
                      -> 采纳规则 -> 实验记录 -> 更新最佳

示例:
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

from cudalab.benchmark import (bench_matrix, annotate_speedups, save_bench,
                               BENCH_MATRIX, HARNESS_VERSION)
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
    print(f"--- 正确性 [{variant}]: 通过 {summary['n_pass']}/{summary['n_total']}")
    for r in summary["failed"][:10]:
        print("   FAIL:", json.dumps(r))
    return summary


def _bench(ext, variants: list[str], tag: str):
    recs = bench_matrix(variants, ext)
    recs = annotate_speedups(recs)
    jp, cp = save_bench(recs, ROOT / "benchmarks", tag)
    print(f"--- 基准矩阵 [{tag}]: {len(recs)} 条记录")
    for r in recs:
        print(f"   {r['variant']:>10}  {r['shape']}  中位={r['median_us']:>8.2f}us "
              f"p95={r['p95_us']:>8.2f}us  带宽={r['effective_bw_gbps']:>6.1f} GB/s")
    return recs


def _profile(variant: str, M=128, H=4096):
    s = profile_variant(variant, M, H)
    if "error" in s:
        print(f"--- 剖析 [{variant}]: 错误 {s['error'][:300]}")
        return s
    print(f"--- 剖析 [{variant}] {s['kernel_duration_us']}us "
          f"DRAM={s['dram_throughput_pct']}% SM={s['sm_throughput_pct']}% "
          f"占用率={s['achieved_occupancy_pct']}% 寄存器={s['registers_per_thread']} "
          f"共享内存={s['shared_memory_bytes']}B")
    top = list(s["warp_stalls"].items())[:3]
    for k, v in top:
        print(f"   停顿 {k}: {v['stalled_cycles_per_issue']} 周期 "
              f"({v['pct_of_stalls']}%)")
    return s


def cmd_baseline(args):
    ext = get_ext(verbose=args.verbose)
    if "baseline" not in ext.variants():
        sys.exit("未构建 baseline 变体")
    cs = _correctness(ext, "baseline")
    recs = _bench(ext, ["baseline"], tag="baseline")
    prof = _profile("baseline")
    rec = next(r for r in recs if r["shape"] == [128, 4096] and r["variant"] == "baseline")
    set_best("baseline", "initial baseline", median_us_target=rec["median_us"])
    save_experiment({
        "id": None,  # 自动分配
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
        "harness": HARNESS_VERSION,
        "notes": "baseline accepted as current best",
    })
    print("[基线] 当前最佳 = baseline")


def cmd_evaluate(args):
    ext = get_ext(verbose=args.verbose)
    best = current_best()
    parent = args.parent if args.parent else best["variant"]
    if args.variant not in ext.variants():
        sys.exit(f"变体 {args.variant!r} 未构建; 可用: {ext.variants()}")

    cs = _correctness(ext, args.variant)
    variants = [parent, args.variant] if parent != args.variant else [args.variant]
    tag = f"exp_{args.variant}_vs_{parent}"
    recs = _bench(ext, variants, tag=tag)
    prof = _profile(args.variant, args.M, args.H)

    verdict = evaluate(recs, parent, cs["all_pass"], cs,
                       primary_shape=(args.M, args.H), dtype="float16")
    decision = verdict["decision"]

    # 相对父变体的完整矩阵差异（从不挑拣）
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
        "harness": HARNESS_VERSION,
        "timestamp": time.time(),
    }
    p = save_experiment(exp)
    print(f"[实验] {p.name}: {decision} — {verdict['detail'].get('rule')}")

    if decision == KEEP:
        rec = next(r for r in recs
                   if r["shape"] == [args.M, args.H] and r["variant"] == args.variant)
        set_best(args.variant,
                 f"{exp['id']} kept: {verdict['detail'].get('rule')}",
                 median_us_target=rec["median_us"])
        print(f"[实验] 当前最佳 -> {args.variant}")
    elif decision == "NEUTRAL":
        print(f"[实验] 当前最佳保持 {parent}")
    else:
        print(f"[实验] 当前最佳保持 {parent}（REJECT）")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("baseline")
    p1.add_argument("--verbose", action="store_true")
    p2 = sub.add_parser("evaluate")
    p2.add_argument("--variant", required=True)
    p2.add_argument("--parent", default=None,
                    help="与该变体比较（默认: 当前最佳）")
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
