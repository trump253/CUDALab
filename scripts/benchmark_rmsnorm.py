#!/usr/bin/env python
"""Run the CUDALab benchmark matrix for RMSNorm variants.

Examples:
    $PYTHON scripts/benchmark_rmsnorm.py --variants baseline
    $PYTHON scripts/benchmark_rmsnorm.py --tag v01_final
"""
import argparse
import sys
from pathlib import Path

from _common import ROOT, get_ext, print_table
from cudalab.benchmark import bench_matrix, annotate_speedups, save_bench, BENCH_MATRIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=None,
                    help="comma-separated variants (default: all built)")
    ap.add_argument("--shapes", default=None,
                    help="comma-separated 'MxH' list, e.g. 128x4096,1024x4096")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--tag", default="matrix", help="output file tag")
    args = ap.parse_args()

    ext = get_ext()
    variants = (args.variants.split(",") if args.variants else ext.variants())
    for v in variants:
        if v not in ext.variants():
            sys.exit(f"unknown variant {v!r}; available: {ext.variants()}")
    shapes = BENCH_MATRIX
    if args.shapes:
        shapes = [tuple(int(x) for x in s.split("x")) for s in args.shapes.split(",")]
    dtype = getattr(torch_dtype(), args.dtype)

    recs = bench_matrix(variants, ext, shapes=shapes, dtype=dtype)
    recs = annotate_speedups(recs)
    jp, cp = save_bench(recs, ROOT / "benchmarks", args.tag)

    rows = [[r["variant"], r["shape"][0], r["shape"][1],
             f'{r["median_us"]:.2f}', f'{r["p95_us"]:.2f}',
             f'{r["min_us"]:.2f}', f'{r["max_us"]:.2f}',
             f'{r["effective_bw_gbps"]:.1f}',
             f'{r["speedup_vs_cuda_baseline"]:.3f}'
             if r["speedup_vs_cuda_baseline"] else "-"]
            for r in recs]
    print_table(["variant", "M", "H", "median_us", "p95_us", "min_us", "max_us",
                 "eff_BW_GBs", "speedup_vs_baseline"], rows)
    print(f"[bench] json -> {jp}\n[bench] csv  -> {cp}")


def torch_dtype():
    import torch
    return torch


if __name__ == "__main__":
    main()
