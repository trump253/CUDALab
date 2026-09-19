#!/usr/bin/env python
"""对 RMSNorm 变体运行 CUDALab 基准矩阵。

示例:
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
                    help="逗号分隔的变体列表（默认: 已构建的全部）")
    ap.add_argument("--shapes", default=None,
                    help="逗号分隔的 'MxH' 列表，例如 128x4096,1024x4096")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--tag", default="matrix", help="输出文件标签")
    args = ap.parse_args()

    ext = get_ext()
    variants = (args.variants.split(",") if args.variants else ext.variants())
    for v in variants:
        if v not in ext.variants():
            sys.exit(f"未知变体 {v!r}; 可用: {ext.variants()}")
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
    print_table(["变体", "M", "H", "中位_us", "p95_us", "最小_us", "最大_us",
                 "有效带宽_GBs", "对baseline加速"], rows)
    print(f"[基准] json -> {jp}\n[基准] csv  -> {cp}")


def torch_dtype():
    import torch
    return torch


if __name__ == "__main__":
    main()
