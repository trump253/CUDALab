#!/usr/bin/env python
"""Run the CUDALab correctness suite for one or all RMSNorm variants.

Examples:
    $PYTHON scripts/test_rmsnorm.py --variant baseline
    $PYTHON scripts/test_rmsnorm.py
"""
import argparse
import json
import sys
from pathlib import Path

from _common import ROOT, get_ext, print_table
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None, help="one variant (default: all)")
    ap.add_argument("--dtypes", default="float16,float32")
    ap.add_argument("--no-edge", action="store_true", help="skip edge cases")
    ap.add_argument("--out", default=None, help="explicit JSON output path")
    args = ap.parse_args()

    ext = get_ext()
    variants = [args.variant] if args.variant else ext.variants()
    dtypes = tuple(args.dtypes.split(","))
    for v in variants:
        if v not in ext.variants():
            sys.exit(f"unknown variant {v!r}; available: {ext.variants()}")

    from cudalab.correctness import run_suite, summarize, save_results
    for v in variants:
        results = run_suite(v, ext, dtypes=dtypes, edge=not args.no_edge)
        s = summarize(results)
        rows = [[r.variant, r.shape, r.dtype, r.mode, r.seed,
                 "PASS" if r.passed else "FAIL",
                 f"{r.max_abs_error:.2e}", f"{r.max_rel_error:.2e}"]
                for r in results]
        print_table(["variant", "shape", "dtype", "mode", "seed", "result",
                     "max_abs_err", "max_rel_err"], rows)
        print(f"[{v}] {s['n_pass']}/{s['n_total']} pass, "
              f"max_abs={s['max_abs_error']:.2e} max_rel={s['max_rel_error']:.2e}")
        out = Path(args.out) if args.out else \
            ROOT / "experiments" / "rmsnorm" / "correctness" / f"{v}.json"
        p = save_results(results, out)
        print(f"[{v}] results -> {p}")
    ok = True
    for v in variants:
        p = ROOT / "experiments" / "rmsnorm" / "correctness" / f"{v}.json"
        if p.exists() and not json.loads(p.read_text())["summary"]["all_pass"]:
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
