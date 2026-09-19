#!/usr/bin/env python
"""对单个（或全部）RMSNorm 变体运行 CUDALab 正确性套件。

示例:
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
    ap.add_argument("--variant", default=None, help="单个变体（默认: 全部）")
    ap.add_argument("--dtypes", default="float16,float32")
    ap.add_argument("--no-edge", action="store_true", help="跳过边界用例")
    ap.add_argument("--out", default=None, help="显式指定 JSON 输出路径")
    args = ap.parse_args()

    ext = get_ext()
    variants = [args.variant] if args.variant else ext.variants()
    dtypes = tuple(args.dtypes.split(","))
    for v in variants:
        if v not in ext.variants():
            sys.exit(f"未知变体 {v!r}; 可用: {ext.variants()}")

    from cudalab.correctness import run_suite, summarize, save_results
    for v in variants:
        results = run_suite(v, ext, dtypes=dtypes, edge=not args.no_edge)
        s = summarize(results)
        rows = [[r.variant, r.shape, r.dtype, r.mode, r.seed,
                 "PASS" if r.passed else "FAIL",
                 f"{r.max_abs_error:.2e}", f"{r.max_rel_error:.2e}"]
                for r in results]
        print_table(["变体", "形状", "dtype", "模式", "种子", "结果",
                     "max_abs_err", "max_rel_err"], rows)
        print(f"[{v}] 通过 {s['n_pass']}/{s['n_total']}, "
              f"max_abs={s['max_abs_error']:.2e} max_rel={s['max_rel_error']:.2e}")
        out = Path(args.out) if args.out else \
            ROOT / "experiments" / "rmsnorm" / "correctness" / f"{v}.json"
        p = save_results(results, out)
        print(f"[{v}] 结果 -> {p}")
    ok = True
    for v in variants:
        p = ROOT / "experiments" / "rmsnorm" / "correctness" / f"{v}.json"
        if p.exists() and not json.loads(p.read_text())["summary"]["all_pass"]:
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
