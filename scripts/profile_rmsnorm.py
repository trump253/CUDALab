#!/usr/bin/env python
"""Profile one RMSNorm variant with ncu and save a structured summary.

Example:
    $PYTHON scripts/profile_rmsnorm.py --variant baseline --M 128 --H 4096
"""
import argparse
import json
import sys

from _common import ROOT, get_ext
from cudalab.profiler import profile_variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--H", type=int, default=4096)
    args = ap.parse_args()

    ext = get_ext()  # build BEFORE profiling (ncu driver also builds itself)
    if args.variant not in ext.variants():
        sys.exit(f"unknown variant {args.variant!r}; available: {ext.variants()}")

    s = profile_variant(args.variant, args.M, args.H)
    if "error" in s:
        print(json.dumps(s, indent=2), file=sys.stderr)
        sys.exit(2)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
