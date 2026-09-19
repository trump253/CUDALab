#!/usr/bin/env python
"""用 ncu 剖析单个 RMSNorm 变体，并保存结构化摘要。

示例:
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

    ext = get_ext()  # 先构建再剖析（ncu 驱动程序自己也会构建）
    if args.variant not in ext.variants():
        sys.exit(f"未知变体 {args.variant!r}; 可用: {ext.variants()}")

    s = profile_variant(args.variant, args.M, args.H)
    if "error" in s:
        print(json.dumps(s, indent=2), file=sys.stderr)
        sys.exit(2)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
