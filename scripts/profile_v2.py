#!/usr/bin/env python
"""v0.2 NCU 剖析：5 变体 × {cache_control: all, none}，主形状 (128, 4096) fp16。

输出:
    profiles/rmsnorm/v0.2/<variant>_M128_H4096_cc<mode>_clkbase.json
    profiles/rmsnorm/v0.2/v02_profile_comparison.json

方法学说明见 cudalab/profiler.py 模块头：
- cc=all  = ncu 默认，剖析前不失效缓存（热 L2）；
- cc=none = 剖析前失效全部缓存（冷 L2）。
v0.1 的所有 ncu 数据均为默认 cc=all（"cold L2" 说法无配置依据）。
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from _common import ROOT, get_ext
from cudalab.profiler import profile_variant

V02_DIR = ROOT / "profiles" / "rmsnorm" / "v0.2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=None, help="逗号分隔，默认全部")
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--H", type=int, default=4096)
    ap.add_argument("--clock-control", default="base",
                    choices=["base", "none", "reset"])
    args = ap.parse_args()

    ext = get_ext()
    avail = ext.variants()
    variants = (args.variants.split(",") if args.variants else avail)
    for v in variants:
        if v not in avail:
            sys.exit(f"未知变体 {v!r}; 可用: {avail}")

    V02_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for cc in ("all", "none"):
        for v in variants:
            out = V02_DIR / f"{v}_M{args.M}_H{args.H}_cc{cc}_clk{args.clock_control}.json"
            print(f"== profiling {v} cache_control={cc} ==")
            s = profile_variant(v, args.M, args.H, out_path=out,
                                cache_control=cc,
                                clock_control=args.clock_control)
            if "error" in s:
                print(f"   ERROR: {s['error'][:200]}", file=sys.stderr)
            else:
                print(f"   duration={s['kernel_duration_us']}us "
                      f"dram={s['dram_throughput_pct']}% "
                      f"l2_read={s['l2_read_hit_rate']} "
                      f"l1={s['l1_hit_rate']}")
            results.setdefault(v, {})[cc] = s

    cmp_path = V02_DIR / "v02_profile_comparison.json"
    cmp = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "shape": [args.M, args.H],
        "dtype": "float16",
        "clock_control": args.clock_control,
        "clock_lock_warnings": {
            v: cc.get("clock_lock_warning")
            for v, modes in results.items()
            for cc_key, cc in modes.items()
            if cc.get("clock_lock_warning")
        },
        "note": ("cc=all 为 ncu 默认（热缓存，v0.1 所有数据即此模式）；"
                 "cc=none 剖析前失效全部缓存（冷）。kernel_duration 在两种"
                 "缓存状态下不可直接跨模式比较绝对值，只作各自模式内的"
                 "变体相对比较。"),
        "variants": results,
    }
    cmp_path.write_text(json.dumps(cmp, indent=2))
    print(f"comparison -> {cmp_path}")


if __name__ == "__main__":
    main()
