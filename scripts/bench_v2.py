"""CUDALab v0.2 paired benchmark CLI（paired-streaming-v2 harness）。

子命令:
  pair     主目标精测: parent vs candidate，每 round 相邻 + 顺序交替
  matrix   单形状全变体 round-robin 矩阵
  full     完整矩阵: 全部 shape × dtype × {hot, streaming}
  winners  从已保存的矩阵记录生成 shape-specific winner

示例:
  source tools/env.sh
  $PYTHON scripts/bench_v2.py pair --parent v4_vec_reg --candidate v1_vec \
      --M 128 --H 4096 --dtype float16 --mode streaming \
      --tag v02_pair_v4_vs_v1_primary_streaming
  $PYTHON scripts/bench_v2.py full --tag v02_full
  $PYTHON scripts/bench_v2.py winners --dir benchmarks/v0.2 --tag v02_full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cudalab.build import build               # noqa: E402
from cudalab.bench_v2 import (                # noqa: E402
    BENCH_MATRIX_V2, bench_pair, bench_matrix, analyze_shape_winners,
    pytorch_ref_latency, save_record, BENCH_DIR,
)

import torch  # noqa: E402


def _dtype(s: str) -> torch.dtype:
    return torch.float16 if s == "float16" else torch.float32


def cmd_pair(a) -> int:
    ext = build()
    rec = bench_pair(a.parent, a.candidate, a.M, a.H, _dtype(a.dtype),
                     mode=a.mode, rounds=a.rounds, ext=ext)
    p = save_record(rec, BENCH_DIR, a.tag)
    _print_pair_summary(rec)
    print(f"保存 -> {p}")
    return 0


def cmd_matrix(a) -> int:
    ext = build()
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    rec = bench_matrix(variants, a.M, a.H, _dtype(a.dtype), mode=a.mode,
                       rounds=a.rounds, ext=ext)
    p = save_record(rec, BENCH_DIR, a.tag)
    _print_matrix_summary(rec)
    print(f"保存 -> {p}")
    return 0


def cmd_full(a) -> int:
    ext = build()
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    records = []
    for (M, H) in BENCH_MATRIX_V2:
        for dtype_name in ("float16", "float32"):
            for mode in ("hot", "streaming"):
                tag = f"v02_full_M{M}_H{H}_{dtype_name}_{mode}"
                print(f"=== {tag} ===", flush=True)
                rec = bench_matrix(variants, M, H, _dtype(dtype_name),
                                   mode=mode, rounds=a.rounds, ext=ext)
                p = save_record(rec, BENCH_DIR, tag)
                _print_matrix_summary(rec)
                records.append(rec)
    winners = analyze_shape_winners(records)
    wp = save_record({"harness": "paired-streaming-v2",
                      "shape_winners": winners}, BENCH_DIR, "shape_winners")
    print(f"shape winners -> {wp}")
    return 0


def cmd_winners(a) -> int:
    d = Path(a.dir)
    recs = []
    for p in sorted(d.glob(f"{a.tag}_M*_H*.json")):
        r = json.loads(p.read_text())
        if "per_variant" in r:
            recs.append(r)
    winners = analyze_shape_winners(recs)
    out = d / f"{a.tag}_shape_winners.json"
    out.write_text(json.dumps({"shape_winners": winners}, indent=2))
    _print_winners(winners)
    print(f"保存 -> {out}")
    return 0


def _print_pair_summary(rec: dict):
    print(f"[pair] {rec['parent']} vs {rec['candidate']} "
          f"M={rec['shape'][0]} H={rec['shape'][1]} "
          f"{rec['dtype']} mode={rec['cache_mode']}")
    print(f"  valid rounds: {rec['valid_rounds']}/{rec['n_rounds']} "
          f"(invalid DVFS: {rec['invalid_dvfs_rounds']})")
    print(f"  parent    median: {rec['parent_median_us']} us "
          f"  algo BW: {rec['algorithmic_bw_gbps_parent']} GB/s")
    print(f"  candidate median: {rec['candidate_median_us']} us "
          f"  algo BW: {rec['algorithmic_bw_gbps_candidate']} GB/s")
    print(f"  median speedup (parent/cand): {rec['median_speedup']}  "
          f"CI95: {rec['bootstrap_ci_95']}  faster rounds: {rec['faster_rounds']}")
    c = rec["clocks"] if "clocks" in rec else None
    sms = [r["clocks"].get("eff_sm_parent_mhz") for r in rec["rounds"]
           if r["valid"]]
    if sms:
        print(f"  parent 有效 SM clock 范围: {min(sms):.0f}-{max(sms):.0f} MHz")


def _print_matrix_summary(rec: dict):
    print(f"[matrix] M={rec['shape'][0]} H={rec['shape'][1]} "
          f"{rec['dtype']} mode={rec['cache_mode']} "
          f"valid {rec['valid_rounds']}/{rec['n_rounds']} rounds")
    for v in rec["variants"]:
        pv = rec["per_variant"][v]
        print(f"  {v:<14} median {pv['median_us']} us   "
              f"algo BW {pv['algorithmic_bw_gbps']} GB/s")


def _print_winners(winners: list[dict]):
    for w in winners:
        print(f"M={w['shape'][0]:<5} H={w['shape'][1]:<5} {w['dtype']:<8} "
              f"{w['cache_mode']:<10} winner={w['winner']} "
              f"runner_up={w['runner_up']} "
              f"ratio={w['median_ratio_runner_over_winner']} "
              f"CI95={w['bootstrap_ci_95']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="CUDALab v0.2 paired benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--dtype", default="float16",
                       choices=["float16", "float32"])
        p.add_argument("--mode", default="streaming", choices=["hot", "streaming"])
        p.add_argument("--rounds", type=int, default=9)
        p.add_argument("--tag", required=True, help="输出文件名（benchmarks/v0.2/<tag>.json）")

    p1 = sub.add_parser("pair", help="parent vs candidate 精测")
    p1.add_argument("--parent", required=True)
    p1.add_argument("--candidate", required=True)
    p1.add_argument("--M", type=int, required=True)
    p1.add_argument("--H", type=int, required=True)
    add_common(p1)
    p1.set_defaults(fn=cmd_pair)

    p2 = sub.add_parser("matrix", help="单形状全变体矩阵")
    p2.add_argument("--variants", default="baseline,v1_vec,v2_reg,v3_wideblock,v4_vec_reg")
    p2.add_argument("--M", type=int, required=True)
    p2.add_argument("--H", type=int, required=True)
    add_common(p2)
    p2.set_defaults(fn=cmd_matrix)

    p3 = sub.add_parser("full", help="完整矩阵（全部 shape × dtype × mode）")
    p3.add_argument("--variants", default="baseline,v1_vec,v2_reg,v3_wideblock,v4_vec_reg")
    p3.add_argument("--rounds", type=int, default=9)
    p3.set_defaults(fn=cmd_full)

    p4 = sub.add_parser("winners", help="从矩阵记录生成 shape winner")
    p4.add_argument("--dir", default=str(BENCH_DIR))
    p4.add_argument("--tag", required=True)
    p4.set_defaults(fn=cmd_winners)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
