"""v0.5 Phase 7: GEMV 全 shape matrix 驱动（fp16 全 5 变体 / fp32 子集）。

cmd_bench_full 循环 op.dtypes 全量 dtype; 本驱动按 v0.5 决策范围裁剪:
  --dtype float16  → 全部 5 变体（决策主路径）
  --dtype float32  → baseline + vec4_row（fp32 "natural if supported" 子集;
                     Phase 4 已有 fp32 baseline 全 matrix, 这里只补候选）
tag 规则与 cmd_bench_full 相同: {tag}_M{M}_H{H}_{dtype}_{mode},
winners 写入 {tag}_shape_winners.json（本 dtype 的记录范围内取 winner）。

只新增记录; 不覆盖任何已提交的历史 JSON。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cudalab.operators import get as get_op  # noqa: E402
from cudalab.evaluator.bench import (  # noqa: E402
    bench_matrix, analyze_shape_winners, save_record,
)

ALL5 = ["gemv_baseline", "gemv_vec4_row", "gemv_warp_vec4_b256",
        "gemv_warp_vec4_b512", "gemv_splitk4"]
FP32_SUBSET = ["gemv_baseline", "gemv_vec4_row"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", required=True, choices=["float16", "float32"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--variants", default=None,
                    help="逗号分隔; 缺省按 dtype 取（fp16 全 5 / fp32 子集）")
    a = ap.parse_args()

    import torch
    dtype = (torch.float16 if a.dtype == "float16" else torch.float32)
    op = get_op("gemv")
    ext = op.build()
    variants = ([v.strip() for v in a.variants.split(",") if v.strip()]
                if a.variants else
                (ALL5 if a.dtype == "float16" else FP32_SUBSET))

    records = []
    for (M, H) in op.bench_shapes:
        for mode in ("hot", "streaming"):
            tag = f"{a.tag}_M{M}_H{H}_{a.dtype}_{mode}"
            print(f"=== {tag} ===", flush=True)
            rec = bench_matrix(op, ext, variants, M, H, dtype, mode=mode,
                               rounds=a.rounds)
            save_record(rec, op.bench_dir, tag)
            records.append(rec)
    winners = analyze_shape_winners(records)
    wp = save_record({"harness": "paired-streaming-v2.3",
                      "operator": op.name, "dtype": a.dtype,
                      "shape_winners": winners}, op.bench_dir,
                     f"{a.tag}_shape_winners_{a.dtype}")
    print(f"shape winners -> {wp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
