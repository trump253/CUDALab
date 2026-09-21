"""v0.5 Phase 8: 最终 incumbent (gemv_vec4_row) 独立复核。

新进程、新 9-round paired run（streaming = 决策口径, hot = 副口径）,
不引用 GEMV-0001 的任何已记录数字; 另跑一次完整 correctness + negative
与 PyTorch context。输出独立 JSON 到
experiments/gemv/revalidation/gemv_vec4_row_revalidation.json。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from cudalab.operators import get as get_op  # noqa: E402
from cudalab.evaluator.bench import bench_pair, HARNESS_VERSION  # noqa: E402
from cudalab.evaluator.decision import (  # noqa: E402
    decide_v2, apply_filter_gate, statistical_relation,
)


def main() -> int:
    op = get_op("gemv")
    ext = op.build()
    M, H = 4096, 4096  # 主目标 (N=4096, K=4096)

    corr = op.run_correctness(ext, "gemv_vec4_row")
    neg = op.run_negative(ext)

    out = {
        "revalidation_of": "GEMV-0001 (gemv_vec4_row, KEEP)",
        "harness": HARNESS_VERSION,
        "operator": op.name,
        "shape": [M, H],
        "dtype": "float16",
        "ts": datetime.now(timezone.utc).isoformat(),
        "correctness": {"all_pass": corr["all_pass"],
                        "summary": corr["summary"], "saved": corr["saved"]},
        "negative": neg["summary"],
        "modes": {},
    }

    ok = corr["all_pass"] and neg["summary"]["all_pass"]
    for mode in ("streaming", "hot"):
        paired = bench_pair(op, ext, "gemv_baseline", "gemv_vec4_row",
                            M, H, torch.float16, mode=mode, rounds=9)
        decision, detail = decide_v2(
            ok, paired["valid_rounds"], paired["speedups"],
            paired["bootstrap_ci_95"])
        decision, detail = apply_filter_gate(
            decision, detail, paired.get("filter_sensitive", False),
            paired.get("filter_sensitive_reason", "未评估"))
        detail["statistical_relation"] = statistical_relation(
            paired["bootstrap_ci_95"])
        detail["policy_decision"] = decision
        out["modes"][mode] = {
            "parent_median_us": paired["parent_median_us"],
            "candidate_median_us": paired["candidate_median_us"],
            "valid_rounds": paired["valid_rounds"],
            "speedups": paired["speedups"],
            "bootstrap_ci_95": paired["bootstrap_ci_95"],
            "filter_sensitive": paired.get("filter_sensitive"),
            "decision_block": detail,
        }
        print(f"[{mode}] baseline={paired['parent_median_us']:.3f}us "
              f"vec4_row={paired['candidate_median_us']:.3f}us "
              f"CI95={paired['bootstrap_ci_95']} "
              f"rel={detail['statistical_relation']} "
              f"policy={decision}")

    out["pytorch_context"] = op.pytorch_ref_latency(
        M, H, torch.float16)
    print("[pytorch] ", json.dumps(out["pytorch_context"],
                                   ensure_ascii=False)[:300])

    p = Path("experiments/gemv/revalidation")
    p.mkdir(parents=True, exist_ok=True)
    outp = p / "gemv_vec4_row_revalidation.json"
    outp.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("->", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
