#!/usr/bin/env python
# CUDALab v0.7 回归验证生成器（append-only, 可复现）
#
# 输出目录: experiments/regression/v0.7/（v0.7 专属 append-only 目录;
# 历史 experiment artifact 不可变, v0.5/v0.6 回归目录不写入）。
#
# 覆盖:
#   int4gemv/: 全部正常变体（baseline / vec16_row / rowtile4 /
#              rowtile4_hx / rowtile8）correctness + per-variant negative
#   gemv/:     4 个正常变体（gemv_splitk4 被隔离, 不在正常列表, 不运行;
#              隔离状态在 quarantine_audit.json 记录）
#   qgemv/:    5 个正常变体（quarantine 集为空）
#   quarantine_audit.json: 3 个算子的 variants()/all_variants()/
#              quarantined_variants() 绑定列表审计
#
# 用法: $PYTHON scripts/regression_v07.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cudalab.operators.int4gemv import Int4gemvOperator
from cudalab.operators.gemv import GemvOperator
from cudalab.operators.qgemv import QgemvOperator

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "experiments" / "regression" / "v0.7"

OPS = [
    ("int4gemv", Int4gemvOperator),
    ("gemv", GemvOperator),
    ("qgemv", QgemvOperator),
]


def main() -> None:
    summary = {"ops": {}, "quarantine": {}}
    for name, opcls in OPS:
        op = opcls()
        ext = op.build()
        out_dir = REG / name
        out_dir.mkdir(parents=True, exist_ok=True)
        vs = op.variants(ext)
        ok_all = True
        per_variant = {}
        for v in vs:
            c = op.run_correctness(ext, v, out_dir=out_dir)
            n = op.run_negative(ext, v, out_dir=out_dir)
            c_ok = bool(c.get("all_pass"))
            n_ok = bool(n.get("summary", {}).get("all_pass"))
            ok_all = ok_all and c_ok and n_ok
            per_variant[v] = {
                "correctness_all_pass": c_ok,
                "correctness_n_pass": c.get("summary", {}).get("n_pass"),
                "negative_all_pass": n_ok,
                "negative_n_passed": n.get("summary", {}).get("n_passed"),
            }
            print(f"[{name}] {v}: correctness={c_ok} "
                  f"({per_variant[v]['correctness_n_pass']}), "
                  f"negative={n_ok} ({per_variant[v]['negative_n_passed']})")
        normal = vs
        allv = ext.all_variants()
        qv = op.unsafe_variants(ext)
        leaked = [v for v in qv if v in normal]
        summary["ops"][name] = {
            "variants_run": per_variant,
            "all_pass": ok_all,
            "quarantine_leaked_into_normal": leaked,
        }
        summary["quarantine"][name] = {
            "variants()": normal,
            "all_variants()": allv,
            "quarantined_variants()": qv,
        }
    (REG / "quarantine_audit.json").write_text(
        json.dumps(summary["quarantine"], indent=2))
    (REG / "summary.json").write_text(json.dumps(summary["ops"], indent=2))
    bad = [k for k, d in summary["ops"].items() if not d["all_pass"]]
    print("SUMMARY all_pass" if not bad else f"SUMMARY FAILED: {bad}")


if __name__ == "__main__":
    main()
