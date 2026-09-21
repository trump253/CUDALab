#!/usr/bin/env python
"""CUDALab v0.3 — 统一 CLI（命令 → 算子，实现形式从简，无框架）。

用法（先 `source tools/env.sh`）:

  # 测试: 完整正确性套件 + negative suite（单变体）
  $PYTHON scripts/cudalab.py test softmax --variant softmax_baseline
  $PYTHON scripts/cudalab.py test rmsnorm --variant v4_vec_reg

  # 基准（paired-streaming-v2 引擎）
  $PYTHON scripts/cudalab.py benchmark softmax pair \
      --parent softmax_baseline --candidate sfm01_vec --M 128 --H 4096 \
      --dtype float16 --mode streaming --tag sfm01_pair_primary_streaming
  $PYTHON scripts/cudalab.py benchmark softmax matrix \
      --variants softmax_baseline,sfm01_vec --M 128 --H 4096 --tag ...
  $PYTHON scripts/cudalab.py benchmark softmax full --rounds 9
  $PYTHON scripts/cudalab.py benchmark softmax winners --tag sfm_full

  # NCU 剖析（cache_control all/none 语义见 cudalab/evaluator/profiler.py）
  $PYTHON scripts/cudalab.py profile softmax --variants softmax_baseline \
      --M 128 --H 4096

  # PyTorch implementation context（非决策依据）
  $PYTHON scripts/cudalab.py pytorch softmax --M 128 --H 4096 --dtype float16

  # 一次优化实验: 正确性 + negative + paired 精测（+ 可选 NCU）
  # → 固定决策规则 → 实验记录（失败实验同样保存）
  $PYTHON scripts/cudalab.py optimize softmax --id SFM-0001 \
      --parent softmax_baseline --candidate sfm01_vec \
      --hypothesis "..." --changes "..." \
      --M 128 --H 4096 --dtype float16 --mode streaming [--profile]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from cudalab.operators import get as get_op  # noqa: E402
from cudalab.evaluator.bench import (  # noqa: E402
    bench_pair, bench_matrix, analyze_shape_winners, save_record,
    HARNESS_VERSION,
)
from cudalab.evaluator.decision import (  # noqa: E402
    decide_v2, apply_filter_gate, statistical_relation,
)
from cudalab.evaluator.experiment import save_experiment  # noqa: E402
from cudalab.evaluator import profiler as ncu  # noqa: E402


def _dtype(s: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[s]


def _variant_gate_error(op, ext, v: str) -> str | None:
    """统一 CLI 的变体门禁（v0.3.1 quarantine 语义）: 正常可 dispatch
    变体返回 None；被隔离变体返回隔离提示；未知变体返回 None（由
    调用方按原逻辑报 "未知变体"）。"""
    if v in op.variants(ext):
        return None
    if v in op.unsafe_variants(ext):
        return (f"变体 {v!r} 已被隔离（UNSAFE_HISTORICAL_EXPERIMENT / "
                f"REJECTED / NOT_FOR_NORMAL_DISPATCH），不能进入正常"
                f"测试 / 基准 / 剖析路径；隔离理由见该变体的实验记录")
    return None


def _print_pair(rec: dict):
    print(f"[pair] {rec['parent']} vs {rec['candidate']} "
          f"M={rec['shape'][0]} H={rec['shape'][1]} "
          f"{rec['dtype']} mode={rec['cache_mode']}")
    n_invalid = rec.get("invalid_environment_rounds",
                        rec.get("invalid_dvfs_rounds"))
    print(f"  valid rounds: {rec['valid_rounds']}/{rec['n_rounds']} "
          f"(invalid env: {n_invalid})")
    print(f"  parent    median: {rec['parent_median_us']} us "
          f"  algo BW: {rec['algorithmic_bw_gbps_parent']} GB/s")
    print(f"  candidate median: {rec['candidate_median_us']} us "
          f"  algo BW: {rec['algorithmic_bw_gbps_candidate']} GB/s")
    print(f"  median speedup (parent/cand): {rec['median_speedup']}  "
          f"CI95: {rec['bootstrap_ci_95']}  faster rounds: {rec['faster_rounds']}")
    # v2.3: raw / filtered 双轨 + filter-sensitivity
    if "filtered_speedup" in rec:
        print(f"  raw_speedup:     {rec['raw_speedup']}  "
              f"(parent {rec['raw']['parent_median_us']} us / "
              f"candidate {rec['raw']['candidate_median_us']} us)  "
              f"CI95: {rec['raw']['bootstrap_ci_95']}")
        print(f"  filtered_speedup: {rec['filtered_speedup']}  "
              f"CI95: {rec['filtered']['bootstrap_ci_95']}")
        print(f"  filter_sensitive: {rec['filter_sensitive']} — "
              f"{rec['filter_sensitive_reason']}")
        print(f"  rejected_samples: {rec['rejected_samples']}")


def _print_matrix(rec: dict):
    print(f"[matrix] M={rec['shape'][0]} H={rec['shape'][1]} "
          f"{rec['dtype']} mode={rec['cache_mode']} "
          f"valid {rec['valid_rounds']}/{rec['n_rounds']} rounds")
    for v in rec["variants"]:
        pv = rec["per_variant"][v]
        print(f"  {v:<20} median {pv['median_us']} us   "
              f"algo BW {pv['algorithmic_bw_gbps']} GB/s")


# ---- test ------------------------------------------------------------------

def cmd_test(a) -> int:
    op = get_op(a.op)
    ext = op.build()
    avail = op.variants(ext)
    gate = _variant_gate_error(op, ext, a.variant)
    if gate:
        print(gate, file=sys.stderr)
        return 2
    if a.variant not in avail:
        print(f"未知变体 {a.variant!r}; 可用: {avail}", file=sys.stderr)
        return 2
    res = op.run_correctness(ext, a.variant,
                             out_dir=Path(a.out_dir) if a.out_dir else None)
    print(f"[correctness] {a.variant}: all_pass={res['all_pass']} "
          f"n_pass={res['summary']['n_pass']}/{res['summary']['n_total']} "
          f"max_abs={res['summary']['max_abs_error']} "
          f"max_rel={res['summary']['max_rel_error']}")
    if "table_check_all_pass" in res:  # rope v0.4 review: 独立表值核对
        print(f"  table_check_all_pass={res['table_check_all_pass']}")
    print(f"  -> {res['saved']}")
    neg = op.run_negative(ext)
    s = neg["summary"]
    print(f"[negative] all_pass={s['all_pass']} "
          f"n_passed={s['n_passed']}/{s['n_total']} "
          f"(reject_ok={s['n_reject_ok']}/{s['n_reject_expected']}, "
          f"pass_ok={s['n_pass_ok']}/{s['n_pass_expected']}, "
          f"skipped={s['n_skipped']})")
    return 0 if (res["all_pass"] and s["all_pass"]) else 1


# ---- benchmark --------------------------------------------------------------

def cmd_bench_pair(a) -> int:
    op = get_op(a.op)
    ext = op.build()
    rec = bench_pair(op, ext, a.parent, a.candidate, a.M, a.H,
                     _dtype(a.dtype), mode=a.mode, rounds=a.rounds)
    p = save_record(rec, op.bench_dir, a.tag)
    _print_pair(rec)
    print(f"保存 -> {p}")
    return 0


def cmd_bench_matrix(a) -> int:
    op = get_op(a.op)
    ext = op.build()
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    rec = bench_matrix(op, ext, variants, a.M, a.H, _dtype(a.dtype),
                       mode=a.mode, rounds=a.rounds)
    p = save_record(rec, op.bench_dir, a.tag)
    _print_matrix(rec)
    print(f"保存 -> {p}")
    return 0


def cmd_bench_full(a) -> int:
    op = get_op(a.op)
    ext = op.build()
    variants = ([v.strip() for v in a.variants.split(",") if v.strip()]
                if a.variants else op.variants(ext))
    records = []
    for (M, H) in op.bench_shapes:
        for dtype_name in op.dtypes:
            for mode in ("hot", "streaming"):
                tag = f"{a.tag}_M{M}_H{H}_{dtype_name}_{mode}"
                print(f"=== {tag} ===", flush=True)
                rec = bench_matrix(op, ext, variants, M, H, _dtype(dtype_name),
                                   mode=mode, rounds=a.rounds)
                p = save_record(rec, op.bench_dir, tag)
                _print_matrix(rec)
                records.append(rec)
    winners = analyze_shape_winners(records)
    wp = save_record({"harness": HARNESS_VERSION, "operator": op.name,
                      "shape_winners": winners}, op.bench_dir,
                     f"{a.tag}_shape_winners")
    print(f"shape winners -> {wp}")
    return 0


def cmd_bench_winners(a) -> int:
    op = get_op(a.op)
    recs = []
    for p in sorted(op.bench_dir.glob(f"{a.tag}_M*_H*.json")):
        r = json.loads(p.read_text())
        if "per_variant" in r:
            recs.append(r)
    winners = analyze_shape_winners(recs)
    out = op.bench_dir / f"{a.tag}_shape_winners.json"
    out.write_text(json.dumps({"operator": op.name,
                               "shape_winners": winners}, indent=2))
    for w in winners:
        print(f"M={w['shape'][0]:<5} H={w['shape'][1]:<5} "
              f"{w['dtype']:<8} {w['cache_mode']:<10} "
              f"winner={w['winner']} runner_up={w['runner_up']} "
              f"ratio={w['median_ratio_runner_over_winner']} "
              f"CI95={w['bootstrap_ci_95']}")
    print(f"保存 -> {out}")
    return 0


# ---- profile ----------------------------------------------------------------

def cmd_profile(a) -> int:
    op = get_op(a.op)
    ext = op.build()
    avail = op.variants(ext)
    variants = (a.variants.split(",") if a.variants else avail)
    for v in variants:
        gate = _variant_gate_error(op, ext, v)
        if gate:
            print(gate, file=sys.stderr)
            return 2
        if v not in avail:
            print(f"未知变体 {v!r}; 可用: {avail}", file=sys.stderr)
            return 2
    op.profiles_dir.mkdir(parents=True, exist_ok=True)
    for cc in ("all", "none"):
        for v in variants:
            out = (op.profiles_dir /
                   f"{v}_M{a.M}_H{a.H}_cc{cc}_clk{a.clock_control}.json")
            print(f"== profiling {v} cache_control={cc} ==", flush=True)
            s = ncu.profile_variant(v, a.M, a.H,
                                    driver_src=op.ncu_driver_source(v, a.M, a.H),
                                    kernel_regex=op.ncu_kernel_regex,
                                    out_path=out, cache_control=cc,
                                    clock_control=a.clock_control)
            if "error" in s:
                print(f"   ERROR: {s['error'][:300]}", file=sys.stderr)
            else:
                print(f"   duration={s['kernel_duration_us']}us "
                      f"dram={s['dram_throughput_pct']}% "
                      f"l2_read={s['l2_read_hit_rate']} "
                      f"l1={s['l1_hit_rate']} "
                      f"regs={s['registers_per_thread']}")
    return 0


# ---- pytorch context ---------------------------------------------------------

def cmd_pytorch(a) -> int:
    op = get_op(a.op)
    d = op.pytorch_ref_latency(a.M, a.H, _dtype(a.dtype))
    print(json.dumps(d, indent=2, ensure_ascii=False))
    return 0


# ---- optimize ----------------------------------------------------------------

def cmd_optimize(a) -> int:
    op = get_op(a.op)
    ext = op.build()
    avail = op.variants(ext)
    for v in (a.parent, a.candidate):
        gate = _variant_gate_error(op, ext, v)
        if gate:
            print(gate, file=sys.stderr)
            return 2
        if v not in avail:
            print(f"未知变体 {v!r}; 可用: {avail}", file=sys.stderr)
            return 2

    print(f"== [{a.id}] correctness: {a.candidate} ==", flush=True)
    corr = op.run_correctness(ext, a.candidate)
    print(f"   all_pass={corr['all_pass']} -> {corr['saved']}")

    print(f"== [{a.id}] negative suite ==", flush=True)
    neg = op.run_negative(ext)
    neg_sum = neg["summary"]
    print(f"   all_pass={neg_sum['all_pass']} "
          f"({neg_sum['n_passed']}/{neg_sum['n_total']})")

    print(f"== [{a.id}] paired bench: {a.parent} vs {a.candidate} "
          f"({a.M},{a.H}) {a.dtype} {a.mode} ==", flush=True)
    tag = (f"{a.id}_pair_{a.parent}_vs_{a.candidate}_"
           f"M{a.M}_H{a.H}_{a.dtype}_{a.mode}")
    paired = bench_pair(op, ext, a.parent, a.candidate, a.M, a.H,
                        _dtype(a.dtype), mode=a.mode, rounds=a.rounds)
    bp = save_record(paired, op.bench_dir, tag)
    _print_pair(paired)

    decision, detail = decide_v2(
        corr["all_pass"] and neg_sum["all_pass"],
        paired["valid_rounds"], paired["speedups"],
        paired["bootstrap_ci_95"])
    # v2.3 filter-sensitivity gate: raw 与 filtered 方向翻转或差 >10%
    # 时, KEEP/REJECT 降级 UNSTABLE（不强行 KEEP/REJECT）
    decision, detail = apply_filter_gate(
        decision, detail,
        paired.get("filter_sensitive", False),
        paired.get("filter_sensitive_reason", "未评估（v2.2 或更早记录）"))
    # v0.4.1: 形式分离 — statistical_relation 只基于 CI95（与 5% 政策
    # 阈值无关）; policy_decision = filter gate 之后的 decision。两者
    # 分别写入实验记录的 decision 块。
    rel = statistical_relation(paired["bootstrap_ci_95"])
    detail["statistical_relation"] = rel
    detail["policy_decision"] = decision
    print(f"== [{a.id}] decision: {decision} ==")
    print(f"   {detail.get('rule')}")
    print(f"   statistical_relation: {rel}（只基于 CI95，与 5% 政策阈值无关）")
    print(f"   policy_decision: {decision}（acceptance policy，filter gate 之后）")

    profile_obs = None
    if a.profile:
        out = (op.profiles_dir /
               f"{a.candidate}_M{a.M}_H{a.H}_ccall_clkbase.json")
        s = ncu.profile_variant(a.candidate, a.M, a.H,
                                driver_src=op.ncu_driver_source(a.candidate,
                                                                a.M, a.H),
                                kernel_regex=op.ncu_kernel_regex,
                                out_path=out)
        profile_obs = {
            "file": str(out),
            "kernel_duration_us": s.get("kernel_duration_us"),
            "dram_throughput_pct": s.get("dram_throughput_pct"),
            "l2_read_hit_rate": s.get("l2_read_hit_rate"),
            "l1_hit_rate": s.get("l1_hit_rate"),
            "registers_per_thread": s.get("registers_per_thread"),
            "warp_stalls": s.get("warp_stalls"),
        }
        print(f"   profile -> {out}")

    record = {
        "id": a.id,
        "operator": op.name,
        "parent": a.parent,
        "candidate": a.candidate,
        "hypothesis": a.hypothesis,
        "changes": a.changes,
        "correctness": corr,
        "negative": neg_sum,
        "paired_benchmark": {
            "file": str(bp),
            "harness": paired["harness"],
            "shape": [a.M, a.H], "dtype": a.dtype, "cache_mode": a.mode,
            "parent": a.parent, "candidate": a.candidate,
            "paired_rounds": paired["n_rounds"],
            "valid_rounds": paired["valid_rounds"],
            "invalid_environment_rounds": paired["invalid_environment_rounds"],
            # legacy alias（v0.2–v0.3 字段名）
            "invalid_dvfs_rounds": paired["invalid_dvfs_rounds"],
            "median_speedup_parent_over_candidate": paired["median_speedup"],
            "bootstrap_ci_95": paired["bootstrap_ci_95"],
            "faster_rounds_candidate": paired["faster_rounds"],
            "parent_median_us": paired["parent_median_us"],
            "candidate_median_us": paired["candidate_median_us"],
            # v2.3: raw / filtered 双轨 + filter-sensitivity
            "raw_speedup": paired.get("raw_speedup"),
            "filtered_speedup": paired.get("filtered_speedup"),
            "filter_sensitive": paired.get("filter_sensitive"),
            "filter_sensitive_reason": paired.get("filter_sensitive_reason"),
            "rejected_samples": paired.get("rejected_samples"),
        },
        "profile_observation": profile_obs,
        "decision": detail,
    }
    p = save_experiment(record, op.experiments_dir, op.experiment_prefix())
    print(f"实验记录 -> {p}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="CUDALab v0.3 统一 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("test", help="正确性套件 + negative suite")
    p.add_argument("op")
    p.add_argument("--variant", required=True)
    p.add_argument("--out-dir", default=None,
                   help="correctness 输出目录（默认按算子约定）")
    p.set_defaults(fn=cmd_test)

    pb = sub.add_parser("benchmark", help="paired benchmark 引擎")
    bsub = pb.add_subparsers(dest="bcmd", required=True)

    def add_common(q):
        q.add_argument("--dtype", default="float16",
                       choices=["float16", "float32"])
        q.add_argument("--mode", default="streaming", choices=["hot", "streaming"])
        q.add_argument("--rounds", type=int, default=9)
        q.add_argument("--tag", required=True)

    q = bsub.add_parser("pair")
    q.add_argument("op")
    q.add_argument("--parent", required=True)
    q.add_argument("--candidate", required=True)
    q.add_argument("--M", type=int, required=True)
    q.add_argument("--H", type=int, required=True)
    add_common(q)
    q.set_defaults(fn=cmd_bench_pair)

    q = bsub.add_parser("matrix")
    q.add_argument("op")
    q.add_argument("--variants", required=True)
    q.add_argument("--M", type=int, required=True)
    q.add_argument("--H", type=int, required=True)
    add_common(q)
    q.set_defaults(fn=cmd_bench_matrix)

    q = bsub.add_parser("full")
    q.add_argument("op")
    q.add_argument("--variants", default=None)
    q.add_argument("--rounds", type=int, default=9)
    q.add_argument("--tag", required=True)
    q.set_defaults(fn=cmd_bench_full)

    q = bsub.add_parser("winners")
    q.add_argument("op")
    q.add_argument("--tag", required=True)
    q.set_defaults(fn=cmd_bench_winners)

    p = sub.add_parser("profile", help="NCU 剖析（cc=all + cc=none）")
    p.add_argument("op")
    p.add_argument("--variants", default=None)
    p.add_argument("--M", type=int, default=128)
    p.add_argument("--H", type=int, default=4096)
    p.add_argument("--clock-control", default="base",
                   choices=["base", "none", "reset"])
    p.set_defaults(fn=cmd_profile)

    p = sub.add_parser("pytorch", help="PyTorch implementation context")
    p.add_argument("op")
    p.add_argument("--M", type=int, required=True)
    p.add_argument("--H", type=int, required=True)
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "float32"])
    p.set_defaults(fn=cmd_pytorch)

    p = sub.add_parser("optimize", help="一次优化实验（记录含失败）")
    p.add_argument("op")
    p.add_argument("--id", required=True, help="实验 ID，如 SFM-0001")
    p.add_argument("--parent", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--changes", required=True)
    p.add_argument("--M", type=int, default=128)
    p.add_argument("--H", type=int, default=4096)
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "float32"])
    p.add_argument("--mode", default="streaming", choices=["hot", "streaming"])
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--profile", action="store_true",
                   help="对候选追加一次 NCU 剖析")
    p.set_defaults(fn=cmd_optimize)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
