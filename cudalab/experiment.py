"""CUDALab experiment tracking + acceptance rules.

One optimization attempt = one experiment record (JSON), including failed
ones. Failed experiments are first-class data and are never dropped.

Acceptance rules (v0.1, fixed):
- correctness FAIL            -> REJECT  (unconditional)
- else compare candidate vs CURRENT BEST at the primary target shape:
    median speedup >= 1.05 AND at least 3 of 5 rounds faster  -> KEEP
    median speedup <= 0.95 AND at least 3 of 5 rounds slower  -> REJECT
    otherwise (within +/-5%, or mixed rounds)                  -> NEUTRAL
- The full benchmark matrix of the candidate is always stored; decisions
  are explained from the primary target, matrix evidence is kept.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments" / "rmsnorm"
BEST_FILE = EXP_DIR / "best.json"

KEEP, REJECT, NEUTRAL = "KEEP", "REJECT", "NEUTRAL"

KEEP_THRESHOLD = 0.05      # >= +5%
REJECT_THRESHOLD = -0.05   # <= -5%
MAJORITY_ROUNDS = 3        # of ROUNDS (5)


def _next_id() -> str:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    ids = [int(p.stem.split("-")[1]) for p in EXP_DIR.glob("EXP-*.json")]
    return f"EXP-{(max(ids) + 1 if ids else 1):04d}"


def current_best() -> dict:
    if BEST_FILE.exists():
        return json.loads(BEST_FILE.read_text())
    return {"variant": "baseline", "reason": "initial", "updated": time.time()}


def set_best(variant: str, reason: str, median_us_target: float | None = None):
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    BEST_FILE.write_text(json.dumps({
        "variant": variant,
        "reason": reason,
        "median_us_target": median_us_target,
        "updated": time.time(),
    }, indent=2))


def _primary(recs: list[dict], shape: tuple, dtype: str):
    out = {}
    for r in recs:
        if tuple(r["shape"]) == shape and r["dtype"] == dtype:
            out[r["variant"]] = r
    return out


def evaluate(candidate_bench: list[dict],
             current_best_variant: str,
             correctness_pass: bool,
             correctness_summary: dict,
             primary_shape: tuple = (128, 4096),
             dtype: str = "float16") -> dict:
    """Apply the fixed acceptance rules.

    candidate_bench must contain records for BOTH the current-best variant
    and the candidate (same shapes/dtype, same harness settings).
    Returns {"decision": ..., "detail": {...}}.
    """
    detail: dict = {
        "primary_shape": list(primary_shape),
        "dtype": dtype,
        "correctness_pass": correctness_pass,
        "correctness": correctness_summary,
    }
    if not correctness_pass:
        detail["rule"] = "correctness FAIL -> REJECT (unconditional)"
        return {"decision": REJECT, "detail": detail}

    cands = _primary(candidate_bench, primary_shape, dtype)
    best_rec = cands.get(current_best_variant)
    cand_variants = [v for v in cands if v != current_best_variant]
    if not cand_variants or best_rec is None:
        detail["rule"] = "no comparable candidate record"
        return {"decision": NEUTRAL, "detail": detail}
    cand_rec = cands[cand_variants[0]]

    b_med, c_med = best_rec["median_us"], cand_rec["median_us"]
    speedup = b_med / c_med if c_med > 0 else float("inf")
    b_rounds = best_rec.get("round_medians_us", [])
    c_rounds = cand_rec.get("round_medians_us", [])
    faster_rounds = sum(1 for b, c in zip(b_rounds, c_rounds) if c < b)
    n_rounds = len(b_rounds)
    detail.update({
        "candidate": cand_rec["variant"],
        "current_best": current_best_variant,
        "current_best_median_us": b_med,
        "candidate_median_us": c_med,
        "speedup_vs_current_best": round(speedup, 4),
        "round_medians_current_best": b_rounds,
        "round_medians_candidate": c_rounds,
        "rounds_candidate_faster": f"{faster_rounds}/{n_rounds}",
    })

    if speedup >= 1 + KEEP_THRESHOLD and faster_rounds >= MAJORITY_ROUNDS:
        detail["rule"] = (f"speedup {speedup:.3f}x >= 1.05 AND "
                          f"{faster_rounds}/{n_rounds} rounds faster -> KEEP")
        return {"decision": KEEP, "detail": detail}
    if speedup <= 1 + REJECT_THRESHOLD and faster_rounds <= n_rounds - MAJORITY_ROUNDS:
        detail["rule"] = (f"speedup {speedup:.3f}x <= 0.95 AND "
                          f"only {faster_rounds}/{n_rounds} rounds faster -> REJECT")
        return {"decision": REJECT, "detail": detail}
    detail["rule"] = (f"speedup {speedup:.3f}x within +/-5% or mixed rounds "
                      f"({faster_rounds}/{n_rounds} faster) -> NEUTRAL")
    return {"decision": NEUTRAL, "detail": detail}


def save_experiment(record: dict) -> Path:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    if not record.get("id"):
        record["id"] = _next_id()
    p = EXP_DIR / f"{record['id']}.json"
    p.write_text(json.dumps(record, indent=2))
    return p
