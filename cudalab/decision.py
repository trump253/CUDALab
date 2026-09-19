"""CUDALab v0.2 — 性能决策规则（纯 CPU，无 GPU / torch 依赖）。

输入:
- correctness_pass: 完整正确性套件是否通过（含 edge cases）
- valid_rounds: 通过 DVFS guard 的独立 round 数
- speedups: valid round 的 per-round paired speedup
  （parent / candidate，>1 = candidate 更快）
- ci95: 基于 round-level speedup 的 95% bootstrap CI [lo, hi]（可为 None）

决策（固定，先检查的先赢）:
1. correctness FAIL          -> REJECT（无条件）
2. valid_rounds < MIN_VALID  -> UNSTABLE（环境不稳定不是 NEUTRAL）
3. KEEP:   median >= 1.05 且 candidate 更快的 round 占比 >= 70%
           且 CI 下界 > 1.00（CI 缺失时该条件不满足）
4. REJECT: median <= 0.95 且 candidate 更快的 round 占比 <= 30%
           且 CI 上界 < 1.00（CI 缺失时该条件不满足）
5. 其余                      -> NEUTRAL
"""
from __future__ import annotations

from . import stats as _stats

KEEP, REJECT, NEUTRAL, UNSTABLE = "KEEP", "REJECT", "NEUTRAL", "UNSTABLE"

MIN_VALID_ROUNDS = 5      # 少于该数量的 DVFS 稳定 round -> UNSTABLE
KEEP_MEDIAN = 1.05        # median paired speedup >= 1.05
REJECT_MEDIAN = 0.95      # median paired speedup <= 0.95
KEEP_FASTER_FRAC = 0.70   # 至少 70% 的 valid round candidate 更快
REJECT_FASTER_FRAC = 0.30  # 至多 30% 的 valid round candidate 更快


def decide_v2(correctness_pass: bool,
              valid_rounds: int,
              speedups: list[float],
              ci95: list[float] | None) -> tuple[str, dict]:
    """返回 (decision, detail)。detail 记录触发规则的各数值，可审计。"""
    s = _stats.summarize(speedups)
    detail = {
        "correctness_pass": correctness_pass,
        "valid_rounds": valid_rounds,
        "min_valid_rounds": MIN_VALID_ROUNDS,
        "speedups": s,
        "ci95": ci95,
    }

    def _decide(decision: str, rule: str) -> tuple[str, dict]:
        detail["rule"] = rule
        detail["decision"] = decision
        return decision, detail

    if not correctness_pass:
        return _decide(REJECT, "correctness FAIL -> REJECT (unconditional)")

    if valid_rounds < MIN_VALID_ROUNDS:
        return _decide(
            UNSTABLE,
            f"valid DVFS-stable rounds {valid_rounds} < {MIN_VALID_ROUNDS} "
            f"-> UNSTABLE (环境不稳定，不强行 KEEP/REJECT)")

    if s["median"] is None:
        return _decide(UNSTABLE, "no valid speedup samples -> UNSTABLE")

    faster_frac = s["faster_fraction"]
    ci_lo = ci95[0] if ci95 else None
    ci_hi = ci95[1] if ci95 else None

    if (s["median"] >= KEEP_MEDIAN
            and faster_frac is not None and faster_frac >= KEEP_FASTER_FRAC
            and ci_lo is not None and ci_lo > 1.00):
        return _decide(
            KEEP,
            f"median {s['median']:.4f} >= {KEEP_MEDIAN} AND "
            f"{s['faster_count']}/{s['n']} rounds faster (>= {KEEP_FASTER_FRAC:.0%}) "
            f"AND CI95 lower {ci_lo:.4f} > 1.00 -> KEEP")

    if (s["median"] <= REJECT_MEDIAN
            and faster_frac is not None and faster_frac <= REJECT_FASTER_FRAC
            and ci_hi is not None and ci_hi < 1.00):
        return _decide(
            REJECT,
            f"median {s['median']:.4f} <= {REJECT_MEDIAN} AND "
            f"{s['faster_count']}/{s['n']} rounds faster (<= {REJECT_FASTER_FRAC:.0%}) "
            f"AND CI95 upper {ci_hi:.4f} < 1.00 -> REJECT")

    return _decide(
        NEUTRAL,
        f"median {s['median']:.4f} within ({REJECT_MEDIAN}, {KEEP_MEDIAN}) "
        f"or mixed rounds ({s['faster_count']}/{s['n']} faster) or CI not "
        f"excluded 1.00 (CI95={ci95}) -> NEUTRAL")
