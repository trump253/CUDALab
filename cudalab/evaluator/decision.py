"""CUDALab v0.2 — 性能决策规则（纯 CPU，无 GPU / torch 依赖）。

v0.3: 原样移入通用 evaluator 核心（cudalab/evaluator/decision.py），
代码不变；cudalab/decision.py 保留为兼容 re-export。

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

v2.3 追加（harness paired-streaming-v2.3，纯 CPU 可单测）:
6. FILTER_SENSITIVE gate（在 decide_v2 之后应用，见
   apply_filter_gate）: 记录被标记 filter_sensitive（raw 与 filtered
   speedup 方向翻转，或 |log(filtered/raw)| > log(1.10)，判据见
   stats.filter_sensitive）时，**无论原决策是 KEEP / REJECT 还是
   NEUTRAL，最终 policy_decision 一律 UNSTABLE**——"guard 改变了
   结论"的记录不强行给出任何 policy 判定；原决策记入
   original_decision。仅原决策已是 UNSTABLE 时保持不变（只记录
   标记）。

v0.4.1 追加（形式分离 statistical_relation / policy_decision）:
7. `statistical_relation(ci95)`（纯函数，只基于 CI95，与 5% 政策
   阈值无关）: CI95 下界 > 1.00 → FASTER；CI95 上界 < 1.00 →
   SLOWER；否则（CI 含 1.00 或 CI 缺失）→ UNRESOLVED。
   policy_decision（KEEP/REJECT/NEUTRAL/UNSTABLE，即 decide_v2 +
   filter gate 的结果）是 acceptance policy 判定，受 5% 替换阈值、
   faster-round 占比、有效轮数与 filter gate 约束。两者可背离：
   例如 CI [1.001, 1.04] → 统计 FASTER，但 median 1.01 < 1.05 →
   policy NEUTRAL。实验 schema（cmd_optimize 的 decision 块与
   classify_cell 输出）从 v0.4.1 起分别记录这两个字段。
"""
from __future__ import annotations

from . import stats as _stats

KEEP, REJECT, NEUTRAL, UNSTABLE = "KEEP", "REJECT", "NEUTRAL", "UNSTABLE"
# v2.3: 记录级标记（不是决策值本身；v0.4.1 起决策层见到它会把
# KEEP/REJECT/NEUTRAL 一律降级 UNSTABLE）
FILTER_SENSITIVE = "FILTER_SENSITIVE"

# v0.4.1: statistical_relation 取值（纯统计陈述，基于 CI95；与
# policy_decision 的 5% 阈值等政策约束无关，见模块 docstring 第 7 条）。
FASTER = "FASTER"
SLOWER = "SLOWER"
UNRESOLVED = "UNRESOLVED"

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


def apply_filter_gate(decision: str, detail: dict,
                      filter_sensitive: bool,
                      reason: str) -> tuple[str, dict]:
    """v2.3 filter-sensitivity gate（纯 CPU，可单测）。

    在 decide_v2 之后应用（v0.4.1 语义收紧）：记录被标记
    filter_sensitive（raw 与 filtered speedup 方向翻转，或
    |log(filtered/raw)| > log(1.10)）时，**无论原决策是 KEEP / REJECT
    还是 NEUTRAL，最终 policy_decision 一律 UNSTABLE**——"guard 改变
    了结论"的记录不强行给出任何 policy 判定（v0.4 的旧语义只降级
    KEEP/REJECT、NEUTRAL 留标记；v0.4.1 起 NEUTRAL 同样降级，因为
    raw/filtered 已分歧时该记录连"中性"都不可信）。仅原决策已是
    UNSTABLE 时保持不变。detail 增加 filter_sensitive /
    filter_sensitive_reason；降级时额外记录 original_decision 与新
    rule。输入 detail 不被修改。
    """
    detail = dict(detail)
    detail["filter_sensitive"] = bool(filter_sensitive)
    detail["filter_sensitive_reason"] = reason
    if filter_sensitive and decision in (KEEP, REJECT, NEUTRAL):
        detail["original_decision"] = decision
        detail["rule"] = (
            f"FILTER_SENSITIVE: {reason} — 原决策 {decision} 降级为 "
            "UNSTABLE（filter_sensitive ⇒ 最终 policy_decision 一律 "
            "UNSTABLE，不强行 KEEP/REJECT/NEUTRAL）")
        return UNSTABLE, detail
    return decision, detail


def statistical_relation(ci95: list[float] | None) -> str:
    """v0.4.1 统计关系（纯 CPU，只基于 CI95；5% 政策阈值不参与）。

    规则（与 decide_v2 的 speedup 约定一致: parent/candidate，>1 =
    candidate 更快）:
    - CI95 下界 > 1.00        -> FASTER（整个 CI 在 1.00 之上，candidate
      显著更快）
    - CI95 上界 < 1.00        -> SLOWER（整个 CI 在 1.00 之下，candidate
      显著更慢）
    - 其余（CI 含 1.00，含恰好触到 1.00 的边界；或 CI 缺失/None）
                             -> UNRESOLVED

    该陈述与 policy_decision 形式分离: policy 的 5% 替换阈值、
    faster-round 占比、有效轮数、filter gate 只影响
    policy_decision（KEEP/REJECT/NEUTRAL/UNSTABLE），不影响
    statistical_relation。两者可背离（见模块 docstring 第 7 条的
    例子: CI [1.001, 1.04] → FASTER 但 median < 1.05 → policy NEUTRAL）。
    """
    if ci95:
        lo = float(ci95[0])
        hi = float(ci95[1])
        if lo > 1.00:
            return FASTER
        if hi < 1.00:
            return SLOWER
    return UNRESOLVED
