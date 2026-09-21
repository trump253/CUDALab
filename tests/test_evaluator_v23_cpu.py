"""CUDALab v2.3 — evaluator 对称 guard / raw+filtered / filter-sensitivity
纯 CPU 单元测试（无 GPU、无 torch 依赖）。

覆盖（Phase 4 要求的四类构造 + 纯函数判据）:
1. 对称 spike（10µs 基线: 15.1 → 慢 spike 拒; 6.5 → 快 spike 拒）;
2. 对称 block（运行中位数 10: 11.6 → 均匀慢块无效; 8.6 → 均匀快块无效）;
3. 无偏 swap（parent 快 outlier vs candidate 慢 outlier，互换后 guard
   行为对称）;
4. filter-sensitive 构造（raw 看 candidate 慢、filtered 看 candidate
   快 → 必须 FILTER_SENSITIVE / UNSTABLE，不得强行 KEEP/REJECT）;
5. v0.4.1: statistical_relation / policy_decision 形式分离
   （CI 边界与严格不等式、5% 阈值独立性、classify_cell 双字段与
   早退路径 UNRESOLVED/None）;
6. v0.4.1: filter gate 收紧（NEUTRAL + filter_sensitive → 最终
   policy_decision 一律 UNSTABLE, 双向必测）。

运行:
    source tools/env.sh
    $PYTHON tests/test_evaluator_v23_cpu.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cudalab.evaluator import decision as D      # noqa: E402
from cudalab.evaluator import experiment as E    # noqa: E402
from cudalab.evaluator import stats as S         # noqa: E402


# ---- 常量钉死（固定规则，见 docs/evaluator_v2_3.md） ------------------------

def test_guard_constants():
    assert S.SPIKE_FACTOR == 1.5
    assert S.SPIKE_WINDOW == 50
    assert S.CROSSBLOCK_FACTOR == 1.15
    assert S.CROSSBLOCK_WARMUP == 3
    assert abs(S.FILTER_LOG_DELTA - 0.0953102) < 1e-6  # log(1.10)


# ---- 1. 对称 per-sample spike guard -----------------------------------------

def _baseline(n: int = 50, v: float = 10.0) -> list[float]:
    return [v] * n


def test_spike_guard_slow():
    clean = _baseline()
    assert S.apply_spike_guard(clean, 15.1) == (False, "slow")
    assert S.apply_spike_guard(clean, 14.9) == (True, None)   # 阈值内
    assert S.apply_spike_guard(clean, 15.0) == (True, None)   # 严格不等式: 边界接受


def test_spike_guard_fast():
    clean = _baseline()
    assert S.apply_spike_guard(clean, 6.5) == (False, "fast")
    assert S.apply_spike_guard(clean, 6.7) == (True, None)    # 阈值内
    # 边界 10/1.5 = 6.6667: 严格 < 才拒
    assert S.apply_spike_guard(clean, 10.0 / 1.5) == (True, None)


def test_spike_guard_first_sample_always_accepted():
    assert S.apply_spike_guard([], 1e9) == (True, None)
    assert S.apply_spike_guard([], 1e-9) == (True, None)


def test_spike_guard_window_uses_last_accepted():
    # 前 60 个 10.0（全部接受），随后 40 个 11.0（阈值内, 接受）;
    # 窗口 = 最近 50 个 accepted = 全 11.0 → ref=11。
    clean = _baseline(60, 10.0) + [11.0] * 40
    # 16.5 == 11*1.5: 严格 > 才拒 → 接受（若误用全历史基线 10 则会拒）
    assert S.apply_spike_guard(clean, 16.5) == (True, None)
    assert S.apply_spike_guard(clean, 16.6) == (False, "slow")


# ---- 2. 对称 cross-block guard ------------------------------------------------

def test_crossblock_slow_flag():
    hist = [10.0, 10.0, 10.0]  # warmup 已满（3 个 block）
    info = S.crossblock_flag(hist, 11.6)
    assert info["flagged"] is True and info["direction"] == "slow"
    assert abs(info["ratio"] - 1.16) < 1e-9
    assert hist[-1] == 11.6  # 判定后 med 入 hist


def test_crossblock_fast_flag():
    hist = [10.0, 10.0, 10.0]
    info = S.crossblock_flag(hist, 8.6)
    # 1/1.15 = 0.86957; 8.6/10 = 0.86 < 0.86957 → 均匀快块
    assert info["flagged"] is True and info["direction"] == "fast"


def test_crossblock_within_factor_not_flagged():
    for med in (11.4, 8.7):  # 1.14 < 1.15; 0.87 > 1/1.15
        info = S.crossblock_flag([10.0, 10.0, 10.0], med)
        assert info["flagged"] is False and info["direction"] is None, med


def test_crossblock_symmetric_reciprocal():
    # 同一 1.16 倍的向上偏差与 1/1.16 倍的向下偏差都必须判（对称核心）
    info_up = S.crossblock_flag([10.0] * 3, 10.0 * 1.16)
    info_dn = S.crossblock_flag([10.0] * 3, 10.0 / 1.16)
    assert info_up["flagged"] and info_up["direction"] == "slow"
    assert info_dn["flagged"] and info_dn["direction"] == "fast"


def test_crossblock_warmup_not_flagged():
    # 前 3 个 block 不判（即使偏差大），但同样入 hist
    h = []
    for _ in range(3):
        info = S.crossblock_flag(h, 11.6)
        assert info["flagged"] is False and info["ratio"] is None
    assert len(h) == 3
    info = S.crossblock_flag(h, 11.6)  # 第 4 个: 运行中位数 11.6 → ratio 1.0 不判
    assert info["flagged"] is False
    assert len(h) == 4


def test_crossblock_none_med_not_appended():
    h = [10.0, 10.0, 10.0]
    info = S.crossblock_flag(h, None)
    assert info["flagged"] is False and info["ratio"] is None
    assert len(h) == 3  # 不入 hist


# ---- block_stats: raw + filtered 双套统计 ------------------------------------

def test_block_stats_fast_outliers():
    # 50×10.0 + 50×6.5: 6.5 全部 fast 拒（ref=10, 6.5 < 10/1.5）
    st = S.block_stats([10.0] * 50 + [6.5] * 50)
    assert st["n_raw"] == 100
    assert st["n_accepted"] == 50
    assert st["n_rejected_fast"] == 50
    assert st["n_rejected_slow"] == 0
    assert st["raw_median_us"] == 8.25    # (10+6.5)/2
    assert st["median_us"] == 10.0        # filtered 轨
    # legacy alias
    assert st["n_clean"] == 50 and st["n_spike"] == 50


def test_block_stats_slow_outliers():
    st = S.block_stats([10.0] * 50 + [15.1] * 50)
    assert st["n_rejected_fast"] == 0
    assert st["n_rejected_slow"] == 50
    assert st["raw_median_us"] == 12.55   # (10+15.1)/2
    assert st["median_us"] == 10.0


def test_block_stats_empty():
    st = S.block_stats([])
    assert st["n_raw"] == 0 and st["n_accepted"] == 0
    assert st["median_us"] is None and st["raw_median_us"] is None


def test_block_stats_clean_untouched():
    st = S.block_stats([10.0] * 100)
    assert st["n_accepted"] == 100
    assert st["n_rejected_fast"] == 0 and st["n_rejected_slow"] == 0
    assert st["raw_median_us"] == st["median_us"] == 10.0
    assert st["raw_samples_us"] == [10.0] * 100
    assert st["accepted_samples_us"] == [10.0] * 100


# ---- 3. 无偏 swap（parent/candidate 对称性） ---------------------------------

def test_swap_no_bias():
    """parent 带 50 个快 outlier vs candidate 带 50 个慢 outlier，
    互换后 guard 行为必须对称: 拒绝计数同构（只是方向标签互换）、
    filtered 决策统计完全不变（speedup 恒 1.0）、raw speedup 互为倒数。"""
    fast_block = [10.0] * 50 + [6.5] * 50    # 50 个 fast outlier
    slow_block = [10.0] * 50 + [15.1] * 50   # 50 个 slow outlier

    # 情形 A: parent=快 outlier, candidate=慢 outlier
    pa, ca = S.block_stats(fast_block), S.block_stats(slow_block)
    assert (pa["n_rejected_fast"], pa["n_rejected_slow"]) == (50, 0)
    assert (ca["n_rejected_fast"], ca["n_rejected_slow"]) == (0, 50)
    speedup_filtered_a = pa["median_us"] / ca["median_us"]
    speedup_raw_a = pa["raw_median_us"] / ca["raw_median_us"]

    # 情形 B: 互换（parent=慢 outlier, candidate=快 outlier）
    pb, cb = S.block_stats(slow_block), S.block_stats(fast_block)
    assert (pb["n_rejected_fast"], pb["n_rejected_slow"]) == (0, 50)
    assert (cb["n_rejected_fast"], cb["n_rejected_slow"]) == (50, 0)
    speedup_filtered_b = pb["median_us"] / cb["median_us"]
    speedup_raw_b = pb["raw_median_us"] / cb["raw_median_us"]

    # guard 对两个 variant 一视同仁: 每个 block 恰拒 50、filtered 中位数
    # 都是 10.0 → filtered speedup 在两个方向上都恒为 1.0（无偏）
    assert speedup_filtered_a == 1.0 and speedup_filtered_b == 1.0
    # raw speedup 互为倒数（raw 轨会"看到"方向，但这是数据的性质,
    # guard 本身没有偏向任何 variant）
    assert abs(speedup_raw_a * speedup_raw_b - 1.0) < 1e-9
    assert speedup_raw_a < 1.0 < speedup_raw_b


# ---- filter_sensitive 纯函数 ---------------------------------------------------

def test_filter_sensitive_flip():
    # 方向翻转（跨 1.0）: raw 看 candidate 慢, filtered 看 candidate 快
    ok, why = S.filter_sensitive(0.6846, 1.1111)
    assert ok is True and "翻转" in why


def test_filter_sensitive_flip_reverse():
    ok, _ = S.filter_sensitive(1.20, 0.90)
    assert ok is True


def test_filter_sensitive_within_threshold():
    # |log(1.11/1.10)| ≈ 0.0091 < log(1.10) ≈ 0.0953 → 不敏感
    ok, why = S.filter_sensitive(1.10, 1.11)
    assert ok is False and "阈值内" in why


def test_filter_sensitive_delta_exceeds():
    # |log(1.11/1.00)| = log(1.11) ≈ 0.1044 > log(1.10) → 敏感
    ok, why = S.filter_sensitive(1.00, 1.11)
    assert ok is True and "log" in why


def test_filter_sensitive_boundary_exact():
    # filtered/raw 恰好 1.10: delta == log(1.10), 严格 > 才敏感 → 不敏感
    ok, _ = S.filter_sensitive(1.00, 1.10)
    assert ok is False


def test_filter_sensitive_missing_not_evaluated():
    ok, why = S.filter_sensitive(None, 1.11)
    assert ok is False and "未评估" in why
    ok2, _ = S.filter_sensitive(0.0, 1.11)  # 非法（<=0）
    assert ok2 is False


# ---- apply_filter_gate（决策层）-------------------------------------------------

def test_filter_gate_keep_downgraded():
    dec, _ = D.decide_v2(True, 9, [1.1111] * 9, [1.1111, 1.1111])
    assert dec == D.KEEP  # filtered 轨本可 KEEP
    dec2, detail = D.apply_filter_gate(dec, _, True, "方向翻转: 构造案例")
    assert dec2 == D.UNSTABLE, "filter_sensitive 时 KEEP 必须降级 UNSTABLE"
    assert detail["original_decision"] == D.KEEP
    assert detail["filter_sensitive"] is True


def test_filter_gate_reject_downgraded():
    dec, detail = D.decide_v2(True, 9, [0.80] * 9, [0.80, 0.80])
    assert dec == D.REJECT
    dec2, detail2 = D.apply_filter_gate(dec, detail, True, "winner flip")
    assert dec2 == D.UNSTABLE
    assert detail2["original_decision"] == D.REJECT


def test_filter_gate_neutral_not_sensitive_unchanged():
    # v0.4.1: 非敏感的 NEUTRAL 不受 gate 影响（只有敏感才降级）
    dec, detail = D.decide_v2(True, 9, [1.02] * 9, [1.0, 1.04])
    assert dec == D.NEUTRAL
    dec2, detail2 = D.apply_filter_gate(
        dec, detail, False, "在阈值内（构造案例）")
    assert dec2 == D.NEUTRAL
    assert detail2["filter_sensitive"] is False
    assert "original_decision" not in detail2


def test_filter_gate_neutral_sensitive_raw_faster_downgraded():
    """v0.4.1 必测: raw 明显更快 + filtered 轨 NEUTRAL → FILTER_SENSITIVE
    → 最终 policy_decision 一律 UNSTABLE（v0.4 旧语义对 NEUTRAL 只留
    标记、不降级 —— 该语义已收紧）。"""
    # filtered 轨: median 1.02（5% 政策带内）+ CI 含 1.00 → NEUTRAL
    dec, detail = D.decide_v2(True, 9, [1.02] * 9, [1.00, 1.04])
    assert dec == D.NEUTRAL
    raw_med, filtered_med = 1.15, 1.02  # raw 明显更快（≥1.05 带）
    sensitive, reason = S.filter_sensitive(raw_med, filtered_med)
    assert sensitive is True  # |log(1.02/1.15)| ≈ 0.120 > log(1.10)
    dec2, detail2 = D.apply_filter_gate(dec, detail, sensitive, reason)
    assert dec2 == D.UNSTABLE, "敏感 + NEUTRAL 必须最终 UNSTABLE（v0.4.1）"
    assert detail2["original_decision"] == D.NEUTRAL
    assert detail2["filter_sensitive"] is True


def test_filter_gate_neutral_sensitive_raw_slower_downgraded():
    """v0.4.1 必测: raw 明显更慢 + filtered 轨 NEUTRAL → FILTER_SENSITIVE
    → 最终 policy_decision 一律 UNSTABLE。"""
    dec, detail = D.decide_v2(True, 9, [1.02] * 9, [1.00, 1.04])
    assert dec == D.NEUTRAL
    raw_med, filtered_med = 0.85, 1.02  # raw 明显更慢（≤0.95 带, 方向翻转）
    sensitive, reason = S.filter_sensitive(raw_med, filtered_med)
    assert sensitive is True  # raw<1<filtered 方向翻转
    dec2, detail2 = D.apply_filter_gate(dec, detail, sensitive, reason)
    assert dec2 == D.UNSTABLE, "敏感 + NEUTRAL 必须最终 UNSTABLE（v0.4.1）"
    assert detail2["original_decision"] == D.NEUTRAL


def test_filter_gate_not_sensitive_unchanged():
    dec, detail = D.decide_v2(True, 9, [1.1111] * 9, [1.1111, 1.1111])
    assert dec == D.KEEP
    dec2, detail2 = D.apply_filter_gate(
        dec, detail, False, "在阈值内（|log 差| 0.0000 <= 0.0953）")
    assert dec2 == D.KEEP
    assert detail2["filter_sensitive"] is False
    assert "original_decision" not in detail2


def test_filter_gate_does_not_mutate_input():
    dec, detail = D.decide_v2(True, 9, [1.1111] * 9, [1.1111, 1.1111])
    before = dict(detail)
    D.apply_filter_gate(dec, detail, True, "方向翻转")
    assert detail == before  # 输入 detail 不被修改


# ---- 4. filter-sensitive 端到端构造（Phase 4 必测）-----------------------------

def test_constructed_filter_sensitive_end_to_end():
    """parent: 50×10.0 + 50×6.5（fast 拒; raw 8.25 / filtered 10.0）
    candidate: 50×9.0 + 50×15.1（slow 拒; raw 12.05 / filtered 9.0）

    raw speedup = 8.25/12.05 ≈ 0.6846  < 1（raw 看 candidate 慢）
    filtered speedup = 10.0/9.0 ≈ 1.1111 > 1（filtered 看 candidate 快）
    → 方向翻转 → FILTER_SENSITIVE; filtered 轨可判 KEEP, 但 gate 后
    必须 UNSTABLE, 不得强行 KEEP。"""
    p = S.block_stats([10.0] * 50 + [6.5] * 50)
    c = S.block_stats([9.0] * 50 + [15.1] * 50)
    assert p["n_rejected_fast"] == 50 and p["n_rejected_slow"] == 0
    assert c["n_rejected_fast"] == 0 and c["n_rejected_slow"] == 50
    assert p["raw_median_us"] == 8.25 and p["median_us"] == 10.0
    assert c["raw_median_us"] == 12.05 and c["median_us"] == 9.0

    # 9 个 valid round, 每轮同值（确定性）
    raw_speedups = [p["raw_median_us"] / c["raw_median_us"]] * 9
    filtered_speedups = [p["median_us"] / c["median_us"]] * 9
    raw_med = S.summarize(raw_speedups)["median"]
    filtered_med = S.summarize(filtered_speedups)["median"]
    assert raw_med < 1.0 < filtered_med

    sensitive, reason = S.filter_sensitive(raw_med, filtered_med)
    assert sensitive is True, f"必须判 FILTER_SENSITIVE: {reason}"

    ci = S.bootstrap_ci(filtered_speedups)
    dec, detail = D.decide_v2(True, 9, filtered_speedups, ci)
    assert dec == D.KEEP  # 前提: filtered 轨单独看会 KEEP
    dec2, detail2 = D.apply_filter_gate(dec, detail, sensitive, reason)
    assert dec2 == D.UNSTABLE
    assert detail2["original_decision"] == D.KEEP
    assert "FILTER_SENSITIVE" in detail2["rule"]


# ---- v0.4 review 更正: classify_cell 的 raw 轨聚合约定 -------------------------

def test_matrix_raw_convention_median_of_round_ratios():
    """v0.4 methodology review（finding 4, nit）钉死: classify_cell 的
    raw 侧必须与 filtered 侧使用**同一聚合约定** —— per-round runner/
    winner 比值的中位数（与 s["median"] 一致）, 而不是跨 round 中位数
    之比（旧版实现）。两约定在 round 双峰格会背离, 使 filter_sensitive
    判定依赖约定而非数据。

    构造 7 个 valid round（n=7 ≥ MIN_VALID_ROUNDS）, 两变体 w/r:
      filtered us: 每轮 (w=10.0, r=10.6) → per-round 比值恒 1.06
                   → s["median"]=1.06 → decide_v2 = KEEP
      us_raw 双峰: 5 轮 (10.0; 10.6/10.6/10.6/11.7) + 2 轮 (50.0, 50.0)
        旧约定（跨 round 中位数之比）= med(r)/med(w) = 11.7/10.0 = 1.17
             |log(1.17/1.06)| = 0.0998 > log(1.10) = 0.0953
             → 会误判 FILTER_SENSITIVE → KEEP 被降级 UNSTABLE
        新约定（per-round 比值的中位数）
             = median(1.06×3, 1.17, 1.0×3) = 1.06
             → |log(1.06/1.06)| = 0 → 不敏感 → KEEP 成立

    同一份数据两约定结论相反（UNSTABLE vs SIGNIFICANT_WINNER）,
    本测试钉住新约定。
    """
    raws = [(10.0, 10.6), (10.0, 10.6), (10.0, 10.6), (10.0, 11.7),
            (50.0, 50.0), (50.0, 50.0), (50.0, 50.0)]
    rounds = [{"valid": True,
               "us": {"w": 10.0, "r": 10.6},
               "us_raw": {"w": w, "r": r}} for w, r in raws]

    # 先证明测试数据能区分两约定（旧约定确实会判敏感）
    old_ratio = statistics.median(r for _, r in raws) / \
        statistics.median(w for w, _ in raws)
    assert abs(old_ratio - 1.17) < 1e-9
    old_sensitive, _ = S.filter_sensitive(old_ratio, 1.06)
    assert old_sensitive is True, "测试数据必须能区分两种约定"

    out = E.classify_cell(["w", "r"], rounds)
    assert out["filter_sensitive"] is False
    assert out["decision"] == D.KEEP
    assert out["status"] == E.SIGNIFICANT_WINNER
    assert out["winner"] == "w" and out["runner_up"] == "r"
    assert out["winner_raw"] == "w"
    # 审计字段（per-variant raw 中位数）仍照常记录
    assert out["all_variants_raw_median_us"] == {"w": 10.0, "r": 11.7}


# ---- v0.4.1: statistical_relation / policy_decision 形式分离 --------------------

def test_statistical_relation_faster():
    # CI95 下界 > 1.00 → FASTER（整个 CI 在 1.00 之上, candidate 显著快）
    assert D.statistical_relation([1.01, 1.20]) == D.FASTER
    assert D.statistical_relation([1.001, 1.04]) == D.FASTER


def test_statistical_relation_slower():
    # CI95 上界 < 1.00 → SLOWER（整个 CI 在 1.00 之下, candidate 显著慢）
    assert D.statistical_relation([0.80, 0.99]) == D.SLOWER
    assert D.statistical_relation([0.95, 0.999]) == D.SLOWER


def test_statistical_relation_unresolved():
    # CI 含 1.00 → UNRESOLVED
    assert D.statistical_relation([0.98, 1.02]) == D.UNRESOLVED
    # 严格不等式: 恰好触到 1.00 的边界也是 UNRESOLVED（下界 > 1 / 上界 < 1）
    assert D.statistical_relation([1.00, 1.20]) == D.UNRESOLVED
    assert D.statistical_relation([0.80, 1.00]) == D.UNRESOLVED
    # CI 缺失 → UNRESOLVED
    assert D.statistical_relation(None) == D.UNRESOLVED


def test_statistical_relation_independent_of_policy_threshold():
    """v0.4.1 核心: 5% 阈值只影响 policy_decision, 不影响
    statistical_relation。CI [1.001, 1.04] → 统计 FASTER, 但
    median 1.01 < 1.05 → policy NEUTRAL —— 两者可背离。"""
    rel = D.statistical_relation([1.001, 1.04])
    dec, _ = D.decide_v2(True, 9, [1.01] * 9, [1.001, 1.04])
    assert rel == D.FASTER, "CI 下界 > 1 必须是统计 FASTER（与 5% 无关）"
    assert dec == D.NEUTRAL, "median < 1.05 必须是 policy NEUTRAL"


def test_classify_cell_emits_relation_and_policy():
    """v0.4.1: classify_cell 输出分别记录 statistical_relation 与
    policy_decision; 早退路径（无 CI、无 policy 判定）为
    UNRESOLVED / None。"""
    # (a) KEEP 格: 与 raw-convention 测试同构的数据 → FASTER + KEEP
    raws = [(10.0, 10.6), (10.0, 10.6), (10.0, 10.6), (10.0, 11.7),
            (50.0, 50.0), (50.0, 50.0), (50.0, 50.0)]
    rounds = [{"valid": True,
               "us": {"w": 10.0, "r": 10.6},
               "us_raw": {"w": w, "r": r}} for w, r in raws]
    out = E.classify_cell(["w", "r"], rounds)
    assert out["decision"] == D.KEEP
    assert out["statistical_relation"] == D.FASTER
    assert out["policy_decision"] == D.KEEP

    # (b) 背离格: per-round 比值恒 1.01（CI [1.01, 1.01]）→ 统计
    # FASTER, 但 median < 1.05 → policy NEUTRAL
    rounds_b = [{"valid": True, "us": {"w": 10.0, "r": 10.1}}
                for _ in range(9)]
    out_b = E.classify_cell(["w", "r"], rounds_b)
    assert out_b["statistical_relation"] == D.FASTER
    assert out_b["policy_decision"] == D.NEUTRAL
    assert out_b["decision"] == D.NEUTRAL
    assert out_b["status"] == E.NO_UNIQUE_WINNER

    # (c) 早退格: 可比较 variant 不足 → UNRESOLVED / None
    out_c = E.classify_cell(["w"], [{"valid": True, "us": {"w": 10.0}}] * 9)
    assert out_c["statistical_relation"] == D.UNRESOLVED
    assert out_c["policy_decision"] is None

    # (d) 早退格: valid rounds 不足（全部 invalid）→ UNRESOLVED / None
    out_d = E.classify_cell(
        ["w", "r"], [{"valid": False, "us": {}}] * 3)
    assert out_d["statistical_relation"] == D.UNRESOLVED
    assert out_d["policy_decision"] is None


# ---- runner --------------------------------------------------------------------

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
