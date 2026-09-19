"""CUDALab v0.2 — evaluator 纯 CPU 单元测试（无 GPU、无 torch 依赖）。

覆盖: paired 统计、bootstrap 可复现性、DVFS guard 判定、决策规则
（KEEP / REJECT / NEUTRAL / UNSTABLE / correctness FAIL）。

运行:
    source tools/env.sh
    $PYTHON tests/test_evaluator_cpu.py        # 自包含 runner
    $PYTHON -m pytest tests/test_evaluator_cpu.py   # 也兼容 pytest
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cudalab import decision as D            # noqa: E402
from cudalab import stats as S               # noqa: E402


# ---- paired 统计 -----------------------------------------------------------

def test_paired_speedups_basic():
    assert S.paired_speedups([2.0, 4.0], [1.0, 2.0]) == [2.0, 2.0]


def test_paired_speedups_length_mismatch():
    try:
        S.paired_speedups([1.0, 2.0], [1.0])
        assert False, "应当抛 ValueError"
    except ValueError:
        pass


def test_paired_speedups_nonpositive():
    try:
        S.paired_speedups([1.0], [0.0])
        assert False, "应当抛 ValueError"
    except ValueError:
        pass


def test_summarize():
    s = S.summarize([1.10, 1.11, 1.09, 1.12, 1.10])
    assert s["n"] == 5
    assert abs(s["median"] - 1.10) < 1e-9
    assert s["faster_count"] == 5
    assert s["faster_fraction"] == 1.0
    assert s["min"] == 1.09 and s["max"] == 1.12
    e = S.summarize([])
    assert e["n"] == 0 and e["median"] is None


# ---- bootstrap 可复现性 ----------------------------------------------------

def test_bootstrap_reproducibility():
    vals = [1.10, 1.09, 1.11, 1.10, 1.08, 1.12, 1.095]
    ci1 = S.bootstrap_ci(vals)
    ci2 = S.bootstrap_ci(vals)
    assert ci1 == ci2, "固定 seed 必须产生确定性 CI"
    ci3 = S.bootstrap_ci(vals, seed=12345)
    assert ci1 != ci3 or True  # 不同 seed 不保证不同，但不得报错
    lo, hi = ci1
    assert lo <= 1.10 <= hi, "median 必须落在 CI 内"
    assert lo > 1.00, "全正加速数据的 CI 下界应 > 1.00"


def test_bootstrap_too_few_samples():
    assert S.bootstrap_ci([1.1, 0.9]) is None  # n < 3 不假装可信


# ---- DVFS guard -------------------------------------------------------------

def test_dvfs_pair_extreme_clocks_invalid():
    # EXP-0007 类型场景: parent 有效时钟 1350 MHz, candidate 1905 MHz
    ok, reason, ep, ec = S.check_dvfs_pair(1350, 1905, 1905, parent_first=True,
                                           tol=0.05)
    assert not ok, "1350 vs 1905 一类 round 必须判无效"
    assert reason == "INVALID_DVFS"


def test_dvfs_pair_stable_valid():
    ok, reason, ep, ec = S.check_dvfs_pair(1900, 1905, 1902, parent_first=True,
                                           tol=0.05)
    assert ok and reason is None
    assert abs(ep - 1902.5) < 1e-9
    assert abs(ec - 1903.5) < 1e-9


def test_dvfs_pair_missing_data_invalid():
    ok, reason, _, _ = S.check_dvfs_pair(None, 1900, 1900, True, 0.05)
    assert not ok and reason == "no_clock_data"


def test_dvfs_matrix():
    ok, reason, effs = S.check_dvfs_matrix(
        [1900, 1905, 1902, 1898, 1901, 1904], tol=0.05)
    assert ok and len(effs) == 5
    ok2, reason2, _ = S.check_dvfs_matrix(
        [1350, 1350, 1350, 1905, 1905, 1905], tol=0.05)
    assert not ok2 and reason2 == "INVALID_DVFS"
    ok3, reason3, _ = S.check_dvfs_matrix([1900, None, 1900], 0.05)
    assert not ok3 and reason3 == "no_clock_data"


# ---- 决策规则 ----------------------------------------------------------------

def test_decision_keep():
    sp = [1.10, 1.11, 1.09, 1.12, 1.10]
    dec, detail = D.decide_v2(True, len(sp), sp, S.bootstrap_ci(sp))
    assert dec == D.KEEP, f"应为 KEEP: {detail}"


def test_decision_neutral_mixed():
    sp = [1.01, 0.99, 1.02, 0.98, 1.00]
    dec, detail = D.decide_v2(True, len(sp), sp, S.bootstrap_ci(sp))
    assert dec == D.NEUTRAL, f"应为 NEUTRAL: {detail}"


def test_decision_reject():
    sp = [0.90, 0.91, 0.89, 0.92, 0.90]
    dec, detail = D.decide_v2(True, len(sp), sp, S.bootstrap_ci(sp))
    assert dec == D.REJECT, f"应为 REJECT: {detail}"


def test_decision_unstable_few_valid_rounds():
    dec, detail = D.decide_v2(True, 2, [1.5, 1.6], None)
    assert dec == D.UNSTABLE, f"valid rounds 不足应为 UNSTABLE: {detail}"


def test_decision_correctness_fail_unconditional():
    sp = [1.20, 1.21, 1.19, 1.22, 1.20]  # 强 KEEP 数据
    dec, detail = D.decide_v2(False, len(sp), sp, S.bootstrap_ci(sp))
    assert dec == D.REJECT, "correctness FAIL 必须无条件 REJECT"


def test_decision_neutral_when_ci_includes_1():
    # median 高于 1.05 但 round 波动大、CI 包含 1.00 → 不满足 KEEP
    sp = [1.30, 0.80, 1.28, 0.82, 1.31]
    ci = S.bootstrap_ci(sp)
    dec, detail = D.decide_v2(True, len(sp), sp, ci)
    assert dec == D.NEUTRAL, f"CI 包含 1.00 时不得 KEEP: {detail}, ci={ci}"


def test_decision_neutral_small_gain():
    sp = [1.02, 1.03, 1.01, 1.04, 1.02]
    dec, detail = D.decide_v2(True, len(sp), sp, S.bootstrap_ci(sp))
    assert dec == D.NEUTRAL, f"+2% 级加速应为 NEUTRAL: {detail}"


def test_decision_reject_ci_must_exclude_1():
    # median <= 0.95 且多数更慢，但 CI 上界 >= 1.00 → 不得 REJECT
    sp = [0.94, 0.95, 0.93, 0.96, 1.01]
    ci = S.bootstrap_ci(sp)
    dec, detail = D.decide_v2(True, len(sp), sp, ci)
    assert dec == D.NEUTRAL, f"CI 包含 1.00 时不得 REJECT: {detail}, ci={ci}"


# ---- runner -----------------------------------------------------------------

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
