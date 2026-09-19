"""CUDALab v0.3 — Softmax 纯 CPU 单元测试（无 GPU、无扩展构建）。

覆盖: softmax_ref 的 dtype 语义 / 行和 / 数值稳定性（FP16 溢出安全）、
make_input 全部模式的确定性不变量、compute_metrics + row_sum_extra
指标路径、summarize 汇总、形状矩阵一致性。

运行:
    source tools/env.sh
    $PYTHON tests/test_softmax_cpu.py          # 自包含 runner
    $PYTHON -m pytest tests/test_softmax_cpu.py  # 也兼容 pytest
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from cudalab.operators.softmax import (  # noqa: E402
    softmax_ref, make_input, BENCH_MATRIX_SFM, PRIMARY_TARGET,
)
from cudalab.softmax_correctness import (  # noqa: E402
    SHAPE_MATRIX, ROW_SUM_TOL, SEEDS,
    _row_sum_extra, summarize, _jsonable,
)
from cudalab.evaluator.correctness import (  # noqa: E402
    TOLERANCES, compute_metrics,
)

DEV = "cpu"


# ---- softmax_ref 语义 ------------------------------------------------------

def test_ref_dtype_semantics():
    x16 = torch.randn(8, 512, dtype=torch.float16, device=DEV)
    y16 = softmax_ref(x16)
    assert y16.dtype == torch.float16, "FP16 输入必须返回 FP16"
    x32 = torch.randn(8, 512, dtype=torch.float32, device=DEV)
    y32 = softmax_ref(x32)
    assert y32.dtype == torch.float32, "FP32 输入必须返回 FP32"


def test_ref_row_sums_and_range():
    for dtype in (torch.float16, torch.float32):
        x = torch.randn(16, 1024, dtype=dtype, device=DEV)
        y = softmax_ref(x)
        s = y.float().sum(dim=-1)
        assert torch.all(s > 0) and torch.all(s <= 1.0 + 1e-3)
        assert float(s.abs().sub(1.0).max().item()) < 1e-2
        assert float(y.float().min().item()) >= 0.0
        assert float(y.float().max().item()) <= 1.0 + 1e-3
        assert y.shape == x.shape


def test_ref_dim_check():
    try:
        softmax_ref(torch.randn(4096, device=DEV))
        assert False, "1 维输入应抛 ValueError"
    except ValueError:
        pass


def test_fp16_overflow_safety():
    """x ∈ [-80,80]: 朴素 exp(x) 在 FP16 下溢出（inf）；
    softmax_ref（max 减除 + FP32 内部）必须全程有限。"""
    x = (torch.rand(8, 2048, device=DEV) * 2.0 - 1.0) * 80.0
    x16 = x.to(torch.float16)
    # 朴素路径确实在 fp16 溢出（记录该前提）:
    assert bool(torch.isinf(torch.exp(x16.float().to(torch.float16).float())
                            .float()).any()) or \
        float(torch.exp(x.float()).max().item()) > 65504.0
    y = softmax_ref(x16)
    assert not bool(torch.isnan(y).any()), "FP16 极值输入不得产生 NaN"
    assert not bool(torch.isinf(y).any()), "FP16 极值输入不得产生 Inf"
    # 行和 ≈ 1（FP32 求和）
    err = float((y.float().sum(dim=-1) - 1.0).abs().max().item())
    assert err <= ROW_SUM_TOL["float16"], f"row_sum_error={err}"


def test_ref_constant_row():
    x = torch.full((4, 64), 3.0, dtype=torch.float32, device=DEV)
    y = softmax_ref(x)
    assert float((y - 1.0 / 64).abs().max().item()) < 1e-6


# ---- make_input 模式不变量 --------------------------------------------------

def test_make_input_deterministic():
    a = make_input(2, 256, dtype=torch.float16, device=DEV, seed=5,
                   mode="normal")
    b = make_input(2, 256, dtype=torch.float16, device=DEV, seed=5,
                   mode="normal")
    assert torch.equal(a, b)
    c = make_input(2, 256, dtype=torch.float16, device=DEV, seed=6,
                   mode="normal")
    assert not torch.equal(a, c)


def test_make_input_modes():
    x = make_input(4, 1024, dtype=torch.float32, device=DEV, seed=1,
                   mode="large_pos")
    assert abs(float(x.mean()) - 50.0) < 0.2, "large_pos 应 ≈ +50"
    x = make_input(4, 1024, dtype=torch.float32, device=DEV, seed=1,
                   mode="large_neg")
    assert abs(float(x.mean()) + 50.0) < 0.2, "large_neg 应 ≈ -50"
    x = make_input(4, 1024, dtype=torch.float32, device=DEV, seed=1,
                   mode="mixed_extremes")
    assert float(x.min()) >= -80.0 - 1e-6 and float(x.max()) <= 80.0 + 1e-6
    x = make_input(4, 1024, dtype=torch.float32, device=DEV, seed=1,
                   mode="zeros")
    assert float(x.abs().max()) == 0.0
    x = make_input(4, 1024, dtype=torch.float32, device=DEV, seed=1,
                   mode="constant")
    assert float(x.abs().sub(3.0).max()) == 0.0
    x = make_input(4, 1024, dtype=torch.float32, device=DEV, seed=1,
                   mode="single_dominant")
    assert bool((x[:, 0] == 30.0).all())
    x = make_input(4, 1024, dtype=torch.float32, device=DEV, seed=1,
                   mode="alternating")
    assert float(x[:, 0].abs().sub(3.0).max()) == 0.0
    assert bool((x[:, 0] > 0).all()) and bool((x[:, 1] < 0).all())
    x = make_input(4, 1024, dtype=torch.float32, device=DEV, seed=1,
                   mode="tiny")
    assert float(x.std()) < 1e-3, "tiny 应 ≈ N(0, 1e-4)"


def test_make_input_dtype():
    for mode in ("normal", "zeros", "constant", "single_dominant",
                 "alternating", "mixed_extremes"):
        x = make_input(2, 128, dtype=torch.float16, device=DEV, seed=3,
                       mode=mode)
        assert x.dtype == torch.float16
        assert x.is_contiguous()
        assert x.shape == (2, 128)


def test_make_input_unknown_mode():
    try:
        make_input(2, 128, dtype=torch.float32, device=DEV, seed=3,
                   mode="nope")
        assert False, "未知 mode 应抛 ValueError"
    except ValueError:
        pass


# ---- 指标路径（compute_metrics + row_sum_extra）-----------------------------

def test_metrics_exact_match():
    x = torch.randn(8, 256, dtype=torch.float32, device=DEV)
    y = softmax_ref(x)
    m = compute_metrics(y, y, "float32", extra=_row_sum_extra("float32"))
    assert m["max_abs_error"] == 0.0 and m["max_rel_error"] == 0.0
    assert m["ok_close"] and not m["has_nan"] and not m["has_inf"]
    assert m["row_sum_error"] < 1e-6


def test_metrics_detects_row_sum_drift():
    """elementwise 容差可能放行的系统性归一化错误（整体缩放）
    必须被 row_sum_error 捕捉。"""
    x = torch.randn(8, 512, dtype=torch.float32, device=DEV)
    y = softmax_ref(x) * 0.99  # 每行和 ≈ 0.99: elementwise 差 ~1e-2·y
    m = compute_metrics(y, softmax_ref(x), "float32",
                        extra=_row_sum_extra("float32"))
    assert m["row_sum_error"] > ROW_SUM_TOL["float32"], \
        "0.99 缩放的行和偏移必须超过 fp32 row_sum 阈值"
    assert m["ref_row_sum_error"] < 1e-6


def test_metrics_nan_inf():
    x = torch.randn(4, 64, dtype=torch.float32, device=DEV)
    y = softmax_ref(x).clone()
    y[0, 0] = float("nan")
    m = compute_metrics(y, softmax_ref(x), "float32")
    assert m["has_nan"]
    y2 = softmax_ref(x).clone()
    y2[1, 1] = float("inf")
    m2 = compute_metrics(y2, softmax_ref(x), "float32")
    assert m2["has_inf"]


def test_tolerance_table_fixed():
    assert TOLERANCES["float16"] == {"atol": 2e-3, "rtol": 5e-3}
    assert TOLERANCES["float32"] == {"atol": 1e-5, "rtol": 1e-4}
    assert ROW_SUM_TOL["float16"] > ROW_SUM_TOL["float32"]


# ---- 形状矩阵 / 汇总 --------------------------------------------------------

def test_shape_matrix_consistency():
    assert SHAPE_MATRIX == list(BENCH_MATRIX_SFM)
    assert len(SHAPE_MATRIX) == 9
    assert PRIMARY_TARGET in SHAPE_MATRIX
    assert SEEDS == [0, 1, 42]


def test_summarize_and_jsonable():
    # 手工构造两条记录（一条通过、一条 row_sum 失败）
    from cudalab.softmax_correctness import CheckResult
    ok = CheckResult(variant="v", shape=[4, 64], dtype="float32", seed=0,
                     mode="normal", passed=True, max_abs_error=1e-7,
                     max_rel_error=1e-7, row_sum_error=1e-8,
                     has_nan=False, has_inf=False,
                     atol=1e-5, rtol=1e-4)
    bad = CheckResult(variant="v", shape=[4, 64], dtype="float32", seed=0,
                      mode="normal", passed=False, max_abs_error=1e-7,
                      max_rel_error=1e-7, row_sum_error=0.02,
                      has_nan=False, has_inf=False,
                      atol=1e-5, rtol=1e-4, note="scaled 0.98")
    s = summarize([ok, bad])
    assert s["n_total"] == 2 and s["n_pass"] == 1 and s["n_fail"] == 1
    assert not s["all_pass"]
    assert s["max_row_sum_error"] == 0.02
    d = _jsonable(ok)
    assert "pass" in d and "passed" not in d


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    total = len(fns)
    print(f"\n{total - failed}/{total} passed")
    sys.exit(1 if failed else 0)
