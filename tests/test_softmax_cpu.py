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


# ---- online (m, l) merge 恒等（docs/softmax_algorithm.md §6 门禁）-----------
#
# 这些测试在任何 online/单遍 CUDA 变体进入仓库之前必须全部通过。
# 推导见 docs/softmax_algorithm.md: (m_S, l_S) = (max, Σexp(x−max))
# 是段的充分统计量，合并恒等
#     (m,l) ⊕ (m',l') = (M, l·exp(m−M) + l'·exp(m'−M)),  M = max(m,m')

import math  # noqa: E402


def _merge(m, l, m2, l2):
    M = m if m >= m2 else m2
    return M, l * math.exp(m - M) + l2 * math.exp(m2 - M)


def _online_scan_py(x_vals):
    """逐元素在线累积（Python float ≈ float64）。"""
    m, l = float("-inf"), 0.0
    for v in x_vals:
        m2 = m if m >= v else v
        l = l * math.exp(m - m2) + math.exp(v - m2)
        m = m2
    return m, l


def _direct_m_l(x_vals):
    M = max(x_vals)
    L = sum(math.exp(v - M) for v in x_vals)
    return M, L


def _online_scan_f32(x: torch.Tensor):
    """逐元素在线累积（真实 float32 算术，0 维张量）。"""
    m = torch.full((), float("-inf"), dtype=torch.float32)
    l = torch.zeros((), dtype=torch.float32)
    for i in range(x.numel()):
        v = x.reshape(-1)[i].float()
        m2 = torch.maximum(m, v)
        l = l * torch.exp(m - m2) + torch.exp(v - m2)
        m = m2
    return m, l


def test_online_scan_matches_direct():
    g = torch.Generator(device=DEV); g.manual_seed(0)
    for (n, scale) in ((128, 1.0), (4096, 1.0), (1024, 80.0),
                       (333, 50.0), (1, 7.0), (16, 0.0)):
        if scale == 0.0:
            x = torch.zeros(n, dtype=torch.float64, device=DEV)
        else:
            x = torch.randn(n, generator=g, dtype=torch.float64,
                            device=DEV) * scale
        m_o, l_o = _online_scan_py(x.tolist())
        M_d, L_d = _direct_m_l(x.tolist())
        assert abs(m_o - M_d) == 0.0, f"max 不一致 n={n} scale={scale}"
        rel = abs(l_o - L_d) / L_d
        # 恒等本体: 在线累积 vs 直接 (max, Σexp) —— float64 求和顺序
        # 差异量级（~2e-15），远小于任何内核容差
        assert rel <= 1e-12, f"l 相对误差 {rel} n={n} scale={scale}"
        # 归一化结果: 先对自己的 (M_d, L_d) 严格对拍（恒等 ⇒ 逐位一致），
        # 再对 torch.softmax 松对拍（float64 参考实现内部求和顺序不同，
        # 实测差异 ≤ ~1e-9；此门只要求远超内核容差 1e-5 的一致性）
        y = torch.tensor([math.exp(v - m_o) / l_o for v in x.tolist()],
                         dtype=torch.float64)
        y_direct = torch.tensor([math.exp(v - M_d) / L_d
                                 for v in x.tolist()], dtype=torch.float64)
        # y vs y_direct 的相对差 = l_o vs L_d 的相对差（≤1e-12）
        assert float((y - y_direct).abs().max()) <= 1e-11
        ref = torch.softmax(x, dim=0)
        assert float((y - ref).abs().max()) <= 1e-9


def test_online_scan_f32_within_kernel_tolerance():
    """float32 在线路径的数值必须在 CUDA 内核使用的固定容差内。"""
    g = torch.Generator(device=DEV); g.manual_seed(1)
    for scale in (1.0, 80.0):
        x = torch.randn(4096, generator=g, dtype=torch.float32,
                        device=DEV) * scale
        m, l = _online_scan_f32(x)
        y = torch.exp(x - m) / l
        ref = torch.softmax(x, dim=0)
        err = float((y - ref).abs().max())
        assert err <= TOLERANCES["float32"]["atol"] + 1e-9, \
            f"fp32 online max_abs={err}"
        rs = float((y.sum() - 1.0).abs())
        assert rs <= ROW_SUM_TOL["float32"], f"row_sum={rs}"


def test_merge_multisegment_random_splits():
    g = torch.Generator(device=DEV); g.manual_seed(2)
    for n, k in ((4096, 1), (4096, 2), (4096, 5), (300, 17), (16, 16)):
        x = torch.randn(n, generator=g, dtype=torch.float64,
                        device=DEV) * (1.0 if n > 100 else 80.0)
        bounds = sorted(torch.randperm(n - 1)[:k - 1].tolist()) if k > 1 else []
        cuts = [0] + bounds + [n]
        segs = [_online_scan_py(x[c1:c2].tolist())
                for c1, c2 in zip(cuts[:-1], cuts[1:])]
        # 二分树两两 merge（与 CUDA 归约的成对合并同构）
        while len(segs) > 1:
            nxt = []
            for i in range(0, len(segs) - 1, 2):
                nxt.append(_merge(segs[i][0], segs[i][1],
                                  segs[i + 1][0], segs[i + 1][1]))
            if len(segs) % 2 == 1:
                nxt.append(segs[-1])
            segs = nxt
        M, L = segs[0]
        M_d, L_d = _direct_m_l(x.tolist())
        assert abs(M - M_d) == 0.0
        rel = abs(L - L_d) / L_d
        assert rel <= 1e-12, f"merge L 相对误差 {rel} n={n} k={k}"


def test_merge_empty_segment_identity():
    m, l = _online_scan_py([1.0, -2.0, 0.5, 3.0])
    M2, L2 = _merge(float("-inf"), 0.0, m, l)
    assert M2 == m and L2 == l
    M3, L3 = _merge(m, l, float("-inf"), 0.0)
    assert M3 == m and L3 == l


def test_merge_determinism():
    g = torch.Generator(device=DEV); g.manual_seed(3)
    x = torch.randn(2048, generator=g, dtype=torch.float64, device=DEV)
    r = []
    for _ in range(3):
        m, l = _online_scan_py(x.tolist())
        r.append((m, l))
    assert r[0] == r[1] == r[2], "同一输入必须逐位可复现"


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
