"""CUDALab v0.3 — Softmax 正确性校验套件。

原则（与 RMSNorm 套件一致）:
- 容差对所有变体固定，并记录在每一份结果中。绝不为某个候选单独放宽
  使其通过。
- 正确性 FAIL 的内核永远没有资格成为性能胜者（在 experiment.py 中强制）。
- 所有结果（通过和失败）都报告并保存；失败用例绝不删除。

参考实现（显式、FP32 内部、与 PyTorch 版本无关）:

    softmax_ref(x) = torch.softmax(x.float(), dim=-1).to(x.dtype)

固定容差:
- float16: atol=2e-3, rtol=5e-3（与 evaluator 核心共享同一策略：
  FP16 约 3 位十进制精度，y 只做一次 fp16 就近舍入，中间量 FP32
  累加，实测最大误差留有充分余量）。
- float32: atol=1e-5, rtol=1e-4。
- row_sum_error（单独报告、单独判定）: max_i |sum_j y[i,j] - 1|，
  在 FP32 中求和。固定阈值 ROW_SUM_TOL:
    float16: 5e-3, float32: 1e-4
  依据: 每行 H 个 fp16 输出元素的就近舍入误差（单元素 ≤ 0.5 ulp
  ≈ 2.4e-4 相对）按随机游走叠加，H ≤ 8192 时远小于阈值；该检查
  捕捉"和不为 1"这一 elementwise 容差可能漏掉的系统性归一化错误。

边界模式（make_input 的 mode）:
  normal / tiny / biased / large_pos（x≈+50）/ large_neg（x≈-50）/
  mixed_extremes（x∈[-80,80]，exp(x) 在 fp16 下会溢出 —— 必须依赖
  max 减除才安全）/ zeros / constant / single_dominant / alternating。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch

from .evaluator.correctness import (
    TOLERANCES,
    REL_EPS_GUARD,
    compute_metrics,
    summarize_results,
    save_suite,
)
from .operators.softmax import softmax_ref, make_input, BENCH_MATRIX_SFM

ROOT = Path(__file__).resolve().parent.parent

# 形状矩阵: 用户指定的 9 个 Softmax 基准形状（含主目标 (128, 4096)）。
SHAPE_MATRIX = list(BENCH_MATRIX_SFM)

# 边界用例矩阵: (M, H, mode)。scale/constant/dominant 用 make_input 默认值。
EDGE_CASES = [
    (4, 4096, "tiny"),
    (4, 4096, "biased"),
    (4, 4096, "large_pos"),
    (4, 4096, "large_neg"),
    (4, 4096, "mixed_extremes"),
    (4, 4096, "zeros"),
    (4, 4096, "constant"),
    (4, 4096, "single_dominant"),
    (4, 4096, "alternating"),
]
SEEDS = [0, 1, 42]

# row_sum_error 的固定判定阈值（对所有变体一致，不随候选放宽）。
ROW_SUM_TOL = {"float16": 5e-3, "float32": 1e-4}


@dataclass
class CheckResult:
    variant: str
    shape: list
    dtype: str
    seed: int
    mode: str
    passed: bool
    max_abs_error: float
    max_rel_error: float
    row_sum_error: float
    has_nan: bool
    has_inf: bool
    atol: float
    rtol: float
    note: str = ""


def _row_sum_extra(dtype_name: str):
    def extra(y: torch.Tensor, ref: torch.Tensor) -> dict:
        # 用候选输出 y 在 FP32 中求行和，与 1 比较（ref 的行和作为
        # 旁路信息一并报告；判定只看候选自身的归一化程度）。
        row_sum_err = float((y.float().sum(dim=-1) - 1.0).abs().max().item())
        ref_row_sum_err = float((ref.float().sum(dim=-1) - 1.0).abs().max().item())
        return {
            "row_sum_error": round(row_sum_err, 8),
            "ref_row_sum_error": round(ref_row_sum_err, 8),
            "row_sum_tol": ROW_SUM_TOL[dtype_name],
        }
    return extra


def check_one(variant: str, ext, x: torch.Tensor, seed: int = 0,
              mode: str = "normal", note: str = "") -> CheckResult:
    y = ext.forward(variant, x)
    y = y.contiguous()
    ref = softmax_ref(x)
    dtype_name = str(x.dtype).split(".")[-1]

    m = compute_metrics(y, ref, dtype_name,
                        extra=_row_sum_extra(dtype_name))
    row_sum_ok = m["row_sum_error"] <= ROW_SUM_TOL[dtype_name]

    M, H = x.shape
    return CheckResult(
        variant=variant,
        shape=[int(M), int(H)],
        dtype=dtype_name,
        seed=seed, mode=mode,
        passed=(m["ok_close"] and not m["has_nan"] and not m["has_inf"]
                and row_sum_ok),
        max_abs_error=m["max_abs_error"],
        max_rel_error=m["max_rel_error"],
        row_sum_error=m["row_sum_error"],
        has_nan=m["has_nan"],
        has_inf=m["has_inf"],
        atol=m["atol"], rtol=m["rtol"],
        note=note,
    )


def run_suite(variant: str, ext, dtypes=("float16", "float32"),
              shapes: Optional[list] = None, edge: bool = True,
              seeds: tuple = SEEDS) -> list[CheckResult]:
    """单个变体的完整正确性套件。"""
    results: list[CheckResult] = []
    shapes = shapes if shapes is not None else SHAPE_MATRIX
    for dtype_name in dtypes:
        dtype = getattr(torch, dtype_name)
        for (M, H) in shapes:
            for seed in seeds:
                x = make_input(M, H, dtype=dtype, seed=seed, mode="normal")
                results.append(check_one(variant, ext, x, seed=seed))
        if edge:
            for (M, H, mode) in EDGE_CASES:
                x = make_input(M, H, dtype=dtype, seed=7, mode=mode)
                results.append(check_one(variant, ext, x, seed=7, mode=mode,
                                         note=f"edge mode={mode}"))
    return results


def _jsonable(r: CheckResult) -> dict:
    d = asdict(r)
    d["pass"] = d.pop("passed")
    return d


def summarize(results: list[CheckResult]) -> dict:
    s = summarize_results([_jsonable(r) for r in results])
    s["max_row_sum_error"] = max((r.row_sum_error for r in results),
                                 default=0.0)
    return s


def save_results(results: list[CheckResult], out_path: Path) -> Path:
    return save_suite(out_path, {
        "operator": "softmax",
        "suite": "softmax-correctness-v0.3",
        "reference": "torch.softmax(x.float(), dim=-1).to(x.dtype)",
        "tolerances": TOLERANCES,
        "row_sum_tol": ROW_SUM_TOL,
        "rel_eps_guard": REL_EPS_GUARD,
        "summary": summarize(results),
        "results": [_jsonable(r) for r in results],
    })
