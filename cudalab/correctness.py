"""CUDALab 正确性校验（RMSNorm 套件；v0.3 起机制复用 evaluator 核心）。

原则:
- 容差对所有变体固定，并记录在每一份结果中。绝不为某个候选单独放宽
  使其通过。
- 正确性 FAIL 的内核永远没有资格成为性能胜者（在 experiment.py 中强制）。
- 所有结果（通过和失败）都报告并保存；失败用例绝不删除。

FP16 容差的依据（已记录，不谈判）:
  - FP16 约 3 位十进制精度（eps_rel = 2^-11 ~ 4.9e-4）。
  - y 只做一次 fp16 舍入（就近舍入），单个输出元素最多偏差约
    0.5 ulp ~ 2.4e-4（相对）。
  - x 和 w 是精确的（同一输入），ss 以 FP32 累加，因此残余误差预算
    由最终 fp16 舍入 + fp32 归约顺序（极小）主导。
  - atol=2e-3、rtol=5e-3 对 N(0,1) 输入的实测最大误差留有充分余量，
    并已对照实测结果验证。

v0.3: 指标计算 / 汇总 / 保存机制移至 cudalab/evaluator/correctness.py
（compute_metrics / summarize_results / save_suite）；本模块保留
RMSNorm 的 CheckResult schema、形状/边界矩阵与套件循环，输出 JSON
schema 与 v0.2 完全一致。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch

from .reference import rmsnorm_ref, make_inputs, DEFAULT_EPS
from .evaluator.correctness import (
    TOLERANCES,
    REL_EPS_GUARD,
    compute_metrics,
    summarize_results,
    save_suite,
)

ROOT = Path(__file__).resolve().parent.parent

# 标准形状矩阵 (M, H)
SHAPE_MATRIX = [
    (1, 1024),
    (1, 2048),
    (1, 4096),
    (1, 8192),
    (16, 1024),
    (16, 4096),
    (128, 1024),
    (128, 4096),
    (128, 8192),
    (1024, 1024),
    (1024, 4096),
]
# 边界用例矩阵: (M, H, mode, scale)
EDGE_CASES = [
    (4, 4096, "zeros", 1.0),
    (4, 4096, "tiny", 1.0),
    (4, 4096, "normal", 10.0),
    (4, 4096, "normal", 0.1),
    (4, 4096, "biased", 1.0),
]
SEEDS = [0, 1, 42]


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
    has_nan: bool
    has_inf: bool
    atol: float
    rtol: float
    note: str = ""


def check_one(variant: str, ext, x: torch.Tensor, w: torch.Tensor,
              eps: float = DEFAULT_EPS, seed: int = 0, mode: str = "normal",
              note: str = "") -> CheckResult:
    y = ext.forward(variant, x, w, eps)
    y = y.contiguous()
    ref = rmsnorm_ref(x, w, eps)
    dtype_name = str(x.dtype).split(".")[-1]

    m = compute_metrics(y, ref, dtype_name)

    M, H = x.shape
    return CheckResult(
        variant=variant,
        shape=[int(M), int(H)],
        dtype=dtype_name,
        seed=seed, mode=mode,
        passed=m["ok_close"] and not m["has_nan"] and not m["has_inf"],
        max_abs_error=m["max_abs_error"],
        max_rel_error=m["max_rel_error"],
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
                x, w = make_inputs(M, H, dtype=dtype, seed=seed, mode="normal")
                results.append(check_one(variant, ext, x, w, seed=seed))
        if edge:
            for (M, H, mode, scale) in EDGE_CASES:
                x, w = make_inputs(M, H, dtype=dtype, seed=7, scale=scale, mode=mode)
                results.append(check_one(variant, ext, x, w, seed=7, mode=mode,
                                         note=f"edge mode={mode} scale={scale}"))
    return results


def _jsonable(r: CheckResult) -> dict:
    d = asdict(r)
    d["pass"] = d.pop("passed")
    return d


def summarize(results: list[CheckResult]) -> dict:
    return summarize_results([_jsonable(r) for r in results])


def save_results(results: list[CheckResult], out_path: Path) -> Path:
    return save_suite(out_path, {
        "tolerances": TOLERANCES,
        "rel_eps_guard": REL_EPS_GUARD,
        "summary": summarize(results),
        "results": [_jsonable(r) for r in results],
    })
