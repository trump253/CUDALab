"""CUDALab correctness harness.

Principles:
- The tolerance is FIXED for every variant and recorded in every result.
  It must never be relaxed for a specific candidate to make it pass.
- A kernel that FAILS correctness is never eligible to be a performance
  winner (enforced in experiment.py).
- Every result (pass AND fail) is reported and saved; failed cases are
  never dropped.

FP16 tolerance rationale (recorded, not negotiated):
  - FP16 has ~3 decimal digits of precision (eps_rel = 2^-11 ~ 4.9e-4).
  - y is rounded to fp16 once (round-to-nearest), so a single output
    element may deviate by up to ~0.5 ulp ~ 2.4e-4 relative.
  - x and w are exact (same input), ss is accumulated in FP32, so the
    residual error budget is dominated by the final fp16 rounding plus
    the fp32 reduction order (tiny).
  - atol=2e-3, rtol=5e-3 covers observed max errors on N(0,1) inputs with
    a comfortable margin, and was verified against measured results.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch

from .reference import rmsnorm_ref, make_inputs, DEFAULT_EPS

ROOT = Path(__file__).resolve().parent.parent

# Fixed tolerance policy — identical for ALL variants.
TOLERANCES = {
    "float16": {"atol": 2e-3, "rtol": 5e-3},
    "float32": {"atol": 1e-5, "rtol": 1e-4},
}
# guard for relative error to avoid amplifying noise near zero
REL_EPS_GUARD = 1e-3

# Standard shape matrix (M, H)
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
# Edge-case matrix: (M, H, mode, scale)
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
    tol = TOLERANCES[str(x.dtype).split(".")[-1]]
    y = ext.forward(variant, x, w, eps)
    y = y.contiguous()
    ref = rmsnorm_ref(x, w, eps)

    diff = (y.float() - ref.float()).abs()
    max_abs = float(diff.max().item()) if x.numel() else 0.0

    denom = ref.float().abs()
    rel = diff / denom.clamp_min(REL_EPS_GUARD)
    max_rel = float(rel.max().item()) if x.numel() else 0.0

    has_nan = bool(torch.isnan(y).any().item())
    has_inf = bool(torch.isinf(y).any().item())

    # allclose with the fixed recorded tolerance
    ok_close = bool(torch.allclose(y.float(), ref.float(),
                                   atol=tol["atol"], rtol=tol["rtol"]))
    ok = ok_close and not has_nan and not has_inf

    M, H = x.shape
    return CheckResult(
        variant=variant,
        shape=[int(M), int(H)],
        dtype=str(x.dtype).split(".")[-1],
        seed=seed, mode=mode,
        passed=ok,
        max_abs_error=round(max_abs, 8),
        max_rel_error=round(max_rel, 8),
        has_nan=has_nan,
        has_inf=has_inf,
        atol=tol["atol"], rtol=tol["rtol"],
        note=note,
    )


def run_suite(variant: str, ext, dtypes=("float16", "float32"),
              shapes: Optional[list] = None, edge: bool = True,
              seeds: tuple = SEEDS) -> list[CheckResult]:
    """Full correctness suite for one variant."""
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
    failed = [r for r in results if not r.passed]
    return {
        "n_total": len(results),
        "n_pass": len(results) - len(failed),
        "n_fail": len(failed),
        "all_pass": not failed,
        "max_abs_error": max((r.max_abs_error for r in results), default=0.0),
        "max_rel_error": max((r.max_rel_error for r in results), default=0.0),
        "failed": [_jsonable(r) for r in failed[:20]],
    }


def save_results(results: list[CheckResult], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "tolerances": TOLERANCES,
        "rel_eps_guard": REL_EPS_GUARD,
        "summary": summarize(results),
        "results": [_jsonable(r) for r in results],
    }, indent=2))
    return out_path
