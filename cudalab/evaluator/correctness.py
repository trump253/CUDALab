"""CUDALab v0.3 — 正确性校验通用机制（算子无关）。

从 v0.2 `cudalab/correctness.py` 抽出的共享部分：
- `TOLERANCES` / `REL_EPS_GUARD`：固定容差策略（FP16 容差依据见
  v0.2 原文，原样保留：FP16 约 3 位十进制精度，y 只做一次 fp16 就近
  舍入，x 精确、中间量 FP32 累加，atol=2e-3 / rtol=5e-3 对实测最大
  误差留有充分余量）。**容差对所有变体固定，绝不为某个候选单独放宽
  使其通过。**
- `compute_metrics`：y vs ref 的 max_abs / max_rel / NaN / Inf /
  allclose 判定（额外指标由调用方以 `extra` 调用函数注入，例如
  Softmax 的 row_sum_error）。
- `summarize_results` / `save_suite`：结构化汇总与保存（通过和失败
  都报告；失败用例绝不删除）。

算子特有的部分（参考实现、输入生成、形状/边界矩阵、套件循环）留在
各自的 operator 模块中（cudalab/operators/*、cudalab/softmax_*）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import torch

# 固定容差策略 —— 对所有变体完全一致。
TOLERANCES = {
    "float16": {"atol": 2e-3, "rtol": 5e-3},
    "float32": {"atol": 1e-5, "rtol": 1e-4},
}
# 相对误差分母保护，避免在零附近放大噪声
REL_EPS_GUARD = 1e-3


def compute_metrics(y: torch.Tensor, ref: torch.Tensor, dtype_name: str,
                    rel_eps_guard: float = REL_EPS_GUARD,
                    extra: Optional[Callable[[torch.Tensor, torch.Tensor], dict]]
                    = None) -> dict:
    """y（候选输出）vs ref（参考输出）的固定指标。

    `extra(y, ref)` 可返回附加指标 dict（合并进结果，原样保留其值，
    不做 pass 判定 —— 是否纳入判定由套件层用固定阈值决定并记录）。
    """
    tol = TOLERANCES[dtype_name]
    diff = (y.float() - ref.float()).abs()
    max_abs = float(diff.max().item()) if y.numel() else 0.0

    denom = ref.float().abs()
    rel = diff / denom.clamp_min(rel_eps_guard)
    max_rel = float(rel.max().item()) if y.numel() else 0.0

    has_nan = bool(torch.isnan(y).any().item())
    has_inf = bool(torch.isinf(y).any().item())

    # 用固定且已记录的容差做 allclose
    ok_close = bool(torch.allclose(y.float(), ref.float(),
                                   atol=tol["atol"], rtol=tol["rtol"]))
    m = {
        "max_abs_error": round(max_abs, 8),
        "max_rel_error": round(max_rel, 8),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "ok_close": ok_close,
        "atol": tol["atol"],
        "rtol": tol["rtol"],
    }
    if extra is not None:
        m.update(extra(y, ref))
    return m


def summarize_results(results: list[dict]) -> dict:
    """results: 每条记录至少含 "pass"。失败记录原样保留（前 20 条）。"""
    failed = [r for r in results if not r["pass"]]
    return {
        "n_total": len(results),
        "n_pass": len(results) - len(failed),
        "n_fail": len(failed),
        "all_pass": not failed,
        "max_abs_error": max((r["max_abs_error"] for r in results), default=0.0),
        "max_rel_error": max((r["max_rel_error"] for r in results), default=0.0),
        "failed": failed[:20],
    }


def save_suite(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))
    return path
