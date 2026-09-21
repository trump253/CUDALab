"""CUDALab v0.6 — INT8 weight-only 量化合同（对称 per-row, zero_point=0）。

用户 §1 合同（逐字）:

    scale[n] = max(abs(W[n,:])) / 127
    q[n,k]   = clamp(round(W[n,k] / scale[n]), -127, 127)

- 量化在 **benchmark 计时区外** 预先完成（池构造期 / 正确性套件构造
  期调用; 计时 launch 只读已量化的 W_q / scale）。
- scale = 0 的行必须安全: 不除零、无 NaN, 输出 q ≡ 0（反量化恒 0,
  y = 0）。
- 舍入模式: `torch.round` = **round-half-to-even**（确定性, 逐元素,
  无 RNG; 同输入同输出, 跨平台同一 torch 版本可复现）。
- 本模块只接受 **有限** FP16 W（非有限输入在量化器内显式拒绝, 与
  内核路径的元数据-only 验证分层: 量化器 = 离线严格路径, 内核 =
  在线元数据路径, 与 v0.5 GEMV 的 NaN/Inf W 语义一致）。
- v0.6 范围外（用户显式排除）: INT4 / group-wise / GPTQ / AWQ /
  activation 量化 / Tensor Core GEMM。

两层正确性的第 (b) 层（量化保真度）指标也由本模块提供
（`fidelity_metrics`）: 量化误差与 kernel bug **不用同一种容差**
（kernel 正确性 = 固定 arith 界 + 有限性门, 见 qgemv_correctness.py;
量化保真度 = max_abs / max_rel / RMSE / cosine similarity, 只报告
不判定）。
"""
from __future__ import annotations

import torch

INT8_MAX = 127  # 对称量化使用的最大绝对值（zero_point = 0）


def quantize_w(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对称 per-row INT8 量化（合同见模块 docstring）。

    参数:
        W: FP16 (N, K) 连续（CPU / CUDA 均可, 量化器不在计时区）。
    返回:
        (W_q, scale): W_q = int8 (N, K), scale = fp32 (N,)。
        scale[n] = max_k |W[n,k]| / 127（FP32 计算, 避免 fp16 下溢/
        溢出）; 全零行 scale = 0, W_q[n] ≡ 0。
    拒绝:
        非 2 维 / 非连续 / 非 FP16 / 含 NaN 或 Inf 的输入
        （ValueError, 量化器是离线严格路径）。
    """
    if W.dim() != 2:
        raise ValueError(f"期望 2 维 W (N, K)，实际 {W.dim()}D")
    if W.dtype != torch.float16:
        raise ValueError(f"量化器只接受 FP16 权重, 实际 {W.dtype}")
    if not W.is_contiguous():
        raise ValueError("量化器要求 W 连续")
    if not torch.isfinite(W.float()).all():
        raise ValueError("量化器要求 W 有限（NaN/Inf 由上游生成器保证; "
                         "非有限权重不走量化合同）")
    Wf = W.float()
    amax = Wf.abs().amax(dim=1)                 # (N,) fp32
    scale = amax / INT8_MAX
    zero = (scale == 0)
    safe = torch.where(zero, torch.ones_like(scale), scale)
    q = torch.round(Wf / safe.unsqueeze(1))     # round-half-to-even
    q = q.clamp_(-INT8_MAX, INT8_MAX)
    if bool(zero.any()):
        q = torch.where(zero.unsqueeze(1), torch.zeros_like(q), q)
    return q.to(torch.int8), scale


def dequantize_w(W_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """反量化 W_dequant = scale[:, None] · W_q（FP32）。仅用于参考 /
    保真度计算（内核自己在线反量化, 不走这里）。"""
    return W_q.float() * scale.unsqueeze(1)


def fidelity_metrics(W: torch.Tensor, x: torch.Tensor,
                     W_q: torch.Tensor,
                     scale: torch.Tensor) -> dict:
    """量化保真度（两层正确性的第 (b) 层, **只报告不判定**）。

    对比同一 x 下:
        y_orig  = W @ x            （原始 FP16 权重, FP32 累加）
        y_quant = (W_q·scale) @ x  （量化-反量化权重, FP32 累加）
    指标（用户指定四项）: max_abs / max_rel / RMSE / cosine similarity
    （cosine 在零向量上定义为零 → 记 None, 不伪造 1.0）。

    另附两个量化器自检量（审计用）:
        max_dequant_abs_err:  max_{n,k} |W[n,k] − q·scale|（应 ≤
            scale[n]/2, 全零行为 0）
        max_dequant_step_err: max_{n,k} |W − q·scale| / scale[n]
            （精确算术意义下 ≤ 0.5 的构造性界; FP32 实现路径因 scale 与
            q·scale 两次舍入实测 ≈ 0.5000019, 偏差 O(2^-24·127) 量级;
            本量为 report-only 自检量, 不参与任何判定; 全零行贡献 0）
    """
    xf = x.float()
    y_orig = torch.mv(W.float(), xf)
    Wd = dequantize_w(W_q, scale)
    y_quant = torch.mv(Wd, xf)
    diff = y_quant - y_orig
    max_abs = float(diff.abs().max().item())
    # max_rel 的分母保护（同 v0.5 REL_EPS_GUARD 语义: 小分母不放大）
    max_rel = float((diff.abs() / y_orig.abs().clamp_min(1e-3)).max().item())
    rmse = float(diff.pow(2).mean().sqrt().item())
    no, nq = float(y_orig.norm().item()), float(y_quant.norm().item())
    cosine = (float(torch.dot(y_orig.flatten(), y_quant.flatten()).item()
                    / (no * nq)) if (no > 0 and nq > 0) else None)
    deq_abs = (W.float() - Wd).abs()
    scale_c = scale.clamp_min(1e-30)
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "rmse": rmse,
        "cosine": cosine,
        "y_orig_norm": no,
        "y_quant_norm": nq,
        "max_dequant_abs_err": float(deq_abs.max().item()),
        "max_dequant_step_err": float((deq_abs / scale_c.unsqueeze(1)).max().item()),
        "n_zero_scale_rows": int((scale == 0).sum().item()),
        "max_abs_q": int(W_q.abs().max().item()) if W_q.numel() else 0,
        "note": "量化保真度: 量化误差, 不是 kernel bug —— 只报告, "
                "不作 kernel 正确性判定门（两层容差分离, 用户 §2）",
    }
