"""RMSNorm 的显式、与 PyTorch 版本无关的参考实现。

公式（作用于最后一维 H）：

    rms  = sqrt(mean(x_i^2) + eps)
    y_i  = x_i / rms * weight_i

所有中间运算均在 FP32 下进行（float32 累加），与 CUDA 内核的累加策略
一致。显式公式是主参考；它不依赖任何数值行为可能随 PyTorch 版本变化
的融合算子。
"""
from __future__ import annotations

import torch

DEFAULT_EPS = 1e-5


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """参考 RMSNorm。

    参数:
        x: (M, H) 连续输入，float16 或 float32。
        weight: (H,) 连续，与 x 同 dtype。
        eps: epsilon，默认 1e-5。

    返回:
        (M, H) 张量，与 x 同 dtype。
    """
    if x.dim() != 2 or weight.dim() != 1:
        raise ValueError(f"期望 2 维 x 与 1 维 weight，实际 x={x.dim()}D w={weight.dim()}D")
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if x.dtype != weight.dtype:
        raise ValueError(f"dtype 不一致: x={x.dtype} w={weight.dtype}")
    if x.size(-1) != weight.size(0):
        raise ValueError(f"维度不匹配: H={x.size(-1)} vs weight={weight.size(0)}")

    x32 = x.float()
    variance = x32.pow(2).mean(dim=-1, keepdim=True)  # FP32 累加
    y = x32 * torch.rsqrt(variance + eps)
    y = y * weight.float()
    return y.to(x.dtype)


def make_inputs(M: int, H: int, dtype: torch.dtype, device: str = "cuda",
                seed: int = 0, scale: float = 1.0, mode: str = "normal") -> tuple[torch.Tensor, torch.Tensor]:
    """正确性与基准两条路径共享的确定性输入生成器。

    mode: 'normal' | 'zeros' | 'tiny' | 'biased'
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    w = torch.randn(H, generator=g, dtype=torch.float32, device=device) * 0.5 + 1.0
    if mode == "zeros":
        x = torch.zeros(M, H, dtype=torch.float32, device=device)
    elif mode == "tiny":
        x = torch.randn(M, H, generator=g, dtype=torch.float32, device=device) * 1e-4
    elif mode == "biased":
        x = torch.randn(M, H, generator=g, dtype=torch.float32, device=device) * scale + 3.0
    else:
        x = torch.randn(M, H, generator=g, dtype=torch.float32, device=device) * scale
    return x.to(dtype).contiguous(), w.to(dtype).contiguous()
