"""Explicit, version-independent PyTorch reference for RMSNorm.

Formula (applied on the last dimension, H):

    rms  = sqrt(mean(x_i^2) + eps)
    y_i  = x_i / rms * weight_i

All intermediate math is done in FP32 (float32 accumulation), mirroring the
CUDA kernels' accumulation policy. The explicit formula is the primary
reference; it does not rely on any fused operator whose numerics may vary
between PyTorch versions.
"""
from __future__ import annotations

import torch

DEFAULT_EPS = 1e-5


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """Reference RMSNorm.

    Args:
        x: (M, H) contiguous input, float16 or float32.
        weight: (H,) contiguous, same dtype as x.
        eps: epsilon, default 1e-5.

    Returns:
        (M, H) tensor, same dtype as x.
    """
    if x.dim() != 2 or weight.dim() != 1:
        raise ValueError(f"expected 2-D x and 1-D weight, got x={x.dim()}D w={weight.dim()}D")
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if x.dtype != weight.dtype:
        raise ValueError(f"dtype mismatch: x={x.dtype} w={weight.dtype}")
    if x.size(-1) != weight.size(0):
        raise ValueError(f"dim mismatch: H={x.size(-1)} vs weight={weight.size(0)}")

    x32 = x.float()
    variance = x32.pow(2).mean(dim=-1, keepdim=True)  # FP32 accumulation
    y = x32 * torch.rsqrt(variance + eps)
    y = y * weight.float()
    return y.to(x.dtype)


def make_inputs(M: int, H: int, dtype: torch.dtype, device: str = "cuda",
                seed: int = 0, scale: float = 1.0, mode: str = "normal") -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic input generator shared by correctness and benchmark paths.

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
