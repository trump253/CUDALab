"""CUDALab v0.7 — W4A16 group-wise INT4 量化合同（对称, zero_point=0）。

用户 §1 合同（逐字）:

    W_fp16:   [N,K]     原始权重，仅量化阶段使用
    W_packed: uint8     两个 INT4 / byte
    scale:    fp16      [N, K/G]
    group_size G = 128
    accumulation = FP32

    q ∈ [-7, 7],  zero_point = 0
    scale[n,g] = max(abs(W_group)) / 7
    q = clamp(round(W / scale), -7, 7)

- 零 group 必须安全处理: 全零 group → scale=0, q≡0（不除零、无 NaN,
  反量化恒 0, y 贡献 0）。
- K 要求: K % 128 == 0（量化器显式拒绝, 内核 host 验证同样拒绝）。
- 量化和 packing 全部在 benchmark 计时区外（池构造期 / 正确性套件
  构造期调用; 计时 launch 只读已量化的 W_packed / scale）。
- 舍入模式: `torch.round` = round-half-to-even（确定性, 同 v0.6）。
- 本模块只接受有限 FP16 W（非有限输入在量化器内显式拒绝 —— 离线
  严格路径; 内核 = 在线元数据路径, 与 v0.5/v0.6 语义一致）。
- **scale 以 FP16 存储**（合同指定）: 量化器在 FP32 中计算
  amax/7, 然后 cast 到 fp16 存储; 内核与参考层 A 都读**存储的
  fp16 scale**（即 scale 的 fp16 量化是算子数据合同的一部分, 不是
  参考与实现的偏差）。第 (b) 层保真度同样用存储 fp16 scale 计算
  y_quant。

Nibble 打包合同（`pack_q` / `unpack_w`, 设备端镜像实现见
kernels/int4gemv/int4gemv_common.h）:
    W_packed[n, b] 的低 4 bit  = 元素 k = 2b   （low nibble）
    W_packed[n, b] 的高 4 bit  = 元素 k = 2b+1  （high nibble）
    编码 = 4-bit two's complement: nibble = v & 0xF（v ∈ [-8, 7];
    量化域 [-7, 7] 是它的子集）。unpack 必须做符号扩展（负数高 4
    bit 为 1 → -8..-1）。
"""
from __future__ import annotations

import torch

INT4_MAX = 7        # 对称量化最大绝对值（zero_point = 0, 不用 ±8）
GROUP_SIZE = 128    # 固定 group size（合同）


def _check_w(W: torch.Tensor) -> None:
    if W.dim() != 2:
        raise ValueError(f"期望 2 维 W (N, K)，实际 {W.dim()}D")
    if W.dtype != torch.float16:
        raise ValueError(f"量化器只接受 FP16 权重, 实际 {W.dtype}")
    if not W.is_contiguous():
        raise ValueError("量化器要求 W 连续")
    if W.size(1) % GROUP_SIZE != 0:
        raise ValueError(
            f"K 必须可被 {GROUP_SIZE} 整除（group_size 合同）, "
            f"实际 K={W.size(1)}")
    if not torch.isfinite(W.float()).all():
        raise ValueError("量化器要求 W 有限（NaN/Inf 由上游生成器保证; "
                         "非有限权重不走量化合同）")


def quantize_w(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对称 group-wise INT4 量化（合同见模块 docstring）。

    参数:
        W: FP16 (N, K) 连续（CPU / CUDA 均可, K % 128 == 0）。
    返回:
        (W_packed, scale): W_packed = uint8 (N, K/2)（两 INT4/byte,
        低 nibble = k=2b, 高 nibble = k=2b+1）; scale = fp16
        (N, K/128)（scale[n,g] = max|W_group|/7, FP32 计算后 cast
        fp16 存储; 全零 group scale=0, 对应 q≡0）。
    拒绝:
        非 2 维 / 非连续 / 非 FP16 / K%128!=0 / 含 NaN 或 Inf
        （ValueError, 量化器是离线严格路径）。
    """
    _check_w(W)
    N, K = W.shape
    G = K // GROUP_SIZE
    Wg = W.float().view(N, G, GROUP_SIZE)
    amax = Wg.abs().amax(dim=2)                  # (N, G) fp32
    scale32 = amax / INT4_MAX
    zero = (scale32 == 0)
    safe = torch.where(zero, torch.ones_like(scale32), scale32)
    q = torch.round(Wg / safe.unsqueeze(2))      # round-half-to-even
    q = q.clamp_(-INT4_MAX, INT4_MAX)
    if bool(zero.any()):
        q = torch.where(zero.unsqueeze(2), torch.zeros_like(q), q)
    W_packed = pack_q(q.view(N, K).to(torch.int32))
    scale16 = scale32.to(torch.float16).view(N, G).contiguous()
    return W_packed, scale16


def pack_q(q: torch.Tensor) -> torch.Tensor:
    """INT4 值域张量 → uint8 packed（两元素/byte, 见模块 docstring
    的 nibble 合同）。q: int32/int64 (N, K), K 偶, 值域 [-8, 7]
    （量化输出 [-7, 7]）。

    low  nibble = q[..., 2b]   （元素 2b）
    high nibble = q[..., 2b+1] （元素 2b+1）
    nibble = v & 0xF（two's complement 低 4 位, 负数自动符号扩展
    语义由 unpack 恢复）。
    """
    if q.dim() != 2:
        raise ValueError(f"pack_q 期望 2 维 (N, K), 实际 {q.dim()}D")
    N, K = q.shape
    if K % 2 != 0:
        raise ValueError(f"pack_q 要求 K 偶数, 实际 K={K}")
    if not (-8 <= int(q.min().item())) or (int(q.max().item()) > 7):
        raise ValueError("pack_q 值域越界: 期望 INT4 two's complement "
                         "[-8, 7]")
    q2 = q.to(torch.int32).view(N, K // 2, 2)
    lo = q2[..., 0] & 15          # two's complement 低 4 位
    hi = q2[..., 1] & 15
    return (((hi << 4) | lo).to(torch.uint8)
            .view(N, K // 2).contiguous())


def unpack_w(W_packed: torch.Tensor) -> torch.Tensor:
    """uint8 packed → INT4 值域张量（`pack_q` 的精确逆, CPU 参考）。

    W_packed: uint8 (N, K/2) 连续。返回 int32 (N, K), 值域
    [-8, 7]（符号扩展）。与设备端 int4gemv_unpack_byte 逐元素一致
    （由 kernel 正确性层 A 在 GPU 上钉死）。
    """
    if W_packed.dim() != 2:
        raise ValueError(f"unpack_w 期望 2 维 (N, K/2), 实际 "
                         f"{W_packed.dim()}D")
    if W_packed.dtype != torch.uint8:
        raise ValueError(f"unpack_w 只接受 uint8, 实际 {W_packed.dtype}")
    N, H = W_packed.shape
    lo = (W_packed & 15).to(torch.int32)
    hi = ((W_packed >> 4) & 15).to(torch.int32)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    return torch.stack((lo, hi), dim=-1).view(N, 2 * H).contiguous()


def dequantize_w(W_packed: torch.Tensor,
                 scale: torch.Tensor) -> torch.Tensor:
    """反量化 W_dequant[n,k] = unpack(W_packed)[n,k] · scale[n, k/G]
    （FP32; scale 是**存储的 fp16 值**提升 fp32 —— 算子数据合同）。
    仅用于参考 / 保真度计算（内核自己在线反量化, 不走这里）。
    """
    Wq = unpack_w(W_packed)
    N, K = Wq.shape
    if scale.shape != (N, K // GROUP_SIZE):
        raise ValueError(f"scale 形状 {tuple(scale.shape)} 不匹配 "
                         f"({N}, {K // GROUP_SIZE})")
    s = scale.float().repeat_interleave(GROUP_SIZE, dim=1)  # (N, K)
    return Wq.float() * s


def fidelity_metrics(W: torch.Tensor, x: torch.Tensor,
                     W_packed: torch.Tensor,
                     scale: torch.Tensor) -> dict:
    """量化保真度（两层正确性的第 (c) 层, **只报告不判定**）。

    对比同一 x 下:
        y_orig  = W @ x              （原始 FP16 权重, FP32 累加）
        y_quant = W_dequant @ x      （unpack + group 反量化, 用
                                       **存储的 fp16 scale**, FP32 累加）
    指标（用户指定三项）: max_abs / RMSE / cosine similarity
    （cosine 在零向量上定义为零 → 记 None, 不伪造 1.0）。

    另附两个量化器自检量（审计用）:
        max_dequant_abs_err:  max_{n,k} |W[n,k] − q·s16|
        max_dequant_step_err: max_{n,k} |W − q·s16| / s（分母 clamp
            1e-30 防除零; FP32 实现路径下构造性界 ≈ 0.5 + 7·2^-11
            —— 0.5 为 round 步长, 7·2^-11 为 fp16 scale 存储量化
            放大; report-only 自检量, 不参与任何判定）
    """
    xf = x.float()
    y_orig = torch.mv(W.float(), xf)
    Wd = dequantize_w(W_packed, scale)
    y_quant = torch.mv(Wd, xf)
    diff = y_quant - y_orig
    max_abs = float(diff.abs().max().item())
    rmse = float(diff.pow(2).mean().sqrt().item())
    no, nq = float(y_orig.norm().item()), float(y_quant.norm().item())
    cosine = (float(torch.dot(y_orig.flatten(), y_quant.flatten()).item()
                    / (no * nq)) if (no > 0 and nq > 0) else None)
    N, K = W.shape
    s = scale.float().clamp_min(1e-30).repeat_interleave(
        GROUP_SIZE, dim=1)
    deq_abs = (W.float() - Wd).abs()
    return {
        "max_abs": max_abs,
        "rmse": rmse,
        "cosine": cosine,
        "y_orig_norm": no,
        "y_quant_norm": nq,
        "max_dequant_abs_err": float(deq_abs.max().item()),
        "max_dequant_step_err": float((deq_abs / s).max().item()),
        "n_zero_scale_groups": int((scale == 0).sum().item()),
        "max_abs_q": int(unpack_w(W_packed).abs().max().item())
        if W_packed.numel() else 0,
        "note": "量化保真度: 量化误差, 不是 kernel bug —— 只报告, "
                "不作 kernel 正确性判定门（用户 §2 三层分离）",
    }
