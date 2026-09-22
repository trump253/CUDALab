"""CUDALab v0.7 — W4A16 量化合同 CPU 单测（用户 §2 第 (b) 层:
pack/unpack correctness, 独立于 GPU kernel）。

覆盖:
  1. nibble 编码全字节双射: 全部 256 个 byte 值 unpack → (lo, hi)
     → repack 必须逐字节一致（low/high nibble 位置、符号扩展、
     边界 -7/0/+7 全部在其中）;
  2. pack_q / unpack_w 全值域往返: v ∈ [-8, 7]（two's complement
     INT4 全域）与量化域 [-7, 7], 低/高 nibble 位置独立验证;
  3. 量化合同: scale = max|W_group|/7（fp32 计算 cast fp16 存储）,
     q = clamp(round-half-to-even(W/scale), -7, 7), 零 group 安全
     （scale=0, q≡0, 无 NaN）, 重构误差 ≤ 0.5·scale（fp32 语义）;
  4. 拒绝路径: 非 2 维 / 非 FP16 / 非连续 / K%128!=0 / NaN / Inf;
  5. fidelity_metrics 小例（有限性 + 零向量 cosine=None 语义）。

CPU-only, 不依赖 CUDA / 扩展构建（同 test_evaluator_cpu 约定,
直接 `python tests/test_int4gemv_cpu.py` 运行）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cudalab.int4gemv_quantize import (
    GROUP_SIZE,
    INT4_MAX,
    dequantize_w,
    fidelity_metrics,
    pack_q,
    quantize_w,
    unpack_w,
)

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_exhaustive_byte_bijection() -> None:
    """全部 256 个 byte: unpack 的 (lo,hi) 必须都在 [-8,7], 且
    repack 出同一 byte（双射 + 符号扩展 + 高低 nibble 位置）。"""
    print("[1] exhaustive 256-byte pack/unpack bijection")
    Wp = torch.arange(256, dtype=torch.uint8).view(1, 256)
    Wq = unpack_w(Wp)
    check("unpack shape (1,512)", tuple(Wq.shape) == (1, 512),
          str(tuple(Wq.shape)))
    check("unpack range [-8,7]",
          int(Wq.min()) >= -8 and int(Wq.max()) <= 7,
          f"[{int(Wq.min())}, {int(Wq.max())}]")
    lo = Wq[0, 0::2]
    hi = Wq[0, 1::2]
    # 手工符号扩展参照（纯 Python 算术, 独立实现）
    ok = True
    for b in range(256):
        exp_lo = (b & 0x0F) - 16 if (b & 0x08) else (b & 0x0F)
        exp_hi = ((b >> 4) & 0x0F) - 16 if (b & 0x80) else ((b >> 4) & 0x0F)
        if int(lo[b]) != exp_lo or int(hi[b]) != exp_hi:
            ok = False
            print(f"    byte {b}: got ({int(lo[b])}, {int(hi[b])}) "
                  f"expected ({exp_lo}, {exp_hi})")
    check("per-byte sign extension matches python reference", ok)
    check("repack byte-identical (bijection)",
          torch.equal(pack_q(Wq), Wp))


def test_value_roundtrip_all_positions() -> None:
    """v ∈ [-8..7] 全值域在低/高 nibble 位置逐值往返。"""
    print("[2] full-domain value round-trip (low & high nibble)")
    vals = list(range(-8, 8))
    v = torch.tensor(vals, dtype=torch.int32)
    # 低位: 偶数位置放 v, 奇数位置放 0
    N = len(vals)
    even = torch.zeros(2 * N, dtype=torch.int32)
    even[0::2] = v
    odd = torch.zeros(2 * N, dtype=torch.int32)
    odd[1::2] = v
    back_even = unpack_w(pack_q(even.view(1, -1)))[0]
    back_odd = unpack_w(pack_q(odd.view(1, -1)))[0]
    check("low-nibble round-trip all 16 values",
          torch.equal(back_even[0::2], v),
          str(back_even[0::2].tolist()))
    check("high-nibble round-trip all 16 values",
          torch.equal(back_odd[1::2], v),
          str(back_odd[1::2].tolist()))
    # 边界三连: -7 / 0 / +7
    b = torch.tensor([-7, 0, 7], dtype=torch.int32)
    e = torch.zeros(6, dtype=torch.int32)
    e[0::2] = b
    o = torch.zeros(6, dtype=torch.int32)
    o[1::2] = b
    be = unpack_w(pack_q(e.view(1, -1)))[0]
    bo = unpack_w(pack_q(o.view(1, -1)))[0]
    check("boundary -7/0/+7 low", torch.equal(be[0::2], b))
    check("boundary -7/0/+7 high", torch.equal(bo[1::2], b))


def test_quantizer_contract() -> None:
    """量化合同: scale 公式 / q 公式 / 零 group / 重构误差界。"""
    print("[3] quantization contract")
    torch.manual_seed(0)
    N, G, K = 4, 3, 384
    W = (torch.randn(N, K) * 0.5).to(torch.float16)
    # 手工构造一个零 group（row 2, group 1）
    W.view(N, G, GROUP_SIZE)[:, 1, :].fill_(0)
    Wp, s16 = quantize_w(W)
    check("packed shape (N, K/2)", tuple(Wp.shape) == (N, K // 2))
    check("packed dtype uint8", Wp.dtype == torch.uint8)
    check("scale shape (N, K/G)", tuple(s16.shape) == (N, G))
    check("scale dtype fp16", s16.dtype == torch.float16)
    Wq = unpack_w(Wp)
    check("q range within [-7, 7]",
          int(Wq.abs().max()) <= INT4_MAX,
          f"max|q|={int(Wq.abs().max())}")
    # 零 group: scale=0, q≡0
    check("zero group scale==0",
          torch.equal(s16[2, 1:2], torch.zeros(1, dtype=torch.float16)))
    check("zero group q==0", bool((Wq[2, 128:256] == 0).all()))
    # 非零 group: scale16 == fl16(amax/7)（独立重算）
    Wg = W.float().view(N, G, GROUP_SIZE)
    amax = Wg.abs().amax(dim=2)
    s_ref = (amax / 7).to(torch.float16)
    check("scale == fp16(amax/7) exactly", torch.equal(s16, s_ref))
    # q == clamp(round-half-to-even(W/s32), -7, 7)（独立重算,
    # 用**fp32 计算 scale** —— 合同: 量化在 fp32 域, 存储才降 fp16）
    s32 = amax / 7
    q_ref = torch.round(Wg / s32.unsqueeze(2)).clamp_(-7, 7)
    q_ref = torch.where(s32.unsqueeze(2) == 0,
                        torch.zeros_like(q_ref), q_ref)
    check("q == clamp(round(W/s32), -7, 7) exactly",
          torch.equal(Wq.view(N, G, GROUP_SIZE), q_ref.to(torch.int32)))
    # 重构误差: |W - q*s32| ≤ 0.5·s32 + eps（fp32 计算域界, 用
    # fp32 scale 而非 fp16 存储值 —— 存储量化是独立效应, 见 [5]）
    s32c = torch.where(s32 == 0, torch.ones_like(s32), s32)
    err = (Wg - q_ref.to(torch.float32) * s32c.unsqueeze(2)).abs()
    bound = 0.5 * s32c + 1e-6
    check("|W - q*s32| <= 0.5*s32 (fp32-domain step bound)",
          bool((err <= bound.repeat_interleave(GROUP_SIZE, dim=1)
                .view(N, G, GROUP_SIZE)).all()))
    # 全零行（整行零 → 全部 group scale=0）
    Wz = torch.zeros(2, K, dtype=torch.float16)
    Wpz, sz = quantize_w(Wz)
    check("all-zero row: scale all 0", bool((sz == 0).all()))
    check("all-zero row: packed all 0", bool((Wpz == 0).all()))
    check("all-zero row: no NaN in scale", bool(torch.isfinite(sz).all()))


def test_quantizer_rejections() -> None:
    print("[4] quantizer rejection paths")
    K = 256
    ok2 = lambda fn: _expect_reject(fn)  # noqa: E731
    W = torch.randn(2, K, dtype=torch.float16)
    check("reject 1D", ok2(lambda: quantize_w(W.view(-1))))
    check("reject 3D", ok2(lambda: quantize_w(W.view(2, 2, K // 2))))
    check("reject fp32",
          ok2(lambda: quantize_w(W.float())))
    check("reject non-contiguous",
          ok2(lambda: quantize_w(torch.randn(K, 2, dtype=torch.float16)
                                 .t())))
    check("reject K%128!=0 (K=200)",
          ok2(lambda: quantize_w(
              torch.randn(2, 200, dtype=torch.float16))))
    Wnan = torch.randn(2, K, dtype=torch.float16)
    Wnan[0, 0] = float("nan")
    check("reject NaN", ok2(lambda: quantize_w(Wnan)))
    Winf = torch.randn(2, K, dtype=torch.float16)
    Winf[1, 5] = float("inf")
    check("reject Inf", ok2(lambda: quantize_w(Winf)))
    check("pack_q reject odd K",
          ok2(lambda: pack_q(torch.zeros(1, 3, dtype=torch.int32))))
    check("pack_q reject out-of-range (+8)",
          ok2(lambda: pack_q(torch.tensor([[8]], dtype=torch.int32))))
    check("pack_q reject out-of-range (-9)",
          ok2(lambda: pack_q(torch.tensor([[-9]], dtype=torch.int32))))


def _expect_reject(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    except Exception as e:  # 错误类型不对
        print(f"    意外异常类型: {type(e).__name__}: {e}")
        return False
    return False


def test_fidelity_small() -> None:
    print("[5] fidelity metrics (small, CPU)")
    torch.manual_seed(1)
    N, K = 4, 256
    W = torch.randn(N, K, dtype=torch.float16)
    x = torch.randn(K, dtype=torch.float16)
    Wp, s16 = quantize_w(W)
    fid = fidelity_metrics(W, x, Wp, s16)
    check("metrics finite",
          all(math.isfinite(fid[k]) for k in
              ("max_abs", "rmse", "max_dequant_abs_err",
               "max_dequant_step_err")))
    check("cosine in (0,1]", fid["cosine"] is not None
          and 0.0 < fid["cosine"] <= 1.0 + 1e-9,
          str(fid["cosine"]))
    # INT4 保真度期望: cosine 通常 > 0.98（randn 权重, 信息性,
    # 不是判定门 —— 这里只钉死"不是灾难"下限 0.9）
    check("cosine sanity > 0.9 (informational)", fid["cosine"] > 0.9,
          str(fid["cosine"]))
    check("step err sanity < 0.52", fid["max_dequant_step_err"] < 0.52,
          str(fid["max_dequant_step_err"]))
    # 零向量 x: cosine 记 None（不伪造 1.0）
    xz = torch.zeros(K, dtype=torch.float16)
    fidz = fidelity_metrics(W, xz, Wp, s16)
    check("zero x → cosine None", fidz["cosine"] is None)
    # 全零 W: scale 全 0, y 全 0, cosine None, 无 NaN
    Wz = torch.zeros(N, K, dtype=torch.float16)
    Wpz, sz = quantize_w(Wz)
    fidwz = fidelity_metrics(Wz, x, Wpz, sz)
    check("all-zero W: finite + cosine None",
          math.isfinite(fidwz["max_abs"]) and fidwz["cosine"] is None
          and fidwz["max_abs"] == 0.0)


def test_dequantize_matches_kernel_reference_semantics() -> None:
    """dequantize_w 必须用**存储的 fp16 scale**（合同: 内核读
    fp16 scale → fp32 提升）。独立构造: 先 fp32 scale, 再强制
    fp16 往返, 验证 dequantize 读的是 fp16 值。"""
    print("[6] dequantize uses stored fp16 scale")
    N, G, K = 2, 2, 256
    Wq = torch.randint(-7, 8, (N, K), dtype=torch.int32)
    Wp = pack_q(Wq)
    s32 = torch.rand(N, G, dtype=torch.float32) * 0.1 + 0.01
    s16 = s32.to(torch.float16)
    Wd = dequantize_w(Wp, s16)
    s_exp = s16.float().repeat_interleave(GROUP_SIZE, dim=1)
    check("Wd == Wq * fp16scale→fp32 exactly",
          torch.equal(Wd, Wq.float() * s_exp))
    check("Wd shape (N,K)", tuple(Wd.shape) == (N, K))


def main() -> int:
    print("CUDALab v0.7 int4gemv quantizer CPU tests")
    print("=" * 60)
    test_exhaustive_byte_bijection()
    test_value_roundtrip_all_positions()
    test_quantizer_contract()
    test_quantizer_rejections()
    test_fidelity_small()
    test_dequantize_matches_kernel_reference_semantics()
    print("=" * 60)
    print(f"TOTAL: {PASS + FAIL}  PASS: {PASS}  FAIL: {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
