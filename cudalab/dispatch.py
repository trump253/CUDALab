"""CUDALab v0.2 — 形状/dtype 分发表（Phase 9，纯 CPU，无 GPU 依赖）。

依据 v0.2 复验数据（EXP-0008、benchmarks/v0.2/shape_winners.json）:

设计原则（保守、可审计）:
1. 只为 v0.2 实测过的 28 个 (M,H,dtype) 单元格指定变体；
2. 单元格指定依据 = 该单元格实测中位数 winner 相对 baseline 的加速，
   且 winner 在 hot 与 streaming 两种缓存模式下一致（或仅该模式有数据
   且 paired 验证显著）；winner-vs-runner 未达显著且对 baseline 优势
   <2% 的单元格诚实回退 baseline；
3. 未实测 (M,H) 组合走显式 fallback 策略（只外推 (128,8192) 的 v2 证据，
   其余一律回退 baseline，不做无证据外推）；
4. 任何 H 不被所选变体支持时回退 baseline（baseline 支持任意 H）。

关键证据（详见 EXP-0008 / benchmarks/v0.2/）:
- (128,4096) fp16: v4 为 incumbent（与 v2 统计平局 NEUTRAL；1.56× vs baseline）
- (128,4096) fp32: v2 KEEP（paired 1.37×/1.62×，CI 不跨 1.0）
- (128,8192) fp16/fp32: v2 KEEP（paired 1.38×；v4 在 H=8192 退化）
- (1,4096) fp16: v3（launch-bound 区域，hot 1.11× vs runner CI 不跨 1.0）
- (1024,4096) 两 dtype: v4（1.29–1.30× vs baseline，两模式一致）
"""
from __future__ import annotations

import torch

# 变体对 H 的支持条件（与 kernels 中 TORCH_CHECK 一致）
_V4_HS = {1024, 2048, 4096, 8192}      # H/256 ∈ {4,8,16,32}
_V2_HS = {512, 1024, 2048, 4096, 8192}  # H/256 ∈ {2,4,8,16,32}
_V3_H_MOD = 512
_V1_H_MOD = (8, 4)  # (fp16, fp32)


def _supported(variant: str, H: int, dtype: torch.dtype) -> bool:
    if variant == "baseline":
        return True
    if variant == "v4_vec_reg":
        return H in _V4_HS
    if variant == "v2_reg":
        return H in _V2_HS
    if variant == "v3_wideblock":
        return H % _V3_H_MOD == 0
    if variant == "v1_vec":
        return H % (_V1_H_MOD[0] if dtype == torch.float16 else _V1_H_MOD[1]) == 0
    return False


# v0.2 实测 28 单元格分发表（M, H, dtype名）-> variant
# 依据与回退理由见模块头；未实测组合不在表内。
TABLE = {
    # ---- fp16 ----
    (1, 1024, "float16"): ("baseline", "无显著 winner（v3/v1 均 <2% 优势）"),
    (128, 1024, "float16"): ("baseline", "对 baseline 优势 5.8%，未 paired 验证显著"),
    (1, 4096, "float16"): ("v3_wideblock", "1.51× vs baseline；hot 1.11× vs runner CI 不跨 1.0"),
    (16, 4096, "float16"): ("v4_vec_reg", "1.52× vs baseline；两模式一致"),
    (128, 4096, "float16"): ("v4_vec_reg", "1.56× vs baseline；v2 平局 NEUTRAL 保留 incumbent"),
    (1024, 4096, "float16"): ("v4_vec_reg", "1.30× vs baseline；两模式显著（CI 不跨 1.0）"),
    (128, 8192, "float16"): ("v2_reg", "1.65× vs baseline；paired 1.38× KEEP"),
    # ---- fp32 ----
    (1, 1024, "float32"): ("baseline", "无显著 winner"),
    (128, 1024, "float32"): ("baseline", "优势 ~1%，无显著性"),
    (1, 4096, "float32"): ("baseline", "baseline 实测最快（6.097 vs 6.275µs）"),
    (16, 4096, "float32"): ("baseline", "无显著 winner（baseline 实测最快）"),
    (128, 4096, "float32"): ("v2_reg", "paired 1.37×/1.62× KEEP（streaming/hot）"),
    (1024, 4096, "float32"): ("v4_vec_reg", "1.28× vs baseline；两模式显著（CI 不跨 1.0）"),
    (128, 8192, "float32"): ("v2_reg", "1.48× vs baseline；两模式显著（CI 不跨 1.0）"),
}


def select_variant(M: int, H: int,
                   dtype: torch.dtype = torch.float16) -> str:
    """按 (M,H,dtype) 选择变体。未实测组合走保守 fallback。"""
    dname = str(dtype).replace("torch.", "")
    if (M, H, dname) in TABLE:
        v = TABLE[(M, H, dname)][0]
        if _supported(v, H, dtype):
            return v
        return "baseline"
    # ---- 未实测组合的 fallback（保守，不外推无证据的单元格）----
    if dtype == torch.float16 and M >= 16:
        if H in _V4_HS and H < 8192:
            return "v4_vec_reg"   # incumbent，实测中从未显著更差
        if H == 8192:
            return "v2_reg"       # 外推 (128,8192) 证据（v4 在 H=8192 退化）
        if H % _V3_H_MOD == 0 and M >= 128:
            return "v2_reg" if H in _V2_HS else "baseline"
    if dtype == torch.float32 and M >= 128:
        if H in (4096, 8192):
            return "v2_reg"       # fp32 两 dtype 显著 winner
    return "baseline"


def dispatch_info(M: int, H: int,
                  dtype: torch.dtype = torch.float16) -> dict:
    """select_variant + 可审计理由（实测/外推/回退）。"""
    dname = str(dtype).replace("torch.", "")
    v = select_variant(M, H, dtype)
    key = (M, H, dname)
    if key in TABLE:
        source, why = "measured-v0.2", TABLE[key][1]
    elif v != "baseline":
        source, why = "fallback", "未实测组合的保守 fallback（见 dispatch.py 模块头）"
    else:
        source, why = "fallback", "未实测且无显著证据 -> baseline"
    return {"M": M, "H": H, "dtype": dname, "variant": v,
            "evidence_source": source, "reason": why}
