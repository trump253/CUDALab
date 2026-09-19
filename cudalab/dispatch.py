"""CUDALab v0.2.1 — 形状/dtype 分发表（Phase 9 修订，纯 CPU，无 GPU 依赖）。

v0.2.1 证据政策（review Finding 4）：**evidence > coverage**。
旧 v0.2 分发表有两类越界：
  (a) 把 v4（fp16 incumbent）/ v2（fp32）外推到未实测的 (M,H) 组合；
  (b) 在矩阵两模式 winner 冲突的单元格硬编码变体
      （如 (16,4096) fp16：hot winner=v4 / streaming winner=v1）。
修订政策：

1. **只有清晰的 v0.2 paired 证据才路由优化变体**
   （paired A/B 9 rounds；KEEP = median 加速比 ≥1.05 且 faster rounds ≥70%
   且 CI95 不跨 1.0）：
   - (128,4096) fp32 → v2_reg  （paired 1.3666× streaming / 1.6177× hot，均 KEEP）
   - (128,8192) fp16 → v2_reg  （paired 1.3803× streaming KEEP；v4 在 H=8192 退化）
2. **主形状 (128,4096) fp16**：无统计唯一胜出者
   （NO_UNIQUE_WINNER：streaming v1/v2/v4 两两 NEUTRAL；hot v4 vs v1 REJECT v1）
   → 路由显式 v0.1 incumbent v4_vec_reg（incumbent-fallback）。
   这是证据一致的选择，但 v4 并非经统计确认的唯一最佳。
3. **matrix-only / hot-streaming 冲突 / 未实测 → baseline**：
   - 矩阵（模式级中位数）winner 未经 paired 验证不构成路由证据
     （(1024,4096) v4 1.28–1.30×、(1,4096) fp16 v3 1.51× 等）；
   - hot/streaming 冲突的单元格不允许声称任何变体稳定
     （(16,4096) fp16：hot winner=v4 / streaming winner=v1 → baseline）；
   - 未实测 (M,H) 组合一律 baseline，不做外推。
4. 任何 H 不被所选变体支持时回退 baseline（baseline 支持任意 H）。

evidence_source 分类（dispatch_info）:
- paired-evidence:    paired KEEP → 优化变体
- incumbent-fallback: 主形状 NO_UNIQUE_WINNER → 显式 incumbent v4
- matrix-only:        矩阵实测单元格、无 paired 证据 → baseline
- baseline-fallback:  未实测单元格 → baseline
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


# v0.2 实测 14 (M,H,dtype) 单元格 -> (variant, evidence_source, reason)
# v0.2.1：仅 3 格路由优化变体（2 格 paired-evidence + 1 格 incumbent-fallback），
# 其余全部 baseline —— evidence > coverage。
TABLE = {
    # ---- fp16 ----
    (1, 1024, "float16"): ("baseline", "matrix-only",
        "无显著 winner（v3/v1 均 <2% 优势）"),
    (128, 1024, "float16"): ("baseline", "matrix-only",
        "对 baseline 优势 5.8%，未 paired 验证显著 → baseline"),
    (1, 4096, "float16"): ("baseline", "matrix-only",
        "矩阵 winner v3（1.51× vs baseline；hot 1.11× vs runner CI 不跨 1.0）"
        "但未 paired 验证；launch-bound 区域 → baseline"),
    (16, 4096, "float16"): ("baseline", "matrix-only",
        "hot/streaming 两模式 winner 冲突（hot winner=v4 / streaming winner=v1），"
        "无 paired 验证 → baseline；本单元格不声称 v4 稳定"),
    (128, 4096, "float16"): ("v4_vec_reg", "incumbent-fallback",
        "NO_UNIQUE_WINNER：streaming v1/v2/v4 两两 NEUTRAL；hot v4 vs v1 REJECT v1"
        "（median v4/v1 0.9327）、v4 vs v2 NEUTRAL → 保留 v0.1 incumbent v4"
        "（非统计确认唯一最佳，v1/v2 仍为竞争性变体）"),
    (1024, 4096, "float16"): ("baseline", "matrix-only",
        "矩阵 v4 1.30×（两模式 CI 不跨 1.0）但未 paired 验证 → baseline"),
    (128, 8192, "float16"): ("v2_reg", "paired-evidence",
        "paired 1.3803×（streaming，9/9 轮，CI95 [1.3489, 1.3995] 不跨 1.0）→ KEEP；"
        "v4 在 H=8192 退化（per=32 寄存器压力）"),
    # ---- fp32 ----
    (1, 1024, "float32"): ("baseline", "matrix-only",
        "无显著 winner"),
    (128, 1024, "float32"): ("baseline", "matrix-only",
        "优势 ~1%，无显著性"),
    (1, 4096, "float32"): ("baseline", "matrix-only",
        "baseline 实测最快（6.097 vs 6.275µs）"),
    (16, 4096, "float32"): ("baseline", "matrix-only",
        "无显著 winner（baseline 实测最快）"),
    (128, 4096, "float32"): ("v2_reg", "paired-evidence",
        "paired 1.3666×（streaming）/ 1.6177×（hot），9/9 轮，CI95 均不跨 1.0 → KEEP"),
    (1024, 4096, "float32"): ("baseline", "matrix-only",
        "矩阵 v4 1.28×（两模式 CI 不跨 1.0）但未 paired 验证 → baseline"),
    (128, 8192, "float32"): ("baseline", "matrix-only",
        "矩阵 winner v2（1.48×，两模式 CI 不跨 1.0）但未 paired 验证 → baseline"),
}


def select_variant(M: int, H: int,
                   dtype: torch.dtype = torch.float16) -> str:
    """按 (M,H,dtype) 选择变体。

    v0.2.1 政策：仅 paired 证据格与显式 incumbent 格路由优化变体；
    矩阵-only / 模式冲突 / 未实测一律 baseline（不做外推）。
    """
    dname = str(dtype).replace("torch.", "")
    entry = TABLE.get((M, H, dname))
    if entry is not None:
        v = entry[0]
    else:
        v = "baseline"  # 未实测：不外推
    if v != "baseline" and not _supported(v, H, dtype):
        return "baseline"  # 安全网：所选变体不支持该 H
    return v


def dispatch_info(M: int, H: int,
                  dtype: torch.dtype = torch.float16) -> dict:
    """select_variant + 可审计理由（evidence_source 四分类）。

    evidence_source ∈ {paired-evidence, incumbent-fallback,
                       matrix-only, baseline-fallback}
    """
    dname = str(dtype).replace("torch.", "")
    v = select_variant(M, H, dtype)
    entry = TABLE.get((M, H, dname))
    if entry is not None:
        tbl_v, source, why = entry
        if v == "baseline" and tbl_v != "baseline":
            source, why = ("baseline-fallback",
                           f"表格所选变体不支持 H={H}，安全网回退 baseline")
    else:
        source, why = "baseline-fallback", \
            "未实测 (M,H,dtype)：v0.2.1 政策不做无证据外推 → baseline"
    return {"M": M, "H": H, "dtype": dname, "variant": v,
            "evidence_source": source, "reason": why}
