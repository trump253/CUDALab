"""CUDALab v0.4 — RoPE 正确性校验套件（interleaved pair 约定）。

原则（与 RMSNorm / Softmax 套件一致）:
- 容差/误差界对所有变体固定，并记录在每一份结果中。绝不为某个候选
  单独放宽使其通过。
- 正确性 FAIL 的内核永远没有资格成为性能胜者（experiment.py 强制）。
- 所有结果（通过和失败）都报告并保存；失败用例绝不删除。

套件矩阵（用户指定）:
- dtype:   float16 / float32
- D:       64 / 128
- M:       1 / 32 / 128 / 1024
- positions: 0 / 1 / max_seq_len-1（全常量）/ sequential / random /
             repeated（高重复, 固定 seed）
- input:   normal / zeros / tiny(N(0,1e-4)) / large(N(0,1000))
共 2 × 2 × 4 × 6 × 4 = 384 项检查, seed=0（确定性）。

判定合同（v0.4 首发即修正, 两个固定不变量, 对所有变体一致）:

(1) 双舍入误差界（arith bound, 对**精确 float64 旋转**）
    设 a, b, c, s 为 x 与 cos/sin 表**存储值的精确提升**（fp16/fp32 →
    fp64 无损）, exact = (a*c − b*s, a*s + b*c) 在实数中计算。任何
    "FP32 中间 + 最终 cast 到 dtype" 的实现（无论 nvcc 是否 FMA 收缩、
    无论 torch 参考的逐元素双舍入路径）都满足:
        |impl − exact| ≤ 2^-23·(|a*c| + |b*s|) + 0.5·ulp_dtype(|impl|)
    推导: FP32 乘/加各引入 ≤ 0.5·ulp32 ≤ 2^-23·|值| 的绝对误差
    （双舍入路径: 两次乘 + 一次减, 最坏 2^-23·(|a*c|+|b*s|+|y|);
    FMA 路径: |a*c − b*s| 一次精确舍入, 更紧）; 最终 cast 引入
    ≤ 0.5·ulp_dtype。
    套件对每元素取 K=2 安全余量并覆盖 exact 与 impl 两个可能的
    binade:
        tol = 2·(2^-23·(|a*c|+|b*s|) + 0.5·ulp_dtype(|y|)
                + 0.5·ulp_dtype(|exact|))
        arith_ratio = max_elem |y − exact| / tol ≤ 1
    该界在小幅值输入下**严于**经典 allclose 合同（scale 1 时
    tol ~ 1e-3(fp16) / 1e-6(fp32), 远小于 atol+rtol 的可用部分）,
    在 large（scale 1000）下捕获 gross 错误（错误缩放、错用 cos²、
    张错 pair 的偏差 ~1e-1 相对 ≫ 界）, 而**不**把两个同样合法的
    FP32 舍入路径（FMA vs 双舍入）在深度抵消元素上的 ~1 ulp 差异
    误判为错误 —— 这是 v0.4 首跑发现的合同缺陷（fp32+large 的
    7 项假失败, 最大差恰为 1 ulp fp32 @ |y|≈3400）, 修正记录在
    本文件与实验记录中。
    y 与 torch 参考 `rope_ref` 的 elementwise 差（max_abs/max_rel/
    allclose, 共享 TOLERANCES: fp16 atol=2e-3/rtol=5e-3, fp32
    atol=1e-5/rtol=1e-4）照常**报告**, 但不再作为判定门（在
    无界抵消深度下它对合法 FP32 实现不是正确合同）。

(2) 范数保持（norm preservation）
    RoPE 是旋转, 每个 pair 的模长必须保持: a² + b² ≈ y[2i]² +
    y[2i+1]², 在 FP32 中计算两侧, 指标为
        max_pair |n_in − n_out| / max(n_in, n_out, floor)
    固定阈值 NRM_REL_TOL: float16 5e-3 / float32 1e-5。
    依据: 输出逐元素 |e_i| ≤ 0.5·ulp_dtype(|y_i|)（fp16 ≈ 2.4e-4·
    |y_i|; fp32 的 FP32 路径误差 ≤ 2^-23·(|a*c|+|b*s|), 与 |y_i|
    无关但在 scale 1000 下仍使相对范数误差 ≤ 2^-23·~11.5 ≈ 1.4e-6
    < 1e-5）, 旋转精确保模 → 相对范数误差上界 ~1e-3(fp16) /
    1.4e-6(fp32), 阈值各留 ~3.5x 余量, 同时足以捕获 gross 错误。
    floor（分母下限）:
      - float16: 4e-9 —— fp16 次正规边界 ≈ 6.1e-5（平方 ≈ 3.7e-9）。
        "tiny"（scale 1e-4）部分输出落入次正规区, 绝对量化步长 6e-8
        主导, 相对误差不是旋转数学的属性; 分母钳位后这类 pair 贡献
        有界: 被钳位 pair 满足 |y_i| ≤ 6.1e-5, |n_in−n_out| ≤
        2·(2·|y|·3e-8) + 2·(3e-8)² ≤ 7.3e-12, 比值 ≤ 1.8e-3 < 5e-3。
      - float32: 1e-30 —— 仅为避免 zeros 输入 0/0 = NaN。
    ref 的范数误差一并报告（ref 由本项目构造, 旁路信息, 不判定）。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch

from .evaluator.correctness import (
    TOLERANCES,
    REL_EPS_GUARD,
    compute_metrics,
    summarize_results,
    save_suite,
)
from .operators.rope import (
    rope_ref,
    make_input,
    make_positions,
    make_rotary_table,
    MAX_SEQ_LEN,
    ROPE_BASE,
)

ROOT = Path(__file__).resolve().parent.parent

DTYPES = ("float16", "float32")
DS = (64, 128)
MS = (1, 32, 128, 1024)
POSITION_PATTERNS = ("pos0", "pos1", "pos_max_seq_len-1",
                     "sequential", "random", "repeated")
INPUT_MODES = ("normal", "zeros", "tiny", "large")
SEED = 0

# 范数保持的固定判定阈值（对所有变体一致, 不随候选放宽）。
NRM_REL_TOL = {"float16": 5e-3, "float32": 1e-5}
# 范数保持分母下限（见模块 docstring 的依据说明）。
NRM_DENOM_FLOOR = {"float16": 4e-9, "float32": 1e-30}
# 双舍入误差界的安全余量 K（见模块 docstring 推导）。
ARITH_K = 2.0
_NEG23 = 2.0 ** -23


@dataclass
class CheckResult:
    variant: str
    shape: list            # [M, D]
    dtype: str
    seed: int
    pattern: str           # positions 模式
    mode: str              # 输入模式
    passed: bool
    # 判定门: (1) 双舍入误差界 (2) 范数保持 (3) 有限性
    arith_max_ratio: float       # max_elem |y − exact64| / tol ≤ 1
    norm_rel_error: float        # 内核输出的范数保持相对误差
    has_nan: bool
    has_inf: bool
    # 报告（不判定）: y vs torch 参考 rope_ref 的 elementwise 差
    max_abs_error: float
    max_rel_error: float
    ok_close: bool
    # 报告（不判定）
    arith_tol_max: float         # tol 的最大值（审计用）
    ref_norm_rel_error: float    # 参考实现的范数保持相对误差
    atol: float
    rtol: float
    nrm_rel_tol: float
    note: str = ""


def _ref64_parts(x: torch.Tensor, positions: torch.Tensor,
                 cos_t: torch.Tensor, sin_t: torch.Tensor):
    """a, b, c, s 的精确 fp64 提升（输入 = 存储值本身）。"""
    a = x.double()[:, 0::2]
    b = x.double()[:, 1::2]
    c = cos_t.double()[positions]
    s = sin_t.double()[positions]
    return a, b, c, s


def rope_ref64(x: torch.Tensor, positions: torch.Tensor,
               cos_t: torch.Tensor, sin_t: torch.Tensor) -> torch.Tensor:
    """精确 float64 旋转（exact 参考; 输入/表的存储值无损提升）。"""
    a, b, c, s = _ref64_parts(x, positions, cos_t, sin_t)
    out_even = a * c - b * s
    out_odd = a * s + b * c
    return torch.stack((out_even, out_odd), dim=-1).reshape(x.shape)


def _ulp(v: torch.Tensor, dtype_name: str) -> torch.Tensor:
    """逐元素 ulp（**目标 dtype 网格**的间距; ulp(0) = 最小次正规）。

    fp32: nextafter 差（CUDA 支持）。
    fp16: CUDA 的 nextafter 不支持 Half, 用 binade 公式
    2^(floor(log2 v) − 10)（次正规/0: 2^-24）。本函数输入恒在 fp16
    网格上（y/exact 先 .half() 取格点）, 网格间距 2^(k−10) 与 fp32
    log2 误差（~1e-7 相对）相差 > 2^11 倍, floor(log2) 不会越 binade
    边界 —— 公式精确, 不引入界偏差。
    """
    if dtype_name == "float32":
        vf = v.float()
        posinf = torch.full_like(vf, float("inf"))
        return torch.nextafter(vf, posinf) - vf
    v32 = v.half().float().abs()
    tiny16 = torch.finfo(torch.float16).tiny  # 2^-14
    e = torch.floor(torch.log2(v32.clamp_min(tiny16)))
    ulp = torch.pow(2.0, e - 10)
    return torch.where(v32 >= tiny16, ulp,
                       torch.full_like(v32, 2.0 ** -24))


def arith_ratio(y: torch.Tensor, x: torch.Tensor, positions: torch.Tensor,
                cos_t: torch.Tensor, sin_t: torch.Tensor,
                dtype_name: str) -> tuple[float, float]:
    """固定双舍入误差界（见模块 docstring (1)）。

    返回 (max_ratio, tol_max); 非有限元素置 0（由 has_nan/has_inf 门
    独立判定, 保证 JSON 记录为有限数）。
    """
    if x.numel() == 0:
        return 0.0, 0.0
    a, b, c, s = _ref64_parts(x, positions, cos_t, sin_t)
    ac = a * c
    bs = b * s
    exact = torch.stack((ac - bs, a * s + b * c), dim=-1).reshape(x.shape)
    # FP32 路径误差按 pair 计, 展开到逐元素（每个 pair 的两个输出共享）
    pair_base = _NEG23 * (ac.abs() + bs.abs())
    tol = ARITH_K * (
        pair_base.repeat_interleave(2, dim=1)
        + 0.5 * _ulp(y.abs(), dtype_name).double()
        + 0.5 * _ulp(exact.abs(), dtype_name).double()
    )
    diff = (y.double() - exact).abs()
    ratio = torch.where(diff.isfinite() & tol.isfinite(),
                        diff / tol.double(),
                        torch.zeros_like(diff))
    return float(ratio.max().item()), float(tol.float().max().item())


def _norm_rel(y: torch.Tensor, x: torch.Tensor,
              floor: float) -> float:
    """max_pair |‖(a,b)‖² − ‖(y0,y1)‖²| / max(‖in‖², ‖out‖², floor)（FP32）。"""
    if x.numel() == 0:
        return 0.0
    xf = x.float()
    yf = y.float()
    n_in = xf[:, 0::2].pow(2) + xf[:, 1::2].pow(2)
    n_out = yf[:, 0::2].pow(2) + yf[:, 1::2].pow(2)
    denom = torch.maximum(n_in, n_out).clamp_min(floor)
    return float(((n_in - n_out).abs() / denom).max().item())


def _build_positions(M: int, pattern: str) -> torch.Tensor:
    """构建 6 种确定性位置模式（见模块 docstring; 恒为 int64）。"""
    dev = "cuda"
    dt = torch.int64
    if pattern == "pos0":
        return torch.full((M,), 0, dtype=dt, device=dev)
    if pattern == "pos1":
        return torch.full((M,), 1, dtype=dt, device=dev)
    if pattern == "pos_max_seq_len-1":
        return torch.full((M,), MAX_SEQ_LEN - 1, dtype=dt, device=dev)
    if pattern in ("sequential", "random", "repeated"):
        return make_positions(M, MAX_SEQ_LEN, dtype=dt, device=dev,
                              seed=SEED, pattern=pattern)
    raise ValueError(f"未知位置模式: {pattern}")


def check_one(variant: str, ext, x: torch.Tensor,
              positions: torch.Tensor, cos_t: torch.Tensor,
              sin_t: torch.Tensor, pattern: str, mode: str,
              seed: int = SEED, note: str = "") -> CheckResult:
    y = ext.forward(variant, x, positions, cos_t, sin_t)
    y = y.contiguous()
    ref = rope_ref(x, positions, cos_t, sin_t)
    dtype_name = str(x.dtype).split(".")[-1]
    floor = NRM_DENOM_FLOOR[dtype_name]

    ratio, tol_max = arith_ratio(y, x, positions, cos_t, sin_t, dtype_name)
    nrm = _norm_rel(y, x, floor)
    nrm_ref = _norm_rel(ref, x, floor)
    nrm_ok = nrm <= NRM_REL_TOL[dtype_name]
    arith_ok = ratio <= 1.0

    m = compute_metrics(y, ref, dtype_name)
    M, D = x.shape
    return CheckResult(
        variant=variant,
        shape=[int(M), int(D)],
        dtype=dtype_name,
        seed=seed, pattern=pattern, mode=mode,
        passed=(arith_ok and nrm_ok and not m["has_nan"]
                and not m["has_inf"]),
        arith_max_ratio=round(ratio, 8),
        norm_rel_error=round(nrm, 8),
        has_nan=m["has_nan"],
        has_inf=m["has_inf"],
        max_abs_error=m["max_abs_error"],
        max_rel_error=m["max_rel_error"],
        ok_close=m["ok_close"],
        arith_tol_max=round(tol_max, 8),
        ref_norm_rel_error=round(nrm_ref, 8),
        atol=m["atol"], rtol=m["rtol"],
        nrm_rel_tol=NRM_REL_TOL[dtype_name],
        note=note,
    )


def run_suite(variant: str, ext,
              dtypes: tuple = DTYPES, ds: tuple = DS, ms: tuple = MS,
              patterns: tuple = POSITION_PATTERNS,
              modes: tuple = INPUT_MODES) -> list[CheckResult]:
    """单个变体的完整 RoPE 正确性套件（384 项检查）。"""
    results: list[CheckResult] = []
    for dtype_name in dtypes:
        dtype = getattr(torch, dtype_name)
        for D in ds:
            # 每个 (dtype, D) 一张确定性 cos/sin 表（计时/测试共享构造）
            cos_t, sin_t = make_rotary_table(MAX_SEQ_LEN, D, dtype)
            for M in ms:
                for pattern in patterns:
                    pos = _build_positions(M, pattern)
                    for mode in modes:
                        x = make_input(M, D, dtype=dtype, seed=SEED,
                                       mode=mode)
                        results.append(check_one(variant, ext, x, pos,
                                                 cos_t, sin_t, pattern,
                                                 mode, seed=SEED))
    return results


def _jsonable(r: CheckResult) -> dict:
    d = asdict(r)
    d["pass"] = d.pop("passed")
    return d


def summarize(results: list[CheckResult]) -> dict:
    s = summarize_results([_jsonable(r) for r in results])
    s["max_arith_max_ratio"] = max((r.arith_max_ratio for r in results),
                                   default=0.0)
    s["max_norm_rel_error"] = max((r.norm_rel_error for r in results),
                                  default=0.0)
    s["max_ref_norm_rel_error"] = max((r.ref_norm_rel_error for r in results),
                                      default=0.0)
    s["ok_close_vs_ref_all"] = all(r.ok_close for r in results)
    return s


def save_results(results: list[CheckResult], out_path: Path) -> Path:
    return save_suite(out_path, {
        "operator": "rope",
        "suite": "rope-correctness-v0.4",
        "convention": ("interleaved RoPE: a=x[2i], b=x[2i+1], "
                       "c=cos[pos,i], s=sin[pos,i]; y[2i]=a*c-b*s, "
                       "y[2i+1]=a*s+b*c（FP32 中间, 输出 dtype = x dtype）; "
                       "不使用 NeoX half-split"),
        "gate": ("pass = 有限性 AND arith_max_ratio <= 1（固定双舍入误差界, "
                 "对精确 float64 旋转, K=2 余量）AND norm_rel_error <= "
                 "NRM_REL_TOL; y vs rope_ref 的 elementwise allclose 只报告"
                 "不判定（无界抵消深度下对合法 FP32 实现不是正确合同, "
                 "见模块 docstring (1) 的 v0.4 首跑修正记录）"),
        "reference": ("rope_ref: x.float() 提升, 与内核同一张 cos/sin 表"
                      "（FP32 提升后运算）, 旋转后 cast 回 x dtype; "
                      "exact 参考 = rope_ref64（fp64 精确旋转）"),
        "cos_sin_table": f"L={MAX_SEQ_LEN}, base={ROPE_BASE}, "
                         "FP32 计算后 cast 到 dtype",
        "matrix": {
            "dtypes": list(DTYPES),
            "D": list(DS),
            "M": list(MS),
            "position_patterns": list(POSITION_PATTERNS),
            "input_modes": list(INPUT_MODES),
            "seed": SEED,
        },
        "tolerances_reported": TOLERANCES,
        "arith_k": ARITH_K,
        "nrm_rel_tol": NRM_REL_TOL,
        "nrm_denom_floor": NRM_DENOM_FLOOR,
        "rel_eps_guard": REL_EPS_GUARD,
        "summary": summarize(results),
        "results": [_jsonable(r) for r in results],
    })
