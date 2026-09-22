"""CUDALab v0.7 — INT4GEMV 三层正确性校验套件（用户 §2, 容差不混用）。

**第 (a) 层 — kernel 正确性（判定门, per-variant）**:
    y vs ref = unpack(W_packed).float() · scale_fp16_expanded @
    x.float() cast FP16
    （用户指定的反量化参考; W_packed / scale 是 kernel 的**输入**,
    unpack 用 CPU 镜像实现, 量化误差不在这一层 —— 它是第 (c) 层的事;
    **fp16 scale 存储量化是数据合同**, 参考与内核读同一个 fp16 值）。
    判定 = 固定累加误差界（arith_max_ratio ≤ 1）AND y 有限 AND
    ref 有限。所有变体同一界, 绝不放宽。

**第 (b) 层 — pack/unpack correctness（CPU 独立层, variant 无关）**:
    由 tests/test_int4gemv_cpu.py 覆盖（256-byte 全双射 / 全值域
    -8..7 高低 nibble 往返 / 负数符号扩展 / 边界 -7/0/+7 / 量化
    合同公式 / 零 group 安全 / 拒绝路径）。本套件不重复运行, 但
    第 (a) 层的 GPU 参考与 kernel 的逐元素一致性同时钉死了 CPU
    镜像 unpack 与设备端 int4gemv_unpack_byte 的等价性。

**第 (c) 层 — 量化保真度（只报告, 不判定, variant 无关）**:
    INT4 量化后的 ref vs 原 FP16 W @ x:
    max_abs / RMSE / cosine similarity。
    量化误差不是 kernel bug —— 不用 kernel 容差衡量（用户 §2
    "不要把量化误差当成 kernel bug"）。每 case 记录参考数值 +
    独立 fidelity 套件（5 基准形状 × 5 模式）。

原则（与 RMSNorm / Softmax / RoPE / GEMV / QGEMV 套件一致）:
- 容差/误差界对所有变体固定，并记录在每一份结果中。
- 正确性 FAIL 的内核永远没有资格成为性能胜者（决策层强制）。
- 所有结果（通过和失败）都报告并保存；失败用例绝不删除。

算子合同:
    W_dequant[n,k] = scale[n,k/128] · unpack(W_packed)[n,k]
    y[n] = Σ_k W_dequant[n,k] · x[k]   （FP32 累加, cast fp16）
    参考 (a): unpack(W_packed).float() * scale_fp16.float()
              .repeat_interleave(128, dim=1) @ x.float() → fp16
              exact = fp64 反量化 GEMV（误差界参照, 用**存储的
              fp16 scale** 的精确 fp64 提升）

第 (a) 层判定合同（v0.7 首发, 固定不变量, 对所有变体一致）:

(1) 累加误差界（arith bound, 对**精确 fp64 反量化 GEMV**）
    设 a_k = q[n,k]·s16[n,k/128]·x[k]（存储值的精确 fp64 提升
    乘积）, exact = Σ_k a_k 在 fp64 中计算。INT4GEMV 的每 term
    计算路径（baseline 逐字: q→fp32, ×scale(fp16→fp32), ×x,
    累加）最多引入 **3 次** FP32 舍入: fl(q·s) 乘 1 次 + fl(·x)
    乘 1 次 + 加法 1 次（FMA 收缩更少）; 其他候选变体（向量 /
    warp-per-row / scale 提升 / group 驻留）舍入次数均 ≤ 每 term
    3 次。因此对**任意合法 FP32 累加顺序**:
        |impl − exact| ≤ 3·K·2^-24·Σ_k |a_k| + 0.5·ulp_dtype(|·|)
    （标准向后误差归纳: 每 term 3 个相对扰动 2^-24 + 最终 cast
    的 0.5·ulp_dtype; 对 impl 与 exact 两个可能 binade 各计一个
    ulp 项）。套件取 TOL_K=2 安全余量:
        tol = 2·(3K·2^-24·S_n + 0.5·ulp16(|y|) + 0.5·ulp16(|exact|))
        arith_ratio = max_elem |y − exact| / tol ≤ 1
    其中 S_n = Σ_k |a_k|（fp64, 逐行）。
    该界在 normal（K=4096）下的量级与 v0.6 QGEMV 相同（|y| 量级
    由 scale·|q|·|x| 决定, 与代际量化位宽无关）, 足以捕获 gross
    错误（漏项、张错行、错误缩放、nibble 高低位错、符号扩展错、
    group 偏移、转置、归约缺失 → O(1) 相对偏差 ≫ 界）, 而**不**
    把两个合法 FP32 累加顺序在深度抵消元素上的 ~γ_K·S 差异误判
    为错误。
    边界自检:
    - zeros: S=0, y=exact=0 → tol = 2·(0.5·ulp16(0)+0.5·ulp16(0))
      = 2^-23 ≈ 1.2e-7, 任何正确实现 diff=0 精确通过（零 group
      的 q≡0 合同也在此钉死）;
    - tiny: |y|~1e-8 落 fp16 次正规区, 界中的 0.5·ulp_dtype 项
      覆盖该量化, 不误判（同 v0.5/v0.6 论证）。

(2) 有限性门
    y 无 NaN / Inf; 参考值同样必须有限（若违反说明套件构造错误,
    该用例判 FAIL 并记录 note —— 绝不把"参考溢出"静默当通过）。
    （量化器本身要求有限 W, 见 int4gemv_quantize.py; large 的
    scale 选择保证 fp16 W 与 fp16 y 有限。）

y 与 ref 的 elementwise 差（max_abs/max_rel/allclose, 共享
TOLERANCES: fp16 atol=2e-3/rtol=5e-3）照常**报告**, 但**不作为
判定门**（同 v0.5 GEMV / v0.6 QGEMV 合同修正: 深度抵消行下对
合法 FP32 实现不是正确合同）。

形状约束: 合同要求 K % 128 == 0, 因此本套件的形状矩阵全部满足
（基准 5 形状 + 边界形状; 无 K=1 之类的小 K —— K 最小 128, 由
group_size 合同决定）。
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
from .operators.int4gemv import (
    BENCH_MATRIX_INT4GEMV,
    make_w4,
    make_x4,
)
from .int4gemv_quantize import (
    GROUP_SIZE,
    dequantize_w,
    fidelity_metrics,
    unpack_w,
)

ROOT = Path(__file__).resolve().parent.parent

DTYPES = ("float16",)
# 基准 5 形状（与 v0.5/v0.6 相同, K 均 %128==0）+ 5 边界形状
# （小 N / 最小 K=128 / 非 2 的幂 N / 小 K=256; K%128 合同约束下
# 的最小合法 K 就是 128）
EDGE_SHAPES = [(1, 128), (1, 4096), (3, 128), (16, 256), (8, 128)]
SHAPES = list(BENCH_MATRIX_INT4GEMV) + EDGE_SHAPES
INPUT_MODES = ("normal", "zeros", "tiny", "large", "mixed_sign")
LARGE_SCALE = 10.0
SEED = 0

# 固定判定参数（对所有变体一致, 不随候选放宽）。
TOL_K = 2.0       # 安全余量（覆盖 impl/exact 两个 binade + 无 FMA 最坏情形）
_PER_TERM_ROUNDINGS = 3.0  # 每 term 最多 3 次 FP32 舍入（见 docstring (1)）
_NEG24 = 2.0 ** -24


def _ulp(v: torch.Tensor, dtype_name: str) -> torch.Tensor:
    """逐元素 ulp（**目标 dtype 网格**的间距; ulp(0) = 最小次正规）。
    同 gemv_correctness._ulp / qgemv_correctness._ulp。"""
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


@dataclass
class CheckResult:
    variant: str
    shape: list            # [N, K]
    dtype: str
    seed: int
    mode: str              # 输入模式
    passed: bool
    # 第 (a) 层判定门: (1) 累加误差界 (2) 有限性（y 与 ref）
    arith_max_ratio: float       # max_elem |y − exact64| / tol ≤ 1
    has_nan: bool                # 内核输出
    has_inf: bool                # 内核输出
    ref_has_nan: bool            # 参考值（违反 = 套件构造错误）
    ref_has_inf: bool            # 参考值
    # 报告（不判定）: y vs 反量化参考 ref 的 elementwise 差
    max_abs_error: float
    max_rel_error: float
    ok_close: bool
    # 报告（不判定）
    arith_tol_max: float         # tol 的最大值（审计用）
    S_max: float                 # max_n Σ_k |a_k|（fp64, 审计用）
    # 第 (c) 层量化保真度（只报告, 不判定; 对 kernel 输入
    # W_packed/scale vs 原始 FP16 W —— 量化误差, 不是 kernel bug）
    fid_max_abs: Optional[float]
    fid_rmse: Optional[float]
    fid_cosine: Optional[float]
    atol: float
    rtol: float
    note: str = ""


def arith_ratio(y: torch.Tensor, W_packed: torch.Tensor,
                scale: torch.Tensor, x: torch.Tensor,
                dtype_name: str) -> tuple[float, float, float]:
    """固定累加误差界（见模块 docstring (1)）。

    存储值: q = unpack(W_packed)（int）, s = **存储的 fp16 scale**
    （合同: 内核读 fp16 scale）。a_k = q·s·x 的精确 fp64 提升乘积。

    返回 (max_ratio, tol_max, S_max); 非有限元素 ratio 置 0（由
    有限性门独立判定, 保证 JSON 记录为有限数）。
    """
    if W_packed.numel() == 0:
        return 0.0, 0.0, 0.0
    N, K2 = W_packed.shape
    K = 2 * K2
    Wq = unpack_w(W_packed).double()              # (N, K) fp64
    s = scale.double().repeat_interleave(
        GROUP_SIZE, dim=1)                        # (N, K) fp64
    xd = x.double()
    # S_n = Σ_k |a_k|, a_k = q·s·x —— 三个因子都取绝对值（x 可负;
    # s 恒 ≥ 0 但零 group 时 s=0, 乘积为 0, 无符号问题 —— 同
    # v0.6 首跑实证的两因子取绝对值教训, 三因子同理）。
    S = (Wq.abs() * s.abs() * xd.abs().unsqueeze(0)).sum(dim=1)  # (N,)
    exact = torch.mv(Wq * s, xd)                  # (N,) fp64
    tol = TOL_K * (
        _PER_TERM_ROUNDINGS * K * _NEG24 * S
        + 0.5 * _ulp(y, dtype_name).double()
        + 0.5 * _ulp(exact, dtype_name).double()
    )
    diff = (y.double() - exact).abs()
    ratio = torch.where(diff.isfinite() & tol.isfinite(),
                        diff / tol,
                        torch.zeros_like(diff))
    return (float(ratio.max().item()), float(tol.max().item()),
            float(S.max().item()))


def check_one(variant: str, ext, W_packed: torch.Tensor,
              scale: torch.Tensor, W_fp16: torch.Tensor, x: torch.Tensor,
              mode: str, seed: int = SEED,
              note: str = "") -> CheckResult:
    y = ext.forward(variant, W_packed, scale, x)
    y = y.contiguous()
    # 第 (a) 层参考（用户指定）: unpack + group 反量化 GEMV cast
    # FP16 —— 用**存储的 fp16 scale**（合同: 与内核读同一份值）
    Wd = dequantize_w(W_packed, scale)            # (N, K) fp32
    ref = (torch.mv(Wd, x.float()).to(torch.float16))
    dtype_name = str(y.dtype).split(".")[-1]

    ratio, tol_max, S_max = arith_ratio(y, W_packed, scale, x,
                                        dtype_name)
    ref_nan = bool(torch.isnan(ref).any().item())
    ref_inf = bool(torch.isinf(ref).any().item())
    if ref_nan or ref_inf:
        note = (note + "; 参考值非有限（large 的 scale 选择应已保证 "
                  "有限 —— 套件构造问题, 该用例判 FAIL）").strip("; ")

    m = compute_metrics(y, ref, dtype_name)
    N, K = W_packed.shape[0], 2 * W_packed.shape[1]
    # 第 (c) 层量化保真度（只报告; 量化器对有限 W, 参考有限）
    fid = fidelity_metrics(W_fp16, x, W_packed, scale)
    return CheckResult(
        variant=variant,
        shape=[int(N), int(K)],
        dtype=dtype_name,
        seed=seed, mode=mode,
        passed=(ratio <= 1.0 and not m["has_nan"] and not m["has_inf"]
                and not ref_nan and not ref_inf),
        # 12 位小数（v0.4 review F6: 8 位会把 tiny 量级审计值舍成 0）
        arith_max_ratio=round(ratio, 12),
        has_nan=m["has_nan"],
        has_inf=m["has_inf"],
        ref_has_nan=ref_nan,
        ref_has_inf=ref_inf,
        max_abs_error=m["max_abs_error"],
        max_rel_error=m["max_rel_error"],
        ok_close=m["ok_close"],
        arith_tol_max=round(tol_max, 12),
        S_max=round(S_max, 12),
        fid_max_abs=round(fid["max_abs"], 12),
        fid_rmse=round(fid["rmse"], 12),
        fid_cosine=(round(fid["cosine"], 9)
                    if fid["cosine"] is not None else None),
        atol=m["atol"], rtol=m["rtol"],
        note=note,
    )


def run_suite(variant: str, ext,
              dtypes: tuple = DTYPES, shapes: tuple = SHAPES,
              modes: tuple = INPUT_MODES) -> list[CheckResult]:
    """单个变体的完整 INT4GEMV kernel 正确性套件（50 项检查,
    确定性）。

    W / x 独立随机流（W seed=SEED, x seed=SEED+1; 同 v0.5/v0.6
    教训: 同 seed 会使 x 成为 W 展平布局的前缀）。量化 + packing
    在每次检查构造期完成（离线, 不在任何计时区）。
    """
    results: list[CheckResult] = []
    for dtype_name in dtypes:
        for (N, K) in shapes:
            for mode in modes:
                W_packed, scale, W_fp16 = make_w4(N, K, seed=SEED,
                                                  mode=mode,
                                                  large=LARGE_SCALE)
                x = make_x4(K, seed=SEED + 1, mode=mode,
                            large=LARGE_SCALE)
                results.append(check_one(variant, ext, W_packed, scale,
                                         W_fp16, x, mode, seed=SEED))
    return results


def _jsonable(r: CheckResult) -> dict:
    d = asdict(r)
    d["pass"] = d.pop("passed")
    return d


def summarize(results: list[CheckResult]) -> dict:
    s = summarize_results([_jsonable(r) for r in results])
    s["max_arith_max_ratio"] = max((r.arith_max_ratio for r in results),
                                   default=0.0)
    s["ok_close_vs_ref_all"] = all(r.ok_close for r in results)
    return s


def save_results(results: list[CheckResult], out_path: Path) -> Path:
    return save_suite(out_path, {
        "operator": "int4gemv",
        "suite": "int4gemv-correctness-v0.7",
        "layer": ("(a) kernel 正确性（判定门, per-variant）; (b) "
                  "pack/unpack correctness 由 tests/test_int4gemv_cpu.py "
                  "CPU 独立层覆盖（256-byte 双射 / 全值域往返 / 符号"
                  "扩展 / 边界 -7/0/+7）, 且本层 GPU 参考同时钉死 CPU "
                  "镜像 unpack 与设备端 unpack 的等价性; (c) 量化保真"
                  "度数值逐 case 记录在 fid_* 字段（只报告）, 独立 "
                  "fidelity 套件见 quantization_fidelity.json"),
        "convention": ("y[n] = Σ_k (scale[n,k/128]·unpack(W_packed)"
                       "[n,k])·x[k]（FP32 累加, cast fp16）; W_packed "
                       "(N,K/2) uint8 row-major 连续（low nibble = "
                       "k=2b, high nibble = k=2b+1, 4-bit two's "
                       "complement）, scale (N,K/128) fp16 连续, x "
                       "(K,) fp16 连续, K % 128 == 0"),
        "gate": ("pass = 有限性（y 与 ref 均无 NaN/Inf）AND "
                 "arith_max_ratio <= 1（固定累加误差界, 对精确 fp64 "
                 "反量化 GEMV, 每 term 最多 3 次 FP32 舍入 → 系数 "
                 "3K·2^-24, TOL_K=2 余量, 对任意合法 FP32 累加顺序"
                 "成立, 见模块 docstring (1)）。y vs 反量化 ref 的 "
                 "elementwise allclose 只报告不判定（同 v0.5 GEMV / "
                 "v0.6 QGEMV 合同修正）"),
        "reference": ("(a) 层 ref = unpack(W_packed).float() * "
                      "scale_fp16.float().repeat_interleave(128, "
                      "dim=1) @ x.float() → fp16（用户指定合同; "
                      "scale 用**存储的 fp16 值** —— 合同的一部分）; "
                      "exact 参考 = unpack(W_packed).double() * "
                      "scale_fp16.double().repeat_interleave(128, "
                      "dim=1) @ x.double()（fp64, 无舍入）"),
        "quantization": ("对称 group-wise INT4, zero_point=0, "
                         "G=128: scale = fp16(max|W_group|/7), "
                         "q = clamp(round-half-to-even(W/scale32), "
                         "-7, 7), 零 group q≡0; 量化 + packing 在"
                         "计时区外（int4gemv_quantize.py）"),
        "matrix": {
            "dtypes": list(DTYPES),
            "shapes_NxK": [list(s) for s in SHAPES],
            "input_modes": list(INPUT_MODES),
            "seed_w": SEED,
            "seed_x": SEED + 1,
        },
        "tolerances_reported": TOLERANCES,
        "tol_k": TOL_K,
        "per_term_roundings": _PER_TERM_ROUNDINGS,
        "rel_eps_guard": REL_EPS_GUARD,
        "summary": summarize(results),
        "results": [_jsonable(r) for r in results],
    })


# ---- 第 (c) 层: 独立量化保真度套件（variant 无关, 只报告）--------------

FIDELITY_SHAPES = list(BENCH_MATRIX_INT4GEMV)


def run_fidelity_suite(seed: int = SEED) -> dict:
    """5 基准形状 × 5 输入模式的量化保真度（FP32 域, 与 kernel 无关:
    同一组 W_packed/scale 对任何 kernel 变体都相同）。

    对比 y_orig = W @ x 与 y_quant = W_dequant @ x（均 FP32 累加,
    W_dequant 用**存储的 fp16 scale**）: max_abs / RMSE / cosine
    （用户指定三项）+ 量化器自检量（max_dequant_step_err 构造性界
    ≈ 0.5 + 7·2^-11, 只报告）。
    """
    per_shape_mode = []
    for (N, K) in FIDELITY_SHAPES:
        for mode in INPUT_MODES:
            W_packed, scale, W_fp16 = make_w4(N, K, seed=seed,
                                              mode=mode,
                                              large=LARGE_SCALE)
            x = make_x4(K, seed=seed + 1, mode=mode, large=LARGE_SCALE)
            fid = fidelity_metrics(W_fp16, x, W_packed, scale)
            per_shape_mode.append({"shape": [int(N), int(K)],
                                   "mode": mode, **fid})
    # 汇总（zeros 模式的 cosine 为 None —— 零向量, 不伪造 1.0）
    cosines = [e["cosine"] for e in per_shape_mode
               if e["cosine"] is not None]
    return {
        "layer": "(c) 量化保真度（只报告, 不判定; 量化误差 ≠ kernel "
                 "bug, 与第 (a) 层容差分离, 用户 §2）",
        "quantization": ("对称 group-wise INT4, zero_point=0, G=128, "
                         "scale = fp16(max|W_group|/7), "
                         "q = clamp(round-half-to-even(W/scale32), "
                         "-7, 7), 零 group q≡0"),
        "reference": ("y_orig = W_fp16.float() @ x.float(); y_quant = "
                      "W_dequant @ x.float()（W_dequant = unpack · "
                      "存储的 fp16 group scale, 均 FP32 累加, "
                      "torch.mv）"),
        "metrics": "max_abs / RMSE / cosine similarity（零向量 → "
                   "null）+ 量化器自检（max_dequant_abs_err / "
                   "max_dequant_step_err 构造性上界 ≈ 0.5+7·2^-11 / "
                   "n_zero_scale_groups / max_abs_q）",
        "seed_w": seed,
        "seed_x": seed + 1,
        "matrix": {"shapes_NxK": [list(s) for s in FIDELITY_SHAPES],
                   "input_modes": list(INPUT_MODES)},
        "summary": {
            "n_cases": len(per_shape_mode),
            "max_fid_max_abs": max(e["max_abs"] for e in per_shape_mode),
            "max_fid_rmse": max(e["rmse"] for e in per_shape_mode),
            "min_cosine": (min(cosines) if cosines else None),
            "max_dequant_step_err": max(e["max_dequant_step_err"]
                                        for e in per_shape_mode),
            "max_abs_q": max(e["max_abs_q"] for e in per_shape_mode),
        },
        "cases": per_shape_mode,
    }


def save_fidelity(fid: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    from .evaluator.gpu import now_iso
    doc = {"operator": "int4gemv",
           "suite": "int4gemv-quantization-fidelity-v0.7",
           "generated": now_iso(), **fid}
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    return out_path
