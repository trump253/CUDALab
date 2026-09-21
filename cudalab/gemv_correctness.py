"""CUDALab v0.5 — GEMV 正确性校验套件。

原则（与 RMSNorm / Softmax / RoPE 套件一致）:
- 容差/误差界对所有变体固定，并记录在每一份结果中。绝不为某个候选
  单独放宽使其通过。
- 正确性 FAIL 的内核永远没有资格成为性能胜者（experiment.py 强制）。
- 所有结果（通过和失败）都报告并保存；失败用例绝不删除。

算子合同:
    y[n] = Σ_k W[n,k]·x[k]   （FP32 累加, 最后 cast 到输出 dtype）
    参考: fp16 → torch.mv(W.float(), x.float()).to(torch.float16)
          fp32 → torch.mv(W, x)

套件矩阵（用户指定: random / zeros / small / large / mixed-sign +
不同 N/K; 基准 5 形状 + 5 个边界形状）:
- dtype:   float16 / float32
- (N, K):  基准矩阵 (1024,4096) (4096,1024) (4096,4096) (11008,4096)
           (4096,11008) + 边界 (1,1) (1,4096) (4096,1) (16,13) (8,7)
- input:   normal(N(0,1)) / zeros / tiny(N(0,1e-4)) / large(N(0,10)) /
           mixed_sign（W 用 (i+j)%2 交替符号, x 用独立 Bernoulli 符号
           流 → 乘积符号独立, 行求和出现真实正负抵消, 检验抵消路径;
           幅度 |N(0,1)|, 该模式不适用 large 参数）
共 2 × 10 × 5 = 100 项检查, W seed=0 / x seed=1（确定性, 独立随机流）。

**W 与 x 必须用独立随机流（W seed=SEED, x seed=SEED+1）**:
make_w / make_x 同 seed 同 device 时产生**同一段**随机数 ——
x 恰好是 W 展平布局的前缀（x[k] == W[0,k]）, 于是
y[0] = Σ_k W[0,k]² ≈ K·E[W²]（large 下 ≈ 4096·100 = 409600
≫ fp16 上限 65504）—— 这不是 GEMV 的数值边界, 是套件构造缺陷
（v0.5 首跑 94/100 的 6 个 FAIL 全部源于此, 已修复并记录）。

**large 模式 scale=10**（与 RoPE 的 1000 不同, 原因记录）:
GEMV 输出幅度 ~ N(0, K)·scale²（K=4096 时 std = 64·scale²,
前提 W / x 独立）, fp16 上限 65504: K=4096 时 8σ = 51200 <
65504; 最大归约形状 (4096, 11008) 时 σ = 100·√11008 ≈ 10492,
4096 行内 |y| 的实际最大值 ~4.5σ ≈ 47000 < 65504（固定 seed
确定性 —— 有限性门 (2) 对每用例双重保证, 溢出即 FAIL）。
取 10（normal 下 |y| ~ 64–105, large 放大 ~100× 至数千量级,
是真正的大幅度压力）。RoPE 是旋转（范数保持）所以可以
scale=1000 —— 两个算子的 large 语义不同, 不共用数值。

判定合同（v0.5 首发, 固定不变量, 对所有变体一致）:

(1) 双舍入/累加误差界（arith bound, 对**精确 float64 GEMV**）
    设 a_k = W[n,k]·x[k]（存储值的精确 fp64 提升乘积）, exact =
    Σ_k a_k 在实数（fp64）中计算。任何"FP32 累加 + 最终 cast 到
    dtype"的实现都满足:
        |impl − exact| ≤ 2·K·2^-24·Σ_k |a_k| + 0.5·ulp_dtype(|impl|)
    推导: K 个乘积 + K-1 次加法, 每次 FP32 舍入 ≤ 0.5·ulp32 ≤
    2^-24·|参与值|（|v| ≥ 2^-126）; 无 FMA 收缩的最坏情形（乘/加
    各自舍入, 每轮 2 次）归纳得上界 2(K−1)·2^-24·Σ|a_k| ≤
    2K·2^-24·Σ|a_k| —— 该界**对任意合法 FP32 累加顺序成立**
    （每线程步长部分和 + warp 树 / warp-per-row / split-K 等
    变体顺序都只改变结合律, 舍入次数不变）; FMA 收缩情形更紧
    （K 次舍入, 半系数量级）。最终 cast 引入 ≤ 0.5·ulp_dtype。
    套件对每元素取 TOL_K=2 安全余量并覆盖 impl 与 exact 两个可能
    的 binade:
        tol = 2·(2K·2^-24·S_n + 0.5·ulp_dtype(|y|)
                 + 0.5·ulp_dtype(|exact|))
        arith_ratio = max_elem |y − exact| / tol ≤ 1
    其中 S_n = Σ_k |W[n,k]·x[k]|（fp64, 逐行）。
    该界在 normal（scale 1, K=4096）下 ~ 2.5 绝对（|y|~64 时
    ~4e-2 相对）, 足以捕获 gross 错误（漏项、张错行/列、错误
    缩放、转置、归约缺失 → O(1) 相对偏差 ≫ 界）, 而**不**把
    两个同样合法的 FP32 累加顺序在深度抵消元素上的 ~γ_K·S 差异
    （γ_K = K·2^-24 ≈ 2.4e-4 @ K=4096）误判为错误。
    边界自检:
    - zeros: S=0, y=exact=0 → tol = 2·(0.5·ulp16(0)+0.5·ulp16(0))
      = 2^-23 ≈ 1.2e-7, 任何正确实现 diff=0 精确通过;
    - tiny: |y|~1e-8 落 fp16 次正规区（步长 2^-24 ≈ 6.1e-5）,
      y 常被舍入到 0 —— 界中的 0.5·ulp_dtype(|exact|) 项
      （次正规步长的一半 ≈ 3e-5）覆盖该量化, 不误判。
    y 与 torch 参考 `gemv_ref` 的 elementwise 差（max_abs/max_rel/
    allclose, 共享 TOLERANCES: fp16 atol=2e-3/rtol=5e-3, fp32
    atol=1e-5/rtol=1e-4）照常**报告**, 但**不作为判定门** ——
    深度抵消行（ref≈0 而 |y−ref| 可达 γ_K·S ~ 0.6 @ scale 1）
    会超出 allclose 的绝对容差, 对合法 FP32 实现不是正确合同
    （同 v0.4 RoPE 首跑的合同修正）。

(2) 有限性门
    y 无 NaN / Inf; 参考值同样必须有限（large 的 scale 选择已
    保证, 若违反说明套件构造错误, 该用例判 FAIL 并记录 note ——
    绝不把"参考溢出"静默当通过）。

判定: pass = (arith_max_ratio ≤ 1) AND y 有限 AND ref 有限。
max_abs_error / max_rel_error（vs gemv_ref, REL_EPS_GUARD=1e-3
分母保护）恒报告, 汇总在 summary 中供审查（用户要求的
max_abs/max_rel/NaN-Inf 三查齐备）。
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
from .operators.gemv import (
    BENCH_MATRIX_GEMV,
    make_w,
    make_x,
    gemv_ref,
)

ROOT = Path(__file__).resolve().parent.parent

DTYPES = ("float16", "float32")
# 基准 5 形状（用户指定）+ 5 边界形状（小 N / 小 K / 奇数 K / 单行单列）
EDGE_SHAPES = [(1, 1), (1, 4096), (4096, 1), (16, 13), (8, 7)]
SHAPES = list(BENCH_MATRIX_GEMV) + EDGE_SHAPES
INPUT_MODES = ("normal", "zeros", "tiny", "large", "mixed_sign")
LARGE_SCALE = 10.0
SEED = 0

# 固定判定参数（对所有变体一致, 不随候选放宽）。
TOL_K = 2.0      # 安全余量（覆盖 impl/exact 两个 binade + 无 FMA 最坏情形）
_NEG24 = 2.0 ** -24


def _ulp(v: torch.Tensor, dtype_name: str) -> torch.Tensor:
    """逐元素 ulp（**目标 dtype 网格**的间距; ulp(0) = 最小次正规）。

    fp32: nextafter 差。fp16: binade 公式 2^(floor(log2 v) − 10)
    （次正规/0: 2^-24）—— 同 rope_correctness._ulp 的论证: 输入
    在目标网格上或其精确提升上, floor(log2) 的 fp 误差（~1e-7
    相对）不会越 binade 边界, 公式精确。
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


@dataclass
class CheckResult:
    variant: str
    shape: list            # [N, K]
    dtype: str
    seed: int
    mode: str              # 输入模式
    passed: bool
    # 判定门: (1) 累加误差界 (2) 有限性（y 与 ref）
    arith_max_ratio: float       # max_elem |y − exact64| / tol ≤ 1
    has_nan: bool                # 内核输出
    has_inf: bool                # 内核输出
    ref_has_nan: bool            # 参考值（违反 = 套件构造错误）
    ref_has_inf: bool            # 参考值
    # 报告（不判定）: y vs torch 参考 gemv_ref 的 elementwise 差
    max_abs_error: float
    max_rel_error: float
    ok_close: bool
    # 报告（不判定）
    arith_tol_max: float         # tol 的最大值（审计用）
    S_max: float                 # max_n Σ_k |W[n,k]·x[k]|（fp64, 审计用）
    atol: float
    rtol: float
    note: str = ""


def arith_ratio(y: torch.Tensor, W: torch.Tensor, x: torch.Tensor,
                dtype_name: str) -> tuple[float, float, float]:
    """固定累加误差界（见模块 docstring (1)）。

    返回 (max_ratio, tol_max, S_max); 非有限元素 ratio 置 0（由
    有限性门独立判定, 保证 JSON 记录为有限数）。
    """
    if W.numel() == 0:
        return 0.0, 0.0, 0.0
    K = W.size(1)
    wd = W.double()
    xd = x.double()
    S = (wd.abs() * xd.abs()).sum(dim=1)          # (N,) fp64
    exact = torch.mv(wd, xd)                      # (N,) fp64
    tol = TOL_K * (
        2.0 * K * _NEG24 * S
        + 0.5 * _ulp(y, dtype_name).double()
        + 0.5 * _ulp(exact, dtype_name).double()
    )
    diff = (y.double() - exact).abs()
    ratio = torch.where(diff.isfinite() & tol.isfinite(),
                        diff / tol,
                        torch.zeros_like(diff))
    return (float(ratio.max().item()), float(tol.max().item()),
            float(S.max().item()))


def check_one(variant: str, ext, W: torch.Tensor, x: torch.Tensor,
              mode: str, seed: int = SEED,
              note: str = "") -> CheckResult:
    y = ext.forward(variant, W, x)
    y = y.contiguous()
    ref = gemv_ref(W, x)
    dtype_name = str(W.dtype).split(".")[-1]

    ratio, tol_max, S_max = arith_ratio(y, W, x, dtype_name)
    ref_nan = bool(torch.isnan(ref).any().item())
    ref_inf = bool(torch.isinf(ref).any().item())
    if ref_nan or ref_inf:
        note = (note + "; 参考值非有限（large 的 scale 选择应已保证 "
                 "有限 —— 套件构造问题, 该用例判 FAIL）").strip("; ")

    m = compute_metrics(y, ref, dtype_name)
    N, K = W.shape
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
        atol=m["atol"], rtol=m["rtol"],
        note=note,
    )


def run_suite(variant: str, ext,
              dtypes: tuple = DTYPES, shapes: tuple = SHAPES,
              modes: tuple = INPUT_MODES) -> list[CheckResult]:
    """单个变体的完整 GEMV 正确性套件（100 项检查, 确定性）。"""
    results: list[CheckResult] = []
    for dtype_name in dtypes:
        dtype = getattr(torch, dtype_name)
        for (N, K) in shapes:
            for mode in modes:
                # W / x 独立随机流（同 seed 会使 x 成为 W 的前缀,
                # 见模块 docstring）; 记录中的 seed 字段 = W 的 seed。
                W = make_w(N, K, dtype=dtype, seed=SEED, mode=mode,
                           large=LARGE_SCALE)
                x = make_x(K, dtype=dtype, seed=SEED + 1, mode=mode,
                           large=LARGE_SCALE)
                results.append(check_one(variant, ext, W, x, mode,
                                         seed=SEED))
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
        "operator": "gemv",
        "suite": "gemv-correctness-v0.5",
        "convention": ("y[n] = Σ_k W[n,k]·x[k]（FP32 累加, 最后 cast "
                       "到输出 dtype）; W (N,K) row-major 连续, x (K,) "
                       "连续"),
        "gate": ("pass = 有限性（y 与 ref 均无 NaN/Inf）AND "
                 "arith_max_ratio <= 1（固定累加误差界, 对精确 float64 "
                 "GEMV, TOL_K=2 余量, 对任意合法 FP32 累加顺序成立, "
                 "见模块 docstring (1)）。y vs gemv_ref 的 elementwise "
                 "allclose 只报告不判定（深度抵消行下对合法 FP32 实现"
                 "不是正确合同, 同 v0.4 RoPE 合同修正）"),
        "reference": ("gemv_ref: fp16 → torch.mv(W.float(), x.float())"
                      ".to(torch.float16)（用户指定合同）; fp32 → "
                      "torch.mv(W, x); exact 参考 = W.double() @ "
                      "x.double()（fp64, 无舍入）"),
        "large_scale_rationale": ("large 模式 scale=10（非 RoPE 的 1000）: "
                                  "y ~ N(0,K)·scale²（W/x 独立）, "
                                  "K=4096 时 8σ=51200 < 65504, "
                                  "K=11008 时 4.5σ≈47000 < 65504, 取 10; "
                                  "W seed=SEED / x seed=SEED+1（独立流; "
                                  "同 seed 会使 x 成为 W 前缀 → y[0]=ΣW[0,k]² "
                                  "溢出, 首跑 6 FAIL 根因, 已修复）"),
        "matrix": {
            "dtypes": list(DTYPES),
            "shapes_NxK": [list(s) for s in SHAPES],
            "input_modes": list(INPUT_MODES),
            "seed_w": SEED,
            "seed_x": SEED + 1,
        },
        "tolerances_reported": TOLERANCES,
        "tol_k": TOL_K,
        "rel_eps_guard": REL_EPS_GUARD,
        "summary": summarize(results),
        "results": [_jsonable(r) for r in results],
    })
