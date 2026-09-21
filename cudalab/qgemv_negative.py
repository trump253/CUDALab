"""CUDALab v0.6 — QGEMV negative correctness suite（per-variant, 29 例）。

目标: 非法输入必须在 **kernel launch 之前** 以明确异常被稳定、安全地
拒绝 —— 而不是静默算出错误结果，也不是产生异步 CUDA 运行时错误。

原则（与 RMSNorm / Softmax / RoPE / GEMV negative 套件一致）:
- 不制造危险的 OOB / 非法访问；所有用例依赖 validation 先行拒绝。
- 每个用例之后验证 CUDA 上下文仍然健康（synchronize + 控制 forward），
  确认拒绝没有污染后续运行。
- 对每个用例记录异常类型与消息；关键用例额外断言消息来自我们自己的
  预启动 validation 文本（expect_msg_contains），而非运行时错误。
- 时间戳由程序生成（ISO 8601，带时区），不手填历史日期。

对齐 / 整除性契约说明（v0.6, 见 qgemv_common.h 头部）:
qgemv_baseline 是**纯标量访存**（每 thread 1B int8 + 2B fp16 load,
无向量化）, **没有对齐契约** —— `valid_offset_view_Wq_control` /
`valid_offset_view_x_control` 两个 control 用例钉死: storage offset
视图（连续但未 16B 对齐）对 baseline 是合法输入, 必须成功。
向量化候选变体（QGEMV-0001 起, U16Q 16B 打包 load）有显式对齐契约:
W_q 基址 16B ∧ x 基址 16B ∧ K % 16 == 0; 契约不满足时不得拒绝 ——
必须回退 `qgemv_scalar_kernel`（与 baseline 同一代码源, 同输入下
输出与 baseline **bit-identical**）。三个 per-variant 回归用例
fallback_Wq_misaligned / fallback_x_misaligned / fallback_K_not_mult16
钉死该契约于每一个受测的向量化变体（与 v0.4.1 rope_v3_half2 /
v0.5 gemv_vec4_row 同一模式）。
**v0.6 当前无隔离变体**（bindings.cpp quarantined_set 为空, 机制
保留）—— 本套件对正常 dispatch 的每一个变体以相同 29 例运行,
baseline 存规范 invalid_inputs.json, 其余变体存
invalid_inputs_<variant>.json（run_negative, 同 v0.5 GEMV 归档
约定）。

29 例构成:
    W_q   8: 1d / 3d / N=0 / K=0 / dtype_fp16 / dtype_int32 /
            cpu / noncontig
    scale 5: len_mismatch / 2d / dtype_fp16 / cpu / noncontig
    x     5: len_mismatch / 2d / dtype_fp32 / cpu / noncontig
    out   4: wrong_shape / wrong_dtype / noncontig / cpu
    变体  1: unknown_variant
    控制  3: valid_forward_into_control /
            valid_offset_view_Wq_control /
            valid_offset_view_x_control
    回退  3: fallback_Wq_misaligned / fallback_x_misaligned /
            fallback_K_not_mult16（bit-identical 到 qgemv_baseline）
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .evaluator.gpu import now_iso as _now_iso
from .evaluator.negative import run_case as _run_case_core, summarize_cases
from .operators.qgemv import make_w_q, make_x_q

ROOT = Path(__file__).resolve().parent.parent

SUITE_VERSION = "negative-v0.6-qgemv"
V = "qgemv_baseline"  # 默认（套件主体）变体


def _post_check_ok(ext, variant: str = V) -> bool:
    """拒绝之后上下文必须仍然健康: 同步 + 一次合法控制 forward。"""
    try:
        torch.cuda.synchronize()
        W_q, scale, _W = make_w_q(2, 16, seed=0)
        x = make_x_q(16, seed=1)
        y = ext.forward(variant, W_q, scale, x)
        torch.cuda.synchronize()
        return bool(torch.isfinite(y.float()).all().item())
    except Exception:
        return False


def _rand_wq(N: int, K: int, seed: int, device: str = "cuda",
             lo: int = -127, hi: int = 128) -> torch.Tensor:
    """合法范围内的随机 int8 W_q（hi 排除 → 值域 [lo, hi-1]）。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randint(lo, hi, (N, K), generator=g,
                      dtype=torch.int8, device="cpu")
    return t.to(device)


def build_cases(ext, variant: str = V) -> list[dict]:
    """构造全部 29 个 negative 用例。每个 dict: id/variant/..."""
    dev = "cuda"
    cases: list[dict] = []

    def add(cid, description, call, expected="reject",
            expect_msg_contains=None, variant_=None):
        cases.append(dict(id=cid, variant=variant_ or variant,
                          description=description, call=call,
                          expected=expected,
                          expect_msg_contains=expect_msg_contains))

    # ---- 合法基线（各用例在此基础上单点破坏）----
    N, K = 8, 64
    W_q, scale, _W = make_w_q(N, K, seed=0)
    x = make_x_q(K, seed=1)
    fwd = lambda Wa, sa, xa: ext.forward(variant, Wa, sa, xa)  # noqa: E731

    # ---- W_q 的契约 ----
    w_1d = _rand_wq(N * K, 1, seed=2).view(N * K)
    add("W_q_1d", "W_q 为 1 维 (N*K,)",
        lambda: fwd(w_1d, scale, x),
        expect_msg_contains="必须是 2 维")
    w_3d = _rand_wq(N, K, seed=3).view(2, N // 2, K)
    add("W_q_3d", "W_q 为 3 维 (2, 2, K)",
        lambda: fwd(w_3d, scale, x),
        expect_msg_contains="必须是 2 维")
    w_n0 = torch.empty(0, K, dtype=torch.int8, device=dev)
    add("W_q_N_zero", "N = 0（空行维）",
        lambda: fwd(w_n0, scale[:0].contiguous(), x),
        expect_msg_contains="N 必须 > 0")
    w_k0 = torch.empty(N, 0, dtype=torch.int8, device=dev)
    add("W_q_K_zero", "K = 0（空归约维）",
        lambda: fwd(w_k0, scale, x),
        expect_msg_contains="K 必须 > 0")
    wf = torch.randn(N, K, dtype=torch.float16, device=dev)
    add("W_q_dtype_fp16", "W_q 为 fp16（合同: 量化权重必须是 int8）",
        lambda: fwd(wf, scale, x),
        expect_msg_contains="必须是 int8")
    wi = torch.randint(-3, 3, (N, K), dtype=torch.int32, device=dev)
    add("W_q_dtype_int32", "W_q 为 int32（非 int8 的整数 dtype）",
        lambda: fwd(wi, scale, x),
        expect_msg_contains="必须是 int8")
    w_cpu = _rand_wq(N, K, seed=4, device="cpu")
    add("W_q_cpu", "W_q 在 CPU",
        lambda: fwd(w_cpu, scale.cpu(), x.cpu()),
        expect_msg_contains="CUDA 张量")
    w_t = _rand_wq(K, N, seed=5).t()
    add("W_q_noncontig", "W_q 为 (K,N).t() 转置视图（非连续）",
        lambda: fwd(w_t, scale, x),
        expect_msg_contains="连续内存")

    # ---- scale 的契约 ----
    s_short = scale[:N - 1].contiguous()
    add("scale_len_mismatch", f"scale 长度 {N-1} ≠ N={N}（shape 不匹配）",
        lambda: fwd(W_q, s_short, x),
        expect_msg_contains="scale 长度必须等于 N")
    s_2d = scale.view(N, 1)
    add("scale_2d", "scale 为 2 维 (N, 1)",
        lambda: fwd(W_q, s_2d, x),
        expect_msg_contains="必须是 1 维")
    s_f16 = scale.half()
    add("scale_dtype_fp16", "scale 为 fp16（合同: scale 必须 fp32）",
        lambda: fwd(W_q, s_f16, x),
        expect_msg_contains="必须是 float32")
    add("scale_cpu", "scale 在 CPU, W_q 在 CUDA（设备错误）",
        lambda: fwd(W_q, scale.cpu(), x),
        expect_msg_contains="CUDA 张量")
    big_s = torch.randn(2 * N, dtype=torch.float32, device=dev)
    s_nc = big_s[::2]
    add("scale_noncontig", "scale 为步长 2 的切片（非连续）",
        lambda: fwd(W_q, s_nc, x),
        expect_msg_contains="连续内存")

    # ---- x 的契约 ----
    x_short = make_x_q(K - 1, seed=6)
    add("x_len_mismatch", f"x 长度 {K-1} ≠ K={K}（shape 不匹配）",
        lambda: fwd(W_q, scale, x_short),
        expect_msg_contains="x 长度必须等于 K")
    x_2d = x.view(1, K)
    add("x_2d", "x 为 2 维 (1, K)",
        lambda: fwd(W_q, scale, x_2d),
        expect_msg_contains="必须是 1 维")
    x_f32 = make_x_q(K, seed=7).float()
    add("x_dtype_fp32", "x 为 fp32（合同: x 必须 fp16, v0.6 主路径）",
        lambda: fwd(W_q, scale, x_f32),
        expect_msg_contains="必须是 float16")
    add("x_cpu", "x 在 CPU, W_q 在 CUDA（设备错误）",
        lambda: fwd(W_q, scale, x.cpu()),
        expect_msg_contains="CUDA 张量")
    big_x = torch.randn(2 * K, dtype=torch.float16, device=dev)
    x_nc = big_x[::2]
    add("x_noncontig", "x 为步长 2 的切片（非连续）",
        lambda: fwd(W_q, scale, x_nc),
        expect_msg_contains="连续内存")

    # ---- forward_into 的 out 契约 ----
    add("out_wrong_shape", f"out 长度 {N-1} ≠ N={N}（shape 不匹配）",
        lambda: ext.forward_into(variant, W_q, scale, x,
                                 torch.empty(N - 1, dtype=torch.float16,
                                             device=dev)),
        expect_msg_contains="out 长度必须等于 N")
    add("out_wrong_dtype", "out=fp32（合同: out 必须 fp16）",
        lambda: ext.forward_into(variant, W_q, scale, x,
                                 torch.empty(N, dtype=torch.float32,
                                             device=dev)),
        expect_msg_contains="必须是 float16")
    big_o = torch.randn(2 * N, dtype=torch.float16, device=dev)
    o_nc = big_o[::2]
    add("out_noncontig", "out 为步长 2 的切片（非连续）",
        lambda: ext.forward_into(variant, W_q, scale, x, o_nc),
        expect_msg_contains="连续内存")
    add("out_cpu", "out 在 CPU, W_q 在 CUDA（设备错误）",
        lambda: ext.forward_into(variant, W_q, scale, x,
                                 torch.empty(N, dtype=torch.float16)),
        expect_msg_contains="CUDA 张量")

    # ---- 变体名契约 ----
    add("unknown_variant", "未知变体名（注册表查找失败, 非 launch 错误）",
        lambda: ext.forward("no_such_qgemv_variant", W_q, scale, x),
        expect_msg_contains="未知 qgemv 变体")

    # ---- control: 合法输入不得被误拒（baseline 为标量访存, 无对齐
    #      契约 —— storage offset 视图是合法输入, 必须成功）----
    out_ok = torch.empty(N, dtype=torch.float16, device=dev)
    add("valid_forward_into_control",
        "control: 全合法输入 forward_into: 必须成功",
        lambda: ext.forward_into(variant, W_q, scale, x, out_ok),
        expected="pass")
    # 基址偏移 1 字节（int8）的 W_q 视图: 1B 未对齐（16B 契约必破坏,
    # 但连续且合法）; x 保持对齐合法数据（randn 有限, 避免回收内存的
    # NaN 位模式使 finite 断言误报 —— 同 v0.5 GEMV negative 教训）。
    big_wq = _rand_wq(N * K + 1, 1, seed=8).view(N * K + 1)
    Wq_mis = big_wq[1:1 + N * K].view(N, K)  # 基址偏移 1 字节
    # x 必须是**有限**数据（make_x_q randn, 见上）
    def _wq_offset_control():
        ext.forward_into(variant, Wq_mis, scale, x, out_ok)
        y = ext.forward(variant, W_q, scale, x)
        assert torch.isfinite(out_ok.float()).all(), (
            "offset-view forward_into 输出含非有限值")
        assert torch.isfinite(y.float()).all(), "control forward 输出非有限"

    if variant == "qgemv_baseline":
        _align_note = "标量 baseline 无对齐契约"
    else:
        _align_note = "对齐契约不满足 → scalar fallback（同源代码, " \
                      "不得拒绝）"
    add("valid_offset_view_Wq_control",
        f"control: W_q 为基址偏移 1 字节（未 16B 对齐）的连续视图, "
        f"is_contiguous()==True: {_align_note}, 不得拒绝此合法"
        f"输入: 必须成功",
        _wq_offset_control,
        expected="pass")
    big_xv = torch.randn(K + 1, dtype=torch.float16, device=dev)
    xo_mis = big_xv[1:1 + K]  # x 基址偏移 1 个 half（2B, 未 16B 对齐）
    add("valid_offset_view_x_control",
        f"control: x 为基址偏移 1 个 half（2B, 未 16B 对齐）的连续切片: "
        f"{_align_note}, 不得拒绝: 必须成功",
        lambda: fwd(W_q, scale, xo_mis),
        expected="pass")

    # ---- per-variant scalar fallback regression（与 v0.5 GEMV /
    #      v0.4.1 rope_v3_half2 同一模式）: 向量化变体的契约
    #      （W_q 基址 16B ∧ x 基址 16B ∧ K%16==0）不满足时, 必须
    #      回退 qgemv_scalar_kernel —— 与 qgemv_baseline 的**同一
    #      代码源**（qgemv_common.h）, 同输入输出 bit-identical。
    #      对 baseline 自身这些用例平凡成立（它就是那个 kernel）。----
    def _fallback_check(cid_desc, Wm, xm, N_):
        out_f = torch.empty(N_, dtype=torch.float16, device=dev)
        ext.forward_into(variant, Wm, scale, xm, out_f)  # 写 out_f
        torch.cuda.synchronize()
        y_base = ext.forward("qgemv_baseline", Wm, scale, xm)
        torch.cuda.synchronize()
        assert torch.equal(out_f, y_base), (
            f"{cid_desc}: fallback output differs from qgemv_baseline "
            f"(must be bit-identical, same scalar kernel source)")

    xa = make_x_q(K, seed=9)
    Wa, sa, _ = make_w_q(N, K, seed=10)
    big_x8 = torch.randn(K + 1, dtype=torch.float16, device=dev)
    xm = big_x8[1:1 + K]  # x 基址偏移 2B, 未 16B 对齐
    add("fallback_Wq_misaligned",
        f"fallback: W_q base offset 1 byte (not 16B-aligned) contiguous "
        f"view, x aligned, K={K} (multiple of 16): alignment contract "
        f"unmet -> must take qgemv_scalar_kernel fallback, must not "
        f"reject, output bit-identical to qgemv_baseline",
        lambda: _fallback_check("fallback_Wq_misaligned", Wq_mis, xa, N),
        expected="pass")
    add("fallback_x_misaligned",
        f"fallback: x base offset 1 half (2B, not 16B-aligned) "
        f"contiguous slice, W_q aligned, K={K} (multiple of 16): "
        f"alignment contract unmet -> must take scalar fallback, must "
        f"not reject, output bit-identical to qgemv_baseline",
        lambda: _fallback_check("fallback_x_misaligned", Wa, xm, N),
        expected="pass")
    # K=13: 13 % 16 = 13（违反向量化契约的整除条件）→ 向量化变体
    # 必须回退 qgemv_scalar_kernel（同一代码源 → bit-identical）;
    # baseline 直接合法执行。
    Wk, sk, _ = make_w_q(16, 13, seed=11)
    xk = make_x_q(13, seed=12)
    # _fallback_check 闭包用了外层 scale; K=13 时 scale 长度不同 ——
    # 内联一个独立版本。
    def _fallback_check_k13(Wm, xm):
        out_f = torch.empty(16, dtype=torch.float16, device=dev)
        ext.forward_into(variant, Wm, sk, xm, out_f)
        torch.cuda.synchronize()
        y_base = ext.forward("qgemv_baseline", Wm, sk, xm)
        torch.cuda.synchronize()
        assert torch.equal(out_f, y_base), (
            "fallback_K_not_mult16: fallback output differs from "
            "qgemv_baseline (must be bit-identical, same scalar "
            "kernel source)")
    add("fallback_K_not_mult16",
        "fallback: K=13 (not a multiple of 16): vectorized contract "
        "unmet -> must take qgemv_scalar_kernel fallback, must not "
        "reject, output bit-identical to qgemv_baseline",
        lambda: _fallback_check_k13(Wk, xk),
        expected="pass")
    return cases


def run_negative_suite(ext, out_path: Path | None = None,
                       variant: str = V) -> dict:
    """运行全部 29 个 negative 用例并保存结构化结果。"""
    cases = build_cases(ext, variant)
    results = []
    for c in cases:
        r = _run_case_core(c["variant"], c["description"], c["call"],
                           c["expected"], c["expect_msg_contains"],
                           lambda v=c["variant"]: _post_check_ok(ext, v))
        r["id"] = c["id"]
        results.append(r)

    summary = summarize_cases(results)
    _note = ("illegal inputs must be rejected with an explicit exception before kernel launch; post_check_ok verifies the CUDA context is not polluted. Alignment contract (vectorized qgemv variants, QGEMV-0001+): W_q base 16B-aligned AND x base 16B-aligned AND K % 16 == 0; unmet inputs MUST fall back to qgemv_scalar_kernel (same code source as qgemv_baseline, so the fallback output is bit-identical -- pinned by the three per-variant regression cases fallback_Wq_misaligned / fallback_x_misaligned / fallback_K_not_mult16, same pattern as v0.4.1 rope_v3_half2 and v0.5 gemv_vec4_row). Alignment is NOT a contract of qgemv_baseline (scalar 1B int8 + 2B fp16 loads); legal inputs (incl. 1-byte offset views) must never be rejected. Quantization (FP16 -> int8 + fp32 per-row scale) is performed offline, outside any timed region; the kernel path validates host metadata only (no D2H), matching v0.5 GEMV")
    doc = {
        "suite": SUITE_VERSION,
        "negative_suite_scope": "per-variant",
        "generated": _now_iso(),
        "note": _note,
        "summary": summary,
        "cases": results,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    return doc
