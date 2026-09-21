"""CUDALab v0.5 — GEMV negative correctness suite。

目标: 非法输入必须在 **kernel launch 之前** 以明确异常被稳定、安全地
拒绝 —— 而不是静默算出错误结果，也不是产生异步 CUDA 运行时错误。

原则（与 RMSNorm / Softmax / RoPE negative 套件一致）:
- 不制造危险的 OOB / 非法访问；所有用例依赖 validation 先行拒绝。
- 每个用例之后验证 CUDA 上下文仍然健康（synchronize + 控制 forward），
  确认拒绝没有污染后续运行。
- 对每个用例记录异常类型与消息；关键用例额外断言消息来自我们自己的
  预启动 validation 文本（expect_msg_contains），而非运行时错误。
- 时间戳由程序生成（ISO 8601，带时区），不手填历史日期。

对齐 / 整除性契约说明（v0.5 总则, 见 gemv_common.h 头部）:
gemv_baseline 是**纯标量访存**（每 thread 2B/4B 元素 load, 无向量化）,
**没有对齐契约** —— `valid_offset_view_control` / `valid_offset_view_x`
两个 control 用例钉死: storage offset 视图（连续但未 16B 对齐）对
baseline 是合法输入, 必须成功。
向量化候选变体（GEMV-0002 起, float4 / __half2 打包 load）将在其
**变体落地时**在本套件中追加 per-variant 对齐回归用例
（未对齐基指针 → host 侧回退标量路径且结果与 baseline 一致; K 不整除
向量宽度 → 标量尾处理; 合法输入不得被拒 —— 与 v0.4.1 rope_v3_half2
的三个回归用例同一模式）。Phase 1 阶段只有 baseline, 故无 per-variant
对齐用例（不是遗漏, 是该契约此刻不存在）。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .evaluator.gpu import now_iso as _now_iso
from .evaluator.negative import run_case as _run_case_core, summarize_cases
from .operators.gemv import make_w, make_x

ROOT = Path(__file__).resolve().parent.parent

SUITE_VERSION = "negative-v0.5-gemv"
V = "gemv_baseline"  # 默认（套件主体）变体


def _post_check_ok(ext, variant: str = V) -> bool:
    """拒绝之后上下文必须仍然健康: 同步 + 一次合法控制 forward。"""
    try:
        torch.cuda.synchronize()
        W = torch.randn(2, 16, dtype=torch.float16, device="cuda")
        x = torch.randn(16, dtype=torch.float16, device="cuda")
        y = ext.forward(variant, W, x)
        torch.cuda.synchronize()
        return bool(torch.isfinite(y.float()).all().item())
    except Exception:
        return False


def build_cases(ext, variant: str = V) -> list[dict]:
    """构造全部 negative 用例。每个 dict: id/variant/description/call/..."""
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
    W = make_w(N, K, torch.float16, device=dev, seed=0)
    x = make_x(K, torch.float16, device=dev, seed=0)
    fwd = lambda Wa, xa: ext.forward(variant, Wa, xa)  # noqa: E731

    # ---- W 的契约 ----
    w_1d = torch.randn(N * K, dtype=torch.float16, device=dev)
    add("W_1d", "W 为 1 维 (N*K,)",
        lambda: fwd(w_1d, x),
        expect_msg_contains="必须是 2 维")
    w_3d = torch.randn(2, 4, 8, dtype=torch.float16, device=dev)
    add("W_3d", "W 为 3 维 (2, 4, 8)",
        lambda: fwd(w_3d, x),
        expect_msg_contains="必须是 2 维")
    w_n0 = torch.empty(0, K, dtype=torch.float16, device=dev)
    add("W_N_zero", "N = 0（空行维）",
        lambda: fwd(w_n0, x),
        expect_msg_contains="N 必须 > 0")
    w_k0 = torch.empty(N, 0, dtype=torch.float16, device=dev)
    add("W_K_zero", "K = 0（空归约维）",
        lambda: fwd(w_k0, x),
        expect_msg_contains="K 必须 > 0")
    wb = make_w(N, K, torch.bfloat16, device=dev, seed=0)
    add("W_bfloat16", "W 为 bfloat16（v0.5 仅支持 FP16/FP32, 无 BF16）",
        lambda: fwd(wb, x),
        expect_msg_contains="float16 / float32")
    wi = torch.randint(-3, 3, (N, K), dtype=torch.int32, device=dev)
    add("W_int32", "W 为 int32（非浮点 dtype）",
        lambda: fwd(wi, x),
        expect_msg_contains="float16 / float32")
    w_cpu = make_w(N, K, torch.float16, device="cpu", seed=0)
    add("W_cpu", "W 在 CPU",
        lambda: fwd(w_cpu, x),
        expect_msg_contains="CUDA 张量")
    w_t = make_w(K, N, torch.float16, device=dev, seed=1).t()
    add("W_transpose_view", "W 为 (K,N).t() 转置视图（非连续）",
        lambda: fwd(w_t, x),
        expect_msg_contains="连续内存")
    add("unknown_variant", "未知变体名（注册表查找失败, 非 launch 错误）",
        lambda: ext.forward("no_such_gemv_variant", W, x),
        expect_msg_contains="未知 gemv 变体")

    # ---- x 的契约 ----
    x_short = make_x(K - 1, torch.float16, device=dev, seed=2)
    add("x_len_mismatch", "x 长度 63 ≠ K=64（shape 不匹配）",
        lambda: fwd(W, x_short),
        expect_msg_contains="长度必须等于 K")
    x_2d = x.view(1, K)
    add("x_2d", "x 为 2 维 (1, 64)",
        lambda: fwd(W, x_2d),
        expect_msg_contains="必须是 1 维")
    x_f32 = make_x(K, torch.float32, device=dev, seed=3)
    add("x_dtype_mismatch", "x 为 float32, W 为 float16（dtype 不匹配）",
        lambda: fwd(W, x_f32),
        expect_msg_contains="dtype 必须与 W 一致")
    x_cpu = make_x(K, torch.float16, device="cpu", seed=4)
    add("x_cpu", "x 在 CPU, W 在 CUDA（设备错误）",
        lambda: fwd(W, x_cpu),
        expect_msg_contains="CUDA 张量")
    big_x = torch.randn(2 * K, dtype=torch.float16, device=dev)
    x_nc = big_x[::2]
    add("x_noncontiguous", "x 为步长 2 的切片（非连续）",
        lambda: fwd(W, x_nc),
        expect_msg_contains="连续内存")

    # ---- forward_into 的 out 契约 ----
    add("out_wrong_shape", "out 长度 7 ≠ N=8（shape 不匹配）",
        lambda: ext.forward_into(variant, W, x,
                                 torch.empty(N - 1, dtype=torch.float16,
                                             device=dev)),
        expect_msg_contains="长度必须等于 N")
    add("out_wrong_dtype", "out=float32, W=float16（dtype 不匹配）",
        lambda: ext.forward_into(variant, W, x,
                                 torch.empty(N, dtype=torch.float32,
                                             device=dev)),
        expect_msg_contains="dtype 必须与 W 一致")
    big_o = torch.randn(2 * N, dtype=torch.float16, device=dev)
    o_nc = big_o[::2]
    add("out_noncontiguous", "out 为步长 2 的切片（非连续）",
        lambda: ext.forward_into(variant, W, x, o_nc),
        expect_msg_contains="连续内存")
    add("out_cpu", "out 在 CPU, W 在 CUDA（设备错误）",
        lambda: ext.forward_into(variant, W, x,
                                 torch.empty(N, dtype=torch.float16)),
        expect_msg_contains="CUDA 张量")

    # ---- control: 合法输入不得被误拒（baseline 为标量访存, 无对齐
    #      契约 —— storage offset 视图是合法输入, 必须成功）----
    out_ok = torch.empty(N, dtype=torch.float16, device=dev)
    add("valid_forward_into_control",
        "control: 全合法输入 forward_into: 必须成功",
        lambda: ext.forward_into(variant, W, x, out_ok),
        expected="pass")
    big_w = torch.randn(N * K + 2, dtype=torch.float16, device=dev)
    Wo = big_w[2:2 + N * K].view(N, K)  # 基址偏移 2 元素（4B, 未 16B 对齐）
    # x 必须是**有限**数据: torch.empty_like 可能落在回收内存上,
    # 其中含 NaN/Inf 位模式（例如 _ulp 的 inf 填充块）—— 首跑在
    # 正确性套件之后运行时命中, 使 finite 断言误报（构造缺陷, 非
    # kernel 错误; kernel 对有限输入输出有限值）。
    xo_ok = make_x(K, torch.float16, device=dev, seed=5)

    def _offset_control():
        ext.forward_into(variant, Wo, xo_ok, out_ok)
        y = ext.forward(variant, W, x)
        # 数据有限（make_x randn）+ kernel 有限输入 → 有限输出;
        # 断言带消息（裸 assert 消息为空, run_case 记录时会 IndexError）
        assert torch.isfinite(out_ok.float()).all(), (
            "offset-view forward_into 输出含非有限值")
        assert torch.isfinite(y.float()).all(), "control forward 输出非有限"

    add("valid_offset_view_control",
        "control: W 为基址偏移 2 元素（4B, 未 16B 对齐）的连续视图, "
        "is_contiguous()==True: 标量 baseline 无对齐契约, 不得拒绝此"
        "合法输入: 必须成功",
        _offset_control,
        expected="pass")
    big_xv = torch.randn(K + 1, dtype=torch.float16, device=dev)
    xo_mis = big_xv[1:1 + K]  # x 基址偏移 1 元素（2B）
    add("valid_offset_view_x",
        "control: x 为基址偏移 1 个 half（2B）的连续切片: 标量 baseline"
        "无对齐契约, 不得拒绝: 必须成功",
        lambda: fwd(W, xo_mis),
        expected="pass")


    # ---- per-variant scalar fallback regression (added after the vectorized
    #      GEMV-0001.. landed; same pattern as v0.4.1 rope_v3_half2): when the
    #      contract is not met (base pointer not 16B-aligned / K not a multiple
    #      of the 16B element count), a vectorized variant must fall back to
    #      gemv_scalar_kernel -- the *same code source* as gemv_baseline
    #      (gemv_common.h), so the output must be **bit-identical** to the
    #      baseline on the same inputs (a stronger contract than "finite +
    #      not rejected"). For the baseline itself these cases are trivially
    #      true (it *is* that kernel).
    def _fallback_check(cid_desc, Wm, xm, N_):
        out_f = torch.empty(N_, dtype=torch.float16, device=dev)
        ext.forward_into(variant, Wm, xm, out_f)  # writes out_f; returns None
        torch.cuda.synchronize()
        y_base = ext.forward("gemv_baseline", Wm, xm)
        torch.cuda.synchronize()
        assert torch.equal(out_f, y_base), (
            f"{cid_desc}: fallback output differs from gemv_baseline "
            f"(must be bit-identical, same scalar kernel source)")

    big_w8 = torch.randn(N * K + 2, dtype=torch.float16, device=dev)
    Wm = big_w8[2:2 + N * K].view(N, K)  # base offset 4B, not 16B-aligned
    xa = make_x(K, torch.float16, device=dev, seed=6)
    add("fallback_W_misaligned",
        f"fallback: W base offset 2 elements (4B, not 16B-aligned) "
        f"contiguous view, x aligned, K={K} multiple of 8: vectorized "
        f"contract unmet -> must take scalar fallback, must not reject, "
        f"output bit-identical to gemv_baseline",
        lambda: _fallback_check("fallback_W_misaligned", Wm, xa, N),
        expected="pass")
    Wa = make_w(N, K, torch.float16, device=dev, seed=7)
    big_x8 = torch.randn(K + 1, dtype=torch.float16, device=dev)
    xm = big_x8[1:1 + K]  # x base offset 2B, not 16B-aligned
    add("fallback_x_misaligned",
        f"fallback: x base offset 1 half (2B, not 16B-aligned) "
        f"contiguous slice, W aligned, K={K} multiple of 8: vectorized "
        f"contract unmet -> must take scalar fallback, must not reject, "
        f"output bit-identical to gemv_baseline",
        lambda: _fallback_check("fallback_x_misaligned", Wa, xm, N),
        expected="pass")
    # K=13: for fp16 K%8=5 (vectorized contract fails) and K%4=1
    # (split-K contract fails) -- covers both fallback classes at once.
    Wk = make_w(16, 13, torch.float16, device=dev, seed=8)
    xk = make_x(13, torch.float16, device=dev, seed=9)
    add("fallback_K_not_mult8",
        "fallback: K=13 (not a multiple of 8 nor of 4): both the "
        "vectorized and split-K contracts unmet -> must take scalar "
        "fallback, must not reject, output bit-identical to "
        "gemv_baseline",
        lambda: _fallback_check("fallback_K_not_mult8", Wk, xk, 16),
        expected="pass")
    return cases


def run_negative_suite(ext, out_path: Path | None = None,
                       variant: str = V) -> dict:
    """运行全部 negative 用例并保存结构化结果。"""
    cases = build_cases(ext, variant)
    results = []
    for c in cases:
        r = _run_case_core(c["variant"], c["description"], c["call"],
                           c["expected"], c["expect_msg_contains"],
                           lambda v=c["variant"]: _post_check_ok(ext, v))
        r["id"] = c["id"]
        results.append(r)

    summary = summarize_cases(results)
    doc = {
        "suite": SUITE_VERSION,
        "generated": _now_iso(),
        "note": ("illegal inputs must be rejected with an explicit exception before kernel launch; post_check_ok verifies the CUDA context is not polluted. Alignment contract (GEMV-0001.. vectorized variants): base pointers 16B-aligned + K a multiple of the 16B element count; unmet inputs MUST fall back to gemv_scalar_kernel (same source as gemv_baseline, so the fallback output is bit-identical -- pinned by the three per-variant regression cases fallback_W_misaligned / fallback_x_misaligned / fallback_K_not_mult8, same pattern as v0.4.1 rope_v3_half2); legal inputs (incl. offset views) must never be rejected"),
        "summary": summary,
        "cases": results,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    return doc
