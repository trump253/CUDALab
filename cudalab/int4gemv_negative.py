"""CUDALab v0.7 — INT4GEMV negative correctness suite（per-variant,
30 例）。

目标: 非法输入必须在 **kernel launch 之前** 以明确异常被稳定、安全地
拒绝 —— 而不是静默算出错误结果，也不是产生异步 CUDA 运行时错误。

原则（与 RMSNorm / Softmax / RoPE / GEMV / QGEMV negative 套件一致）:
- 不制造危险的 OOB / 非法访问；所有用例依赖 validation 先行拒绝。
- 每个用例之后验证 CUDA 上下文仍然健康（synchronize + 控制 forward），
  确认拒绝没有污染后续运行。
- 对每个用例记录异常类型与消息；关键用例额外断言消息来自我们自己的
  预启动 validation 文本（expect_msg_contains），而非运行时错误。
- 时间戳由程序生成（ISO 8601，带时区），不手填历史日期。

对齐 / 整除性契约说明（v0.7, 见 int4gemv_common.h 头部）:
int4gemv_baseline 是**纯标量访存**（每 thread 1B uint8 + 2B fp16
load, 无向量化）, **没有对齐契约** —— `valid_offset_view_Wp_control`
/ `valid_offset_view_x_control` 两个 control 用例钉死: storage
offset 视图（连续但未 16B 对齐）对 baseline 是合法输入, 必须成功。
向量化候选变体（INT4GEMV-0001 起, uint4 16B packed load）有显式
对齐契约: W_packed 基址 16B ∧ x 基址 16B ∧ K%32==0（K%128==0 已
保证）; 契约不满足时不得拒绝 —— 必须回退 `int4gemv_scalar_kernel`
（与 baseline 同一代码源, 同输入下输出与 baseline **bit-identical**）。
per-variant 回归用例 fallback_Wp_misaligned / fallback_x_misaligned
钉死该契约于每一个受测的向量化变体（与 v0.4.1 rope_v3_half2 /
v0.5 gemv_vec4_row / v0.6 qgemv_vec16_row 同一模式）。
K % 128 != 0 是**硬合同拒绝**（不是回退情形）: group_size=128
要求 K 为 128 的倍数, K=64 的输入在 scale 组数检查之前就被
"K 必须可被 128 整除" 拒绝（negative 套件 K_not_mult128 钉死）。
**v0.7 当前无隔离变体**（bindings.cpp quarantined_set 为空, 机制
保留）—— 本套件对正常 dispatch 的每一个变体以相同 30 例运行,
baseline 存规范 invalid_inputs.json, 其余变体存
invalid_inputs_<variant>.json（run_negative, 同 v0.5/v0.6 归档
约定）。

30 例构成（覆盖用户 §6 最低清单: wrong packed shape / wrong scale
shape / K%128!=0 / dtype mismatch / device mismatch / non-contiguous /
misaligned packed pointer / misaligned x / wrong output / invalid
group count）:
    W_packed  8: 1d / 3d / N=0 / K2=0 / dtype_fp16 / dtype_int8 /
                 cpu / noncontig
    K 合同    1: K_not_mult128（K=64, 硬拒绝）
    scale     6: N_mismatch / group_count（invalid group count）/
                1d / dtype_fp32 / cpu / noncontig
    x         5: len_mismatch / 2d / dtype_fp32 / cpu / noncontig
    out       4: wrong_shape / wrong_dtype / noncontig / cpu
    变体      1: unknown_variant
    控制      3: valid_forward_into_control /
                valid_offset_view_Wp_control /
                valid_offset_view_x_control
    回退      2: fallback_Wp_misaligned / fallback_x_misaligned
                （bit-identical 到 int4gemv_baseline）
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .evaluator.gpu import now_iso as _now_iso
from .evaluator.negative import run_case as _run_case_core, summarize_cases
from .operators.int4gemv import make_w4, make_x4

ROOT = Path(__file__).resolve().parent.parent

SUITE_VERSION = "negative-v0.7-int4gemv"
V = "int4gemv_baseline"  # 默认（套件主体）变体


def _post_check_ok(ext, variant: str = V) -> bool:
    """拒绝之后上下文必须仍然健康: 同步 + 一次合法控制 forward。"""
    try:
        torch.cuda.synchronize()
        W_packed, scale, _W = make_w4(2, 128, seed=0)
        x = make_x4(128, seed=1)
        y = ext.forward(variant, W_packed, scale, x)
        torch.cuda.synchronize()
        return bool(torch.isfinite(y.float()).all().item())
    except Exception:
        return False


def _rand_wp(N: int, K: int, seed: int, device: str = "cuda") -> torch.Tensor:
    """随机 uint8 packed W（K 为原始归约维, K % 128 == 0）。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randint(0, 256, (N, K // 2), generator=g,
                      dtype=torch.uint8, device="cpu")
    return t.to(device)


def build_cases(ext, variant: str = V) -> list[dict]:
    """构造全部 30 个 negative 用例。每个 dict: id/variant/..."""
    dev = "cuda"
    cases: list[dict] = []

    def add(cid, description, call, expected="reject",
            expect_msg_contains=None, variant_=None):
        cases.append(dict(id=cid, variant=variant_ or variant,
                          description=description, call=call,
                          expected=expected,
                          expect_msg_contains=expect_msg_contains))

    # ---- 合法基线（各用例在此基础上单点破坏）----
    N, K = 8, 256
    W_packed, scale, _W = make_w4(N, K, seed=0)
    x = make_x4(K, seed=1)
    fwd = lambda Wa, sa, xa: ext.forward(variant, Wa, sa, xa)  # noqa: E731

    # ---- W_packed 的契约 ----
    g2 = torch.Generator(device="cpu").manual_seed(2)
    w_1d = torch.randint(0, 256, (N * K // 2,), generator=g2,
                         dtype=torch.uint8, device="cpu").to(dev)
    add("W_packed_1d", "W_packed 为 1 维 (N*K/2,)",
        lambda: fwd(w_1d, scale, x),
        expect_msg_contains="必须是 2 维")
    w_3d = _rand_wp(N, K, seed=3).view(2, N // 2, K // 2)
    add("W_packed_3d", "W_packed 为 3 维 (2, 4, K/2)",
        lambda: fwd(w_3d, scale, x),
        expect_msg_contains="必须是 2 维")
    w_n0 = torch.empty(0, K // 2, dtype=torch.uint8, device=dev)
    add("W_packed_N_zero", "N = 0（空行维）",
        lambda: fwd(w_n0, scale[:0].contiguous(), x),
        expect_msg_contains="N 必须 > 0")
    w_k0 = torch.empty(N, 0, dtype=torch.uint8, device=dev)
    add("W_packed_K2_zero", "K/2 = 0（空归约维）",
        lambda: fwd(w_k0, scale, x),
        expect_msg_contains="K/2 必须 > 0")
    wf = torch.randn(N, K // 2, dtype=torch.float16, device=dev)
    add("W_packed_dtype_fp16",
        "W_packed 为 fp16（合同: packed 权重必须是 uint8）",
        lambda: fwd(wf, scale, x),
        expect_msg_contains="必须是 uint8")
    wi = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8,
                       device=dev).to(torch.int8)  # 任意字节值, 解释为 int8
    add("W_packed_dtype_int8",
        "W_packed 为 int8（有符号 8 位, 非 uint8）",
        lambda: fwd(wi, scale, x),
        expect_msg_contains="必须是 uint8")
    w_cpu = _rand_wp(N, K, seed=4, device="cpu")
    add("W_packed_cpu", "W_packed 在 CPU",
        lambda: fwd(w_cpu, scale.cpu(), x.cpu()),
        expect_msg_contains="CUDA 张量")
    w_t = _rand_wp(K, N, seed=5).t()
    add("W_packed_noncontig",
        "W_packed 为 (K,N/2).t() 转置视图（非连续）",
        lambda: fwd(w_t, scale, x),
        expect_msg_contains="连续内存")

    # ---- K % 128 合同（硬拒绝, 非回退情形）----
    Wk64 = _rand_wp(N, 64, seed=6)      # K=64, K/2=32
    add("K_not_mult128",
        "K=64 不是 128 的倍数（group_size=128 合同, 硬拒绝）",
        lambda: fwd(Wk64, torch.zeros(N, 1, dtype=torch.float16,
                                       device=dev),
                    make_x4(64, seed=6)),
        expect_msg_contains="必须可被 128 整除")

    # ---- scale 的契约 ----
    s_short = scale[:, :scale.size(1) - 1].contiguous()
    add("scale_group_count",
        f"scale 组数 {scale.size(1) - 1} ≠ K/128={K // 128} "
        f"（invalid group count）",
        lambda: fwd(W_packed, s_short, x),
        expect_msg_contains="scale 组数必须等于 K/128")
    s_nshort = scale[:N - 1].contiguous()
    add("scale_N_mismatch", f"scale 行数 {N - 1} ≠ N={N}",
        lambda: fwd(W_packed, s_nshort, x),
        expect_msg_contains="scale 行数必须等于 N")
    s_1d = scale.view(N * (K // 128))
    add("scale_1d", "scale 为 1 维 (N*K/128,)",
        lambda: fwd(W_packed, s_1d, x),
        expect_msg_contains="必须是 2 维")
    s_f32 = scale.float()
    add("scale_dtype_fp32",
        "scale 为 fp32（合同: group scale 必须 fp16）",
        lambda: fwd(W_packed, s_f32, x),
        expect_msg_contains="必须是 float16")
    add("scale_cpu", "scale 在 CPU, W_packed 在 CUDA（设备错误）",
        lambda: fwd(W_packed, scale.cpu(), x),
        expect_msg_contains="CUDA 张量")
    big_s = torch.randn(2 * N, scale.size(1), dtype=torch.float16,
                        device=dev)
    s_nc = big_s[::2]
    add("scale_noncontig", "scale 为步长 2 的行切片（非连续）",
        lambda: fwd(W_packed, s_nc, x),
        expect_msg_contains="连续内存")

    # ---- x 的契约 ----
    x_short = make_x4(K - 128, seed=7)   # K-128 仍是 128 的倍数
    add("x_len_mismatch", f"x 长度 {K - 128} ≠ K={K}（shape 不匹配）",
        lambda: fwd(W_packed, scale, x_short),
        expect_msg_contains="x 长度必须等于 K")
    x_2d = x.view(1, K)
    add("x_2d", "x 为 2 维 (1, K)",
        lambda: fwd(W_packed, scale, x_2d),
        expect_msg_contains="必须是 1 维")
    x_f32 = make_x4(K, seed=8).float()
    add("x_dtype_fp32", "x 为 fp32（合同: x 必须 fp16, v0.7 主路径）",
        lambda: fwd(W_packed, scale, x_f32),
        expect_msg_contains="必须是 float16")
    add("x_cpu", "x 在 CPU, W_packed 在 CUDA（设备错误）",
        lambda: fwd(W_packed, scale, x.cpu()),
        expect_msg_contains="CUDA 张量")
    big_x = torch.randn(2 * K, dtype=torch.float16, device=dev)
    x_nc = big_x[::2]
    add("x_noncontig", "x 为步长 2 的切片（非连续）",
        lambda: fwd(W_packed, scale, x_nc),
        expect_msg_contains="连续内存")

    # ---- forward_into 的 out 契约 ----
    add("out_wrong_shape", f"out 长度 {N - 1} ≠ N={N}（shape 不匹配）",
        lambda: ext.forward_into(variant, W_packed, scale, x,
                                 torch.empty(N - 1, dtype=torch.float16,
                                             device=dev)),
        expect_msg_contains="out 长度必须等于 N")
    add("out_wrong_dtype", "out=fp32（合同: out 必须 fp16）",
        lambda: ext.forward_into(variant, W_packed, scale, x,
                                 torch.empty(N, dtype=torch.float32,
                                             device=dev)),
        expect_msg_contains="必须是 float16")
    big_o = torch.randn(2 * N, dtype=torch.float16, device=dev)
    o_nc = big_o[::2]
    add("out_noncontig", "out 为步长 2 的切片（非连续）",
        lambda: ext.forward_into(variant, W_packed, scale, x, o_nc),
        expect_msg_contains="连续内存")
    add("out_cpu", "out 在 CPU, W_packed 在 CUDA（设备错误）",
        lambda: ext.forward_into(variant, W_packed, scale, x,
                                 torch.empty(N, dtype=torch.float16)),
        expect_msg_contains="CUDA 张量")

    # ---- 变体名契约 ----
    add("unknown_variant",
        "未知变体名（注册表查找失败, 非 launch 错误）",
        lambda: ext.forward("no_such_int4gemv_variant", W_packed,
                            scale, x),
        expect_msg_contains="未知 int4gemv 变体")

    # ---- control: 合法输入不得被误拒（baseline 为标量访存, 无对齐
    #      契约 —— storage offset 视图是合法输入, 必须成功）----
    out_ok = torch.empty(N, dtype=torch.float16, device=dev)
    add("valid_forward_into_control",
        "control: 全合法输入 forward_into: 必须成功",
        lambda: ext.forward_into(variant, W_packed, scale, x, out_ok),
        expected="pass")
    # 基址偏移 1 字节（uint8）的 W_packed 视图: 1B 未对齐（16B 契约
    # 必破坏, 但连续且合法）; x 保持对齐合法数据（randn 有限, 避免
    # 回收内存的 NaN 位模式使 finite 断言误报 —— 同 v0.5/v0.6 教训）。
    g9 = torch.Generator(device="cpu").manual_seed(9)
    big_wp = torch.randint(0, 256, (N * K // 2 + 1,), dtype=torch.uint8,
                           generator=g9, device="cpu").to(dev)
    Wp_mis = big_wp[1:1 + N * K // 2].view(N, K // 2)  # 基址偏移 1 字节
    # x 必须是**有限**数据（make_x4 randn, 见上）

    def _wp_offset_control():
        ext.forward_into(variant, Wp_mis, scale, x, out_ok)
        y = ext.forward(variant, W_packed, scale, x)
        assert torch.isfinite(out_ok.float()).all(), (
            "offset-view forward_into 输出含非有限值")
        assert torch.isfinite(y.float()).all(), "control forward 输出非有限"

    if variant == "int4gemv_baseline":
        _align_note = "标量 baseline 无对齐契约"
    else:
        _align_note = "对齐契约不满足 → scalar fallback（同源代码, " \
                      "不得拒绝）"
    add("valid_offset_view_Wp_control",
        f"control: W_packed 为基址偏移 1 字节（未 16B 对齐）的连续"
        f"视图, is_contiguous()==True: {_align_note}, 不得拒绝此合法"
        f"输入: 必须成功",
        _wp_offset_control,
        expected="pass")
    big_xv = torch.randn(K + 1, dtype=torch.float16, device=dev)
    xo_mis = big_xv[1:1 + K]  # x 基址偏移 1 个 half（2B, 未 16B 对齐）
    add("valid_offset_view_x_control",
        f"control: x 为基址偏移 1 个 half（2B, 未 16B 对齐）的连续"
        f"切片: {_align_note}, 不得拒绝: 必须成功",
        lambda: fwd(W_packed, scale, xo_mis),
        expected="pass")

    # ---- per-variant scalar fallback regression（与 v0.5 GEMV /
    #      v0.6 QGEMV 同一模式）: 向量化变体的契约
    #      （W_packed 基址 16B ∧ x 基址 16B ∧ K%32==0）不满足时,
    #      必须回退 int4gemv_scalar_kernel —— 与 int4gemv_baseline
    #      的**同一代码源**（int4gemv_common.h）, 同输入输出
    #      bit-identical。对 baseline 自身这些用例平凡成立（它就是
    #      那个 kernel）。----
    def _fallback_check(cid_desc, Wm, xm, N_):
        out_f = torch.empty(N_, dtype=torch.float16, device=dev)
        ext.forward_into(variant, Wm, scale, xm, out_f)  # 写 out_f
        torch.cuda.synchronize()
        y_base = ext.forward("int4gemv_baseline", Wm, scale, xm)
        torch.cuda.synchronize()
        assert torch.equal(out_f, y_base), (
            f"{cid_desc}: fallback output differs from "
            f"int4gemv_baseline (must be bit-identical, same scalar "
            f"kernel source)")

    xa = make_x4(K, seed=10)
    Wa, sa, _ = make_w4(N, K, seed=11)
    big_x8 = torch.randn(K + 1, dtype=torch.float16, device=dev)
    xm = big_x8[1:1 + K]  # x 基址偏移 2B, 未 16B 对齐
    add("fallback_Wp_misaligned",
        "fallback: W_packed base offset 1 byte (not 16B-aligned) "
        "contiguous view, x aligned, K=256 (multiple of 128): "
        "alignment contract unmet -> must take int4gemv_scalar_kernel "
        "fallback, must not reject, output bit-identical to "
        "int4gemv_baseline",
        lambda: _fallback_check("fallback_Wp_misaligned", Wp_mis, xa, N),
        expected="pass")
    add("fallback_x_misaligned",
        "fallback: x base offset 1 half (2B, not 16B-aligned) "
        "contiguous slice, W_packed aligned, K=256 (multiple of "
        "128): alignment contract unmet -> must take scalar "
        "fallback, must not reject, output bit-identical to "
        "int4gemv_baseline",
        lambda: _fallback_check("fallback_x_misaligned", Wa, xm, N),
        expected="pass")
    return cases


def run_negative_suite(ext, out_path: Path | None = None,
                       variant: str = V) -> dict:
    """运行全部 30 个 negative 用例并保存结构化结果。"""
    cases = build_cases(ext, variant)
    results = []
    for c in cases:
        r = _run_case_core(c["variant"], c["description"], c["call"],
                           c["expected"], c["expect_msg_contains"],
                           lambda v=c["variant"]: _post_check_ok(ext, v))
        r["id"] = c["id"]
        results.append(r)

    summary = summarize_cases(results)
    _note = ("illegal inputs must be rejected with an explicit exception before kernel launch; post_check_ok verifies the CUDA context is not polluted. K % 128 != 0 is a HARD contract rejection (group_size=128), not a fallback case. Alignment contract (vectorized int4gemv variants, INT4GEMV-0001+): W_packed base 16B-aligned AND x base 16B-aligned AND K % 32 == 0 (implied by K%128==0); unmet inputs MUST fall back to int4gemv_scalar_kernel (same code source as int4gemv_baseline, so the fallback output is bit-identical -- pinned by the two per-variant regression cases fallback_Wp_misaligned / fallback_x_misaligned, same pattern as v0.4.1 rope_v3_half2 / v0.5 gemv_vec4_row / v0.6 qgemv_vec16_row). Alignment is NOT a contract of int4gemv_baseline (scalar 1B uint8 + 2B fp16 loads); legal inputs (incl. 1-byte offset views) must never be rejected. Quantization + packing (FP16 -> uint8 packed + fp16 group scale, G=128, q in [-7,7]) is performed offline, outside any timed region; the kernel path validates host metadata only (no D2H), matching v0.5 GEMV / v0.6 QGEMV")
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
