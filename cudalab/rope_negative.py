"""CUDALab v0.4 — RoPE negative correctness suite（≥15 例）。

目标: 非法输入必须在 **kernel launch 之前** 以明确异常被稳定、安全地
拒绝 —— 而不是静默算出错误结果，也不是产生异步 CUDA 运行时错误。

原则（与 RMSNorm / Softmax negative 套件一致）:
- 不制造危险的 OOB / 非法访问；所有用例依赖 validation 先行拒绝。
- 每个用例之后验证 CUDA 上下文仍然健康（synchronize + 控制 forward），
  确认拒绝没有污染后续运行。
- 对每个用例记录异常类型与消息；关键用例额外断言消息来自我们自己的
  预启动 validation 文本（expect_msg_contains），而非运行时错误。
- 时间戳由程序生成（ISO 8601，带时区），不手填历史日期。

本套件是 **per-variant** 的（negative_suite_scope = "per-variant"）:
套件主体按 variant 参数化（build_cases / _post_check_ok 全部走
`ext.forward(variant, ...)`）, operator 层对每个受测变体运行全套 37
例 —— 默认变体存规范 `invalid_inputs.json`, 其余变体存
`invalid_inputs_<variant>.json`。

说明: baseline 是标量访存（每 thread 一个 pair, 非向量化），因此
**没有**对齐契约；`valid_offset_view_control` 用例记录这一事实：
storage offset 视图（连续但未 16B 对齐）对 baseline 是合法输入，
必须成功。v3_half2 的 fp16 路径做 4B `__half2` load/store，对 **x 与
out 的基指针** 有 4B 对齐契约（`is_contiguous()==true` 不保证 —— 奇
数元素 storage offset 的连续视图基指针偏移 2B）；未对齐时 host 侧
launch 前回退标量 fp16 路径（baseline 兼容数学，位级一致），合法输入
不得被拒 —— `v3_half2_x_*` / `v3_half2_out_*` 三个回归用例钉死该
契约（v0.4.1 新增）。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .evaluator.gpu import now_iso as _now_iso
from .evaluator.negative import run_case as _run_case_core, summarize_cases
from .operators.rope import make_rotary_table

ROOT = Path(__file__).resolve().parent.parent

SUITE_VERSION = "negative-v0.4-rope"
V = "rope_baseline"  # 默认（套件主体）变体；per-variant 用例见 build_cases


def _post_check_ok(ext, variant: str = V) -> bool:
    """拒绝之后上下文必须仍然健康: 同步 + 一次合法控制 forward。"""
    try:
        torch.cuda.synchronize()
        x = torch.randn(2, 64, dtype=torch.float16, device="cuda")
        pos = torch.arange(2, dtype=torch.int64, device="cuda")
        cos_t, sin_t = make_rotary_table(8, 64, torch.float16)
        y = ext.forward(variant, x, pos, cos_t, sin_t)
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
    M, D, L = 4, 128, 64          # d2 = 64
    x = torch.randn(M, D, dtype=torch.float16, device=dev)
    pos = torch.arange(M, dtype=torch.int64, device=dev)
    cos_t, sin_t = make_rotary_table(L, D, torch.float16)
    fwd = lambda xa, pa, ca, sa: ext.forward(variant, xa, pa, ca, sa)  # noqa: E731

    # ---- x 的契约 ----
    x_1d = torch.randn(D, dtype=torch.float16, device=dev)
    add("x_1d", "x 为 1 维 (128,)",
        lambda: fwd(x_1d, pos, cos_t, sin_t),
        expect_msg_contains="必须是 2 维")
    x_3d = torch.randn(2, 4, 64, dtype=torch.float16, device=dev)
    add("x_3d", "x 为 3 维 (2, 4, 64)",
        lambda: fwd(x_3d, pos, cos_t, sin_t),
        expect_msg_contains="必须是 2 维")
    x_m0 = torch.empty(0, D, dtype=torch.float16, device=dev)
    add("x_M_zero", "M = 0（空行维）",
        lambda: fwd(x_m0, pos, cos_t, sin_t),
        expect_msg_contains="M 必须 > 0")
    x_d0 = torch.empty(M, 0, dtype=torch.float16, device=dev)
    add("x_D_zero", "D = 0（空对维）",
        lambda: fwd(x_d0, pos, cos_t, sin_t),
        expect_msg_contains="D 必须 > 0")
    x_dodd = torch.randn(M, D - 1, dtype=torch.float16, device=dev)
    add("x_D_odd", "D = 127 为奇数（interleaved pair 约定要求偶数）",
        lambda: fwd(x_dodd, pos, cos_t, sin_t),
        expect_msg_contains="偶数")
    xb = torch.randn(M, D, dtype=torch.bfloat16, device=dev)
    add("x_bfloat16", "x 为 bfloat16（v0.4 仅支持 FP16/FP32, 无 BF16）",
        lambda: fwd(xb, pos, cos_t, sin_t),
        expect_msg_contains="float16 / float32")
    xi = torch.randint(-3, 3, (M, D), dtype=torch.int32, device=dev)
    add("x_int32", "x 为 int32（非浮点 dtype）",
        lambda: fwd(xi, pos, cos_t, sin_t),
        expect_msg_contains="float16 / float32")
    x_cpu = torch.randn(M, D, dtype=torch.float16)
    add("x_cpu", "x 在 CPU",
        lambda: fwd(x_cpu, pos, cos_t, sin_t),
        expect_msg_contains="CUDA 张量")
    x_t = torch.randn(D, M, dtype=torch.float16, device=dev).t()
    add("x_transpose_view", "x 为 (D,M).t() 转置视图（非连续）",
        lambda: fwd(x_t, pos, cos_t, sin_t),
        expect_msg_contains="连续内存")
    add("unknown_variant", "未知变体名（注册表查找失败, 非 launch 错误）",
        lambda: ext.forward("no_such_rope_variant", x, pos, cos_t, sin_t),
        expect_msg_contains="未知 rope 变体")

    # ---- per-variant 整除性约束（v1/v2/v4 launcher 的 TORCH_CHECK,
    #      v0.4 review finding 补测: 此前无 negative 用例覆盖）----
    def add_div_case(cid, vname, d, expect_sub, desc):
        xv = torch.randn(M, d, dtype=torch.float16, device=dev)
        pv = torch.arange(M, dtype=torch.int64, device=dev)
        cv, sv = make_rotary_table(L, d, torch.float16)
        add(cid, desc,
            lambda: ext.forward(vname, xv, pv, cv, sv),
            expect_msg_contains=expect_sub, variant_=vname)

    add_div_case("divisibility_v1_D6", "rope_v1_2pair", 6, "D%4==0",
                 "v1_2pair 要求 D%4==0: D=6 (D/2=3, 3%2!=0) 必须在 "
                 "launch 前拒绝")
    add_div_case("divisibility_v2_D12", "rope_v2_4pair", 12, "D%8==0",
                 "v2_4pair 要求 D%8==0: D=12 (D/2=6, 6%4!=0) 必须在 "
                 "launch 前拒绝（D=12 满足 v1 的 D%4==0, 不满足 v2）")
    add_div_case("divisibility_v4_D24", "rope_v4_8pair", 24, "D%16==0",
                 "v4_8pair 要求 D%16==0: D=24 (D/2=12, 12%8!=0) 必须在 "
                 "launch 前拒绝（D=24 满足 v1/v2, 不满足 v4）")

    # ---- positions 的契约 ----
    add("pos_len_mismatch", "positions 长度 3 ≠ M=4",
        lambda: fwd(x, pos[:3], cos_t, sin_t),
        expect_msg_contains="长度必须等于 M")
    add("pos_2d", "positions 为 2 维 (1, 4)",
        lambda: fwd(x, pos.view(1, M), cos_t, sin_t),
        expect_msg_contains="1 维")
    add("pos_float_dtype", "positions 为 float32（要求 int64）",
        lambda: fwd(x, pos.float(), cos_t, sin_t),
        expect_msg_contains="int64")
    add("pos_cpu", "positions 在 CPU",
        lambda: fwd(x, pos.cpu(), cos_t, sin_t),
        expect_msg_contains="CUDA 张量")
    pos_neg = torch.tensor([0, -1, 2, 3], dtype=torch.int64, device=dev)
    add("pos_negative", "positions 含 -1（值域: 0 <= p < L）",
        lambda: fwd(x, pos_neg, cos_t, sin_t),
        expect_msg_contains=">= 0")
    pos_oob = torch.tensor([0, 1, 2, L], dtype=torch.int64, device=dev)
    add("pos_out_of_range", "positions 含 64 = L（值域: p < max_seq_len）",
        lambda: fwd(x, pos_oob, cos_t, sin_t),
        expect_msg_contains="max_seq_len")

    # ---- cos/sin 表的契约 ----
    # cos/sin 必须同行数（cos_sin_shape_mismatch 用例单独覆盖不一致情况）,
    # 这里让两张表都短, 才能走到 positions 值域检查
    cos_small, sin_small = make_rotary_table(16, D, torch.float16)
    pos_far = torch.tensor([0, 1, 2, 31], dtype=torch.int64, device=dev)
    add("cos_rows_too_few",
        "cos/sin 表行数 16 < 使用到的最大 position 31（表行数即 "
        "max_seq_len, 值域检查在 launch 前捕获）",
        lambda: fwd(x, pos_far, cos_small, sin_small),
        expect_msg_contains="max_seq_len")
    cos_cols = torch.randn(L, D // 4, dtype=torch.float16, device=dev)
    add("cos_cols_mismatch", "cos 列数 32 ≠ D/2 = 64",
        lambda: fwd(x, pos, cos_cols, sin_t),
        expect_msg_contains="列数必须等于 D/2")
    cos_1d = torch.randn(L * (D // 2), dtype=torch.float16, device=dev)
    add("cos_1d", "cos 为 1 维 (L*D/2,)",
        lambda: fwd(x, pos, cos_1d, sin_t),
        expect_msg_contains="必须是 2 维")
    add("cos_dtype_mismatch", "cos 为 float32, x 为 float16",
        lambda: fwd(x, pos, cos_t.float(), sin_t),
        expect_msg_contains="dtype 必须与 x 一致")
    cos_t_view = torch.randn(D // 2, L, dtype=torch.float16, device=dev).t()
    add("cos_transpose_view", "cos 为 (D/2,L).t() 转置视图（非连续）",
        lambda: fwd(x, pos, cos_t_view, sin_t),
        expect_msg_contains="连续内存")
    add("sin_cpu", "sin 在 CPU",
        lambda: fwd(x, pos, cos_t, sin_t.cpu()),
        expect_msg_contains="CUDA 张量")
    sin_t_view = torch.randn(D // 2, L, dtype=torch.float16, device=dev).t()
    add("sin_transpose_view", "sin 为 (D/2,L).t() 转置视图（非连续）",
        lambda: fwd(x, pos, cos_t, sin_t_view),
        expect_msg_contains="连续内存")
    sin_short = make_rotary_table(48, D, torch.float16)[1]
    add("cos_sin_shape_mismatch",
        "cos 行数 64 ≠ sin 行数 48（两者各自合法, 互相不一致）",
        lambda: fwd(x, pos, cos_t, sin_short),
        expect_msg_contains="形状必须一致")

    # ---- forward_into 的 out 契约 ----
    add("out_wrong_shape", "out 形状 (4, 127) ≠ x (4, 128)",
        lambda: ext.forward_into(variant, x, pos, cos_t, sin_t,
                                 torch.empty(M, D - 1, dtype=torch.float16,
                                             device=dev)),
        expect_msg_contains="形状")
    add("out_wrong_dtype", "out=float32, x=float16",
        lambda: ext.forward_into(variant, x, pos, cos_t, sin_t,
                                 torch.empty(M, D, dtype=torch.float32,
                                             device=dev)),
        expect_msg_contains="dtype")
    add("out_transpose_view", "out 为 (D,M).t() 转置视图（非连续）",
        lambda: ext.forward_into(variant, x, pos, cos_t, sin_t,
                                 torch.empty(D, M, dtype=torch.float16,
                                             device=dev).t()),
        expect_msg_contains="连续内存")
    add("out_cpu", "out 在 CPU, x 在 CUDA（设备错误）",
        lambda: ext.forward_into(variant, x, pos, cos_t, sin_t,
                                 torch.empty(M, D, dtype=torch.float16)),
        expect_msg_contains="CUDA 张量")

    # ---- 多设备（仅当可见 >1 个 GPU 时执行；CUDA_VISIBLE_DEVICES=0
    #      的 v0.4 环境中安全跳过并记录）----
    if torch.cuda.device_count() > 1:
        x0, o1 = x.to("cuda:0"), torch.empty(M, D, dtype=torch.float16,
                                             device="cuda:1")
        add("x_out_different_device", "x 在 cuda:0, out 在 cuda:1",
            lambda: ext.forward_into(variant, x0, pos, cos_t, sin_t, o1),
            expect_msg_contains="同一 CUDA 设备")
    else:
        cases.append(dict(id="x_out_different_device", variant=variant,
                          description="x/out 不同 CUDA 设备（本环境仅 1 个可见 "
                                      "GPU，CUDA_VISIBLE_DEVICES=0，安全跳过）",
                          call=None, expected="skip",
                          expect_msg_contains=None))

    # ---- control: 合法输入不得被误拒（baseline 为标量访存, 无对齐
    #      契约 —— storage offset 视图是合法输入, 必须成功）----
    out_ok = torch.empty_like(x)
    add("valid_forward_into_control",
        "control: 全合法输入 forward_into: 必须成功",
        lambda: ext.forward_into(variant, x, pos, cos_t, sin_t, out_ok),
        expected="pass")
    big_x = torch.randn(M * D + 2, dtype=torch.float16, device=dev)
    xo = big_x[2:2 + M * D].view(M, D)
    add("valid_offset_view_control",
        "control: 基址偏移 2 元素（4B, 未 16B 对齐）的连续视图; 所有变体"
        "均不得拒绝此合法输入（向量化变体须走其回退/可用路径）: 必须成功",
        lambda: fwd(xo, pos, cos_t, sin_t), expected="pass")

    # ---- v3_half2 对齐契约（v0.4.1）: fp16 路径做 4B __half2
    #      load/store, x/out 基指针未 4B 对齐时 host 侧 launch 前回退
    #      标量 fp16 路径（baseline 兼容数学, 位级一致）。合法输入
    #      不得被拒, 也不得进入 unsafe half2 路径。----
    def _v3_fwd_bitmatch(xa):
        y3 = ext.forward("rope_v3_half2", xa, pos, cos_t, sin_t)
        yb = ext.forward("rope_baseline", xa, pos, cos_t, sin_t)
        if not torch.equal(y3, yb):
            raise AssertionError(
                "v3_half2 结果必须与 baseline 位级一致")
        return y3

    big_x1 = torch.randn(M * D + 1, dtype=torch.float16, device=dev)
    x_mis2b = big_x1[1:1 + M * D].view(M, D)
    add("v3_half2_x_misaligned_2b",
        "v0.4.1 对齐回归: x 基指针偏移 1 个 half（2B, 未 4B 对齐）, "
        "is_contiguous()==True: v3_half2 fp16 路径必须回退标量路径"
        "（不得进入 unsafe half2 路径）, 且结果与 baseline 位级一致: "
        "必须成功",
        lambda: _v3_fwd_bitmatch(x_mis2b),
        expected="pass", variant_="rope_v3_half2")
    big_x2 = torch.randn(M * D + 2, dtype=torch.float16, device=dev)
    x_ok4b = big_x2[2:2 + M * D].view(M, D)
    add("v3_half2_x_aligned_4b",
        "v0.4.1 对齐回归: x 基指针偏移 2 个 half（4B, 已对齐）: "
        "v3_half2 fp16 路径走 __half2 打包 load/store, 且结果与 "
        "baseline 位级一致: 必须成功",
        lambda: _v3_fwd_bitmatch(x_ok4b),
        expected="pass", variant_="rope_v3_half2")
    big_o = torch.randn(M * D + 1, dtype=torch.float16, device=dev)
    o_mis2b = big_o[1:1 + M * D].view(M, D)

    def _v3_into_bitmatch():
        # forward_into 为 void（就地写入 out）, 比较的是被写入的 o_mis2b
        ext.forward_into("rope_v3_half2", x, pos, cos_t, sin_t, o_mis2b)
        yb = ext.forward("rope_baseline", x, pos, cos_t, sin_t)
        if not torch.equal(o_mis2b, yb):
            raise AssertionError(
                "v3_half2 结果必须与 baseline 位级一致")
        return o_mis2b

    add("v3_half2_out_misaligned_2b",
        "v0.4.1 对齐回归: forward_into 的 out 基指针偏移 1 个 half"
        "（2B, 未 4B 对齐）: v3_half2 必须回退（或明确拒绝）—— 本实现"
        "回退标量路径, 且结果与 baseline 位级一致: 必须成功",
        _v3_into_bitmatch,
        expected="pass", variant_="rope_v3_half2")

    return cases


def run_negative_suite(ext, out_path: Path | None = None,
                       variant: str = V) -> dict:
    """运行全部 negative 用例并保存结构化结果。"""
    cases = build_cases(ext, variant)
    results = []
    for c in cases:
        if c["expected"] == "skip":
            r = _run_case_core(c["variant"], c["description"],
                               lambda: None, "skip", None, lambda: True)
        else:
            r = _run_case_core(c["variant"], c["description"], c["call"],
                               c["expected"], c["expect_msg_contains"],
                               lambda: _post_check_ok(ext, c["variant"]))
        r["id"] = c["id"]
        results.append(r)

    summary = summarize_cases(results)
    doc = {
        "suite": SUITE_VERSION,
        "negative_suite_scope": "per-variant",
        "generated": _now_iso(),
        "note": "非法输入必须在 kernel launch 前被明确异常拒绝；"
                "post_check_ok 验证拒绝未污染 CUDA 上下文。"
                "baseline 为标量访存，无对齐契约"
                "（见 valid_offset_view_control）；v3_half2 fp16 路径"
                "对 x/out 基指针有 4B 对齐契约，未对齐时 host 侧回退"
                "标量路径（v0.4.1 回归用例 v3_half2_x_misaligned_2b / "
                "v3_half2_x_aligned_4b / v3_half2_out_misaligned_2b）。",
        "summary": summary,
        "cases": results,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    return doc
