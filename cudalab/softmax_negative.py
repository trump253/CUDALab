"""CUDALab v0.3 — Softmax negative correctness suite。

目标: 非法输入必须在 **kernel launch 之前** 以明确异常被稳定、安全地
拒绝 —— 而不是静默算出错误结果，也不是产生异步 CUDA 运行时错误。

原则（与 RMSNorm negative 套件一致）:
- 不制造危险的 OOB / 非法访问；所有用例依赖 validation 先行拒绝。
- 每个用例之后验证 CUDA 上下文仍然健康（synchronize + 控制 forward），
  确认拒绝没有污染后续运行。
- 对每个用例记录异常类型与消息；关键用例额外断言消息来自我们自己的
  预启动 validation 文本（expect_msg_contains），而非运行时错误。
- 时间戳由程序生成（ISO 8601，带时区），不手填历史日期。

说明（v0.5 merge review 2026-09-21 刷新）:
- 本套件是 **per-variant** 的（negative_suite_scope = "per-variant"）:
  套件主体按 variant 参数化（build_cases / _post_check_ok 全部走
  `ext.forward(variant, ...)`）, operator 层对每个受测变体运行全套
  15 例 —— 默认变体存规范 `invalid_inputs.json`, 其余变体存
  `invalid_inputs_<variant>.json`。
- 对齐语义因变体而异: baseline 是标量访存（无向量化）, **没有**对齐
  契约 —— `align_offset_view_control` 用例钉死：storage offset 视图
  （连续但未 16B 对齐）是合法输入，必须成功。vec4 类向量化变体有
  对齐契约（H%4==0 ∧ x/out 基址按向量宽度对齐）, 不满足时回退标量
  路径（与 baseline 同源, 不得拒绝）—— 同一 control 用例对它们同样
  必须成功（走回退）。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .evaluator.gpu import now_iso as _now_iso
from .evaluator.negative import run_case as _run_case_core, summarize_cases

ROOT = Path(__file__).resolve().parent.parent

SUITE_VERSION = "negative-v0.3-softmax"
V = "softmax_baseline"  # 默认（套件主体）变体；套件按 variant 参数化


def _post_check_ok(ext, variant: str = V) -> bool:
    """拒绝之后上下文必须仍然健康: 同步 + 一次合法控制 forward。"""
    try:
        torch.cuda.synchronize()
        x = torch.randn(2, 4096, dtype=torch.float16, device="cuda")
        y = ext.forward(variant, x)
        torch.cuda.synchronize()
        return bool(torch.isfinite(y.float()).all().item())
    except Exception:
        return False


def build_cases(ext, variant: str = V) -> list[dict]:
    """构造全部 negative 用例。每个 dict: id/variant/description/call/..."""
    dev = "cuda"
    cases: list[dict] = []

    def add(cid, description, call, expected="reject",
            expect_msg_contains=None):
        cases.append(dict(id=cid, variant=variant, description=description,
                          call=call, expected=expected,
                          expect_msg_contains=expect_msg_contains))

    # 1) 维度错误
    x_1d = torch.randn(4096, dtype=torch.float16, device=dev)
    add("x_1d", "x 为 1 维 (4096,)",
        lambda: ext.forward(variant, x_1d), expect_msg_contains="必须是 2 维")
    x_3d = torch.randn(2, 4, 4096, dtype=torch.float16, device=dev)
    add("x_3d", "x 为 3 维 (2, 4, 4096)",
        lambda: ext.forward(variant, x_3d), expect_msg_contains="必须是 2 维")

    # 2) 空维度
    x_m0 = torch.empty(0, 4096, dtype=torch.float16, device=dev)
    add("x_M_zero", "M = 0（空行维）",
        lambda: ext.forward(variant, x_m0), expect_msg_contains="M 必须 > 0")
    x_h0 = torch.empty(4, 0, dtype=torch.float16, device=dev)
    add("x_H_zero", "H = 0（空行内维）",
        lambda: ext.forward(variant, x_h0), expect_msg_contains="H 必须 > 0")

    # 3) 不支持的 dtype（v0.3 范围: FP16 / FP32，**无 BF16**）
    xb = torch.randn(4, 4096, dtype=torch.bfloat16, device=dev)
    add("dtype_bfloat16", "x 为 bfloat16（v0.3 明确不支持 BF16）",
        lambda: ext.forward(variant, xb), expect_msg_contains="float16 / float32")
    xi = torch.randint(-3, 3, (4, 4096), dtype=torch.int32, device=dev)
    add("dtype_int32", "x 为 int32（非浮点 dtype）",
        lambda: ext.forward(variant, xi), expect_msg_contains="float16 / float32")

    # 4) CPU 张量
    x_cpu = torch.randn(4, 4096, dtype=torch.float16)
    add("x_cpu", "x 在 CPU",
        lambda: ext.forward(variant, x_cpu), expect_msg_contains="CUDA 张量")

    # 5) 非连续
    x_t = torch.randn(4096, 4, dtype=torch.float16, device=dev).t()
    add("x_transpose_view", "x 为 (H,M).t() 转置视图（非连续）",
        lambda: ext.forward(variant, x_t), expect_msg_contains="连续内存")
    x_str = torch.randn(4, 8192, dtype=torch.float16, device=dev)[::2]
    add("x_row_stride", "x 为隔行切片 [::2]（非连续）",
        lambda: ext.forward(variant, x_str), expect_msg_contains="连续内存")

    # 6) forward_into 的 out 错误
    x = torch.randn(4, 4096, dtype=torch.float16, device=dev)
    add("out_cpu", "out 在 CPU, x 在 CUDA（设备错误）",
        lambda: ext.forward_into(variant, x,
                                 torch.empty(4, 4096, dtype=torch.float16)),
        expect_msg_contains="CUDA 张量")
    add("out_wrong_shape", "out 形状 (4, 4095) ≠ x (4, 4096)",
        lambda: ext.forward_into(variant, x,
                                 torch.empty(4, 4095, dtype=torch.float16,
                                             device=dev)),
        expect_msg_contains="形状")
    add("out_wrong_dtype", "out=fp32, x=fp16",
        lambda: ext.forward_into(variant, x,
                                 torch.empty(4, 4096, dtype=torch.float32,
                                             device=dev)),
        expect_msg_contains="dtype")
    add("out_transpose_view", "out 为 (H,M).t() 转置视图（非连续）",
        lambda: ext.forward_into(variant, x,
                                 torch.empty(4096, 4, dtype=torch.float16,
                                             device=dev).t()),
        expect_msg_contains="连续内存")

    # 7) 多设备（仅当可见 >1 个 GPU 时执行；CUDA_VISIBLE_DEVICES=0
    #    的 v0.3 环境中安全跳过并记录）
    if torch.cuda.device_count() > 1:
        x0, o1 = x.to("cuda:0"), torch.empty(4, 4096, dtype=torch.float16,
                                             device="cuda:1")
        add("x_out_different_device",
            "x 在 cuda:0, out 在 cuda:1",
            lambda: ext.forward_into(variant, x0, o1),
            expect_msg_contains="同一 CUDA 设备")
    else:
        cases.append(dict(id="x_out_different_device", variant=variant,
                          description="x/out 不同 CUDA 设备（本环境仅 1 个可见 "
                                      "GPU，CUDA_VISIBLE_DEVICES=0，安全跳过）",
                          call=None, expected="skip",
                          expect_msg_contains=None))

    # 8) 对齐说明 control: baseline 为标量访存，无对齐契约 ——
    #    连续但基址偏移 4 元素（8B，未 16B 对齐）的 storage offset
    #    视图是合法输入，必须成功（验证 baseline 不误拒）。
    big_x = torch.randn(4 * 4096 + 4, dtype=torch.float16, device=dev)
    xo = big_x[4:4 + 4 * 4096].view(4, 4096)
    add("align_offset_view_control",
        "control: 基址偏移 8B（未 16B 对齐）的连续视图; 所有变体均不得"
        "拒绝此合法输入（向量化变体须走其回退/可用路径）: 必须成功",
        lambda: ext.forward(variant, xo), expected="pass")

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
                               lambda: _post_check_ok(ext, variant))
        r["id"] = c["id"]
        results.append(r)

    summary = summarize_cases(results)
    doc = {
        "suite": SUITE_VERSION,
        "negative_suite_scope": "per-variant",
        "generated": _now_iso(),
        "note": "非法输入必须在 kernel launch 前被明确异常拒绝；"
                "post_check_ok 验证拒绝未污染 CUDA 上下文。"
                "baseline 为标量访存，无对齐契约（见 align_offset_view_control）；"
                "vec4 类向量化变体有对齐契约（H%4==0 ∧ 基址按向量宽度对齐），"
                "不满足时回退与 baseline 同源的标量内核，不得拒绝"
                "（同一 control 用例对其同样必须成功）。",
        "summary": summary,
        "cases": results,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    return doc
