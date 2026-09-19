"""CUDALab negative correctness suite（v0.2）。

目标: 非法输入必须在 **kernel launch 之前** 以明确异常被稳定、安全地
拒绝 —— 而不是静默算出错误结果，也不是产生异步 CUDA 运行时错误。

原则:
- 不制造危险的 OOB / 非法访问；所有用例依赖 validation 先行拒绝。
- 每个用例之后验证 CUDA 上下文仍然健康（synchronize + 控制 forward），
  确认拒绝没有污染后续运行。
- 对每个用例记录异常类型与消息；对关键用例（非法 H、对齐契约）额外
  断言消息来自我们自己的预启动 validation 文本，而非运行时错误。
- 结构化结果保存至 experiments/rmsnorm/correctness/v0.2/。
- 时间戳由程序生成（ISO 8601，带时区），不手填历史日期。
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Callable, Optional

import torch

from .reference import make_inputs

ROOT = Path(__file__).resolve().parent.parent
NEG_DIR = ROOT / "experiments" / "rmsnorm" / "correctness" / "v0.2"

SUITE_VERSION = "negative-v0.2"

# Finding A 回归集: H 不满足 H % 256 == 0。
# 其中 1025 (per=4) 与 4100 (per=16) 在 v0.1 会误入 switch 分支并
# 静默产生错误输出 —— 是本套件的核心回归用例。
INVALID_HS = [1023, 1025, 4095, 4097, 4100]
INVALID_H_VARIANTS = ["v2_reg", "v4_vec_reg"]


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _post_check_ok(ext, variant: str) -> bool:
    """拒绝之后上下文必须仍然健康: 同步 + 一次合法控制 forward。"""
    try:
        torch.cuda.synchronize()
        x, w = make_inputs(2, 4096, dtype=torch.float16, seed=99)
        y = ext.forward(variant, x, w, 1e-5)
        torch.cuda.synchronize()
        return bool(torch.isfinite(y.float()).all().item())
    except Exception:
        return False


def _run_case(ext, variant: str, description: str,
              call: Callable[[], None], expected: str,
              expect_msg_contains: Optional[str] = None) -> dict:
    torch.cuda.synchronize()
    status = "rejected"
    exc_type = None
    message = None
    try:
        call()
        status = "passed_without_exception"
    except Exception as e:  # noqa: BLE001 — 任何异常都算"拒绝"
        exc_type = type(e).__name__
        message = str(e).splitlines()[0][:300]

    rec = {
        "variant": variant,
        "description": description,
        "expected": expected,          # "reject" | "pass" | "skip"
        "status": status,              # "rejected" | "passed_without_exception" | "skipped"
        "exception_type": exc_type,
        "message": message,
        "expect_msg_contains": expect_msg_contains,
        "msg_match": (expect_msg_contains in message)
                     if (expect_msg_contains and message) else None,
        "post_check_ok": None,         # skip 用例不执行 post check
    }
    if expected == "skip":
        rec["status"] = "skipped"
        return rec
    rec["post_check_ok"] = _post_check_ok(ext, variant)
    if expected == "reject":
        rec["pass"] = (status == "rejected") and bool(rec["post_check_ok"]) \
            and (rec["msg_match"] is not False)
    else:  # expected == "pass"（对齐 control：合法输入不得被误拒）
        rec["pass"] = (status == "passed_without_exception") \
            and bool(rec["post_check_ok"])
    return rec


def build_cases(ext) -> list[dict]:
    """构造全部 negative 用例。每个 dict: id/variant/description/call/..."""
    dev = "cuda"
    cases: list[dict] = []

    def add(cid, variant, description, call, expected="reject",
            expect_msg_contains=None):
        cases.append(dict(id=cid, variant=variant, description=description,
                          call=call, expected=expected,
                          expect_msg_contains=expect_msg_contains))

    # 1) 非法 H（Finding A 回归）: v2/v4 必须预启动明确拒绝
    for H in INVALID_HS:
        for v in INVALID_H_VARIANTS:
            x, w = make_inputs(4, H, dtype=torch.float16, seed=1)
            add(f"H{H}_{v}", v,
                f"非法 H={H}（H % 256 != 0）: PER switch 不得静默误算",
                lambda x=x, w=w, v=v: ext.forward(v, x, w, 1e-5),
                expect_msg_contains="H % 256 == 0")

    # 2) w 错误
    x, w = make_inputs(4, 4096, dtype=torch.float16, seed=2)
    add("w_len_short", "baseline", "w 长度 H-1",
        lambda: ext.forward("baseline", x, w[:-1], 1e-5),
        expect_msg_contains="H 不匹配")
    add("w_dtype_mismatch", "baseline", "x=fp16, w=fp32",
        lambda: ext.forward("baseline", x, w.float(), 1e-5),
        expect_msg_contains="同 dtype")

    # 3) 设备错误
    x_cpu, w_cpu = x.cpu(), w.cpu()
    add("x_cpu", "baseline", "x 在 CPU, w 在 CUDA",
        lambda: ext.forward("baseline", x_cpu, w, 1e-5),
        expect_msg_contains="CUDA 张量")
    add("w_cpu", "baseline", "x 在 CUDA, w 在 CPU",
        lambda: ext.forward("baseline", x, w_cpu, 1e-5),
        expect_msg_contains="CUDA 张量")

    # 4) 非连续
    x_nc = torch.randn(4096, 4, dtype=torch.float16, device=dev).contiguous().t()
    w_nc = torch.randn(8192, dtype=torch.float16, device=dev)[::2]
    add("x_noncontig", "baseline", "x 为 (H,M).t() 转置视图（非连续）",
        lambda: ext.forward("baseline", x_nc, w, 1e-5),
        expect_msg_contains="连续内存")
    add("w_noncontig", "baseline", "w 为步长 2 切片（非连续，长度 H）",
        lambda: ext.forward("baseline", x, w_nc, 1e-5),
        expect_msg_contains="连续内存")

    # 5) forward_into 的 out 错误
    add("out_cpu", "baseline", "out 在 CPU, x 在 CUDA",
        lambda: ext.forward_into("baseline", x, w,
                                 torch.empty(4, 4096, dtype=torch.float16), 1e-5),
        expect_msg_contains="CUDA 张量")
    add("out_wrong_shape", "baseline", "out 形状 (4, 4095) ≠ x (4, 4096)",
        lambda: ext.forward_into("baseline", x, w,
                                 torch.empty(4, 4095, dtype=torch.float16, device=dev), 1e-5),
        expect_msg_contains="形状")
    add("out_wrong_dtype", "baseline", "out=fp32, x=fp16",
        lambda: ext.forward_into("baseline", x, w,
                                 torch.empty(4, 4096, dtype=torch.float32, device=dev), 1e-5),
        expect_msg_contains="dtype")
    add("out_noncontig", "baseline", "out 为 (H,M).t() 转置视图（非连续）",
        lambda: ext.forward_into("baseline", x, w,
                                 torch.empty(4096, 4, dtype=torch.float16, device=dev).t(), 1e-5),
        expect_msg_contains="连续内存")

    # 6) 不支持的 dtype
    xb, wb = x.bfloat16(), w.bfloat16()
    add("dtype_bfloat16", "baseline", "x/w 为 bfloat16（不支持）",
        lambda: ext.forward("baseline", xb, wb, 1e-5),
        expect_msg_contains="float16 / float32")

    # 7) eps
    add("eps_nan", "baseline", "eps = NaN",
        lambda: ext.forward("baseline", x, w, float("nan")),
        expect_msg_contains="eps")
    add("eps_negative", "baseline", "eps = -1.0",
        lambda: ext.forward("baseline", x, w, -1.0),
        expect_msg_contains="eps")

    # 8) 对齐契约（Finding D）: storage offset 破坏 16B 对齐
    #    big[4:] 基址偏移 4 元素 = 8 字节（fp16）→ 未 16B 对齐，
    #    但张量本身连续（1D 切片 + view），先过 contiguity 检查，
    #    必须被对齐契约拒绝。
    big_x = torch.randn(4 * 4096 + 4, dtype=torch.float16, device=dev)
    xm = big_x[4:4 + 4 * 4096].view(4, 4096)
    big_w = torch.randn(4096 + 4, dtype=torch.float16, device=dev)
    wm = big_w[4:]
    for v in ("v1_vec", "v4_vec_reg"):
        add(f"align_misaligned_{v}", v,
            "连续张量但基址偏移 8B（storage offset 视图）: 未 16B 对齐",
            lambda xm=xm, wm=wm, v=v: ext.forward(v, xm, wm, 1e-5),
            expect_msg_contains="对齐契约")
    # 对齐 control: 偏移 8 元素 = 16B → 合法，必须成功（验证不误拒）
    big_x2 = torch.randn(4 * 4096 + 8, dtype=torch.float16, device=dev)
    xa = big_x2[8:8 + 4 * 4096].view(4, 4096)
    big_w2 = torch.randn(4096 + 8, dtype=torch.float16, device=dev)
    wa = big_w2[8:]
    for v in ("v1_vec", "v4_vec_reg"):
        add(f"align_ok_control_{v}", v,
            "control: 基址偏移 16B（仍 16B 对齐）的连续视图: 必须成功",
            lambda xa=xa, wa=wa, v=v: ext.forward(v, xa, wa, 1e-5),
            expected="pass")

    # 8b) v0.2.1 Finding 2 回归: v4 FP32 H=1024（PER=4）恒用 float4 加载，
    #     需 16B 对齐。旧代码按 per%8 判定，FP32 PER=4 被误设为 4B，会放行
    #     未对齐的 float4 加载（静默错误/崩溃）。修正后：
    #     - 基址偏移 1 float（4B，连续）→ 未 16B 对齐，必须被对齐契约拒绝；
    #     - 基址偏移 4 floats（16B 对齐，连续）→ 必须成功。
    M32, H32 = 4, 1024
    big_x32_bad = torch.randn(M32 * H32 + 1, dtype=torch.float32, device=dev)
    xm32 = big_x32_bad[1:1 + M32 * H32].view(M32, H32)   # 基址 +4B: 4B 对齐、未 16B
    big_w32_bad = torch.randn(H32 + 1, dtype=torch.float32, device=dev)
    wm32 = big_w32_bad[1:]                               # 基址 +4B: 4B 对齐、未 16B
    add("align_fp32_h1024_misaligned_v4", "v4_vec_reg",
        "v4 FP32 H=1024（PER=4, float4）: 连续张量但基址偏移 1 float"
        "（4B，未 16B 对齐）: 必须被对齐契约拒绝",
        lambda: ext.forward("v4_vec_reg", xm32, wm32, 1e-5),
        expect_msg_contains="对齐契约")
    big_x32_ok = torch.randn(M32 * H32 + 4, dtype=torch.float32, device=dev)
    xa32 = big_x32_ok[4:4 + M32 * H32].view(M32, H32)    # 基址 +16B: 16B 对齐
    big_w32_ok = torch.randn(H32 + 4, dtype=torch.float32, device=dev)
    wa32 = big_w32_ok[4:]                                # 基址 +16B: 16B 对齐
    add("align_fp32_h1024_aligned_v4", "v4_vec_reg",
        "v4 FP32 H=1024（PER=4, float4）control: 基址偏移 4 floats"
        "（16B 对齐）的连续视图: 必须成功",
        lambda: ext.forward("v4_vec_reg", xa32, wa32, 1e-5),
        expected="pass")

    # 9) 多设备（仅当可见 >1 个 GPU 时执行；CUDA_VISIBLE_DEVICES=0
    #    的 v0.2 环境中安全跳过并记录）
    if torch.cuda.device_count() > 1:
        x0, w1 = x.to("cuda:0"), w.to("cuda:1")
        add("x_w_different_device", "baseline",
            "x 在 cuda:0, w 在 cuda:1",
            lambda: ext.forward("baseline", x0, w1, 1e-5),
            expect_msg_contains="同一 CUDA 设备")
    else:
        cases.append(dict(id="x_w_different_device", variant="baseline",
                          description="x/w 不同 CUDA 设备（本环境仅 1 个可见 GPU，"
                                      "CUDA_VISIBLE_DEVICES=0，安全跳过）",
                          call=None, expected="skip",
                          expect_msg_contains=None))
    return cases


def run_negative_suite(ext, out_path: Path | None = None) -> dict:
    """运行全部 negative 用例并保存结构化结果。"""
    if out_path is None:
        out_path = NEG_DIR / "invalid_inputs.json"
    cases = build_cases(ext)
    results = []
    for c in cases:
        if c["expected"] == "skip":
            r = _run_case(ext, c["variant"], c["description"],
                          lambda: None, "skip", None)
            r["id"] = c["id"]
        else:
            r = _run_case(ext, c["variant"], c["description"], c["call"],
                          c["expected"], c["expect_msg_contains"])
            r["id"] = c["id"]
        results.append(r)

    rejected_expected = [r for r in results if r["expected"] == "reject"]
    passed_expected = [r for r in results if r["expected"] == "pass"]
    skipped = [r for r in results if r["expected"] == "skip"]
    n_ok = sum(1 for r in results if r.get("pass"))
    summary = {
        "n_total": len(results),
        "n_reject_expected": len(rejected_expected),
        "n_reject_ok": sum(1 for r in rejected_expected if r.get("pass")),
        "n_pass_expected": len(passed_expected),
        "n_pass_ok": sum(1 for r in passed_expected if r.get("pass")),
        "n_skipped": len(skipped),
        "n_passed": n_ok,
        # 所有非 skip 用例 pass 即为 all_pass
        "all_pass": all(r.get("pass") for r in results
                        if r["expected"] != "skip"),
    }
    doc = {
        "suite": SUITE_VERSION,
        "generated": _now_iso(),
        "note": "非法输入必须在 kernel launch 前被明确异常拒绝；"
                "post_check_ok 验证拒绝未污染 CUDA 上下文。",
        "summary": summary,
        "cases": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    return doc
