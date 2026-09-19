"""CUDALab v0.3 — negative suite 通用执行器（算子无关）。

从 v0.2 `cudalab/negative_suite.py` 抽出的共享机制：

- 非法输入必须在 **kernel launch 之前** 以明确异常被稳定、安全地拒绝
  —— 而不是静默算出错误结果，也不是产生异步 CUDA 运行时错误。
- 每个用例之后验证 CUDA 上下文仍然健康（`control_ok` 回调：同步 +
  一次合法控制 forward），确认拒绝没有污染后续运行。
- 对每个用例记录异常类型与消息；对关键用例额外断言消息来自我们
  自己的预启动 validation 文本（`expect_msg_contains`）。
- 结构化结果（summary + 全部用例）由调用方保存。

算子特有的用例表留在各自模块（cudalab/negative_suite.py = RMSNorm，
cudalab/softmax_negative.py = Softmax）。
"""
from __future__ import annotations

from typing import Callable, Optional

import torch


def run_case(variant: str, description: str,
             call: Callable[[], None], expected: str,
             expect_msg_contains: Optional[str],
             control_ok: Callable[[], bool]) -> dict:
    """执行单个 negative 用例。

    expected: "reject" | "pass" | "skip"
    control_ok: 仅非 skip 用例使用 —— 拒绝/放行之后上下文必须健康。
    """
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
    rec["post_check_ok"] = bool(control_ok())
    if expected == "reject":
        rec["pass"] = (status == "rejected") and bool(rec["post_check_ok"]) \
            and (rec["msg_match"] is not False)
    else:  # expected == "pass"（对齐 control：合法输入不得被误拒）
        rec["pass"] = (status == "passed_without_exception") \
            and bool(rec["post_check_ok"])
    return rec


def summarize_cases(results: list[dict]) -> dict:
    rejected_expected = [r for r in results if r["expected"] == "reject"]
    passed_expected = [r for r in results if r["expected"] == "pass"]
    skipped = [r for r in results if r["expected"] == "skip"]
    n_ok = sum(1 for r in results if r.get("pass"))
    return {
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
