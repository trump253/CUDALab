"""CUDALab v0.2 — 非法输入 negative suite 入口。

运行:
    source tools/env.sh
    $PYTHON tests/test_invalid_inputs.py

对全部 negative 用例验证: 非法输入必须在 kernel launch 之前被明确
异常拒绝（而非静默计算），且拒绝之后 CUDA 上下文仍然健康。
结构化结果保存至 experiments/rmsnorm/correctness/v0.2/invalid_inputs.json。
退出码: 0 = 全部符合预期, 1 = 存在不符合预期的用例。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cudalab.build import build          # noqa: E402
from cudalab.negative_suite import run_negative_suite  # noqa: E402


def main() -> int:
    ext = build()
    doc = run_negative_suite(ext)
    s = doc["summary"]
    print(f"[negative suite v0.2] {s['n_passed']}/{s['n_total']} 用例符合预期"
          f"（skip {s['n_skipped']}）")
    for r in doc["cases"]:
        if not r.get("pass") and r["expected"] != "skip":
            print(f"  失败: {r['id']} ({r['variant']}): "
                  f"status={r['status']} msg={r['message']!r}")
        if r["expected"] == "skip":
            print(f"  跳过: {r['id']}: {r['description']}")
    print("all_pass:", s["all_pass"])
    return 0 if s["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
