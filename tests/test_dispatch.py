#!/usr/bin/env python
"""v0.2 分发表单元测试（纯 CPU，无需 GPU / 构建）。

用法: $PYTHON tests/test_dispatch.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

from cudalab import dispatch


def test_measured_cells():
    cases = [
        # (M, H, dtype, expected)
        (1, 1024, torch.float16, "baseline"),
        (128, 1024, torch.float16, "baseline"),
        (1, 4096, torch.float16, "v3_wideblock"),
        (16, 4096, torch.float16, "v4_vec_reg"),
        (128, 4096, torch.float16, "v4_vec_reg"),
        (1024, 4096, torch.float16, "v4_vec_reg"),
        (128, 8192, torch.float16, "v2_reg"),
        (1, 1024, torch.float32, "baseline"),
        (128, 1024, torch.float32, "baseline"),
        (1, 4096, torch.float32, "baseline"),
        (16, 4096, torch.float32, "baseline"),
        (128, 4096, torch.float32, "v2_reg"),
        (1024, 4096, torch.float32, "v4_vec_reg"),
        (128, 8192, torch.float32, "v2_reg"),
    ]
    for M, H, dt, exp in cases:
        got = dispatch.select_variant(M, H, dt)
        assert got == exp, f"({M},{H},{dt}): got {got}, want {exp}"


def test_table_matches_selector():
    for (M, H, dname), (v, _why) in dispatch.TABLE.items():
        dt = torch.float16 if dname == "float16" else torch.float32
        assert dispatch.select_variant(M, H, dt) == v, (M, H, dname)


def test_unsupported_h_falls_back():
    # v4 不支持 H=512（H/256=2）；fp16 M>=16 的 fallback 也只在 V4_HS 内选 v4
    v = dispatch.select_variant(256, 512, torch.float16)
    assert v in ("v2_reg", "baseline"), v
    # H=300 任何优化变体都不支持 -> baseline
    assert dispatch.select_variant(128, 300, torch.float16) == "baseline"
    assert dispatch.select_variant(128, 300, torch.float32) == "baseline"


def test_unmeasured_extrapolation():
    # (1024,8192) 未实测：M>=16 fp16 H=8192 -> v2（外推）
    assert dispatch.select_variant(1024, 8192, torch.float16) == "v2_reg"
    assert dispatch.dispatch_info(1024, 8192, torch.float16)["evidence_source"] == "fallback"
    # 未实测 fp32 小 M -> baseline
    assert dispatch.select_variant(16, 8192, torch.float32) == "baseline"


def test_dispatch_info_audit():
    info = dispatch.dispatch_info(128, 4096, torch.float32)
    assert info["variant"] == "v2_reg"
    assert info["evidence_source"] == "measured-v0.2"
    assert "KEEP" in info["reason"]


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    main()
