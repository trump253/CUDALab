#!/usr/bin/env python
"""v0.2.1 分发表单元测试（纯 CPU，无需 GPU / 构建）。

证据政策（review Finding 4）：evidence > coverage ——
仅 paired 证据格（(128,4096) fp32、(128,8192) fp16 → v2_reg）与
显式 incumbent 格（(128,4096) fp16 → v4_vec_reg）路由优化变体；
matrix-only / hot-streaming 冲突 / 未实测一律 baseline。

用法: $PYTHON tests/test_dispatch.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

from cudalab import dispatch


def test_routed_cells():
    cases = [
        # (M, H, dtype, expected)
        # paired-evidence → v2_reg
        (128, 4096, torch.float32, "v2_reg"),
        (128, 8192, torch.float16, "v2_reg"),
        # incumbent-fallback（NO_UNIQUE_WINNER 主形状）
        (128, 4096, torch.float16, "v4_vec_reg"),
        # matrix-only（无 paired 验证）→ baseline
        (1, 1024, torch.float16, "baseline"),
        (128, 1024, torch.float16, "baseline"),
        (1, 4096, torch.float16, "baseline"),
        (16, 4096, torch.float16, "baseline"),
        (1024, 4096, torch.float16, "baseline"),
        (1, 1024, torch.float32, "baseline"),
        (128, 1024, torch.float32, "baseline"),
        (1, 4096, torch.float32, "baseline"),
        (16, 4096, torch.float32, "baseline"),
        (1024, 4096, torch.float32, "baseline"),
        (128, 8192, torch.float32, "baseline"),
        # 未实测 → baseline（不外推）
        (1024, 8192, torch.float16, "baseline"),
        (16, 8192, torch.float32, "baseline"),
        (512, 2048, torch.float16, "baseline"),
    ]
    for M, H, dt, exp in cases:
        got = dispatch.select_variant(M, H, dt)
        assert got == exp, f"({M},{H},{dt}): got {got}, want {exp}"


def test_table_matches_selector():
    for (M, H, dname), (v, _src, _why) in dispatch.TABLE.items():
        dt = torch.float16 if dname == "float16" else torch.float32
        assert dispatch.select_variant(M, H, dt) == v, (M, H, dname)


def test_evidence_sources():
    # paired-evidence
    for M, H, dt in [(128, 4096, torch.float32), (128, 8192, torch.float16)]:
        info = dispatch.dispatch_info(M, H, dt)
        assert info["evidence_source"] == "paired-evidence", (M, H)
        assert info["variant"] == "v2_reg", (M, H)
    # incumbent-fallback（主形状 NO_UNIQUE_WINNER）
    info = dispatch.dispatch_info(128, 4096, torch.float16)
    assert info["evidence_source"] == "incumbent-fallback"
    assert info["variant"] == "v4_vec_reg"
    assert "NO_UNIQUE_WINNER" in info["reason"]
    # matrix-only（实测单元格、无 paired 证据 → baseline）
    for M, H, dt in [(1024, 4096, torch.float16), (1, 4096, torch.float16),
                     (128, 8192, torch.float32), (128, 1024, torch.float16)]:
        info = dispatch.dispatch_info(M, H, dt)
        assert info["evidence_source"] == "matrix-only", (M, H)
        assert info["variant"] == "baseline", (M, H)
    # baseline-fallback（未实测）
    for M, H, dt in [(1024, 8192, torch.float16), (16, 8192, torch.float32),
                     (512, 2048, torch.float16)]:
        info = dispatch.dispatch_info(M, H, dt)
        assert info["evidence_source"] == "baseline-fallback", (M, H)
        assert info["variant"] == "baseline", (M, H)


def test_hot_streaming_conflict_not_claimed_stable():
    # (16,4096) fp16：矩阵 hot winner=v4 / streaming winner=v1（冲突）
    # → 必须路由 baseline，不得声称 v4 稳定
    info = dispatch.dispatch_info(16, 4096, torch.float16)
    assert info["variant"] == "baseline"
    assert info["evidence_source"] == "matrix-only"
    assert "冲突" in info["reason"]


def test_no_unmeasured_extrapolation():
    # 旧版外推：fp16 M>=16 H<8192 → v4；fp32 M>=128 H∈{4096,8192} → v2。
    # v0.2.1：未实测一律 baseline。
    assert dispatch.select_variant(1024, 8192, torch.float16) == "baseline"
    assert dispatch.select_variant(1024, 2048, torch.float16) == "baseline"
    assert dispatch.select_variant(512, 4096, torch.float16) == "baseline"
    assert dispatch.select_variant(512, 4096, torch.float32) == "baseline"
    assert dispatch.select_variant(512, 8192, torch.float32) == "baseline"
    # 实测但 matrix-only 的 fp32 大 shape 同样不外推
    assert dispatch.select_variant(1024, 4096, torch.float32) == "baseline"


def test_unsupported_h_falls_back():
    # H=300：任何优化变体都不支持，且未实测 → baseline
    assert dispatch.select_variant(128, 300, torch.float16) == "baseline"
    assert dispatch.select_variant(128, 300, torch.float32) == "baseline"
    # H=512 未实测：即使 v2 支持也不路由（无 paired 证据）
    assert dispatch.select_variant(256, 512, torch.float16) == "baseline"


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    main()
