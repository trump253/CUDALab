"""CUDALab v0.3 兼容层 — 通用 NCU 核心在 cudalab/evaluator/profiler.py。

本模块保持 v0.2 公开 API 不变（绑定 RMSNorm operator 的 driver 源码
与 kernel regex），`scripts/profile_v2.py` / `scripts/profile_rmsnorm.py`
/ `scripts/optimize_rmsnorm.py` 不受影响。

NCU 方法学说明（--cache-control 语义 v0.2.1 修正、--clock-control、
指标单位、停顿原因解读）见 `cudalab/evaluator/profiler.py` 模块头。
"""
from __future__ import annotations

from pathlib import Path

from .evaluator.profiler import (  # noqa: F401
    NCU,
    PYTHON_REAL,
    METRICS,
    ncu_version,
    profile_variant as _profile_variant,
)
from .operators import get as _get_op  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROF_DIR = ROOT / "profiles" / "rmsnorm"


def profile_variant(variant: str, M: int = 128, H: int = 4096,
                    out_path: Path | None = None, ncu: str = NCU,
                    launch_skip: int = 2, launch_count: int = 4,
                    cache_control: str = "all",
                    clock_control: str = "base") -> dict:
    """剖析 RMSNorm 单个变体（v0.2 签名）。

    cache_control / clock_control 语义见 cudalab/evaluator/profiler.py。
    """
    op = _get_op("rmsnorm")
    if out_path is None:
        out_path = PROF_DIR / f"{variant}_M{M}_H{H}.json"
    return _profile_variant(
        variant, M, H,
        driver_src=op.ncu_driver_source(variant, M, H),
        kernel_regex=op.ncu_kernel_regex,
        out_path=out_path, ncu=ncu,
        launch_skip=launch_skip, launch_count=launch_count,
        cache_control=cache_control, clock_control=clock_control)


if __name__ == "__main__":
    import json
    import sys
    v = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    s = profile_variant(v, 128, 4096)
    print(json.dumps(s, indent=2))
