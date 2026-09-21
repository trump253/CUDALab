"""CUDALab v0.3 兼容层 — 实现已移至 cudalab/evaluator/decision.py。

纯 CPU（无 GPU / torch 依赖），KEEP / REJECT / NEUTRAL / UNSTABLE
固定决策规则。此模块仅 re-export 公开 API，保持 v0.2 导入路径
（tests/test_evaluator_cpu.py 等）不变。
"""
from .evaluator.decision import (  # noqa: F401
    KEEP,
    REJECT,
    NEUTRAL,
    UNSTABLE,
    MIN_VALID_ROUNDS,
    KEEP_MEDIAN,
    REJECT_MEDIAN,
    KEEP_FASTER_FRAC,
    REJECT_FASTER_FRAC,
    decide_v2,
    FASTER,
    SLOWER,
    UNRESOLVED,
    statistical_relation,
)
