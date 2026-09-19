"""CUDALab v0.3 兼容层 — 实现已移至 cudalab/evaluator/stats.py。

纯 CPU（无 GPU / torch 依赖），round-level paired 统计 + bootstrap CI
+ DVFS guard。此模块仅 re-export 公开 API，保持 v0.2 导入路径
（tests/test_evaluator_cpu.py、cudalab/decision.py 等）不变。
"""
from .evaluator.stats import (  # noqa: F401
    BOOTSTRAP_SEED,
    BOOTSTRAP_N,
    CI_ALPHA,
    paired_speedups,
    summarize,
    bootstrap_ci,
    check_dvfs_pair,
    check_dvfs_matrix,
)
