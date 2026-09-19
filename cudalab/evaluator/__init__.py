"""CUDALab v0.3 — 通用 evaluator 核心（算子无关）。

包含（v0.2 方法学原样保留，仅按算子维度参数化）:
- stats:        round-level paired 统计 + bootstrap CI + DVFS guard（纯 CPU）
- decision:     KEEP/REJECT/NEUTRAL/UNSTABLE 固定决策规则（纯 CPU）
- gpu:          nvidia-smi 状态采样
- bench:        paired-streaming-v2 引擎（paired / matrix / winners / 保存）
- correctness:  固定容差指标计算（compute_metrics 等）
- negative:     negative suite 通用执行器（launch 前拒绝验证）
- experiment:   实验 ID/保存 + 按单元格的 SIGNIFICANT_WINNER /
                NO_UNIQUE_WINNER / UNSTABLE 分类
- profiler:     通用 ncu 集成（driver 源码与 kernel regex 由 adapter 提供）

注意: 本包 `__init__` 刻意只暴露纯 CPU 子模块（stats/decision），
避免纯 CPU 测试路径（tests/test_evaluator_cpu.py）被迫导入 torch。
GPU 相关子模块请显式导入: `from cudalab.evaluator import bench` 等。

算子 adapter（输入/launch/算法 IO/driver 源码）在 cudalab/operators/。
"""
from . import decision as decision
from . import stats as stats  # noqa: F401
from .stats import (  # noqa: F401
    BOOTSTRAP_SEED, BOOTSTRAP_N, CI_ALPHA,
    paired_speedups, summarize, bootstrap_ci,
    check_dvfs_pair, check_dvfs_matrix,
)
from .decision import (  # noqa: F401
    KEEP, REJECT, NEUTRAL, UNSTABLE, decide_v2, MIN_VALID_ROUNDS,
)

__all__ = [
    "stats", "decision",
    "BOOTSTRAP_SEED", "BOOTSTRAP_N", "CI_ALPHA",
    "paired_speedups", "summarize", "bootstrap_ci",
    "check_dvfs_pair", "check_dvfs_matrix",
    "KEEP", "REJECT", "NEUTRAL", "UNSTABLE", "decide_v2", "MIN_VALID_ROUNDS",
]
