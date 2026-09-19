"""CUDALab v0.2 paired benchmark harness（paired-streaming-v2）— v0.3 兼容层。

通用引擎已移至 `cudalab/evaluator/bench.py`（算子无关）；RMSNorm 差异
在 `cudalab/operators/rmsnorm.py`。本模块保持 v0.2 公开 API 不变
（绑定 RMSNorm operator），`scripts/bench_v2.py` 与历史引用不受影响。

v0.2 方法学说明（paired 测量 / 顺序去偏 / DVFS guard / cache mode /
algorithmic bandwidth / round-level 统计）见
`cudalab/evaluator/bench.py` 模块头。
"""
from __future__ import annotations

import torch
from pathlib import Path

from .evaluator.bench import (  # noqa: F401
    HARNESS_VERSION,
    WARMUP,
    ITERS,
    BATCH,
    ROUNDS,
    MAX_RETRIES,
    MIN_VALID_ROUNDS,
    DVFS_TOL,
    POOL_SIZE,
    SEED,
    L2_BYTES,
    BenchPool,
    measure_block,
    bench_pair as _bench_pair,
    bench_matrix as _bench_matrix,
    analyze_shape_winners,
    save_record,
)
from .evaluator.stats import check_dvfs_pair, check_dvfs_matrix  # noqa: E402,F401
from .operators import get as _get_op  # noqa: E402
from .operators.rmsnorm import BENCH_MATRIX_V2, PRIMARY_TARGET  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks" / "v0.2"


def bench_pair(parent: str, candidate: str, M: int, H: int,
               dtype: torch.dtype = torch.float16, mode: str = "streaming",
               rounds: int = ROUNDS, warmup: int = WARMUP,
               iters: int = ITERS, batch: int = BATCH,
               max_retries: int = MAX_RETRIES, seed: int = SEED,
               ext=None) -> dict:
    """RMSNorm paired benchmark（v0.2 签名，绑定 RMSNorm adapter）。"""
    if ext is None:
        from .build import build
        ext = build()
    return _bench_pair(_get_op("rmsnorm"), ext, parent, candidate, M, H,
                       dtype, mode=mode, rounds=rounds, warmup=warmup,
                       iters=iters, batch=batch, max_retries=max_retries,
                       seed=seed)


def bench_matrix(variants: list[str], M: int, H: int,
                 dtype: torch.dtype = torch.float16, mode: str = "streaming",
                 rounds: int = ROUNDS, warmup: int = WARMUP,
                 iters: int = ITERS, batch: int = BATCH,
                 max_retries: int = MAX_RETRIES, seed: int = SEED,
                 ext=None) -> dict:
    """RMSNorm 全矩阵 round-robin（v0.2 签名，绑定 RMSNorm adapter）。"""
    if ext is None:
        from .build import build
        ext = build()
    return _bench_matrix(_get_op("rmsnorm"), ext, variants, M, H,
                         dtype, mode=mode, rounds=rounds, warmup=warmup,
                         iters=iters, batch=batch, max_retries=max_retries,
                         seed=seed)


def pytorch_ref_latency(M: int, H: int, dtype: torch.dtype = torch.float16,
                        iters: int = 200, batch: int = 32) -> dict:
    return _get_op("rmsnorm").pytorch_ref_latency(M, H, dtype, iters, batch)
