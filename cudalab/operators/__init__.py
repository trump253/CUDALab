"""CUDALab v0.3 — operator adapter 注册表。

`get(name)` 返回算子单例。注意: 导入本包会导入 torch（adapter 是 GPU
路径）；纯 CPU 测试请只导入 cudalab/evaluator 的 stats/decision。
"""
from __future__ import annotations

from .rmsnorm import rmsnorm as _rmsnorm
from .softmax import softmax as _softmax

_REGISTRY = {op.name: op for op in (_rmsnorm, _softmax)}


def get(name: str):
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"未知 operator {name!r}; 可用: {sorted(_REGISTRY)}")


def names() -> list[str]:
    return sorted(_REGISTRY)
