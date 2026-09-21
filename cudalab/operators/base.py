"""CUDALab v0.3 — operator adapter 协议（算子与 evaluator 核心的边界）。

evaluator 核心（cudalab/evaluator/）不含任何算子代码；算子差异全部
通过 Operator 注入：

- `build()`                          构建/加载 PyTorch CUDA 扩展
- `variants(ext)`                    正常（可 dispatch）内核变体列表
                                     （被隔离变体不在其中，见
                                     `unsafe_variants(ext)`）
- `unsafe_variants(ext)`             被隔离变体: UNSAFE_HISTORICAL_
                                     EXPERIMENT / REJECTED /
                                     NOT_FOR_NORMAL_DISPATCH（仍注册，
                                     仅供显式历史审计入口）
- `make_bench_pool(M,H,dtype,mode,seed,pool_size)`
                                     预分配计时张量池（BenchPool；
                                     launch(ext, variant, i) 为一次
                                     计时启动，计时区域内绝不
                                     malloc / 随机数 / copy）
- `algorithmic_bytes(M,H,element_size)`
                                     算子最小有用 IO（逻辑算法流量；
                                     例如 Softmax = M*H*es*2 = 读 x
                                     一次 + 写 y 一次）
- `ncu_kernel_regex`                 ncu `-k regex:` 过滤
- `ncu_driver_source(variant,M,H)`   ncu 驱动脚本源码
- `pytorch_ref_latency(M,H,dtype,iters,batch)`
                                     PyTorch 实现的延迟参照
                                     （implementation context，不是
                                     决策依据 —— 决策始终是 candidate
                                     vs incumbent 的 paired 证据）
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..evaluator.bench import BenchPool  # noqa: F401  (re-export 给 adapter)


class Operator:
    """算子 adapter 协议。子类以类属性声明元数据:

    name / dtypes / bench_shapes / primary_target / ncu_kernel_regex /
    profiles_dir / bench_dir / experiments_dir
    """
    name: str                                   # "rmsnorm" | "softmax"
    dtypes: tuple                               # ("float16", "float32")
    bench_shapes: list                          # 基准矩阵 [(M, H), ...]
    primary_target: tuple                       # 主目标 (M, H)
    ncu_kernel_regex: str
    profiles_dir: Path
    bench_dir: Path
    experiments_dir: Path

    # ---- 必须由子类实现 ----
    def build(self):
        raise NotImplementedError

    def variants(self, ext) -> list[str]:
        """正常（可 dispatch）变体列表。所有正常基准 / 测试 / 剖析
        路径使用本列表；被隔离变体不在其中。"""
        return sorted(ext.variants())

    def unsafe_variants(self, ext) -> list[str]:
        """被隔离变体（NOT_FOR_NORMAL_DISPATCH）: 仍注册在扩展里，但
        已从默认 `variants()` 列表移除；显式命名调用（如
        `ext.forward(name, x)`）是受控的历史审计入口，不属于正常
        dispatch。默认实现: 扩展未提供 `quarantined_variants()`
        时返回空列表（rmsnorm 扩展无隔离变体）。"""
        fn = getattr(ext, "quarantined_variants", None)
        return sorted(fn()) if fn is not None else []

    def make_bench_pool(self, M: int, H: int, dtype: torch.dtype,
                        mode: str, seed: int, pool_size: int) -> BenchPool:
        raise NotImplementedError

    def algorithmic_bytes(self, M: int, H: int, element_size: int) -> int:
        raise NotImplementedError

    def ncu_driver_source(self, variant: str, M: int, H: int) -> str:
        raise NotImplementedError

    def pytorch_ref_latency(self, M: int, H: int, dtype: torch.dtype,
                            iters: int = 200, batch: int = 32) -> dict:
        raise NotImplementedError

    # ---- 测试套件钩子（统一 CLI 的 test / optimize 使用）----
    def experiment_prefix(self) -> str:
        """实验 ID 前缀（rmsnorm=EXP, softmax=SFM），绝不混用。"""
        raise NotImplementedError

    def run_correctness(self, ext, variant: str,
                        out_dir: Path | None = None) -> dict:
        """单变体完整正确性套件。返回 {all_pass, summary, saved}。"""
        raise NotImplementedError

    def run_negative(self, ext, variant: str | None = None,
                     out_dir: Path | None = None) -> dict:
        """negative suite。返回保存的 doc（含 summary）。

        套件范围语义（v0.5 merge review 2026-09-21 统一, 以 doc 的
        `negative_suite_scope` 字段为权威标记）:
        - **per-variant**（GEMV / Softmax / RoPE）: 套件本体按
          variant 参数化, 统一 CLI 传入受测变体, 实际测试 CLI 指定的
          候选（每个向量化候选的对齐契约 / 标量回退回归必须对该候选
          自身运行并归档）。默认变体存规范 `invalid_inputs.json`,
          其余变体存 `invalid_inputs_<variant>.json`。
        - **cross-variant**（RMSNorm）: 单跑设计, 同一组用例覆盖全部
          变体（用例自带 variant 字段）, variant 参数被忽略 —— 验证
          算子级共享契约, 不描述为 per-variant。

        out_dir（append-only 约定）: 显式指定时结果存到该目录（自动
        创建）; 默认按算子约定。历史 experiment artifact（main 已发布
        的记录, 如 experiments/{rmsnorm,softmax,rope}/correctness/
        下的 v0.2/v0.3/v0.4 目录）**不可变**, 新验证只追加到
        `experiments/regression/<版本>/`。
        """
        raise NotImplementedError
