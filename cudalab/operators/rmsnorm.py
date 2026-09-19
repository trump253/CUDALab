"""CUDALab v0.3 — RMSNorm operator adapter（v0.1/v0.2 既有实现接入核心）。

输入生成 / 参考实现复用 `cudalab/reference.py`（显式公式参考，
与 PyTorch 版本无关）。计时池构造与 v0.2 `_make_pool` 逐字节等价
（同一 seed 下生成完全相同的张量），launch 为
`ext.forward_into(variant, x, w, out, 1e-5)`。

算法 IO（最小有用流量）: 读 x 一次 + 读 w 一次 + 写 y 一次
= (M*H + H + M*H) * element_size。
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..evaluator.bench import BenchPool, SEED
from ..evaluator.gpu import now_iso  # noqa: F401  (re-export 兼容)
from .base import Operator

ROOT = Path(__file__).resolve().parents[2]

# v0.2 基准矩阵（与 v0.1 相同的 7 个 shape；主目标 (128, 4096)）
BENCH_MATRIX_V2 = [
    (1, 4096),
    (16, 4096),
    (128, 4096),
    (1024, 4096),
    (128, 8192),
    (1, 1024),
    (128, 1024),
]
PRIMARY_TARGET = (128, 4096)


class RMSNormOperator(Operator):
    name = "rmsnorm"
    dtypes = ("float16", "float32")
    bench_shapes = BENCH_MATRIX_V2
    primary_target = PRIMARY_TARGET
    ncu_kernel_regex = "rmsnorm"
    profiles_dir = ROOT / "profiles" / "rmsnorm"
    bench_dir = ROOT / "benchmarks" / "v0.2"
    experiments_dir = ROOT / "experiments" / "rmsnorm"

    def build(self):
        from ..build import build
        return build()

    def make_bench_pool(self, M: int, H: int, dtype: torch.dtype,
                        mode: str, seed: int = SEED,
                        pool_size: int = 16) -> BenchPool:
        """预分配全部计时张量（计时区域内永不 malloc / 随机数 / copy）。

        与 v0.2 `_make_pool` 等价：同一 seed 下 xs/w 生成顺序不变。
        """
        from ..reference import DEFAULT_EPS
        dev = "cuda"
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        n = pool_size if mode == "streaming" else 1
        xs = [(torch.randn(M, H, generator=g, dtype=torch.float32, device=dev)
               .to(dtype).contiguous()) for _ in range(n)]
        w = (torch.randn(H, generator=g, dtype=torch.float32, device=dev) * 0.5 + 1.0
             ).to(dtype).contiguous()
        outs = [torch.empty_like(xs[0]) for _ in xs]
        es = 2 if dtype == torch.float16 else 4
        per = M * H * es
        working_set = len(xs) * 2 * per + H * es  # x pool + out pool + w

        def launch(ext, variant: str, i: int):
            ext.forward_into(variant, xs[i], w, outs[i], DEFAULT_EPS)

        return BenchPool(xs=xs, outs=outs, pool_size=len(xs),
                         working_set_bytes=working_set, element_size=es,
                         mode=mode, launch=launch)

    def algorithmic_bytes(self, M: int, H: int, element_size: int) -> int:
        return (M * H + H + M * H) * element_size

    def ncu_driver_source(self, variant: str, M: int, H: int) -> str:
        return f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import torch
from cudalab.build import build

ext = build()
assert "{variant}" in ext.variants(), ext.variants()
g = torch.Generator(device="cuda"); g.manual_seed(1234)
x = torch.randn({M}, {H}, generator=g, dtype=torch.float32, device="cuda").half().contiguous()
w = (torch.randn({H}, generator=g, dtype=torch.float32, device="cuda") * 0.5 + 1.0).half().contiguous()
out = torch.empty_like(x)
# 2 次不计时的预热启动（ncu --launch-skip 2），之后 4 次被剖析
for _ in range(6):
    ext.forward_into("{variant}", x, w, out, 1e-5)
torch.cuda.synchronize()
print("profile driver done")
"""

    def experiment_prefix(self) -> str:
        return "EXP"

    def run_correctness(self, ext, variant: str,
                        out_dir: Path | None = None) -> dict:
        """完整正确性套件。默认输出到 v0.3 回归目录 ——
        v0.2 的已发布产物（correctness/v0.2/*.json）绝不覆盖。"""
        from ..correctness import run_suite, summarize, save_results
        if out_dir is None:
            out_dir = self.experiments_dir / "correctness" / "v0.3_regression"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = run_suite(variant, ext)
        saved = save_results(results, out_dir / f"{variant}.json")
        s = summarize(results)
        return {"all_pass": s["all_pass"], "summary": s, "saved": str(saved)}

    def run_negative(self, ext) -> dict:
        from ..negative_suite import run_negative_suite
        # 默认输出到 v0.3 回归目录（v0.2 产物不覆盖；内容 schema 一致，
        # 仅 generated 时间戳不同）。
        out = self.experiments_dir / "correctness" / "v0.3_regression" \
            / "invalid_inputs.json"
        return run_negative_suite(ext, out_path=out)

    def pytorch_ref_latency(self, M: int, H: int, dtype: torch.dtype,
                            iters: int = 200, batch: int = 32) -> dict:
        """PyTorch 官方实现的延迟参照（torch 2.4.1 有 F.rms_norm）。

        注意: 这是 "PyTorch implementation context"，不是公平 fused-kernel
        baseline 对比（PyTorch 路径可能含额外 kernel/内存操作）。
        """
        from ..evaluator.bench import time_call, SEED as _SEED
        if isinstance(dtype, str):
            dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
        try:
            from torch.nn.functional import rms_norm
        except ImportError:
            return {"available": False,
                    "note": "torch.nn.functional.rms_norm 不存在"}
        # 输入生成与 v0.2 pytorch_ref_latency 逐字节等价（seed 相同、
        # 抽取顺序 x→w 不变）。
        dev = "cuda"
        g = torch.Generator(device=dev)
        g.manual_seed(_SEED)
        x = (torch.randn(M, H, generator=g, dtype=torch.float32, device=dev)
             .to(dtype).contiguous())
        w = (torch.randn(H, generator=g, dtype=torch.float32, device=dev) * 0.5 + 1.0
             ).to(dtype).contiguous()

        def _call():
            # torch 2.4.1: F.rms_norm(input, normalized_shape, weight, eps)
            return rms_norm(x, (H,), w, 1e-5)

        try:
            _call()
            torch.cuda.synchronize()
        except Exception as e:
            return {"available": True, "error": f"{type(e).__name__}: {e}",
                    "note": "F.rms_norm 在该 dtype/device 上失败；不伪造数字"}
        t = time_call(_call, warmup=100, iters=iters, batch=batch)
        return {"available": True,
                "api": "torch.nn.functional.rms_norm(input, normalized_shape, weight, eps)",
                "eps": 1e-5,
                **t,
                "note": "PyTorch implementation context（非公平 fused-kernel 对比）"}


rmsnorm = RMSNormOperator()
