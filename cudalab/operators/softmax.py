"""CUDALab v0.3 — Softmax operator adapter。

算子: 行-wise Softmax（仅最后一维），数值稳定形式:

    m_i = max_j x[i, j]
    y[i, j] = exp(x[i, j] - m_i) / sum_k exp(x[i, k] - m_i)

中间量（max / exp / sum / normalize）全部 FP32，输出与 x 同 dtype。
**不支持 BF16**（v0.3 范围: FP16 优先，FP32 自然支持）。

v0.3.1 隔离（quarantine）: `softmax_hsplit2`（SFM-0004）被标记为
UNSAFE_HISTORICAL_EXPERIMENT / REJECTED / NOT_FOR_NORMAL_DISPATCH ——
其跨 block spin-wait 合并依赖 CUDA 调度模型不保证的 block 并发驻留
假设，且 HsGlobal scratch 为进程级共享状态（多 stream / 多 device
并发 race 风险）。它不在 `variants(ext)` 正常列表中；正常基准 / 测试
/ 剖析路径均拒绝它。内核源码与全部 SFM-0004 数据保留（历史证据）；
显式 `ext.forward("softmax_hsplit2", x)` 仍是受控历史审计入口。详见
experiments/softmax/SFM-0004.md 与 docs/report_v0.3_result.md。

参考实现（显式、FP32 内部、与 PyTorch 版本无关）:

    softmax_ref(x) = torch.softmax(x.float(), dim=-1).to(x.dtype)

注意 PyTorch 自身 `torch.softmax(x, dim=-1)` 的 dtype 语义: 对 FP16
输入，CUDA SoftMax kernel 使用 FP32 累加器（acc_type<half>=float）并
返回 FP16，数值上与参考一致；但它是 PyTorch 的融合实现，只作
implementation context，不是决策依据（决策 = candidate vs incumbent
的 paired 证据）。

算法 IO（最小有用流量）: 读 x 一次 + 写 y 一次
= M * H * element_size * 2。（内核内部多趟读取的流量**不是**算法
流量，需要时在实验中以不同指标名单独报告；真实 DRAM 行为以 NCU 为准。）
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..evaluator.bench import BenchPool, SEED
from .base import Operator

ROOT = Path(__file__).resolve().parents[2]

# v0.3 Softmax 基准矩阵（用户指定，至少覆盖）
BENCH_MATRIX_SFM = [
    (1, 128),
    (1, 1024),
    (1, 4096),
    (32, 1024),
    (128, 1024),
    (128, 4096),
    (1024, 1024),
    (1024, 4096),
    (128, 8192),
]
PRIMARY_TARGET = (128, 4096)


def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    """参考 Softmax（行-wise，FP32 内部，输出与 x 同 dtype）。"""
    if x.dim() != 2:
        raise ValueError(f"期望 2 维 x (M, H)，实际 {x.dim()}D")
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def make_input(M: int, H: int, dtype: torch.dtype, device: str = "cuda",
               seed: int = 0, mode: str = "normal", scale: float = 1.0,
               constant: float = 3.0, dominant: float = 30.0,
               extreme_range: float = 80.0) -> torch.Tensor:
    """确定性输入生成器（正确性与基准两条路径共享）。

    mode:
      "normal"          N(0, scale)
      "tiny"            N(0, 1e-4)
      "biased"          N(0, scale) + 3.0
      "large_pos"       50 + N(0, 0.5)        （x ≈ +50）
      "large_neg"       -50 + N(0, 0.5)       （x ≈ -50）
      "mixed_extremes"  U(-80, 80)            （x ∈ [-80, 80]，
                                               exp(x) 在 fp16 下会
                                               溢出 —— 必须依赖
                                               max 减除才安全）
      "zeros"           全 0
      "constant"        全 constant
      "single_dominant" N(0,1)，每行一个元素 = dominant
      "alternating"     按列交替 +constant / -constant
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    if mode == "normal":
        x = torch.randn(M, H, generator=g, dtype=torch.float32, device=device) * scale
    elif mode == "tiny":
        x = torch.randn(M, H, generator=g, dtype=torch.float32, device=device) * 1e-4
    elif mode == "biased":
        x = torch.randn(M, H, generator=g, dtype=torch.float32, device=device) * scale + 3.0
    elif mode == "large_pos":
        x = torch.randn(M, H, generator=g, dtype=torch.float32, device=device) * 0.5 + 50.0
    elif mode == "large_neg":
        x = torch.randn(M, H, generator=g, dtype=torch.float32, device=device) * 0.5 - 50.0
    elif mode == "mixed_extremes":
        x = (torch.rand(M, H, generator=g, dtype=torch.float32, device=device)
             * 2.0 - 1.0) * extreme_range
    elif mode == "zeros":
        x = torch.zeros(M, H, dtype=torch.float32, device=device)
    elif mode == "constant":
        x = torch.full((M, H), constant, dtype=torch.float32, device=device)
    elif mode == "single_dominant":
        x = torch.randn(M, H, generator=g, dtype=torch.float32, device=device)
        x[:, 0] = dominant
    elif mode == "alternating":
        sign = torch.where(torch.arange(H, device=device) % 2 == 0,
                           torch.tensor(1.0, device=device),
                           torch.tensor(-1.0, device=device))
        x = sign.unsqueeze(0).expand(M, H).contiguous() * constant
    else:
        raise ValueError(f"未知输入 mode: {mode}")
    return x.to(dtype).contiguous()


class SoftmaxOperator(Operator):
    name = "softmax"
    dtypes = ("float16", "float32")
    bench_shapes = BENCH_MATRIX_SFM
    primary_target = PRIMARY_TARGET
    ncu_kernel_regex = "softmax"
    profiles_dir = ROOT / "profiles" / "softmax"
    bench_dir = ROOT / "benchmarks" / "softmax"
    experiments_dir = ROOT / "experiments" / "softmax"

    def build(self):
        from ..build import build
        return build("softmax")

    def make_bench_pool(self, M: int, H: int, dtype: torch.dtype,
                        mode: str, seed: int = SEED,
                        pool_size: int = 16) -> BenchPool:
        """预分配全部计时张量（计时区域内永不 malloc / 随机数 / copy）。

        Softmax 无辅助权重: 只有 x 池与 out 池。
        """
        dev = "cuda"
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        n = pool_size if mode == "streaming" else 1
        xs = [(torch.randn(M, H, generator=g, dtype=torch.float32, device=dev)
               .to(dtype).contiguous()) for _ in range(n)]
        outs = [torch.empty_like(xs[0]) for _ in xs]
        es = 2 if dtype == torch.float16 else 4
        per = M * H * es
        working_set = len(xs) * 2 * per  # x pool + out pool

        def launch(ext, variant: str, i: int):
            ext.forward_into(variant, xs[i], outs[i])

        return BenchPool(xs=xs, outs=outs, pool_size=len(xs),
                         working_set_bytes=working_set, element_size=es,
                         mode=mode, launch=launch)

    def algorithmic_bytes(self, M: int, H: int, element_size: int) -> int:
        return M * H * element_size * 2

    def ncu_driver_source(self, variant: str, M: int, H: int) -> str:
        return f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import torch
from cudalab.build import build

ext = build("softmax")
# 驱动模板用全量列表断言（含隔离变体），以便显式审计脚本可直接使用;
# 但统一 CLI 的 profile 入口仍拒绝隔离变体（NOT_FOR_NORMAL_DISPATCH，
# 隔离理由见文件头部 v0.3.1 注释与 experiments/softmax/SFM-0004.md）
assert "{variant}" in ext.all_variants(), ext.all_variants()
g = torch.Generator(device="cuda"); g.manual_seed(1234)
x = torch.randn({M}, {H}, generator=g, dtype=torch.float32, device="cuda").half().contiguous()
out = torch.empty_like(x)
# 2 次不计时的预热启动（ncu --launch-skip 2），之后 4 次被剖析
for _ in range(6):
    ext.forward_into("{variant}", x, out)
torch.cuda.synchronize()
print("profile driver done")
"""

    def experiment_prefix(self) -> str:
        return "SFM"

    def run_correctness(self, ext, variant: str,
                        out_dir: Path | None = None) -> dict:
        from ..softmax_correctness import run_suite, summarize, save_results
        if out_dir is None:
            out_dir = self.experiments_dir / "correctness" / "v0.3"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = run_suite(variant, ext)
        saved = save_results(results, out_dir / f"{variant}.json")
        s = summarize(results)
        return {"all_pass": s["all_pass"], "summary": s, "saved": str(saved)}

    def run_negative(self, ext, variant: str | None = None) -> dict:
        # 本套件是单跑设计（对所有变体同一组用例）, variant 参数被忽略
        # （签名与 base.Operator 协议一致, v0.5 独立审查 MAJOR-1）。
        from ..softmax_negative import run_negative_suite
        out = self.experiments_dir / "correctness" / "v0.3" \
            / "invalid_inputs.json"
        return run_negative_suite(ext, out_path=out)

    def pytorch_ref_latency(self, M: int, H: int, dtype: torch.dtype,
                            iters: int = 200, batch: int = 32) -> dict:
        """PyTorch 实现的延迟参照（implementation context）。

        torch 2.4.1: torch.softmax(x, dim=-1)。FP16 输入走 CUDA
        SoftMax kernel（FP32 累加器，返回 FP16）——可能比我们的 fused
        内核多/少若干内存操作，仅作参照，不作决策依据。
        """
        from ..evaluator.bench import time_call
        if isinstance(dtype, str):
            dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
        x = make_input(M, H, dtype=dtype, seed=SEED, mode="normal")

        def _call():
            return torch.softmax(x, dim=-1)

        try:
            _call()
            torch.cuda.synchronize()
        except Exception as e:
            return {"available": True, "error": f"{type(e).__name__}: {e}",
                    "note": "torch.softmax 在该 dtype/device 上失败；不伪造数字"}
        t = time_call(_call, warmup=100, iters=iters, batch=batch)
        return {"available": True,
                "api": "torch.softmax(x, dim=-1)",
                "dtype_semantics": "FP16 输入 -> FP32 累加器 -> FP16 输出"
                                   "（acc_type<half>=float）",
                **t,
                "note": "PyTorch implementation context（非公平 fused-kernel 对比）"}


softmax = SoftmaxOperator()
