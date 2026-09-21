"""CUDALab v0.5 — GEMV operator adapter。

算子: y = W @ x（广义矩阵-向量乘）:

    W  (N, K)   连续, row-major; N = 输出行数, K = 归约维
    x  (K,)     连续
    y  (N,)     连续（预分配, 输出 dtype = W dtype）

    y[n] = Σ_k W[n,k]·x[k]   （FP32 累加, 最后 cast 到输出 dtype）

主路径: W / x / 输出 = FP16, 累加 = FP32; FP32 dtype 同样支持
（"自然"情形, 累加同为 FP32）。

参考实现（用户指定, correctness 与 PyTorch context 两条路径共享）:

    fp16:  torch.mv(W.float(), x.float()).to(torch.float16)
    fp32:  torch.mv(W, x)

PyTorch 的 `torch.mv(W, x)`（cuBLAS Gemv 路径, fp16 输入默认 FP32
compute）**只作 framework context**（pytorch_ref_latency）, 不是决策
依据 —— 决策始终是 current CUDA incumbent vs candidate 的 paired
证据（v0.5 用户指定; v0.4 PyTorch context 语义原样沿用）。

形状约定: 本协议的 (M, H) 参数在 GEMV 中映射为 **(N, K)**
（M := N 输出行数, H := K 归约维）。CLI 的 --M/--H 即 --N/--K。
主目标 (N, K) = (4096, 4096), FP16。

基准矩阵（用户指定, LLM hidden / MLP projection 形状）:
    (1024, 4096), (4096, 1024), (4096, 4096), (11008, 4096),
    (4096, 11008)
其中 11008 = 4×2752（7B 级 MLP intermediate 的典型宽度）, 4096 为
常见 hidden dim。

基准池语义（make_bench_pool）:
- **轮换（计时工作集）**: x 池 + out 池, 各 pool_size 个。
  真实推理语义: **权重常驻, activation 变化** —— 每次计时 launch
  换 x / out 缓冲（同一 token 序列的下一个 step 的 activation）,
  W 固定。
- **共享常驻（shared, 不进轮换工作集）**: W（真实 inference 的
  权重）。其字节数与总逻辑工作集记录在 pool.pool_extra。
- streaming 不意味着 "cold cache"（rotating-buffer / cache-cold-ish,
  同 v0.2 语义）。**关键口径事实**: 轮换池 (x/y) 很小
  （4096×4096 fp16 下 16 池仅 0.25MB ≪ L2 5.5MB）, 但常驻 W
  （32MB ≫ L2 5.8×）每次 launch 都被完整重读且无法驻留 L2 ——
  因此每次 launch 的 DRAM 流量由 W 主导, GEMV 在此形状下是
  DRAM 带宽受限的（NCU 的 dram_throughput_pct 是实证, 见 Phase 4）。
  harness 记录的 working_set_bytes / working_set_gt_l2 按协议只统计
  轮换部分（False）; W 的字节数与结论在 pool_extra 中显式记录,
  两者不混用（v0.3.1 带宽口径教训的延续）。

算法 IO（逻辑流量, 非 DRAM 实测）:
    W 读（N·K·es）+ x 读（K·es, 广播但逻辑上读一次）+ y 写（N·es）
    = (N·K + K + N)·es
这是算子最小有用 IO（cuBLAS/llama.cpp GEMV 的同口径下界, x 的
多次物理重读由 L2 缓存, 不计入算法流量）。
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..evaluator.bench import BenchPool, SEED
from .base import Operator

ROOT = Path(__file__).resolve().parents[2]

# v0.5 GEMV 基准矩阵（用户指定, 5 个 LLM 形状）; 元素为 (N, K)
BENCH_MATRIX_GEMV = [
    (1024, 4096),
    (4096, 1024),
    (4096, 4096),
    (11008, 4096),
    (4096, 11008),
]
# 主目标 (N, K) = (4096, 4096), FP16。理由:
# (1) 用户指定的主目标形状（7B 级 hidden dim 自投影）;
# (2) W = 32MB ≫ L2 5.5MB, 每次 launch 的 DRAM 流量由 W 主导,
#     是干净的 DRAM 带宽受限 regime（算法带宽下界可算:
#     (N·K+K+N)·2B = 33.575MB, 2080 Ti 616GB/s 峰值 → ~54.5us）;
# (3) N=4096 blocks 在 30 SM 上有充足的 block 级并行
#     （4096/30 ≈ 136 waves @ 32 blocks/SM 上限, 或 8 waves
#     @ 256 threads/block × 8 blocks/SM）。
PRIMARY_TARGET = (4096, 4096)


def make_w(N: int, K: int, dtype: torch.dtype, device: str = "cuda",
           seed: int = 0, mode: str = "normal", scale: float = 1.0,
           large: float = 10.0) -> torch.Tensor:
    """确定性 W (N,K) 输入生成器（正确性与基准两条路径共享）。

    mode:
      "normal"      N(0, scale)
      "zeros"       全零
      "tiny"        N(0, 1e-4)
      "large"       N(0, large)。GEMV 的 large 用 scale=10（与 RoPE
                    的 1000 不同）: y ~ N(0, K)·scale², K=4096 时
                    5σ = 32000 < fp16 上限 65504（8σ = 51200 仍在界内）;
                    RoPE 是旋转（范数保持）所以可以 scale=1000。
      "mixed_sign"  绝对值随机 + 确定性交替符号（(i+j)%2 模式, 强制
                    正负各半）; 与 make_x 的独立 Bernoulli 符号流配合,
                    乘积 W[i,j]·x[j] 的逐元素符号独立 → 行求和出现
                    真实正负抵消, 检验抵消路径（v0.5 独立审查
                    MINOR-2: 旧实现的 x 符号 (-1)^j 使乘积符号逐行
                    恒定 = (-1)^i, 该模式下抵消深度为零）
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    if mode == "normal":
        w = torch.randn(N, K, generator=g, dtype=torch.float32,
                        device=device) * scale
    elif mode == "zeros":
        w = torch.zeros(N, K, dtype=torch.float32, device=device)
    elif mode == "tiny":
        w = torch.randn(N, K, generator=g, dtype=torch.float32,
                        device=device) * 1e-4
    elif mode == "large":
        w = torch.randn(N, K, generator=g, dtype=torch.float32,
                        device=device) * large
    elif mode == "mixed_sign":
        a = torch.randn(N, K, generator=g, dtype=torch.float32,
                        device=device).abs()
        i = torch.arange(N, device=device).unsqueeze(1)
        j = torch.arange(K, device=device).unsqueeze(0)
        sign = torch.where(((i + j) % 2) == 0, 1.0, -1.0)
        w = a * sign
    else:
        raise ValueError(f"未知输入 mode: {mode}")
    return w.to(dtype).contiguous()


def make_x(K: int, dtype: torch.dtype, device: str = "cuda",
           seed: int = 0, mode: str = "normal", scale: float = 1.0,
           large: float = 10.0) -> torch.Tensor:
    """确定性 x (K,) 输入生成器。mode 语义同 make_w:

      "mixed_sign": |randn| × 独立 Bernoulli 符号（与 make_w 的
      (i+j)%2 模式独立 → 乘积符号独立, 行求和出现真实正负抵消,
      见 MINOR-2 说明; 幅度固定 |N(0,1)|, 该模式不适用 large
      参数 —— 它是符号模式而非幅度模式）。
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    if mode == "normal":
        x = torch.randn(K, generator=g, dtype=torch.float32,
                        device=device) * scale
    elif mode == "zeros":
        x = torch.zeros(K, dtype=torch.float32, device=device)
    elif mode == "tiny":
        x = torch.randn(K, generator=g, dtype=torch.float32,
                        device=device) * 1e-4
    elif mode == "large":
        x = torch.randn(K, generator=g, dtype=torch.float32,
                        device=device) * large
    elif mode == "mixed_sign":
        # 独立 Bernoulli 符号流（不复用 (i+j)%2 / k%2 模式）: 与
        # make_w 的 (i+j)%2 符号组合后, 乘积 W[i,j]*x[j] 的符号
        # (-1)^(i+j)·s_j 对 j 独立 → 每行求和是 K 个独立符号项的
        # 和, 出现真实正负抵消（深度抵消覆盖来自此处 + normal 模式）。
        # 旧实现 x 符号 = (-1)^j: 乘积符号 = (-1)^i 逐行恒定,
        # |y| = Σ|W·x|, 无抵消 —— v0.5 独立审查 MINOR-2。
        a = torch.randn(K, generator=g, dtype=torch.float32,
                        device=device).abs()
        s = torch.randint(0, 2, (K,), generator=g, dtype=torch.int32,
                          device=device)
        sign = torch.where(s == 0, 1.0, -1.0)
        x = a * sign
    else:
        raise ValueError(f"未知输入 mode: {mode}")
    return x.to(dtype).contiguous()


def gemv_ref(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """参考 GEMV（用户指定合同; FP32 累加, cast 回输入 dtype）。"""
    if W.dim() != 2:
        raise ValueError(f"期望 2 维 W (N, K)，实际 {W.dim()}D")
    if x.dim() != 1:
        raise ValueError(f"期望 1 维 x (K,)，实际 {x.dim()}D")
    if W.dtype == torch.float16:
        return torch.mv(W.float(), x.float()).to(torch.float16)
    return torch.mv(W, x)


class GemvOperator(Operator):
    name = "gemv"
    dtypes = ("float16", "float32")
    bench_shapes = BENCH_MATRIX_GEMV      # (N, K)
    primary_target = PRIMARY_TARGET       # (N, K) = (4096, 4096)
    ncu_kernel_regex = "gemv"
    profiles_dir = ROOT / "profiles" / "gemv"
    bench_dir = ROOT / "benchmarks" / "gemv"
    experiments_dir = ROOT / "experiments" / "gemv"

    def build(self):
        from ..build import build
        return build("gemv")

    def make_bench_pool(self, M: int, K: int, dtype: torch.dtype,
                        mode: str, seed: int = SEED,
                        pool_size: int = 16) -> BenchPool:
        """预分配全部计时张量（计时区域内永不 malloc / 随机数 / copy）。

        协议参数 (M, H) 映射为 (N, K)。

        轮换: x 池 + out 池（working_set_bytes = 只这部分, 见模块
        docstring "基准池语义"）。
        共享常驻: W（权重常驻语义; 字节记录在 shared_bytes 与
        pool_extra）。

        launch 验证契约: GEMV 的 forward_into 验证全部是 host 元数据
        （无 D2H 同步, 与 RoPE 的 positions 值域检查不同）, 池构造期
        已保证全部张量连续/设备/dtype/形状正确, 因此计时 launch 直接
        走完整验证, 无豁免开关、无额外开销。
        """
        N = M  # 协议 M 即 N（W 的行数）
        dev = "cuda"
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        n = pool_size if mode == "streaming" else 1
        xs = [make_x(K, dtype, device=dev, seed=seed + i, mode="normal")
              for i in range(n)]
        outs = [torch.empty(N, dtype=dtype, device=dev) for _ in xs]
        es = 2 if dtype == torch.float16 else 4

        # 共享常驻权重（真实 inference: 权重常驻, activation 轮换）
        # W 的 seed 与 x 池 seed+0..seed+15 错开（同 seed 同 device
        # 会生成同一段随机数, 使 x 成为 W 展平布局的前缀; 基准数据
        # 应当是独立随机流 —— 同 v0.5 correctness 套件的教训）。
        W = make_w(N, K, dtype, device=dev, seed=seed + 100000,
                   mode="normal")
        shared = {"W": W}
        shared_bytes = N * K * es
        x_bytes = n * K * es
        out_bytes = n * N * es
        working_set = x_bytes + out_bytes        # 轮换部分
        total_logical = working_set + shared_bytes

        def launch(ext, variant: str, i: int):
            ext.forward_into(variant, shared["W"], xs[i], outs[i])

        return BenchPool(xs=xs, outs=outs, pool_size=len(xs),
                         working_set_bytes=working_set, element_size=es,
                         mode=mode, launch=launch,
                         shared=shared, shared_bytes=shared_bytes,
                         pool_extra={
                             "weights_resident": True,
                             "rotation": "x 池 + out 池（真实推理: "
                                         "activation 随 step 变化, "
                                         "权重常驻）",
                             "W_bytes": shared_bytes,
                             "W_gt_l2": shared_bytes > 5.5 * 1024 * 1024,
                             "total_logical_working_set_bytes":
                                 total_logical,
                             "note": "harness 的 working_set_bytes 按"
                                     "协议只统计轮换部分（x/y 池, 通常"
                                     "远小于 L2）; 但 W（权重, 每次 "
                                     "launch 完整重读）在 4096×4096 "
                                     "fp16 下为 32MB ≫ L2 5.5MB, 每次 "
                                     "launch 的 DRAM 流量由 W 主导 —— "
                                     "GEMV 的 cache-cold-ish 语义来自 "
                                     "W 而非轮换池, 见 v0.5 报告",
                         })

    def algorithmic_bytes(self, M: int, K: int, element_size: int) -> int:
        """逻辑算法流量: W 读 + x 读 + y 写 = (N·K + K + N)·es。

        x 的跨行重读由 L2 缓存, 逻辑上只读一次（同 RoPE positions
        的口径）; 这是算子最小有用 IO, 不是 DRAM 实测。
        """
        return (M * K + K + M) * element_size

    def ncu_driver_source(self, variant: str, M: int, K: int) -> str:
        """ncu 驱动脚本（FP16 主路径; 与基准池相同构造, 计时区外）。"""
        return f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import torch
from cudalab.build import build
from cudalab.operators.gemv import make_w, make_x

ext = build("gemv")
assert "{variant}" in ext.all_variants(), ext.all_variants()
W = make_w({M}, {K}, torch.float16, seed=1234)
x = make_x({K}, torch.float16, seed=1234)
out = torch.empty({M}, dtype=torch.float16, device="cuda")
# 2 次不计时的预热启动（ncu --launch-skip 2），之后 4 次被剖析。
for _ in range(6):
    ext.forward_into("{variant}", W, x, out)
torch.cuda.synchronize()
print("profile driver done")
"""

    def experiment_prefix(self) -> str:
        return "GEMV"

    def run_correctness(self, ext, variant: str,
                        out_dir: Path | None = None) -> dict:
        from ..gemv_correctness import run_suite, summarize, save_results
        if out_dir is None:
            out_dir = self.experiments_dir / "correctness" / "v0.5"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = run_suite(variant, ext)
        saved = save_results(results, out_dir / f"{variant}.json")
        s = summarize(results)
        return {"all_pass": bool(s["all_pass"]),
                "summary": s, "saved": str(saved)}

    def run_negative(self, ext, variant: str | None = None) -> dict:
        """per-variant negative suite（v0.5 独立审查 MAJOR-1 修复）。

        套件本体自 75c1ccd 起即按 variant 参数化（build_cases /
        _post_check_ok / 三个标量回退回归用例全部走 ext.forward
        (variant, ...)）, 但统一 CLI 此前只以默认 gemv_baseline 调用 ——
        4 个向量化变体的对齐契约/回退回归证据从未归档。修复后:
        variant=None/"gemv_baseline" → 规范 invalid_inputs.json;
        其余变体 → invalid_inputs_<variant>.json。
        """
        from ..gemv_negative import run_negative_suite
        v = variant or "gemv_baseline"
        if v == "gemv_baseline":
            out = self.experiments_dir / "correctness" / "v0.5" \
                / "invalid_inputs.json"
        else:
            out = self.experiments_dir / "correctness" / "v0.5" \
                / f"invalid_inputs_{v}.json"
        return run_negative_suite(ext, out_path=out, variant=v)

    def pytorch_ref_latency(self, M: int, K: int, dtype: torch.dtype,
                            iters: int = 200, batch: int = 32) -> dict:
        """PyTorch torch.mv(W, x) 的延迟参照（framework context）。

        torch.mv 走 cuBLAS Gemv（fp16 输入默认 FP32 compute type）——
        这是成熟库实现, 只作 implementation context / 带宽参照,
        **不是决策依据**（决策 = incumbent vs candidate 的 paired
        证据, v0.5 用户指定）。
        """
        from ..evaluator.bench import time_call
        if isinstance(dtype, str):
            dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
        W = make_w(M, K, dtype=dtype, seed=SEED, mode="normal")
        x = make_x(K, dtype=dtype, seed=SEED, mode="normal")

        def _call():
            return torch.mv(W, x)

        try:
            _call()
            torch.cuda.synchronize()
        except Exception as e:
            return {"available": True, "error": f"{type(e).__name__}: {e}",
                    "note": "torch.mv 路径失败；不伪造数字"}
        t = time_call(_call, warmup=100, iters=iters, batch=batch)
        return {"available": True,
                "api": "torch.mv(W, x)（cuBLAS Gemv; fp16 输入 FP32 "
                       "compute —— 成熟库实现, 非本项目的候选 kernel）",
                **t,
                "note": "PyTorch framework context（带宽参照, 不作"
                        "决策依据）"}


gemv = GemvOperator()
