"""CUDALab v0.6 — QGEMV operator adapter（INT8 weight-only GEMV）。

算子: y = W_dequant @ x（INT8 权重, FP16 activation, FP32 累加）:

    W_q   (N, K)   连续, row-major, **int8**    （量化权重, 权重流量
                                                 减半: 2B/元素 → 1B）
    scale (N,)     连续, **float32**            （对称 per-row scale）
    x     (K,)     连续, **float16**            （activation）
    y     (N,)     连续, **float16**            （输出, 预分配）

    W_dequant[n,k] = scale[n] · W_q[n,k]
    y[n] = Σ_k W_dequant[n,k] · x[k]   （FP32 累加, 最后 cast 到 fp16）

量化合同（§1, 实现见 cudalab/qgemv_quantize.py）: 对称 per-row,
zero_point = 0, scale[n] = max|W[n,:]|/127, q = clamp(round(W/scale),
-127, 127); scale = 0 行安全; **量化在 benchmark 计时区外预先完成**
（池构造期量化一次, 计时 launch 只读 W_q / scale）。v0.6 范围外:
INT4 / group-wise / GPTQ / AWQ / activation 量化 / Tensor Core GEMM。

两层正确性（§2, 实现见 cudalab/qgemv_correctness.py, 两种容差
**不混用**）:
    (a) kernel 正确性: y vs ref = (W_q.float()·scale[:,None]) @
        x.float() cast FP16 —— 固定累加误差界 + 有限性门（判定门）;
    (b) 量化保真度: 量化后 ref vs 原始 FP16 W @ x —— max_abs /
        max_rel / RMSE / cosine（只报告, 信息性）。

参考 / framework context（§6）: 本算子的 pytorch_ref_latency 是
**未量化 FP16 GEMV** 的 torch.mv(W, x)（与 v0.5 相同的 framework
context; PyTorch 无 INT8 weight-only GEMV op, 公平的 INT8 框架
参照不存在 —— 报告 §6 中如实说明）。性能参照主体是 v0.5 FP16 GEMV
incumbent（gemv_vec4_row）—— 它只是**性能参照**, 不是 QGEMV 候选
间 dispatch / 决策的输入（决策始终在 QGEMV 候选之间, 用户指定）。

形状约定: 本协议的 (M, H) 参数在 QGEMV 中映射为 **(N, K)**
（M := N 输出行数, H := K 归约维）, 与 v0.5 GEMV 一致。CLI 的
--M/--H 即 --N/--K。主目标 (N, K) = (4096, 4096), FP16 activation。

基准矩阵（与 v0.5 相同 5 个 LLM 形状）:
    (1024, 4096), (4096, 1024), (4096, 4096), (11008, 4096),
    (4096, 11008)

基准池语义（make_bench_pool）:
- **轮换（计时工作集）**: x 池 + out 池, 各 pool_size 个（真实推理
  语义: 权重常驻, activation 变化）。
- **共享常驻（shared, 不进轮换工作集）**: W_q（int8, 16MB @ 4096²
  ≫ L2 5.5MB → 每次 launch 完整重读, DRAM 流量主导）+ scale
  （fp32, 16KB @ N=4096, 与 W_q 一起常驻）。
- **量化在池构造期完成**（计时区外, 合同 §1）; 池内保存的 W_q /
  scale 就是计时 launch 读到的张量。
- streaming 不意味着 "cold cache"（rotating-buffer / cache-cold-ish,
  同 v0.2 语义）。轮换池很小（4096² fp16 下 16 池 0.25MB ≪ L2）,
  常驻 W_q（16MB ≫ L2 2.9×）每次 launch 都被完整重读且无法驻留 L2
  —— 与 v0.5 FP16 GEMV 相同, DRAM 流量由权重主导; harness 记录的
  working_set_bytes 按协议只统计轮换部分, W_q/scale 字节数与结论
  记录在 pool_extra（v0.3.1 带宽口径教训的延续）。

算法 IO（逻辑流量, 非 DRAM 实测; §6 "algorithmic BW ≠ 实测 DRAM
BW"）:
    W_q 读（N·K·1B）+ x 读（K·2B, 广播但逻辑上读一次, L2 缓存
    跨行重读）+ scale 读（N·4B）+ y 写（N·2B）
    = N·K + 2K + 4N + 2N
4096×4096: 16,777,216 + 8,192 + 16,384 + 8,192 = 16,809,984 B
（≈ 16.03MB, 恰好是 v0.5 FP16 GEMV 33.575MB 的 ~50.4%）。

变体状态（v0.6）:
- 当前**无隔离变体**（bindings.cpp quarantined_set 为空, 机制保留,
  与 v0.5 GEMV 同一协议）。
- **对齐契约的适用范围**: 16B 向量化变体有显式对齐契约（W_q 基址
  16B ∧ x 基址 16B ∧ K % 16 == 0; 不满足 → qgemv_scalar_kernel
  同源代码逐位一致回退, negative 套件的三个 fallback 回归用例钉死）。
  qgemv_baseline 是标量访存（1B int8 + 2B fp16 load）, **无对齐
  契约** —— 任何连续合法输入（含 1 字节 offset 视图）必须成功。
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..evaluator.bench import BenchPool, SEED
from ..qgemv_quantize import quantize_w
from .base import Operator
from .gemv import make_w as _make_w_fp16, make_x as _make_x_fp16

ROOT = Path(__file__).resolve().parents[2]

# v0.6 QGEMV 基准矩阵（与 v0.5 GEMV 相同 5 个 LLM 形状）; 元素为 (N, K)
BENCH_MATRIX_QGEMV = [
    (1024, 4096),
    (4096, 1024),
    (4096, 4096),
    (11008, 4096),
    (4096, 11008),
]
# 主目标 (N, K) = (4096, 4096), FP16 activation。理由:
# (1) 用户指定的主目标形状;
# (2) W_q = 16MB ≫ L2 5.5MB, 每次 launch 的 DRAM 流量由 W_q 主导,
#     是干净的 DRAM 带宽受限 regime（算法带宽下界可算: 16.81MB /
#     616GB/s → ~27.3us, 对比 v0.5 FP16 GEMV 的 33.575MB / 54.5us）;
# (3) N=4096 blocks 在 30 SM 上有充足的 block 级并行（同 v0.5）。
PRIMARY_TARGET = (4096, 4096)


def make_w_q(N: int, K: int, seed: int = 0, mode: str = "normal",
             scale: float = 1.0, large: float = 10.0,
             device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor,
                                            torch.Tensor]:
    """确定性 INT8 权重生成器: 先生成 FP16 W（复用 v0.5 生成器,
    同一 mode 语义 / seed 约定）, 再按 §1 合同量化（**计时区外**）。

    返回 (W_q, scale, W_fp16): W_q int8 (N,K) / scale fp32 (N,) /
    W_fp16 原始 FP16 权重（第 (b) 层量化保真度参照用; 基准池不持有
    W_fp16 —— 计时 launch 只读 W_q / scale）。
    """
    W = _make_w_fp16(N, K, torch.float16, device=device, seed=seed,
                     mode=mode, scale=scale, large=large)
    W_q, s = quantize_w(W)
    return W_q, s, W


def make_x_q(K: int, seed: int = 0, mode: str = "normal",
             large: float = 10.0, device: str = "cuda") -> torch.Tensor:
    """确定性 FP16 activation 生成器（复用 v0.5 生成器）。"""
    return _make_x_fp16(K, torch.float16, device=device, seed=seed,
                        mode=mode, large=large)


class QgemvOperator(Operator):
    name = "qgemv"
    dtypes = ("float16",)     # activation / 输出 dtype; W_q 恒 int8
    bench_shapes = BENCH_MATRIX_QGEMV   # (N, K)
    primary_target = PRIMARY_TARGET     # (N, K) = (4096, 4096)
    ncu_kernel_regex = "qgemv"
    profiles_dir = ROOT / "profiles" / "qgemv"
    bench_dir = ROOT / "benchmarks" / "qgemv"
    experiments_dir = ROOT / "experiments" / "qgemv"

    def build(self):
        from ..build import build
        return build("qgemv")

    def make_bench_pool(self, M: int, K: int, dtype: torch.dtype,
                        mode: str, seed: int = SEED,
                        pool_size: int = 16) -> BenchPool:
        """预分配全部计时张量（计时区域内永不 malloc / 随机数 / copy）。

        协议参数 (M, H) 映射为 (N, K)。dtype = activation dtype
        （v0.6 主路径仅 FP16; FP32 activation 不在本版本合同内）。

        轮换: x 池 + out 池（working_set_bytes = 只这部分）。
        共享常驻: W_q（int8）+ scale（fp32）—— **量化在池构造期
        完成（计时区外, 合同 §1）**, 池内张量即计时 launch 所读。

        launch 验证契约: QGEMV 的 forward_into 验证全部是 host
        元数据（无 D2H 同步）, 池构造期已保证全部张量连续/设备/
        dtype/形状正确, 因此计时 launch 直接走完整验证, 无豁免
        开关、无额外开销。
        """
        N = M  # 协议 M 即 N（W_q 的行数）
        if dtype != torch.float16:
            raise NotImplementedError(
                f"qgemv v0.6 主路径仅 FP16 activation, 实际 dtype={dtype}"
                "（INT8 权重 × FP32 activation 不在本版本合同内）")
        dev = "cuda"
        n = pool_size if mode == "streaming" else 1
        xs = [make_x_q(K, seed=seed + i, mode="normal") for i in range(n)]
        outs = [torch.empty(N, dtype=torch.float16, device=dev) for _ in xs]
        es = 2  # activation / 输出元素字节数（逻辑工作集口径）

        # 共享常驻量化权重（真实 inference: 权重常驻, activation 轮换）。
        # W 的 seed 与 x 池 seed+0..seed+15 错开（同 v0.5 GEMV 池的
        # 独立随机流教训）。
        W_q, scale, _W_fp16 = make_w_q(N, K, seed=seed + 100000,
                                       mode="normal")
        shared = {"W_q": W_q, "scale": scale}
        wq_bytes = N * K          # int8: 1B/元素（权重流量减半）
        scale_bytes = N * 4
        shared_bytes = wq_bytes + scale_bytes
        x_bytes = n * K * es
        out_bytes = n * N * es
        working_set = x_bytes + out_bytes        # 轮换部分
        total_logical = working_set + shared_bytes

        def launch(ext, variant: str, i: int):
            ext.forward_into(variant, shared["W_q"], shared["scale"],
                             xs[i], outs[i])

        return BenchPool(xs=xs, outs=outs, pool_size=len(xs),
                         working_set_bytes=working_set, element_size=es,
                         mode=mode, launch=launch,
                         shared=shared, shared_bytes=shared_bytes,
                         pool_extra={
                             "weights_resident": True,
                             "quantization_offline": True,
                             "rotation": "x 池 + out 池（真实推理: "
                                         "activation 随 step 变化, "
                                         "量化权重常驻）",
                             "W_q_bytes": wq_bytes,
                             "scale_bytes": scale_bytes,
                             "W_q_gt_l2": wq_bytes > 5.5 * 1024 * 1024,
                             "total_logical_working_set_bytes":
                                 total_logical,
                             "note": "量化（FP16 W → int8 W_q + fp32 "
                                     "scale）在池构造期完成, 严格位于"
                                     "计时区外（合同 §1）; harness 的 "
                                     "working_set_bytes 按协议只统计"
                                     "轮换部分（x/y 池, 远小于 L2）; "
                                     "常驻 W_q 在 4096×4096 下为 "
                                     "16MB ≫ L2 5.5MB, 每次 launch "
                                     "的 DRAM 流量由 W_q 主导 —— 与 "
                                     "v0.5 FP16 GEMV 相同 regime",
                         })

    def algorithmic_bytes(self, M: int, K: int, element_size: int) -> int:
        """逻辑算法流量（§6: INT8 W bytes + FP16 x bytes + scale bytes
        + FP16 y bytes）:

            W_q 读 N·K·1B + x 读 K·2B（逻辑一次, L2 缓存跨行重读）
            + scale 读 N·4B + y 写 N·2B
            = M·K + 2K + 4M + 2M

        这是算子最小有用 IO, **不是 DRAM 实测**（algorithmic BW ≠
        实测 DRAM BW, 以 NCU dram_throughput_pct 为准）。
        element_size 参数按协议传入（=2, fp16）, 本公式按算子混合
        dtype 独立计算。
        """
        return M * K + 2 * K + 4 * M + 2 * M

    def ncu_driver_source(self, variant: str, M: int, K: int) -> str:
        """ncu 驱动脚本（FP16 主路径; 与基准池相同构造, 量化在计时
        区外, 计时区只含 kernel launch）。"""
        return f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import torch
from cudalab.build import build
from cudalab.operators.qgemv import make_w_q, make_x_q

ext = build("qgemv")
assert "{variant}" in ext.all_variants(), ext.all_variants()
W_q, scale, _W = make_w_q({M}, {K}, seed=1234)
x = make_x_q({K}, seed=1234)
out = torch.empty({M}, dtype=torch.float16, device="cuda")
# 2 次不计时的预热启动（ncu --launch-skip 2），之后 4 次被剖析。
for _ in range(6):
    ext.forward_into("{variant}", W_q, scale, x, out)
torch.cuda.synchronize()
print("profile driver done")
"""

    def experiment_prefix(self) -> str:
        return "QGEMV"

    def run_correctness(self, ext, variant: str,
                        out_dir: Path | None = None) -> dict:
        """两层正确性: (a) per-variant kernel 正确性套件（50 项,
        判定门）; (b) 量化保真度套件（variant 无关, 只报告）。"""
        from ..qgemv_correctness import (
            run_suite, summarize, save_results,
            run_fidelity_suite, save_fidelity,
        )
        if out_dir is None:
            out_dir = self.experiments_dir / "correctness" / "v0.6"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = run_suite(variant, ext)
        saved = save_results(results, out_dir / f"{variant}.json")
        s = summarize(results)
        fid = run_fidelity_suite()
        fid_saved = save_fidelity(fid, out_dir / "quantization_fidelity.json")
        return {"all_pass": bool(s["all_pass"]),
                "summary": s, "saved": str(saved),
                "quantization_fidelity": {
                    "saved": str(fid_saved),
                    "summary": fid["summary"],
                }}

    def run_negative(self, ext, variant: str | None = None,
                     out_dir: Path | None = None) -> dict:
        """per-variant negative suite（与 v0.5 GEMV 同一语义:
        negative_suite_scope = "per-variant", 契约/回退回归必须对
        受测变体自身运行并归档）。

        variant=None/"qgemv_baseline" → 规范 invalid_inputs.json;
        其余变体 → invalid_inputs_<variant>.json。

        out_dir（append-only 约定, 同 v0.5）: 显式指定时结果存到该
        目录（自动创建）, 不覆盖规范记录 —— 例如
        `experiments/regression/v0.6/qgemv/`。
        """
        from ..qgemv_negative import run_negative_suite
        v = variant or "qgemv_baseline"
        if out_dir is None:
            out_dir = self.experiments_dir / "correctness" / "v0.6"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = ("invalid_inputs.json" if v == "qgemv_baseline"
                else f"invalid_inputs_{v}.json")
        return run_negative_suite(ext, out_path=out_dir / name, variant=v)

    def pytorch_ref_latency(self, M: int, K: int, dtype: torch.dtype,
                            iters: int = 200, batch: int = 32) -> dict:
        """PyTorch framework context（非决策依据, 同 v0.5 语义）。

        v0.6 的 framework context 是**未量化 FP16 GEMV** 的
        torch.mv(W, x)（cuBLAS Gemv; fp16 输入默认 FP32 compute）——
        PyTorch 没有 INT8 weight-only GEMV op, 公平的 INT8 框架参照
        不存在（报告 §6 如实说明; 与 QGEMV kernel 的直接对比对象是
        v0.5 FP16 GEMV incumbent gemv_vec4_row, 它只作性能参照,
        不是 QGEMV 候选间决策的输入）。
        """
        from ..evaluator.bench import time_call
        if isinstance(dtype, str):
            dtype = {"float16": torch.float16}[dtype]
        if dtype != torch.float16:
            return {"available": True,
                    "error": "qgemv v0.6 仅 FP16 activation 主路径",
                    "note": "不伪造数字"}
        _W_q, _scale, W = make_w_q(M, K, seed=SEED, mode="normal")
        x = make_x_q(K, seed=SEED, mode="normal")

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
                "api": "torch.mv(W_fp16, x_fp16)（cuBLAS Gemv; 未量化 "
                       "FP16 GEMV —— framework context, 与 v0.5 相同 "
                       "口径; PyTorch 无 INT8 weight-only GEMV op）",
                **t,
                "note": "PyTorch framework context（带宽参照, 不作"
                        "决策依据）"}


qgemv = QgemvOperator()
