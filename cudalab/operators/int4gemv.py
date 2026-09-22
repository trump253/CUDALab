"""CUDALab v0.7 — INT4GEMV operator adapter（W4A16 group-wise INT4 GEMV）。

算子: y = W_dequant @ x（INT4 权重, FP16 activation, FP32 累加）:

    W_packed (N, K/2)  连续, row-major, **uint8**   （两个 INT4/byte,
                                                     权重流量再减半:
                                                     1B/元素 → 0.5B）
    scale    (N, K/128) 连续, **float16**           （对称 group-wise
                                                     scale, G=128 固定）
    x        (K,)       连续, **float16**           （activation）
    y        (N,)       连续, **float16**           （输出, 预分配）

    W_dequant[n,k] = scale[n, k/128] · unpack(W_packed)[n,k]
    y[n] = Σ_k W_dequant[n,k] · x[k]   （FP32 累加, 最后 cast 到 fp16）

量化合同（用户 §1, 实现见 cudalab/int4gemv_quantize.py）: 对称
group-wise, zero_point = 0, q ∈ [-7, 7], scale[n,g] = max|W_group|/7,
q = clamp(round-half-to-even(W/scale), -7, 7); 零 group 安全（scale=0
→ q≡0）; K % 128 == 0; **scale 以 fp16 存储**（fp32 计算后 cast）;
量化与 packing 在 benchmark 计时区外预先完成（池构造期量化一次, 计时
launch 只读 W_packed / scale）。v0.7 范围外（用户显式排除）:
GPTQ / AWQ / activation 量化 / Tensor Core GEMM / CUDALM 集成。

三层正确性（用户 §2, 实现见 cudalab/int4gemv_correctness.py +
tests/test_int4gemv_cpu.py, 容差**不混用**）:
    (a) kernel 正确性（判定门, per-variant）: y vs ref =
        unpack(W_packed).float()·scale_fp16_expanded @ x.float()
        cast FP16 —— 固定累加误差界 + 有限性门;
    (b) pack/unpack correctness（CPU 独立层）: nibble 编码
        -7..7（+ 全域 -8..7）pack → unpack 逐元素一致, 负数符号
        扩展 / high-low nibble / 边界 -7/0/+7;
    (c) 量化保真度（只报告, 不判定, variant 无关）: INT4 量化后
        reference vs 原 FP16 W @ x —— max_abs / RMSE / cosine。

参考 / framework context: 本算子的 pytorch_ref_latency 是
**未量化 FP16 GEMV** 的 torch.mv(W, x)（与 v0.5/v0.6 相同的
framework context; PyTorch 无 W4A16 GEMV op, 公平的 INT4 框架参照
不存在 —— 报告中如实说明）。性能参照主体是 v0.5 FP16 GEMV
incumbent（gemv_vec4_row）与 v0.6 INT8 QGEMV incumbent
（qgemv_vec16_row）—— 它们只是**性能参照**（三代对比 §8）, 不是
INT4GEMV 候选间 dispatch / 决策的输入。

形状约定: 本协议的 (M, H) 参数在 INT4GEMV 中映射为 **(N, K)**
（M := N 输出行数, H := K 归约维）, 与 v0.5 GEMV / v0.6 QGEMV
一致。CLI 的 --M/--H 即 --N/--K。主目标 (N, K) = (4096, 4096)。

基准矩阵（与 v0.5/v0.6 相同 5 个 LLM 形状, K 均为 128 的倍数）:
    (1024, 4096), (4096, 1024), (4096, 4096), (11008, 4096),
    (4096, 11008)

基准池语义（make_bench_pool）:
- **轮换（计时工作集）**: x 池 + out 池, 各 pool_size 个（真实推理
  语义: 权重常驻, activation 变化）。
- **共享常驻（shared, 不进轮换工作集）**: W_packed（uint8, 8MB @
  4096² ≫ L2 5.5MB → 每次 launch 完整重读, DRAM 流量主导）+ scale
  （fp16, 256KB @ N=4096, 与 W_packed 一起常驻）。
- **量化 + packing 在池构造期完成**（计时区外, 合同 §1）; 池内保存
  的 W_packed / scale 就是计时 launch 读到的张量。
- streaming 不意味着 "cold cache"（rotating-buffer / cache-cold-ish,
  同 v0.2/v0.5/v0.6 语义）。轮换池很小（4096² fp16 下 16 池 0.25MB
  ≪ L2）, 常驻 W_packed（8MB ≫ L2 1.5×）每次 launch 都被完整重读
  且无法完全驻留 L2 —— 与 v0.5/v0.6 相同 regime, DRAM 流量由权重
  主导; harness 记录的 working_set_bytes 按协议只统计轮换部分,
  W_packed/scale 字节数与结论记录在 pool_extra。

算法 IO（逻辑流量, 非 DRAM 实测; "algorithmic BW ≠ 实测 DRAM BW"）:
    W_packed 读（N·K/2 · 1B）+ scale 读（N·K/128 · 2B）
    + x 读（K·2B, 广播但逻辑上读一次, L2 缓存跨行重读）
    + y 写（N·2B）
    = N·K/2 + N·K/64 + 2K + 2N
4096×4096: 8,388,608 + 262,144 + 8,192 + 8,192 = 8,667,136 B
（≈ 8.27MB, 是 v0.5 FP16 GEMV 33.575MB 的 ~25.8%, 恰好是 v0.6
INT8 16.810MB 的 ~51.6% —— 权重流量再减半 + group scale 开销）。
算法带宽下界: 8.67MB / 616GB/s → ~14.1us（vs v0.6 INT8 ~27.3us /
v0.5 FP16 ~54.5us）。

变体状态（v0.7）:
- 当前**无隔离变体**（bindings.cpp quarantined_set 为空, 机制保留,
  与 v0.5/v0.6 同一协议）。
- **对齐契约的适用范围**: 16B 向量化变体有显式对齐契约（W_packed
  基址 16B ∧ x 基址 16B ∧ K%32==0; 不满足 → int4gemv_scalar_kernel
  同源代码逐位一致回退, negative 套件的 fallback 回归用例钉死）。
  int4gemv_baseline 是标量访存（1B uint8 + 2B fp16 load）, **无对齐
  契约** —— 任何连续合法输入（含 1 字节 offset 视图）必须成功。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..evaluator.bench import BenchPool, SEED
from ..int4gemv_quantize import quantize_w
from .base import Operator
from .gemv import make_w as _make_w_fp16, make_x as _make_x_fp16

ROOT = Path(__file__).resolve().parents[2]

# v0.7 INT4GEMV 基准矩阵（与 v0.5/v0.6 相同 5 个 LLM 形状）; 元素为
# (N, K), K 均为 128 的倍数
BENCH_MATRIX_INT4GEMV = [
    (1024, 4096),
    (4096, 1024),
    (4096, 4096),
    (11008, 4096),
    (4096, 11008),
]
# 主目标 (N, K) = (4096, 4096), FP16 activation。理由:
# (1) 用户指定的主目标形状（N=4096, K=4096, G=128）;
# (2) W_packed = 8MB ≫ L2 5.5MB, 每次 launch 的 DRAM 流量由
#     W_packed 主导, 是干净的 DRAM 带宽受限 regime（算法带宽下界:
#     8.67MB / 616GB/s → ~14.1us, 对比 v0.6 INT8 的 16.81MB /
#     ~27.3us 与 v0.5 FP16 的 33.575MB / ~54.5us）;
# (3) N=4096 blocks 在 30 SM 上有充足的 block 级并行（同 v0.5/v0.6）。
PRIMARY_TARGET = (4096, 4096)


def make_w4(N: int, K: int, seed: int = 0, mode: str = "normal",
            scale: float = 1.0, large: float = 10.0,
            device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor,
                                           torch.Tensor]:
    """确定性 W4A16 权重生成器: 先生成 FP16 W（复用 v0.5 生成器,
    同一 mode 语义 / seed 约定）, 再按 §1 合同 group-wise 量化 +
    packing（**计时区外**）。要求 K % 128 == 0（量化器拒绝否则）。

    返回 (W_packed, scale, W_fp16): W_packed uint8 (N, K/2) /
    scale fp16 (N, K/128) / W_fp16 原始 FP16 权重（第 (c) 层量化
    保真度参照用; 基准池不持有 W_fp16 —— 计时 launch 只读
    W_packed / scale）。
    """
    W = _make_w_fp16(N, K, torch.float16, device=device, seed=seed,
                     mode=mode, scale=scale, large=large)
    W_packed, s = quantize_w(W)
    return W_packed, s, W


def make_x4(K: int, seed: int = 0, mode: str = "normal",
            large: float = 10.0, device: str = "cuda") -> torch.Tensor:
    """确定性 FP16 activation 生成器（复用 v0.5 生成器）。"""
    return _make_x_fp16(K, torch.float16, device=device, seed=seed,
                        mode=mode, large=large)


class Int4gemvOperator(Operator):
    name = "int4gemv"
    dtypes = ("float16",)     # activation / 输出 dtype; W_packed 恒 uint8
    bench_shapes = BENCH_MATRIX_INT4GEMV   # (N, K)
    primary_target = PRIMARY_TARGET        # (N, K) = (4096, 4096)
    ncu_kernel_regex = "int4gemv"
    profiles_dir = ROOT / "profiles" / "int4gemv"
    bench_dir = ROOT / "benchmarks" / "int4gemv"
    experiments_dir = ROOT / "experiments" / "int4gemv"

    def build(self):
        from ..build import build
        return build("int4gemv")

    def make_bench_pool(self, M: int, K: int, dtype: torch.dtype,
                        mode: str, seed: int = SEED,
                        pool_size: int = 16) -> BenchPool:
        """预分配全部计时张量（计时区域内永不 malloc / 随机数 / copy）。

        协议参数 (M, H) 映射为 (N, K)。dtype = activation dtype
        （v0.7 主路径仅 FP16; FP32 activation 不在本版本合同内）。

        轮换: x 池 + out 池（working_set_bytes = 只这部分）。
        共享常驻: W_packed（uint8）+ scale（fp16）—— **量化 +
        packing 在池构造期完成（计时区外, 合同 §1）**, 池内张量即
        计时 launch 所读。

        launch 验证契约: INT4GEMV 的 forward_into 验证全部是 host
        元数据（无 D2H 同步）, 池构造期已保证全部张量连续/设备/
        dtype/形状正确, 因此计时 launch 直接走完整验证, 无豁免
        开关、无额外开销。
        """
        N = M  # 协议 M 即 N（W_packed 的行数）
        if dtype != torch.float16:
            raise NotImplementedError(
                f"int4gemv v0.7 主路径仅 FP16 activation, 实际 "
                f"dtype={dtype}（INT4 权重 × FP32 activation 不在"
                "本版本合同内）")
        if K % 128 != 0:
            raise NotImplementedError(
                f"int4gemv 合同要求 K % 128 == 0, 实际 K={K}")
        dev = "cuda"
        n = pool_size if mode == "streaming" else 1
        xs = [make_x4(K, seed=seed + i, mode="normal") for i in range(n)]
        outs = [torch.empty(N, dtype=torch.float16, device=dev) for _ in xs]
        es = 2  # activation / 输出元素字节数（逻辑工作集口径）

        # 共享常量化量化权重（真实 inference: 权重常驻, activation
        # 轮换）。W 的 seed 与 x 池 seed+0..seed+15 错开（同
        # v0.5/v0.6 GEMV/QGEMV 池的独立随机流教训）。
        W_packed, scale, _W_fp16 = make_w4(N, K, seed=seed + 100000,
                                           mode="normal")
        shared = {"W_packed": W_packed, "scale": scale}
        wp_bytes = N * K // 2          # uint8 packed: 0.5B/元素
        scale_bytes = N * (K // 128) * 2
        shared_bytes = wp_bytes + scale_bytes
        x_bytes = n * K * es
        out_bytes = n * N * es
        working_set = x_bytes + out_bytes        # 轮换部分
        total_logical = working_set + shared_bytes

        def launch(ext, variant: str, i: int):
            ext.forward_into(variant, shared["W_packed"], shared["scale"],
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
                             "W_packed_bytes": wp_bytes,
                             "scale_bytes": scale_bytes,
                             "W_packed_gt_l2": wp_bytes > 5.5 * 1024 * 1024,
                             "total_logical_working_set_bytes":
                                 total_logical,
                             "note": "量化 + packing（FP16 W → uint8 "
                                     "W_packed + fp16 group scale）"
                                     "在池构造期完成, 严格位于计时区外"
                                     "（合同 §1）; harness 的 "
                                     "working_set_bytes 按协议只统计"
                                     "轮换部分（x/y 池, 远小于 L2）; "
                                     "常驻 W_packed 在 4096×4096 下为 "
                                     "8MB ≫ L2 5.5MB, 每次 launch 的 "
                                     "DRAM 流量由 W_packed 主导 —— "
                                     "与 v0.5 FP16 GEMV / v0.6 INT8 "
                                     "QGEMV 相同 regime",
                         })

    def algorithmic_bytes(self, M: int, K: int, element_size: int) -> int:
        """逻辑算法流量（"algorithmic BW ≠ 实测 DRAM BW, 以 NCU
        dram_throughput_pct 为准"）:

            W_packed 读 N·K/2·1B + scale 读 N·(K/128)·2B
            + x 读 K·2B（逻辑一次, L2 缓存跨行重读）+ y 写 N·2B
            = M·K/2 + M·K/64 + 2K + 2M

        这是算子最小有用 IO, **不是 DRAM 实测**。element_size 参数
        按协议传入（=2, fp16）, 本公式按算子混合 dtype 独立计算。
        """
        return M * K // 2 + M * (K // 128) * 2 + 2 * K + 2 * M

    def ncu_driver_source(self, variant: str, M: int, K: int) -> str:
        """ncu 驱动脚本（FP16 主路径; 与基准池相同构造, 量化在计时
        区外, 计时区只含 kernel launch）。"""
        return f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import torch
from cudalab.build import build
from cudalab.operators.int4gemv import make_w4, make_x4

ext = build("int4gemv")
assert "{variant}" in ext.all_variants(), ext.all_variants()
W_packed, scale, _W = make_w4({M}, {K}, seed=1234)
x = make_x4({K}, seed=1234)
out = torch.empty({M}, dtype=torch.float16, device="cuda")
# 2 次不计时的预热启动（ncu --launch-skip 2），之后 4 次被剖析。
for _ in range(6):
    ext.forward_into("{variant}", W_packed, scale, x, out)
torch.cuda.synchronize()
print("profile driver done")
"""

    def experiment_prefix(self) -> str:
        return "INT4GEMV"

    def run_correctness(self, ext, variant: str,
                        out_dir: Path | None = None) -> dict:
        """三层正确性: (a) per-variant kernel 正确性套件（判定门）;
        (b) pack/unpack CPU 层（由 tests/test_int4gemv_cpu.py 覆盖,
        此处记录 suite 存在性）; (c) 量化保真度套件（variant 无关,
        只报告）。"""
        from ..int4gemv_correctness import (
            run_suite, summarize, save_results,
            run_fidelity_suite, save_fidelity,
        )
        if out_dir is None:
            out_dir = self.experiments_dir / "correctness" / "v0.7"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = run_suite(variant, ext)
        saved = save_results(results, out_dir / f"{variant}.json")
        s = summarize(results)
        # 量化保真度是 variant 无关层（只报告）: append-only 约定 ——
        # 同一 out_dir 已有记录时不重跑、不刷新（避免每次变体正确性
        # 运行改写历史 artifact 的时间戳）; 新目录（如 regression
        # out_dir）仍会生成。
        fid_path = out_dir / "quantization_fidelity.json"
        if fid_path.exists():
            with open(fid_path, "r", encoding="utf-8") as f:
                fid_summary = json.load(f).get("summary")
            fid_saved = fid_path
        else:
            fid = run_fidelity_suite()
            fid_saved = save_fidelity(fid, fid_path)
            fid_summary = fid["summary"]
        return {"all_pass": bool(s["all_pass"]),
                "summary": s, "saved": str(saved),
                "quantization_fidelity": {
                    "saved": str(fid_saved),
                    "summary": fid_summary,
                }}

    def run_negative(self, ext, variant: str | None = None,
                     out_dir: Path | None = None) -> dict:
        """per-variant negative suite（与 v0.5 GEMV / v0.6 QGEMV
        同一语义: negative_suite_scope = "per-variant", 契约/回退
        回归必须对受测变体自身运行并归档）。

        variant=None/"int4gemv_baseline" → 规范 invalid_inputs.json;
        其余变体 → invalid_inputs_<variant>.json。

        out_dir（append-only 约定, 同 v0.5/v0.6）: 显式指定时结果
        存到该目录（自动创建）, 不覆盖规范记录 —— 例如
        `experiments/regression/v0.7/int4gemv/`。
        """
        from ..int4gemv_negative import run_negative_suite
        v = variant or "int4gemv_baseline"
        if out_dir is None:
            out_dir = self.experiments_dir / "correctness" / "v0.7"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = ("invalid_inputs.json" if v == "int4gemv_baseline"
                else f"invalid_inputs_{v}.json")
        return run_negative_suite(ext, out_path=out_dir / name, variant=v)

    def pytorch_ref_latency(self, M: int, K: int, dtype: torch.dtype,
                            iters: int = 200, batch: int = 32) -> dict:
        """PyTorch framework context（非决策依据, 同 v0.5/v0.6 语义）。

        v0.7 的 framework context 是**未量化 FP16 GEMV** 的
        torch.mv(W, x)（cuBLAS Gemv; fp16 输入默认 FP32 compute）——
        PyTorch 没有 W4A16 GEMV op, 公平的 INT4 框架参照不存在
        （报告如实说明; 与 INT4GEMV kernel 的直接对比对象是 v0.5
        FP16 GEMV incumbent gemv_vec4_row 与 v0.6 INT8 QGEMV
        incumbent qgemv_vec16_row, 它们只作性能参照, 不是 INT4GEMV
        候选间决策的输入）。
        """
        from ..evaluator.bench import time_call
        if isinstance(dtype, str):
            dtype = {"float16": torch.float16}[dtype]
        if dtype != torch.float16:
            return {"available": True,
                    "error": "int4gemv v0.7 仅 FP16 activation 主路径",
                    "note": "不伪造数字"}
        _Wp, _scale, W = make_w4(M, K, seed=SEED, mode="normal")
        x = make_x4(K, seed=SEED, mode="normal")

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
                       "FP16 GEMV —— framework context, 与 v0.5/v0.6 "
                       "相同口径; PyTorch 无 W4A16 GEMV op）",
                **t,
                "note": "PyTorch framework context（带宽参照, 不作"
                        "决策依据）"}


int4gemv = Int4gemvOperator()
