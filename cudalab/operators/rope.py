"""CUDALab v0.4 — RoPE operator adapter（interleaved RoPE）。

算子: Rotary Position Embedding（**interleaved pair 约定**, 不使用 NeoX
half-split）:

    x        (M, D)   连续, D 偶数; M = token-head 行数
    positions (M,)    int64, 0 <= p < max_seq_len
    cos/sin  (L, D/2) 连续; L = max_seq_len, 表常驻 GPU（计时区外预计算）

    对每行 m、每对 i ∈ [0, D/2):
      a = x[m, 2i]          b = x[m, 2i+1]
      c = cos[positions[m], i]   s = sin[positions[m], i]
      y[m, 2i]   = a*c - b*s      （FP32 中间运算）
      y[m, 2i+1] = a*s + b*c      （输出 dtype = x dtype）

参考实现（显式、FP32 中间、与 PyTorch 版本无关）:

    a = x.float()[:, 0::2];  b = x.float()[:, 1::2]
    c = cos[positions].float();  s = sin[positions].float()
    out_even = a*c - b*s;  out_odd = a*s + b*c
    y = interleave(out_even, out_odd).to(x.dtype)

PyTorch 2.4.1 **没有**内置 fused RoPE 算子（无 torch.nn.functional.rope /
torch.ops.rope）——Python 参考只作 correctness / implementation context,
不做 "X× faster than PyTorch" 式 headline 对比。

cos/sin 表构造（确定性, 计时区外）:

    theta_i = base ** (-2*i/D),  base = 10000,  i ∈ [0, D/2)
    freqs[m, i] = positions[m] * theta_i
    cos = cos(freqs).to(x dtype),  sin = sin(freqs).to(x dtype)

表以 FP32 计算后 cast 到 x 的 dtype 存储（与常见推理实现一致）；
正确性参考使用**同一张表**（FP32 提升后运算），隔离旋转数学与表
舍入两个误差源。

基准池语义（make_bench_pool）:
- **轮换（计时工作集）**: x 池 + out 池, 各 pool_size 个
  （working_set_bytes 只统计这部分, working_set_gt_l2 基于它）;
- **不轮换**: positions（真实 inference 中同一 batch 的位置序列不随
  activation 缓冲轮换而变化, 引擎与池设计都固定它）;
- **共享常驻（shared, 不进轮换工作集）**: cos/sin 表
  （真实 inference: 表常驻, activation 变化）。表的字节数与总逻辑
  工作集记录在 pool.pool_extra。
- streaming 不意味着 "cold cache"（rotating-buffer / cache-cold-ish,
  同 v0.2 语义）。

算法 IO（逻辑流量, 非 DRAM 实测）:
    x 读 + y 写 + cos 查表 + sin 查表 + positions 读
    = M*D*es*2 + M*(D/2)*es*2 + M*8
cos/sin 查表在 L2 可缓存（表远小于轮换工作集时）, 这里报告的是
逻辑算法流量, 不声称 DRAM 饱和（v0.3.1 带宽口径教训）。
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..evaluator.bench import BenchPool, SEED
from .base import Operator

ROOT = Path(__file__).resolve().parents[2]

# v0.4 RoPE 基准矩阵（用户指定, 9 个形状; H 参数在本算子中即 D）
BENCH_MATRIX_ROPE = [
    (1, 64),
    (1, 128),
    (32, 64),
    (32, 128),
    (128, 64),
    (128, 128),
    (1024, 64),
    (1024, 128),
    (4096, 128),
]
# 主目标 (M, D) = (1024, 128), FP16。理由:
# (1) 吞吐观测: M=1024, D=128 -> 65536 个 RoPE pair = 65536 线程
#     （65536 / (30 SM × 2048 线程/SM) ≈ 1.07 theoretical-residency
#     waves ≈ 107%, 即略超一个完整 wave, 足以让 kernel 进入稳定的
#     内存/发射节奏而非 launch-bound 极值）;
# (2) D=128 是常见 LLM head_dim（如 7B/13B 级模型的 GQA head dim）,
#     该形状代表真实推理中每 token 每 head 的 RoPE 单元。
PRIMARY_TARGET = (1024, 128)

# cos/sin 表行数（max_seq_len）: 4096（事实记录, 覆盖最大基准形状
# M=4096 的 sequential 位置模式 0..4095）。
MAX_SEQ_LEN = 4096
ROPE_BASE = 10000.0


def make_rotary_table(max_seq_len: int, D: int, dtype: torch.dtype,
                      device: str = "cuda",
                      base: float = ROPE_BASE) -> tuple[torch.Tensor, torch.Tensor]:
    """确定性 cos/sin 表 (L, D/2), FP32 计算后 cast 到 dtype。

    theta_i = base ** (-2*i/D)（RoPE 标准频数）, 全位置外积后
    cos/sin。计时区外调用（表常驻）。
    """
    if D % 2 != 0:
        raise ValueError(f"D 必须是偶数, 实际 {D}")
    d2 = D // 2
    i = torch.arange(d2, dtype=torch.float32)
    theta = base ** (-2.0 * i / D)              # (D/2,)
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = pos.unsqueeze(1) * theta.unsqueeze(0)  # (L, D/2)
    cos_t = torch.cos(freqs).to(dtype).contiguous()
    sin_t = torch.sin(freqs).to(dtype).contiguous()
    return cos_t.to(device), sin_t.to(device)


def rope_ref(x: torch.Tensor, positions: torch.Tensor,
             cos_t: torch.Tensor, sin_t: torch.Tensor) -> torch.Tensor:
    """参考 RoPE（interleaved pair, FP32 中间, 输出 dtype = x dtype）。

    与内核使用**同一张** cos/sin 表（cast 到 FP32 后运算）。
    """
    if x.dim() != 2:
        raise ValueError(f"期望 2 维 x (M, D)，实际 {x.dim()}D")
    M, D = x.shape
    a = x.float()[:, 0::2]
    b = x.float()[:, 1::2]
    c = cos_t[positions].float()
    s = sin_t[positions].float()
    out_even = a * c - b * s
    out_odd = a * s + b * c
    y = torch.stack((out_even, out_odd), dim=-1).reshape(M, D)
    return y.to(x.dtype).contiguous()


def make_input(M: int, D: int, dtype: torch.dtype, device: str = "cuda",
               seed: int = 0, mode: str = "normal",
               scale: float = 1.0, large: float = 1000.0) -> torch.Tensor:
    """确定性输入生成器（正确性与基准两条路径共享）。

    mode: "normal"（N(0, scale)）/ "zeros" / "tiny"（N(0, 1e-4)）/
    "large"（N(0, large), RoPE 是旋转, 大值检验舍入与溢出边界）。
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    if mode == "normal":
        x = torch.randn(M, D, generator=g, dtype=torch.float32,
                        device=device) * scale
    elif mode == "zeros":
        x = torch.zeros(M, D, dtype=torch.float32, device=device)
    elif mode == "tiny":
        x = torch.randn(M, D, generator=g, dtype=torch.float32,
                        device=device) * 1e-4
    elif mode == "large":
        x = torch.randn(M, D, generator=g, dtype=torch.float32,
                        device=device) * large
    else:
        raise ValueError(f"未知输入 mode: {mode}")
    return x.to(dtype).contiguous()


def make_positions(M: int, max_seq_len: int, dtype=torch.int64,
                   device: str = "cuda", seed: int = 0,
                   pattern: str = "sequential") -> torch.Tensor:
    """确定性位置序列（正确性/基准共享）。

    pattern:
      "sequential"    0, 1, ..., M-1（主基准模式; 要求 M <= max_seq_len）
      "same"          全 7（同一位置重复）
      "random"        固定 seed 的 [0, max_seq_len) 均匀随机
      "repeated"      固定 seed, 从 {0, 1, 2, max_seq_len-1} 采样（高重复）
      "boundary"      0 / 1 / max_seq_len-1 轮换（边界位置）
    """
    if pattern == "sequential":
        if M > max_seq_len:
            raise ValueError(
                f"sequential 模式要求 M <= max_seq_len: {M} > {max_seq_len}")
        p = torch.arange(M, dtype=dtype, device=device)
    elif pattern == "same":
        p = torch.full((M,), 7, dtype=dtype, device=device)
    elif pattern == "random":
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        p = torch.randint(0, max_seq_len, (M,), generator=g,
                          dtype=dtype, device=device)
    elif pattern == "repeated":
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        vals = torch.tensor([0, 1, 2, max_seq_len - 1], dtype=dtype,
                            device=device)
        idx = torch.randint(0, 4, (M,), generator=g, dtype=dtype,
                            device=device)
        p = vals[idx]
    elif pattern == "boundary":
        vals = torch.tensor([0, 1, max_seq_len - 1], dtype=dtype,
                            device=device)
        p = vals[torch.arange(M, dtype=dtype, device=device) % 3]
    else:
        raise ValueError(f"未知位置 pattern: {pattern}")
    return p.contiguous()


class RopeOperator(Operator):
    name = "rope"
    dtypes = ("float16", "float32")
    bench_shapes = BENCH_MATRIX_ROPE
    primary_target = PRIMARY_TARGET
    ncu_kernel_regex = "rope"
    profiles_dir = ROOT / "profiles" / "rope"
    bench_dir = ROOT / "benchmarks" / "rope"
    experiments_dir = ROOT / "experiments" / "rope"

    def build(self):
        from ..build import build
        return build("rope")

    def make_bench_pool(self, M: int, D: int, dtype: torch.dtype,
                        mode: str, seed: int = SEED,
                        pool_size: int = 16) -> BenchPool:
        """预分配全部计时张量（计时区域内永不 malloc / 随机数 / copy）。

        轮换: x 池 + out 池（working_set_bytes = 只这部分）。
        不轮换: positions（sequential, 主基准模式; 引擎不轮换它）。
        共享常驻: cos/sin 表（shared, 字节记录在 shared_bytes 与
        pool_extra; 表在 FP32 计算后 cast 到 dtype, 计时区外）。

        validate 契约（为什么计时 launch 用 validate=False）:
        完整验证里 positions 值域检查（0<=p<L）需要一次**同步** D2H
        拷贝, 逐 launch 强制流同步（v0.4 首跑 baseline 28.8 us 的
        根源）。基准池在构造期已保证: 全部张量连续/设备/dtype 正确,
        positions = 0..M-1 必然 < MAX_SEQ_LEN=4096。因此计时 launch
        传 validate=False（仅 host 元数据检查, 无数据访问无同步, 与
        rmsnorm/softmax 验证开销同级）。正常/测试/negative 路径一律
        用默认 validate=True。
        """
        dev = "cuda"
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        n = pool_size if mode == "streaming" else 1
        xs = [(torch.randn(M, D, generator=g, dtype=torch.float32,
                           device=dev).to(dtype).contiguous())
              for _ in range(n)]
        outs = [torch.empty_like(xs[0]) for _ in xs]
        es = 2 if dtype == torch.float16 else 4

        # 位置: sequential（主基准）; 固定（不随 activation 缓冲轮换）
        pos = make_positions(M, MAX_SEQ_LEN, pattern="sequential",
                             device=dev)
        positions = [pos] * n  # 引擎不轮换 positions（见模块 docstring）

        # 共享 cos/sin 表（真实 inference: 表常驻, activation 变化）
        cos_t, sin_t = make_rotary_table(MAX_SEQ_LEN, D, dtype, device=dev)
        shared = {"cos": cos_t, "sin": sin_t}
        shared_bytes = cos_t.numel() * es + sin_t.numel() * es
        table_bytes = 2 * MAX_SEQ_LEN * (D // 2) * es
        pos_bytes = M * 8
        per = M * D * es
        working_set = n * 2 * per  # 轮换部分: x 池 + out 池
        total_logical = working_set + shared_bytes + pos_bytes

        def launch(ext, variant: str, i: int):
            # validate=False: 跳过 positions 值域 D2H 同步拷贝（池已
            # 预验证, positions=0..M-1<L）; 仍执行全部廉价 host 元数据
            # 检查, 与 rmsnorm/softmax 的逐 launch 验证开销同级。
            ext.forward_into(variant, xs[i], positions[i],
                             shared["cos"], shared["sin"], outs[i],
                             validate=False)

        return BenchPool(xs=xs, outs=outs, pool_size=len(xs),
                         working_set_bytes=working_set, element_size=es,
                         mode=mode, launch=launch, positions=positions,
                         shared=shared, shared_bytes=shared_bytes,
                         pool_extra={
                             "positions_pattern": "sequential",
                             "positions_bytes": pos_bytes,
                             "max_seq_len": MAX_SEQ_LEN,
                             "rope_base": ROPE_BASE,
                             "cos_sin_table_bytes": table_bytes,
                             "total_logical_working_set_bytes": total_logical,
                             "note": "cos/sin 查表 L2 可缓存; "
                                     "total_logical 为逻辑流量口径, "
                                     "非 DRAM 实测",
                         })

    def algorithmic_bytes(self, M: int, D: int, element_size: int) -> int:
        """逻辑算法流量: x 读 + y 写 + cos 查表 + sin 查表 + positions。"""
        return M * D * element_size * 2 + M * (D // 2) * element_size * 2 \
            + M * 8

    def ncu_driver_source(self, variant: str, M: int, D: int) -> str:
        return f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import torch
from cudalab.build import build
from cudalab.operators.rope import make_rotary_table, make_positions

ext = build("rope")
assert "{variant}" in ext.all_variants(), ext.all_variants()
x = torch.randn({M}, {D}, dtype=torch.float16, device="cuda").contiguous()
positions = make_positions({M}, {MAX_SEQ_LEN}, pattern="sequential")
cos_t, sin_t = make_rotary_table({MAX_SEQ_LEN}, {D}, torch.float16)
out = torch.empty_like(x)
# 2 次不计时的预热启动（ncu --launch-skip 2），之后 4 次被剖析。
# validate=False: 跳过值域 D2H 同步, 避免 host 同步干扰 kernel 计时。
for _ in range(6):
    ext.forward_into("{variant}", x, positions, cos_t, sin_t, out,
                     validate=False)
torch.cuda.synchronize()
print("profile driver done")
"""

    def experiment_prefix(self) -> str:
        return "ROPE"

    def run_correctness(self, ext, variant: str,
                        out_dir: Path | None = None) -> dict:
        # 默认输出（v0.5 merge review 2026-09-21, append-only 约定）:
        # experiments/regression/v0.5/rope/ —— 旧默认
        # experiments/rope/correctness/v0.4/ 是 main 已发布的历史记录,
        # 历史 experiment artifact 不可变; v0.5 smoke 的覆盖结果已迁移
        # 到 regression 目录（invalid_inputs_rerun_v0.5.json, 与原始
        # 文件除 generated 时间戳外逐字节一致）, 新验证一律只追加到
        # regression 目录。
        from ..rope_correctness import run_suite, summarize, save_results
        if out_dir is None:
            out_dir = ROOT / "experiments" / "regression" / "v0.5" / "rope"
        out_dir.mkdir(parents=True, exist_ok=True)
        # v0.4 review: 独立表值核对（判表不判核, 补 arith/norm 两门的
        # 共同模式盲区）按 (dtype,D) 记录并折叠进 all_pass。
        table_checks: dict = {}
        results = run_suite(variant, ext, table_checks_out=table_checks)
        saved = save_results(results, out_dir / f"{variant}.json",
                             table_checks=table_checks)
        s = summarize(results)
        table_ok = all(tc["passed"] for tc in table_checks.values())
        return {"all_pass": bool(s["all_pass"] and table_ok),
                "table_check_all_pass": table_ok,
                "summary": s, "saved": str(saved)}

    def run_negative(self, ext, variant: str | None = None,
                     out_dir: Path | None = None) -> dict:
        """per-variant negative 套件（negative_suite_scope =
        "per-variant"; v0.5 merge review 2026-09-21 修复）。

        套件本体（cudalab/rope_negative.py 的 build_cases /
        _post_check_ok）按 variant 参数化（另含固定 variant_ 的
        per-variant 整除性 / v3_half2 对齐回归段）, 但 operator 层此前
        忽略 variant, 恒以默认 rope_baseline 跑 —— CLI 指定的候选变体
        从未被 negative 套件实测。修复后实际测试受测变体:
        默认变体 → 规范 invalid_inputs.json, 其余变体 →
        invalid_inputs_<variant>.json。

        out_dir（append-only 约定）: 默认
        experiments/regression/v0.5/rope/ —— 不覆盖 main 已发布的 v0.4
        记录（v0.5 smoke 的覆盖结果已迁移为
        invalid_inputs_rerun_v0.5.json）。
        """
        from ..rope_negative import run_negative_suite, V as _V
        v = variant or _V
        if out_dir is None:
            out_dir = ROOT / "experiments" / "regression" / "v0.5" / "rope"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = ("invalid_inputs.json" if v == _V
                else f"invalid_inputs_{v}.json")
        return run_negative_suite(ext, out_path=out_dir / name, variant=v)

    def pytorch_ref_latency(self, M: int, D: int, dtype: torch.dtype,
                            iters: int = 200, batch: int = 32) -> dict:
        """PyTorch 实现的延迟参照（implementation context）。

        torch 2.4.1 没有内置 fused RoPE 算子 —— 这里用 Python 参考
        （rope_ref, 含索引与 stack 的多次 kernel 启动）作 context,
        不是公平 kernel 对比, 不产生 headline 数字。
        """
        from ..evaluator.bench import time_call
        if isinstance(dtype, str):
            dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
        x = make_input(M, D, dtype=dtype, seed=SEED, mode="normal")
        positions = make_positions(M, MAX_SEQ_LEN, pattern="sequential")
        cos_t, sin_t = make_rotary_table(MAX_SEQ_LEN, D, dtype)

        def _call():
            return rope_ref(x, positions, cos_t, sin_t)

        try:
            _call()
            torch.cuda.synchronize()
        except Exception as e:
            return {"available": True, "error": f"{type(e).__name__}: {e}",
                    "note": "torch 2.4.1 参考路径失败；不伪造数字"}
        t = time_call(_call, warmup=100, iters=iters, batch=batch)
        return {"available": True,
                "api": "python reference rope_ref（torch 2.4.1 无内置 "
                       "RoPE op, 非公平 fused-kernel 对比）",
                **t,
                "note": "PyTorch implementation context（多 kernel 的 "
                        "Python 参考路径, 不作决策依据）"}


rope = RopeOperator()
