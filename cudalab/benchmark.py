"""CUDALab 基准测试框架。

方法论（对所有变体固定 —— 绝不按候选微调测量方式）:
- 使用 torch.cuda.Event 计时并显式同步；GPU 路径上绝不用墙钟
  Python time。
- 编译必须先于任何计时完成（调用方：先构建）。
- 固定输入张量：给定（形状, dtype）的每个变体复用同一组 x/w 张量
  （数值相同）。输入只生成一次。
- 计时方法（v1 框架，"cuda-event-batched"）:
  单发事件计时被发现会引入约 6us 的启动开销与噪声，淹没 5-15us
  的内核（它甚至把一个真实的 2.35x ncu 内核时长改进翻转成了表面上的
  回归 —— 见 EXP-0002 作废记录）。因此每个样本是 `batch` 次连续内核
  启动（夹在两个 cuda 事件之间），每样本之后做一次完整同步；
  样本时间 = 耗时 / batch。同一输入上的连续启动正是模型循环内归一化
  算子的真实稳态，且该方法对所有变体完全一致。
- 每个（变体, 形状, dtype）:
    先 warmup 次不计时启动（>=150），
    然后 `iters`（>=100）个计时样本，每样本 `batch`（>=32）次启动，
    `rounds`（>=5）个独立轮次（每轮重新预热）。
- 主指标：所有单发样本（跨轮次）的中位数（MEDIAN）。
  同时报告 p95、min、max 和每轮中位数（供"多数轮次一致更快"的
  判定规则使用）。
- 有效 DRAM 带宽: (M*H*2 + H*2 + M*H*2) 字节 / 中位时间（fp16）。
  这是内核必须搬动的内存流量；作为派生数字报告，绝不伪造。

公平性:
- 同一 GPU（CUDA_VISIBLE_DEVICES=0）、同一输入、同一形状、同一 dtype。
- 每个变体的套件运行前后各做一次 GPU 状态（nvidia-smi）快照。
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

WARMUP = 150      # 不计时的预热启动次数
ITERS = 100       # 每轮计时样本数
BATCH = 32        # 每计时样本的启动次数
ROUNDS = 5
HARNESS_VERSION = "cuda-event-batched-v1"

# 基准矩阵: (M, H)。主要优化目标: (128, 4096)。
BENCH_MATRIX = [
    (1, 4096),
    (16, 4096),
    (128, 4096),
    (1024, 4096),
    (128, 8192),
    (1, 1024),
    (128, 1024),
]
PRIMARY_TARGET = (128, 4096)


def gpu_state() -> dict:
    """经 nvidia-smi 快照关键 GPU 计数器（尽力而为）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout
        fields = out.strip().splitlines()[0].split(",")
        keys = ["gpu_index", "name", "temp_c", "sm_clock_mhz", "mem_clock_mhz",
                "power_w", "gpu_util_pct", "mem_util_pct"]
        d = {}
        for k, f in zip(keys, fields):
            k2 = k.replace(" ", "_")
            try:
                d[k2] = float(f) if "." in f else int(f)
            except ValueError:
                d[k2] = f
        d["name"] = fields[1]
        d["ts"] = time.time()
        return d
    except Exception as e:  # 尽力而为，永不致命
        return {"error": str(e)}


def _time_one_round(fn, warmup: int, iters: int, batch: int) -> list[float]:
    """一个独立轮次。

    先不计时的预热，然后 `iters` 个样本；每个样本 = 两个 cuda 事件之间
    的 `batch` 次连续启动，同步后 time/batch = 单发时间（us）。
    每样本同步保证每个样本都是独立、完全排空的测量（任何地方都没有
    未同步的计时）。
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        for _ in range(batch):
            fn()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop) * 1e3 / batch)  # 单发 us
    return times


def bench_variant(variant: str, ext, M: int, H: int,
                  dtype: torch.dtype = torch.float16,
                  warmup: int = WARMUP, iters: int = ITERS,
                  batch: int = BATCH, rounds: int = ROUNDS) -> dict:
    """基准测试单个（变体, 形状, dtype）。返回完整记录。"""
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(1234)
    x = (torch.randn(M, H, generator=g, dtype=torch.float32, device=dev)
         .to(dtype).contiguous())
    w = (torch.randn(H, generator=g, dtype=torch.float32, device=dev) * 0.5 + 1.0
         ).to(dtype).contiguous()

    out = torch.empty_like(x)

    def fn():
        # 预分配输出: 计时区域 = 仅内核启动，无分配、无拷贝。
        ext.forward_into(variant, x, w, out, 1e-5)

    state_before = gpu_state()
    all_times: list[float] = []
    round_medians: list[float] = []
    for _ in range(rounds):
        t = _time_one_round(fn, warmup, iters, batch)
        all_times.extend(t)
        round_medians.append(statistics.median(t))
    state_after = gpu_state()

    n = M * H
    bytes_moved = (n * 2) + (H * 2) + (n * 2)  # 读 x + 读 w + 写 y（fp16）
    med_us = statistics.median(all_times)
    p95_us = float(sorted(all_times)[int(0.95 * len(all_times)) - 1])

    rec = {
        "variant": variant,
        "shape": [M, H],
        "dtype": str(dtype).split(".")[-1],
        "median_us": round(med_us, 3),
        "p95_us": round(p95_us, 3),
        "min_us": round(min(all_times), 3),
        "max_us": round(max(all_times), 3),
        "round_medians_us": [round(r, 3) for r in round_medians],
        "n_samples": len(all_times),
        "warmup": warmup, "iters": iters, "batch": batch, "rounds": rounds,
        "harness": HARNESS_VERSION,
        "effective_bw_gbps": round(bytes_moved / (med_us * 1e-6) / 1e9, 1),
        "gpu_state_before": state_before,
        "gpu_state_after": state_after,
    }
    del x, w, out
    torch.cuda.empty_cache()
    return rec


def bench_matrix(variants: list[str], ext, shapes: list = BENCH_MATRIX,
                 dtype: torch.dtype = torch.float16) -> list[dict]:
    recs = []
    for v in variants:
        for (M, H) in shapes:
            recs.append(bench_variant(v, ext, M, H, dtype))
    return recs


def annotate_speedups(recs: list[dict],
                      baseline: str = "baseline",
                      pytorch_ref_fn=None) -> list[dict]:
    """添加 speedup_vs_cuda_baseline / speedup_vs_pytorch_reference 列。"""
    by_shape = {}
    for r in recs:
        by_shape.setdefault((tuple(r["shape"]), r["dtype"]), {})[r["variant"]] = r
    for r in recs:
        base = by_shape.get((tuple(r["shape"]), r["dtype"]), {}).get(baseline)
        r["speedup_vs_cuda_baseline"] = (
            round(base["median_us"] / r["median_us"], 4) if base else None)
        r["speedup_vs_pytorch_reference"] = None
    return recs


def save_bench(recs: list[dict], out_dir: Path, tag: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"bench_{tag}.json"
    jp.write_text(json.dumps(recs, indent=2))
    cp = out_dir / f"bench_{tag}.csv"
    if recs:
        cols = ["variant", "M", "H", "dtype", "median_us", "p95_us", "min_us",
                "max_us", "effective_bw_gbps", "speedup_vs_cuda_baseline",
                "speedup_vs_pytorch_reference"]
        lines = [",".join(cols)]
        for r in recs:
            lines.append(",".join(str({
                "M": r["shape"][0], "H": r["shape"][1],
            }.get(c, r.get(c, ""))) for c in cols))
        cp.write_text("\n".join(lines) + "\n")
    return jp, cp
