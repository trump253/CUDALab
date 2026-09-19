"""CUDALab v0.3 — GPU 状态采样（nvidia-smi，尽力而为）。

从 v0.2 `cudalab/bench_v2.py` 原样抽出（行为不变），供 paired benchmark
引擎与 profiler 共用：

- `now_iso`:        ISO 8601 时间戳（带时区），程序生成，不手填。
- `gpu_state`:      完整 nvidia-smi 快照（round 前/后）。
- `gpu_clocks`:     轻量时钟/温度/功耗采样（round 内高频使用）。
- `condense_clocks`: 记录用精简时钟字段。
- `rel_diff`:       相对差（DVFS guard 使用）。
"""
from __future__ import annotations

import subprocess
import time


def now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def gpu_state() -> dict:
    """完整 nvidia-smi 快照（round 前/后，尽力而为）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,temperature.gpu,clocks.sm,clocks.mem,"
             "power.draw,utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout
        fields = out.strip().splitlines()[0].split(",")
        keys = ["gpu_index", "name", "temp_c", "sm_clock_mhz", "mem_clock_mhz",
                "power_w", "gpu_util_pct", "mem_util_pct"]
        d = {}
        for k, f in zip(keys, fields):
            try:
                d[k] = float(f) if "." in f else int(f)
            except ValueError:
                d[k] = f
        d["ts"] = time.time()
        return d
    except Exception as e:  # 尽力而为
        return {"error": str(e)}


def gpu_clocks() -> dict:
    """轻量 nvidia-smi 采样（round 内高频使用，只取时钟/温度/功耗）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        f = out.strip().splitlines()[0].split(",")
        return {"sm_clock_mhz": int(f[0]), "mem_clock_mhz": int(f[1]),
                "temp_c": int(f[2]), "power_w": float(f[3])}
    except Exception:
        return {}


def condense_clocks(c: dict) -> dict:
    return {k: c.get(k) for k in ("sm_clock_mhz", "temp_c", "power_w")}


def rel_diff(a: float, b: float) -> float:
    m = (a + b) / 2.0
    return abs(a - b) / m if m > 0 else 0.0
