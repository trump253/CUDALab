"""CUDALab v0.3 — 通用 GPU 剖析器集成（Nsight Compute 2022.3，sm_75）。

在小型专用剖析驱动程序上运行 `ncu --csv`，并把报告转换为结构化
JSON 摘要。指标取不到时字段为 null —— 绝不伪造数字。ncu 原始输出
（stdout+stderr）始终保留在目标目录的 raw/ 子目录供审计。

指标名已在本 GPU / 本 ncu 版本上用 `ncu --query-metrics` 核实
（核实过程见 profiles/rmsnorm 的 v0.2 记录）。

v0.2 方法学审计（NCU 2022.3.0；v0.2.1 修正 --cache-control 语义，此前写反）:
- `--cache-control {all,none}`，默认 **all**。语义（依据本机 ncu 2022.3
  --help + 本仓 raw 输出 + NVIDIA 文档核实）：
  - **all**（默认）= cache flush/reset profiling：NCU 在每个 replay pass
    前失效全部 GPU 缓存，得到确定性的 flushed 状态；
  - **none** = no-flush profiling：不失效缓存，缓存状态不受控（可能保留
    前序活动残留），ncu 输出 "Running with uncontrolled GPU caches" 警告。
  为避免歧义，本仓统一称 all 为 "cache flush/reset"、none 为 "no-flush"，
  不称 "hot/cold L2"（除非能从 replay 配置严格推出）。
- v0.1 未传该参数，走默认 **all**（= 失效/flush），其 "cold L2" 说法与
  默认配置**一致**（此前 v0.2 误判为"无配置依据/实际热"，v0.2.1 更正）。
  v0.2 起两种模式都显式记录在 JSON 里。
- `--clock-control {base,none,reset}`，默认 base = 尝试锁到 base clock；
  容器内可能失败，stderr 警告会被提取到 `clock_lock_warning` 字段。

解读说明（sm_75，NCU 2022.3）:
- `gpu__time_duration.sum` 在 --csv 中报告的单位是 **nsecond**；此处
  换算为微秒。
- 停顿原因使用 `smsp__average_warps_issue_stalled_<r>_per_issue_active.ratio`
  = 每 issue-active 周期的停顿 warp 周期数（单位 `inst`）。这些指标的
  派生 `.pct` 字段无意义（>100%），因此这里的百分比按"占全部停顿
  原因之和的份额"计算。

算子差异（driver 源码、ncu kernel regex、输出目录）由 operator
adapter 提供，本模块不含任何算子代码。
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
from pathlib import Path

NCU = "/usr/local/bin/ncu"
# 本容器内 ncu 无法 exec 符号链接的解释器；使用真实二进制。
PYTHON_REAL = "/root/miniconda3/envs/pytorch/bin/python3.10"

# 已核实可用的指标（NCU 2022.3，sm_75）。
METRICS = [
    "gpu__time_duration.sum",
    "l1tex__t_sector_hit_rate",
    "lts__t_sector_op_read_hit_rate",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active",
    "smsp__average_warps_issue_stalled_wait_per_issue_active",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active",
    "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active",
    "smsp__average_warps_issue_stalled_drain_per_issue_active",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active",
    "smsp__average_warps_issue_stalled_selected_per_issue_active",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active",
    "smsp__average_warps_issue_stalled_sleeping_per_issue_active",
    "smsp__average_warps_issue_stalled_membar_per_issue_active",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active",
]

STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active"


def _parse_csv_launches(text: str) -> tuple[list[dict], str | None, list[str | None]]:
    """解析 ncu --csv 输出。

    返回（每次启动的指标 dict 列表, 最后见到的内核名, 每次启动的内核名列表）。
    v0.5: 第三次返回值是增量新增 —— GEMV split-K 一次算子调用发射两个内核
    （partials + combine）, 旧实现只保留"最后一个"内核名, 跨内核平均的
    kernel_duration_us 会误导（把 [131.9µs, 2.4µs] 平均成 67.2µs）。
    """
    # 去掉表头之前的非 CSV 进度行
    lines = text.splitlines()
    start = 0
    for i, ln in enumerate(lines):
        if ln.startswith('"ID"'):
            start = i
            break
    if start == 0 and not text.lstrip().startswith('"ID"'):
        return [], None
    rows = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
    header = rows[0]
    idx = {name: i for i, name in enumerate(header)}
    launches: dict[str, dict] = {}
    names: dict[str, str] = {}
    kernel_name = None
    for r in rows[1:]:
        if len(r) < len(header):
            continue
        lid = r[idx["ID"]]
        m = launches.setdefault(lid, {})
        metric = r[idx["Metric Name"]]
        unit = r[idx["Metric Unit"]]
        try:
            val = float(r[idx["Metric Value"]])
        except ValueError:
            continue
        names[lid] = r[idx["Kernel Name"]]
        kernel_name = r[idx["Kernel Name"]]
        m[(metric, unit)] = val
    lid_order = list(launches.keys())
    return (list(launches.values()), kernel_name,
            [names.get(l) for l in lid_order])


def _avg(launches: list[dict], metric: str, unit: str = None):
    vals = []
    for d in launches:
        for (m, u), v in d.items():
            if m == metric and (unit is None or u == unit):
                vals.append(v)
    if not vals:
        return None
    return sum(vals) / len(vals)


def ncu_version(ncu: str = NCU) -> str:
    try:
        out = subprocess.run([ncu, "--version"], capture_output=True, text=True,
                             timeout=30).stdout
        m = re.search(r"Version\s+([\d.]+)", out)
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"


def profile_variant(variant: str, M: int, H: int,
                    driver_src: str, kernel_regex: str,
                    out_path: Path,
                    ncu: str = NCU,
                    launch_skip: int = 2, launch_count: int = 4,
                    cache_control: str = "all",
                    clock_control: str = "base") -> dict:
    """剖析单个变体在单个形状上的表现；返回（并保存）摘要。

    driver_src:    算子专用的 ncu 驱动脚本源码（adapter 提供）。
    kernel_regex:  ncu `-k regex:` 过滤（如 "rmsnorm" / "softmax"）。
    out_path:      摘要 JSON 路径（raw 输出保存在 out_path 同级的 raw/）。

    cache_control（v0.2.1 修正语义，此前写反）:
      "all"  (ncu 默认) —— cache flush/reset profiling：NCU 在每个
          replay pass 前失效全部 GPU 缓存（确定性 flushed 状态）；
      "none"            —— no-flush profiling：不失效缓存，状态不受控
          （可能保留前序残留），ncu 警告 "Running with uncontrolled GPU
          caches"。
    clock_control:
      "base" (ncu 默认) —— 尝试把 GPU 锁定到 base clock（容器内可能
      失败，原始输出会保留以便审计）；"none" —— 不锁频。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = out_path.parent / "raw"
    raw_dir.mkdir(exist_ok=True)

    tag = f"{variant}_M{M}_H{H}_cc{cache_control}_clk{clock_control}"
    drv = raw_dir / f"{tag}_drv.py"
    drv.write_text(driver_src)
    raw_txt = raw_dir / f"{tag}.ncu.txt"

    cmd = [
        ncu, "--csv",
        "-k", f"regex:{kernel_regex}",
        "--launch-skip", str(launch_skip),
        "--launch-count", str(launch_count),
        "--metrics", ",".join(METRICS),
        "--cache-control", cache_control,
        "--clock-control", clock_control,
        PYTHON_REAL, str(drv),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    except FileNotFoundError:
        raise RuntimeError(f"在 {ncu} 找不到 ncu")
    raw_txt.write_text("$ ncu cmd\n" + " ".join(cmd) + "\n\n=== stdout ===\n"
                       + proc.stdout + "\n=== stderr ===\n" + proc.stderr)

    summary: dict = {
        "variant": variant,
        "shape": [M, H],
        "ncu_version": ncu_version(ncu),
        "cache_control": cache_control,
        "clock_control": clock_control,
        "clock_lock_warning": None,  # 从 stderr 提取的锁频警告（如有）
        "kernel_name": None,
        "kernel_duration_us": None,
        "l1_hit_rate": None,        # {value, unit} —— 不做百分比假设
        "l2_read_hit_rate": None,
        "dram_throughput_pct": None,
        "sm_throughput_pct": None,
        "achieved_occupancy_pct": None,
        "registers_per_thread": None,
        "shared_memory_bytes": None,
        "warp_stalls": {},
        "n_launches_profiled": 0,
        "raw_report": str(raw_txt),
    }

    if proc.returncode != 0:
        summary["error"] = (proc.stderr + proc.stdout).strip()[-2000:]
        out_path.write_text(json.dumps(summary, indent=2))
        return summary

    launches, kernel_name, launch_kernels = _parse_csv_launches(proc.stdout)
    if not launches:
        summary["error"] = "ncu exited 0 but no metric rows parsed; see raw report"
        out_path.write_text(json.dumps(summary, indent=2))
        return summary

    # 锁频是否真正生效：容器内 ncu 无法锁频时 stderr 会有警告。
    for ln in proc.stderr.splitlines():
        low = ln.lower()
        if "clock" in low and ("fail" in low or "unable" in low or "cannot" in low
                               or "not supported" in low or "denied" in low):
            summary["clock_lock_warning"] = ln.strip()
            break

    summary["n_launches_profiled"] = len(launches)
    summary["kernel_name"] = kernel_name

    # v0.5（GEMV split-K 触发）: 一次算子调用发射多个内核时, 上面所有顶层
    # 标量是"跨内核平均"（如 [131.9µs partials, 2.4µs combine] 平均成
    # 67.2µs）, 方向都可能误导。新增逐内核分解; 单内核算子该列表恰好
    # 1 条, 与顶层标量一致。顶层标量字段本身保持不变（向后兼容）。
    def _metric_of(launch: dict, metric: str, unit: str):
        for (m, u), v in launch.items():
            if m == metric and u == unit:
                return v
        return None

    distinct = []
    for nm in launch_kernels:
        if nm is not None and nm not in distinct:
            distinct.append(nm)
    kernels_out = []
    for nm in distinct:
        idxs = [i for i, k in enumerate(launch_kernels) if k == nm]
        durs = [_metric_of(launches[i], "gpu__time_duration.sum", "nsecond")
                for i in idxs]
        durs = [d for d in durs if d is not None]
        ddr = [_metric_of(launches[i],
                          "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                          "%") for i in idxs]
        ddr = [d for d in ddr if d is not None]
        kernels_out.append({
            "kernel_name": nm,
            "n_launches": len(idxs),
            "duration_us": round(sum(durs) / len(durs) / 1e3, 3)
            if durs else None,
            "dram_throughput_pct": round(sum(ddr) / len(ddr), 2)
            if ddr else None,
        })
    summary["kernels"] = kernels_out
    if len(distinct) > 1:
        summary["multi_kernel_note"] = (
            "本次 profile 捕获到 "
            f"{len(distinct)} 个不同内核（算子一次调用发射多次 kernel）; "
            "顶层 kernel_duration_us / dram_throughput_pct 等标量为跨内核"
            "平均, 不可与单内核算子直接比较 —— 以 kernels[] 逐项为准, "
            "算子总时长 = 各内核时长之和")

    def rate(metric: str):
        """平均命中率。ncu --csv 对 ratio 型指标输出三行：
        <m>.max_rate（unit 空）/ <m>.pct（unit %）/ <m>.ratio（unit 空）。
        优先取 .pct 行（百分比数值），缺失时回退 .ratio。"""
        vals = []
        unit = "%"
        for d in launches:
            for (m, u), v in d.items():
                if m == metric + ".pct" and u == "%":
                    vals.append(v)
        if not vals:
            unit = "ratio"
            for d in launches:
                for (m, u), v in d.items():
                    if m == metric + ".ratio":
                        vals.append(v)
        if not vals:
            return None
        return {"value": round(sum(vals) / len(vals), 2), "unit": unit}

    summary["l1_hit_rate"] = rate("l1tex__t_sector_hit_rate")
    summary["l2_read_hit_rate"] = rate("lts__t_sector_op_read_hit_rate")

    def pct(metric: str):
        v = _avg(launches, metric, "%")
        return round(v, 2) if v is not None else None

    dur_ns = _avg(launches, "gpu__time_duration.sum", "nsecond")
    summary["kernel_duration_us"] = round(dur_ns / 1e3, 3) if dur_ns is not None else None
    summary["dram_throughput_pct"] = pct(
        "dram__throughput.avg.pct_of_peak_sustained_elapsed")
    summary["sm_throughput_pct"] = pct(
        "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    summary["achieved_occupancy_pct"] = pct(
        "sm__warps_active.avg.pct_of_peak_sustained_active")
    rgt = _avg(launches, "launch__registers_per_thread", "register/thread")
    summary["registers_per_thread"] = int(rgt) if rgt is not None else None
    sm_s = _avg(launches, "launch__shared_mem_per_block_static", "byte/block")
    sm_d = _avg(launches, "launch__shared_mem_per_block_dynamic", "byte/block")
    if sm_s is not None or sm_d is not None:
        summary["shared_memory_bytes"] = int((sm_s or 0) + (sm_d or 0))

    # 停顿原因: .ratio = 每 issue-active 周期的停顿 warp 周期数
    stalls = {}
    for m in METRICS:
        if m.startswith(STALL_PREFIX) and m.endswith(STALL_SUFFIX):
            reason = m[len(STALL_PREFIX):-len(STALL_SUFFIX)]
            v = _avg(launches, m + ".ratio", "inst")
            if v is not None:
                stalls[reason] = round(v, 3)
    if stalls:
        total = sum(stalls.values())
        summary["warp_stalls"] = {
            k: {"stalled_cycles_per_issue": v,
                "pct_of_stalls": round(100.0 * v / total, 1) if total > 0 else 0.0}
            for k, v in sorted(stalls.items(), key=lambda kv: -kv[1])
        }
    out_path.write_text(json.dumps(summary, indent=2))
    return summary
