"""CUDALab v0.2 — round-level paired 统计（纯 CPU，无 GPU / torch 依赖）。

v0.3: 原样移入通用 evaluator 核心（cudalab/evaluator/stats.py），
代码不变；cudalab/stats.py 保留为兼容 re-export。

关键方法论（v0.1 的问题）:
- 500 个连续 GPU event sample 不能当作 500 个独立样本 —— 同一轮内的
  sample 存在热相关、时钟相关、缓存相关、调度相关。
- v0.2 的统计单位是 **独立 benchmark round**：每个 round 内部
  （parent, candidate）时间相邻、同张量；round 之间才允许做统计推断。
- bootstrap CI 基于 round-level speedup 数组（对 median 做重抽样），
  固定 seed → 确定性、可复现（纯 Python random，跨平台稳定）。
"""
from __future__ import annotations

import random
import statistics

BOOTSTRAP_SEED = 20260919   # 固定 seed，可复现
BOOTSTRAP_N = 10000
CI_ALPHA = 0.05


def paired_speedups(parent_round_us: list[float],
                    candidate_round_us: list[float]) -> list[float]:
    """per-round paired speedup = parent / candidate（>1 = candidate 更快）。

    两个列表必须等长（同一 round 的配对测量）。
    """
    if len(parent_round_us) != len(candidate_round_us):
        raise ValueError(
            f"paired 长度不一致: {len(parent_round_us)} vs {len(candidate_round_us)}")
    out = []
    for p, c in zip(parent_round_us, candidate_round_us):
        if c <= 0 or p <= 0:
            raise ValueError(f"非法时间值: parent={p} candidate={c}")
        out.append(p / c)
    return out


def summarize(speedups: list[float]) -> dict:
    if not speedups:
        return {"n": 0, "median": None, "mean": None, "min": None,
                "max": None, "faster_count": 0, "faster_fraction": None}
    return {
        "n": len(speedups),
        "median": round(statistics.median(speedups), 6),
        "mean": round(statistics.fmean(speedups), 6),
        "min": round(min(speedups), 6),
        "max": round(max(speedups), 6),
        "faster_count": sum(1 for s in speedups if s > 1.0),
        "faster_fraction": round(sum(1 for s in speedups if s > 1.0)
                                 / len(speedups), 4),
    }


def bootstrap_ci(values: list[float], statistic: str = "median",
                 n_boot: int = BOOTSTRAP_N, alpha: float = CI_ALPHA,
                 seed: int = BOOTSTRAP_SEED) -> tuple[float, float] | None:
    """对 round-level speedup 数组做 percentile bootstrap CI。

    固定 seed → 相同输入永远得到相同 CI（单元测试验证）。
    n < 3 时返回 None（样本太少，不假装可信）。
    """
    n = len(values)
    if n < 3:
        return None
    rng = random.Random(seed)
    fn = statistics.median if statistic == "median" else statistics.fmean
    stats = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(fn(sample))
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[int((1.0 - alpha / 2) * n_boot) - 1]
    return [round(lo, 6), round(hi, 6)]


# ---- DVFS guard 纯函数（无 GPU 可单元测试）--------------------------------
# 背景: 本容器无法锁定 GPU 时钟（nvidia-smi 显示 idle 300 MHz，
# max 2100 MHz，DVFS 范围大）。v0.1 的 EXP-0007 曾出现 parent ~1350
# MHz vs candidate ~1905 MHz 的配对测量（相对差 ~34%），1.231× 被
# boost 状态放大。v0.2 规则: 有效时钟差 > tol 的 round 判 INVALID_DVFS，
# 不进入 KEEP/REJECT 统计。


def check_dvfs_pair(sm_before, sm_after_first, sm_after_second,
                    parent_first: bool, tol: float) -> tuple:
    """paired round 的 DVFS 判定。

    每个 variant 的有效时钟 = 其测量区间前后两次 nvidia-smi 采样的
    均值；parent 与 candidate 有效时钟相对差 > tol → round 无效。
    返回 (valid, reason, eff_parent_mhz, eff_candidate_mhz)。
    """
    if not (sm_before and sm_after_first and sm_after_second):
        return False, "no_clock_data", None, None
    eff_first = (sm_before + sm_after_first) / 2.0
    eff_second = (sm_after_first + sm_after_second) / 2.0
    eff_parent = eff_first if parent_first else eff_second
    eff_cand = eff_second if parent_first else eff_first
    m = (eff_parent + eff_cand) / 2.0
    ok = (abs(eff_parent - eff_cand) / m) <= tol if m > 0 else False
    return ok, (None if ok else "INVALID_DVFS"), eff_parent, eff_cand


def check_dvfs_matrix(sms: list, tol: float) -> tuple:
    """矩阵 round 的 DVFS 判定: 该 round 内所有 variant 有效时钟
    （相邻两次采样均值）的极差/均值 <= tol。返回 (valid, reason, effs)。"""
    if not sms or not all(sms):
        return False, "no_clock_data", []
    effs = [(sms[i] + sms[i + 1]) / 2.0 for i in range(len(sms) - 1)]
    spread = (max(effs) - min(effs)) / statistics.fmean(effs)
    ok = spread <= tol
    return ok, (None if ok else "INVALID_DVFS"), effs
