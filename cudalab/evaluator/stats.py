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

import math
import random
import statistics

BOOTSTRAP_SEED = 20260919   # 固定 seed，可复现
BOOTSTRAP_N = 10000
CI_ALPHA = 0.05

# ---- v2.3 filter-sensitivity（raw vs filtered，纯 CPU 可单测）--------------
# 背景: v2.2 的 guard 不对称（只拒慢），且记录只保存过滤后的中位数，
# 无法审计"guard 是否改变了结论"。v2.3 每个 block 同时保存 raw（全部
# 样本）与 filtered（accepted）两套统计；pair 级比较 raw_speedup 与
# filtered_speedup：方向翻转（跨 1.0）或相对差 > 10%（ratio 空间，
# 等价 log 空间 |Δ| > log(1.10)）→ FILTER_SENSITIVE → 决策 UNSTABLE
# （不强行 KEEP/REJECT）。规则固定、确定性、parent/candidate 对称。
FILTER_LOG_DELTA = math.log(1.10)  # ≈ 0.0953；10% 相对差（ratio 空间）

# ---- v2.3 对称 guard 参数与纯函数（无 torch, 可 CPU 单测）-------------------
# 判据: 对称偏差 |log(t/ref)| > log(F)。parent/candidate（或矩阵中任意
# variant）使用完全相同的规则与阈值——固定规则、明确、可解释、可单测。
SPIKE_FACTOR = 1.5      # per-sample guard 的 F（快/慢对称）
SPIKE_WINDOW = 50       # per-sample guard 基线 = 最近 N 个 accepted 样本中位数
CROSSBLOCK_FACTOR = 1.15  # cross-block guard 的 F（均匀快/慢块对称）
CROSSBLOCK_WARMUP = 3     # 每 variant 前 N 个 block 不判（吸收 run 前空隙）


def apply_spike_guard(clean: list[float], t_us: float,
                      spike_factor: float = SPIKE_FACTOR,
                      window: int = SPIKE_WINDOW) -> tuple[bool, str | None]:
    """v2.3 对称 per-sample guard（纯 CPU，可单测）。

    基线 = 最近 `window` 个 accepted 样本的中位数。对称判据（等价 log
    空间 |log(t/ref)| > log(spike_factor)）:
    - t_us > ref * spike_factor  -> 拒绝，reason="slow"
    - t_us < ref / spike_factor  -> 拒绝，reason="fast"
    - 否则接受（reason=None）。
    无 accepted 基线时（block 首个样本）恒接受。边界（恰好 = 阈值）
    接受（严格不等式）。parent/candidate 完全同一规则（对称）。
    """
    if not clean:
        return True, None
    ref = statistics.median(clean[-window:])
    if t_us > ref * spike_factor:
        return False, "slow"
    if t_us < ref / spike_factor:
        return False, "fast"
    return True, None


def block_stats(raw_samples_us: list[float],
                spike_factor: float = SPIKE_FACTOR,
                window: int = SPIKE_WINDOW) -> dict:
    """v2.3 raw + filtered 双套统计（纯 CPU，可单测）。

    逐样本应用 apply_spike_guard：
    - raw: 全部样本（guard 前）；
    - filtered: accepted 样本（guard 后，进入 block 中位数 / round 统计）。

    返回字段（记录原样落盘，供审计 "guard 是否改变结论"）:
    median_us / raw_median_us / n_raw / n_accepted / n_rejected_fast /
    n_rejected_slow + legacy alias（n_clean = n_accepted, n_spike =
    fast+slow，v2.1/v2.2 消费者兼容）+ 全量样本列表（3 位小数）。
    """
    accepted: list[float] = []
    n_fast = n_slow = 0
    for t in raw_samples_us:
        ok, why = apply_spike_guard(accepted, t, spike_factor, window)
        if ok:
            accepted.append(t)
        elif why == "fast":
            n_fast += 1
        else:
            n_slow += 1
    return {
        "median_us": statistics.median(accepted) if accepted else None,
        "raw_median_us": statistics.median(raw_samples_us)
        if raw_samples_us else None,
        "n_raw": len(raw_samples_us),
        "n_accepted": len(accepted),
        "n_rejected_fast": n_fast,
        "n_rejected_slow": n_slow,
        # legacy alias（v2.1/v2.2 记录字段名，只增不改旧语义）
        "n_clean": len(accepted),
        "n_spike": n_fast + n_slow,
        "raw_samples_us": [round(t, 3) for t in raw_samples_us],
        "accepted_samples_us": [round(t, 3) for t in accepted],
    }


def crossblock_flag(hist: list[float], med: float | None,
                    factor: float = CROSSBLOCK_FACTOR,
                    warmup: int = CROSSBLOCK_WARMUP) -> dict:
    """v2.3 对称 cross-block guard（纯 CPU，可单测）。

    hist: 该 variant 本 run 内此前各 block 的 filtered 中位数（调用方
    持有；本函数在判定后把当前 med 追加进 hist，warmup block 同样
    追加——与 v2.2 行为一致，只是判据对称化）:
    - warmup 前（已测 block < warmup）: 不判（flagged=False）；
    - 否则 ratio = med / median(hist)；
      ratio > factor       -> flagged, direction="slow"（均匀慢块）
      ratio < 1 / factor   -> flagged, direction="fast"（均匀快块，v2.3 新增）
      其余                 -> 不判。
    med 为 None（无 accepted 样本）时不判、不入 hist。
    返回 {running_med_us, ratio, flagged, direction}。
    """
    info: dict = {"running_med_us": None, "ratio": None,
                  "flagged": False, "direction": None}
    if med is None:
        return info
    if len(hist) >= warmup:
        running_med = statistics.median(hist)
        ratio = med / running_med if running_med > 0 else float("inf")
        info["running_med_us"] = round(running_med, 3)
        info["ratio"] = round(ratio, 4)
        if ratio > factor:
            info["flagged"] = True
            info["direction"] = "slow"
        elif ratio < 1.0 / factor:
            info["flagged"] = True
            info["direction"] = "fast"
    hist.append(med)
    return info


def filter_sensitive(raw_speedup: float | None,
                     filtered_speedup: float | None,
                     log_delta: float = FILTER_LOG_DELTA) -> tuple[bool, str]:
    """比较 raw（未过滤）与 filtered（guard 过滤后）的 paired speedup
    （均为 parent/candidate，>1 = candidate 更快）。

    返回 (sensitive, reason)：
    - 任一缺失/非法 → (False, 未评估)（不假装可信，也不强行敏感）；
    - 方向翻转（一个 <1、另一个 >1，严格跨 1.0）→ True；
    - |log(filtered/raw)| > log_delta（≈10% 相对差）→ True；
    - 否则 False。
    确定性：相同输入永远相同输出（单元测试覆盖）。
    """
    if (raw_speedup is None or filtered_speedup is None
            or raw_speedup <= 0 or filtered_speedup <= 0):
        return False, "raw/filtered speedup 缺失或非法，未评估"
    if (raw_speedup < 1.0 < filtered_speedup
            or filtered_speedup < 1.0 < raw_speedup):
        return True, (f"方向翻转: raw {raw_speedup:.4f} 与 filtered "
                      f"{filtered_speedup:.4f} 分居 1.0 两侧")
    delta = abs(math.log(filtered_speedup / raw_speedup))
    if delta > log_delta:
        return True, (f"|log(filtered/raw)| = {delta:.4f} > "
                      f"{log_delta:.4f}（≈10% 相对差）")
    return False, f"在阈值内（|log 差| {delta:.4f} <= {log_delta:.4f}）"


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
