"""CUDALab v0.3 — 实验追踪 + winner 分类（通用，算子无关）。

一次优化尝试 = 一条实验记录（JSON），失败的也不例外。失败的实验是
一等数据，绝不丢弃。

- `next_experiment_id` / `save_experiment`：按前缀（EXP / SFM / …）
  生成单调 ID 并保存记录（v0.1 的 `cudalab/experiment.py` 保留为
  RMSNorm 专用历史模块，不动）。
- `classify_cell`（v0.3 best/incumbent 语义，RMSNorm v0.2 教训）:
  winner 判定**按 (shape, dtype, cache_mode) 单元格**进行，绝不强制
  全局 winner：
  - winner = valid round 跨轮中位数最小的 variant；
  - winner vs runner-up 的 round-level paired 比值套用 v0.2 固定决策
    规则（decision.decide_v2，correctness 恒 True —— 正确性在实验层
    单独把关）：
      KEEP      -> SIGNIFICANT_WINNER（winner 统计显著优于 runner-up）
      NEUTRAL   -> NO_UNIQUE_WINNER（顶部变体间无统计显著差异）
      REJECT    -> NO_UNIQUE_WINNER（中位数第一但 paired 证据不足，
                   如实记录）
      UNSTABLE  -> UNSTABLE（valid rounds 不足）
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Optional

from . import decision as _decision
from . import stats as _stats

SIGNIFICANT_WINNER = "SIGNIFICANT_WINNER"
NO_UNIQUE_WINNER = "NO_UNIQUE_WINNER"
UNSTABLE = "UNSTABLE"

_STATUS_BY_DECISION = {
    _decision.KEEP: SIGNIFICANT_WINNER,
    _decision.NEUTRAL: NO_UNIQUE_WINNER,
    _decision.REJECT: NO_UNIQUE_WINNER,
    _decision.UNSTABLE: UNSTABLE,
}


def next_experiment_id(exp_dir: Path, prefix: str) -> str:
    """在 exp_dir 中找 prefix-NNNN.json 的最大编号 + 1。"""
    exp_dir.mkdir(parents=True, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}-(\d+)\.json$")
    ids = []
    for p in exp_dir.glob(f"{prefix}-*.json"):
        m = pat.match(p.name)
        if m:
            ids.append(int(m.group(1)))
    return f"{prefix}-{(max(ids) + 1 if ids else 1):04d}"


def save_experiment(record: dict, exp_dir: Path,
                    prefix: str = "EXP") -> Path:
    exp_dir.mkdir(parents=True, exist_ok=True)
    if not record.get("id"):
        record["id"] = next_experiment_id(exp_dir, prefix)
    p = exp_dir / f"{record['id']}.json"
    p.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return p


def classify_cell(variants: list[str], rounds: list[dict],
                  min_valid_rounds: int = _decision.MIN_VALID_ROUNDS) -> dict:
    """从一个矩阵记录的 rounds（含 valid 标记）分类一个单元格。

    rounds: bench_matrix 记录的 "rounds" 列表（{valid, us: {v: us}}）。
    返回 {status, winner, runner_up, medians, median_ratio, ci95,
    faster_rounds, decision, decision_rule, valid_rounds}。
    """
    valid = [r for r in rounds if r["valid"]]
    per = {v: [r["us"][v] for r in valid if v in r["us"]] for v in variants}
    medians = {}
    for v in variants:
        if per[v]:
            medians[v] = round(statistics.median(per[v]), 3)
    ranked = sorted(medians, key=medians.get)

    out = {
        "valid_rounds": len(valid),
        "min_valid_rounds": min_valid_rounds,
        "all_variants_median_us": medians,
        "winner": ranked[0] if ranked else None,
        "runner_up": ranked[1] if len(ranked) > 1 else None,
        "median_ratio": None,
        "bootstrap_ci_95": None,
        "faster_rounds": None,
        "decision": None,
        "decision_rule": None,
        "status": None,
    }
    if len(ranked) < 2:
        out["status"] = UNSTABLE if len(valid) < min_valid_rounds else NO_UNIQUE_WINNER
        out["decision_rule"] = "可比较 variant 不足 2 个"
        return out
    winner, runner = ranked[0], ranked[1]
    out["winner"] = winner
    out["runner_up"] = runner

    if len(valid) < min_valid_rounds:
        out["status"] = UNSTABLE
        out["decision_rule"] = (f"valid DVFS-stable rounds {len(valid)} "
                                f"< {min_valid_rounds} -> UNSTABLE")
        return out

    # runner-up / winner（>1 = winner 更快），round-level paired
    winner_us = [r["us"][winner] for r in valid if winner in r["us"]
                 and runner in r["us"]]
    runner_us = [r["us"][runner] for r in valid if winner in r["us"]
                 and runner in r["us"]]
    ratios = _stats.paired_speedups(runner_us, winner_us)
    s = _stats.summarize(ratios)
    ci95 = _stats.bootstrap_ci(ratios)
    dec, detail = _decision.decide_v2(True, len(ratios), ratios, ci95)
    out.update({
        "median_ratio": s["median"],
        "bootstrap_ci_95": ci95,
        "faster_rounds": f"{s['faster_count']}/{s['n']}",
        "decision": dec,
        "decision_rule": detail.get("rule"),
        "status": _STATUS_BY_DECISION[dec],
    })
    return out
