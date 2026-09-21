"""v0.5 口径冲突调查: API-path (32-launch 块 + Python 间隙) vs
native kernel-loop (C++ 连续 raw launch) 的持续负载 DVFS 探测。

方法:
  phase A (sustained): 子进程内连续 5000+ 次 raw launch（约 500ms 持续
      满载 DRAM）, 主进程每 100ms 采样 nvidia-smi SM 时钟/功耗/温度。
  phase B (bursty): 32 次 forward_into 为一块, 块间 sleep 200µs（模拟
      bench harness 的 32-launch 块 + Python 轮转开销）, 同样采样。
  比较两阶段的稳态 SM 时钟 —— 若 sustained 显著更低, 说明连续满载触发
  更低的稳态 DVFS 时钟, 解释 native(连续) > API(突发) 的口径差。

只读测量: 不修改任何已提交的 benchmark/profile 记录。
输出: stdout 摘要 + --out 指定的 JSON（存 profiles/gemv/caliber_probe/）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from cudalab.build import build
    from cudalab.operators.gemv import make_w, make_x

    ext = build("gemv")
    N = K = 4096
    W = make_w(N, K, torch.float16, seed=0)
    x = make_x(K, torch.float16, seed=1)
    out = torch.empty(N, dtype=torch.float16, device="cuda")
    variant = "gemv_baseline"

    import threading

    samples: list[dict] = []
    stop_evt = threading.Event()
    t0_global = [0.0]

    def sampler(phase: str):
        while not stop_evt.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--id=0",
                     "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                clk, pw, tmp = (float(v) for v in
                                r.stdout.strip().split(","))
                samples.append({"phase": phase,
                                "t": round(time.time() - t0_global[0], 3),
                                "sm_mhz": clk, "power_w": pw,
                                "temp_c": tmp})
            except Exception as e:  # 采样失败不中断负载
                print(f"[warn] sample failed: {e}", file=sys.stderr)
            time.sleep(0.05)

    def sustained():
        # 与 native_timing 相同的 C++ raw-launch 路径, 拉长为持续负载
        ext.native_timing(variant, W, x, out, warmup=200, n_windows=24,
                          launches_per_window=256)
        torch.cuda.synchronize()

    def bursty():
        for _ in range(240):
            for _ in range(32):
                ext.forward_into(variant, W, x, out)
            time.sleep(0.0002)
        torch.cuda.synchronize()

    # 预热, 让分配与 JIT 稳定
    sustained()
    time.sleep(1.0)

    phases = [("A_sustained", sustained), ("B_bursty", bursty)]
    result = {"variant": variant, "shape": [N, K], "phases": {}}
    for name, fn in phases:
        t0_global[0] = time.time()
        stop_evt.clear()
        th = threading.Thread(target=sampler, args=(name,), daemon=True)
        th.start()
        n0 = len(samples)
        t_start = time.time()
        fn()
        stop_evt.set()
        th.join()
        time.sleep(0.3)  # 让尾部采样落地
        t_end = time.time()
        ph = [s for s in samples[n0:]]
        clks = sorted(s["sm_mhz"] for s in ph)
        n = len(clks)
        result["phases"][name] = {
            "n_samples": n,
            "sm_mhz_min": clks[0],
            "sm_mhz_median": clks[n // 2],
            "sm_mhz_p10": clks[max(0, n // 10)],
            "sm_mhz_p90": clks[min(n - 1, 9 * n // 10)],
            "power_w_median": sorted(s["power_w"] for s in ph)[n // 2],
            "temp_c_max": max(s["temp_c"] for s in ph),
            "duration_s": round(t_end - t_start, 2),
        }
        print(f"[{name}] median={result['phases'][name]['sm_mhz_median']}MHz "
              f"p10={result['phases'][name]['sm_mhz_p10']} "
              f"p90={result['phases'][name]['sm_mhz_p90']}")

    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        result["samples"] = samples
        p.write_text(json.dumps(result, indent=2))
        print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
