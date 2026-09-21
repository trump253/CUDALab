"""v0.5 口径冲突调查 2: native kernel-loop 的短 warmup (200 launches ≈
22ms) 是否把 GPU 留在 nvidia-smi 空闲缺口后的"降级态"里。

paired-streaming-v2.3 引擎用 ≥300ms 时间制 burn 吸收该态（模块头记录
实测衰减 8–10ms 至 >150ms）; native_timing 只有 200 次 launch warmup。

方法:
  phase S: native_timing(warmup=200, n_windows=10, lpw=64) + 全程
      30ms 间隔采样 nvidia-smi SM 时钟（抓低时钟窗口）。
  phase L: native_timing(warmup=5000, n_windows=10, lpw=64) + 同样采样。
  比较两 phase 的 window 中位数与采样到的 SM 时钟分布。

预测（假设成立时）: S 的前若干 window 中位数偏高且伴随较低 SM 时钟,
L 全程接近稳态; baseline（latency-bound）受影响明显大于 vec4_row
（DRAM 饱和）。

只读测量, 不改动已提交记录。
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def main() -> int:
    from cudalab.build import build
    from cudalab.operators.gemv import make_w, make_x

    ext = build("gemv")
    N = K = 4096
    W = make_w(N, K, torch.float16, seed=0)
    x = make_x(K, torch.float16, seed=1)
    out = torch.empty(N, dtype=torch.float16, device="cuda")

    samples: list[dict] = []
    stop_evt = threading.Event()
    t0_global = [0.0]

    def sampler():
        while not stop_evt.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--id=0",
                     "--query-gpu=clocks.sm",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                clk = float(r.stdout.strip())
                samples.append({"t": round(time.time() - t0_global[0], 3),
                                "sm_mhz": clk})
            except Exception as e:
                print(f"[warn] sample failed: {e}", file=sys.stderr)
            time.sleep(0.03)

    def run_phase(tag: str, warmup: int):
        t0_global[0] = time.time()
        stop_evt.clear()
        th = threading.Thread(target=sampler, daemon=True)
        th.start()
        n0 = len(samples)
        res = ext.native_timing("gemv_baseline", W, x, out,
                                warmup=warmup, n_windows=10,
                                launches_per_window=64)
        stop_evt.set()
        th.join()
        ph = samples[n0:]
        clks = sorted(s["sm_mhz"] for s in ph)
        n = len(clks)
        stats = {
            "warmup_launches": warmup,
            "window_median_us": res.get("window_median_us"),
            "median_us": res.get("median_us"),
            "n_samples": n,
            "sm_mhz_min": clks[0] if n else None,
            "sm_mhz_median": clks[n // 2] if n else None,
            "sm_mhz_p10": clks[max(0, n // 10)] if n else None,
            "first_5_samples_ms": ph[:5],
        }
        print(f"[{tag}] median={stats['median_us']:.3f}us "
              f"clock min={stats['sm_mhz_min']} med={stats['sm_mhz_median']} "
              f"p10={stats['sm_mhz_p10']} n={n}")
        return stats

    # 预热到稳态
    ext.native_timing("gemv_baseline", W, x, out, warmup=500, n_windows=1,
                      launches_per_window=64)
    time.sleep(2.0)

    out_doc = {
        "variant": "gemv_baseline",
        "shape": [N, K],
        "dtype": "float16",
        "phases": {},
        "note": ("口径冲突调查: native 短 warmup(200) vs 长 warmup(5000, "
                 "≈API harness 的 300ms burn) 下的 window 中位数与 SM 时钟"),
    }
    for tag, warmup in (("S_short_warmup200", 200),
                        ("L_long_warmup5000", 5000)):
        out_doc["phases"][tag] = run_phase(tag, warmup)
        time.sleep(3.0)  # 让 GPU 回到空闲缺口态, 模拟真实调用前状态

    p = Path("profiles/gemv/caliber_probe") / \
        "warmup_probe_baseline_4096x4096.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    out_doc["all_samples"] = samples
    p.write_text(json.dumps(out_doc, indent=2))
    print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
