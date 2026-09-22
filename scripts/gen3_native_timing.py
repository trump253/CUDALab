#!/usr/bin/env python
# CUDALab v0.7 三代对比（用户 §8）: FP16 / INT8 参照内核的
# native kernel-loop 计时（INT4 侧已有 experiments/int4gemv/
# native_timing_rowtile4_hx_4096.json, 同一 harness 字符串）。
#
# 口径: C++ 内连续 raw launch, CUDA events / 64, 10 windows;
# 输入 seed=1234（与 INT4 侧一致）; 4096x4096。
# 输出: experiments/gemv/native_timing_vec4_row_4096.json
#       experiments/qgemv/native_timing_vec16_row_4096.json
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cudalab.operators.gemv import GemvOperator, make_w, make_x
from cudalab.operators.qgemv import QgemvOperator, make_w_q, make_x_q

ROOT = Path(__file__).resolve().parents[1]
N = K = 4096
HARNESS = ("native kernel-loop: C++ continuous raw launches, "
           "CUDA events / 64, 10 windows")
NOTE = ("原生 kernel-loop 口径: C++ 内连续 launch, CUDA events / N; "
        "不含 Python 边界开销（区别于 bench 引擎 API 路径口径与 "
        "NCU kernel duration）")


def run(ext, name, tensors, out_path: Path) -> None:
    out = torch.empty(N, dtype=torch.float16, device="cuda")
    res = {}
    for tag, warmup in (("w200", 200), ("w5000", 5000)):
        d = ext.native_timing(name, *tensors, out, warmup, 10, 64)
        d = dict(d)
        d["note"] = NOTE
        res[tag] = d
    doc = {
        "variant": name,
        "shape": "4096x4096x4096",
        "harness": HARNESS,
        "windows": 10,
        "launches_per_window": 64,
        "w200": res["w200"],
        "w5000": res["w5000"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2))
    print(out_path)
    print("  w200 median:", res["w200"]["median_us"])
    print("  w5000 median:", res["w5000"]["median_us"])


def main() -> None:
    g = GemvOperator()
    gext = g.build()
    W = make_w(N, K, torch.float16, device="cuda", seed=1234)
    x = make_x(K, torch.float16, device="cuda", seed=1234)
    run(gext, "gemv_vec4_row", (W, x),
        ROOT / "experiments" / "gemv" / "native_timing_vec4_row_4096.json")

    q = QgemvOperator()
    qext = q.build()
    Wq, s, _W = make_w_q(N, K, seed=1234)
    xq = make_x_q(K, seed=1234)
    run(qext, "qgemv_vec16_row", (Wq, s, xq),
        ROOT / "experiments" / "qgemv" /
        "native_timing_vec16_row_4096.json")


if __name__ == "__main__":
    main()
