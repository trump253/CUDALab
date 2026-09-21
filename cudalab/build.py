"""构建（或复用）CUDALab 的 PyTorch CUDA 扩展（v0.3: 算子参数化）。

`build(op="rmsnorm")` 为 v0.1/v0.2 行为；`build("softmax")` 构建
kernels/softmax/ 下的 Softmax 扩展（独立的构建目录与内容指纹）。

策略:
- 构建目录: $TORCH_EXTENSIONS_DIR/cudalab_<op>（默认
  /root/.cache/torch_extensions/cudalab_<op>），重复运行绝不重编译；
  源文件变化时 ninja 做增量重建。
- 将（源文件 + 编译参数）的 SHA-256 记录在构建目录中，并在每次加载时
  可审计。rmsnorm 的指纹覆盖范围与 v0.2 完全一致
  （bindings.cpp + 排序后的 *.cu + *_common.h，同序同字节 → 同哈希），
  历史构建缓存不受影响。
- 编译严格位于任何基准计时窗口之外：调用方必须在开始测量之前完成
  扩展的导入。

编译参数: -O3、-lineinfo（供 ncu 源码关联）、针对 RTX 2080 Ti
（Turing）的显式 sm_75 gencode。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXTRA_CUDA_CFLAGS = [
    "-O3",
    "-lineinfo",
    "--use_fast_math",  # 注意: 影响 rsqrtf 的近似精度；已由正确性套件验证
    "-gencode=arch=compute_75,code=sm_75",
]
EXTRA_CFLAGS = ["-O3"]
EXTRA_LDFLAGS = ["-lcuda"]


def _kernel_dir(op: str) -> Path:
    return ROOT / "kernels" / op


def _sources(op: str) -> list[str]:
    KER = _kernel_dir(op)
    srcs = [str(KER / "bindings.cpp")]
    srcs += sorted(str(p) for p in KER.glob("*.cu"))
    return srcs


def _fingerprint(op: str) -> str:
    h = hashlib.sha256()
    KER = _kernel_dir(op)
    # rmsnorm 时恰为 bindings.cpp + sorted(*.cu) + rmsnorm_common.h
    # （与 v0.2 逐字节同序 → 同哈希）。
    for f in _sources(op) + sorted(str(p) for p in KER.glob("*_common.h")):
        h.update(Path(f).read_bytes())
    h.update(" | ".join(EXTRA_CUDA_CFLAGS + EXTRA_CFLAGS).encode())
    return h.hexdigest()[:16]


def _ensure_ninja_on_path():
    """torch.cpp_extension 会 shell 调用 ninja；确保即使在本机（非交互）
    shell（例如作为 ncu 子进程运行时）也能找到它。"""
    import shutil
    if shutil.which("ninja") is None:
        cand = "/root/miniconda3/envs/pytorch/bin"
        if (Path(cand) / "ninja").exists():
            os.environ["PATH"] = cand + os.pathsep + os.environ.get("PATH", "")


def build(op: str = "rmsnorm", force: bool = False, verbose: bool = False):
    """加载扩展，必要时构建。返回模块。"""
    _ensure_ninja_on_path()
    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError as e:
        raise RuntimeError(f"当前解释器中没有 PyTorch: {e}")

    KER = _kernel_dir(op)
    if not KER.exists():
        raise RuntimeError(f"找不到内核目录 {KER}")
    srcs = _sources(op)
    ext_name = f"cudalab_{op}"
    build_dir = Path(os.environ.get(
        "TORCH_EXTENSIONS_DIR", "/root/.cache/torch_extensions")) / ext_name
    build_dir.mkdir(parents=True, exist_ok=True)

    fp = _fingerprint(op)
    marker = build_dir / ".source_hash.json"
    if force and marker.exists():
        marker.unlink()
    if marker.exists():
        old = json.loads(marker.read_text())
        if old.get("hash") != fp:
            # 源码/参数变化 -> 清空构建目录让 ninja 从零重建
            # （已文档化的确定性行为）。
            import shutil
            shutil.rmtree(build_dir, ignore_errors=True)
            build_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"hash": fp, "operator": op,
                                  "sources": srcs}, indent=2))

    ext = load(
        name=ext_name,
        sources=srcs,
        build_directory=str(build_dir),
        extra_cuda_cflags=EXTRA_CUDA_CFLAGS,
        extra_cflags=EXTRA_CFLAGS,
        extra_ldflags=EXTRA_LDFLAGS,
        verbose=verbose,
    )
    return ext


if __name__ == "__main__":
    import torch
    op = sys.argv[1] if len(sys.argv) > 1 else "rmsnorm"
    ext = build(op, force="--force" in sys.argv, verbose=True)
    print(f"[{op}] 可用变体:", ext.variants())
    q = sorted(getattr(ext, "quarantined_variants", lambda: [])())
    if q:
        print(f"[{op}] 被隔离变体（NOT_FOR_NORMAL_DISPATCH，"
              f"仅供显式历史审计）:", q)
    if op == "rmsnorm":
        x = torch.randn(4, 4096, dtype=torch.float16, device="cuda")
        w = torch.randn(4096, dtype=torch.float16, device="cuda")
        for name in ext.variants():
            y = ext.forward(name, x, w, 1e-5)
            print(name, "ok", y.shape, y.dtype)
    elif op == "rope":
        from cudalab.operators.rope import make_rotary_table, make_positions
        x = torch.randn(4, 128, dtype=torch.float16, device="cuda")
        positions = make_positions(4, 4096, pattern="sequential")
        cos_t, sin_t = make_rotary_table(4096, 128, torch.float16)
        for name in ext.variants():
            y = ext.forward(name, x, positions, cos_t, sin_t)
            print(name, "ok", y.shape, y.dtype)
    elif op == "gemv":
        from cudalab.operators.gemv import make_w, make_x
        W = torch.randn(4, 4096, dtype=torch.float16, device="cuda")
        x = torch.randn(4096, dtype=torch.float16, device="cuda")
        out = torch.empty(4, dtype=torch.float16, device="cuda")
        for name in ext.variants():
            y = ext.forward(name, W, x)
            ext.forward_into(name, W, x, out)
            print(name, "ok", y.shape, y.dtype)
        nt = ext.native_timing("gemv_baseline", W, x, out,
                               warmup=50, n_windows=3,
                               launches_per_window=16)
        print("native_timing(gemv_baseline) median_us =", nt["median_us"])
    elif op == "qgemv":
        from cudalab.operators.qgemv import make_w_q, make_x_q
        W_q, scale, _W = make_w_q(4, 4096, seed=0)
        x = make_x_q(4096, seed=1)
        out = torch.empty(4, dtype=torch.float16, device="cuda")
        for name in ext.variants():
            y = ext.forward(name, W_q, scale, x)
            ext.forward_into(name, W_q, scale, x, out)
            print(name, "ok", y.shape, y.dtype)
        nt = ext.native_timing("qgemv_baseline", W_q, scale, x, out,
                               warmup=50, n_windows=3,
                               launches_per_window=16)
        print("native_timing(qgemv_baseline) median_us =", nt["median_us"])
    else:
        x = torch.randn(4, 4096, dtype=torch.float16, device="cuda")
        for name in ext.variants():
            y = ext.forward(name, x)
            print(name, "ok", y.shape, y.dtype)
    torch.cuda.synchronize()
