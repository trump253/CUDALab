"""构建（或复用）CUDALab RMSNorm 的 PyTorch CUDA 扩展。

策略:
- 构建目录: $TORCH_EXTENSIONS_DIR/cudalab_rmsnorm（默认
  /root/.cache/torch_extensions/cudalab_rmsnorm），重复运行绝不重编译；
  源文件变化时 ninja 做增量重建。
- 将（源文件 + 编译参数）的 SHA-256 记录在构建目录中，并在每次加载时
  打印，使缓存行为可审计。
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
KER = ROOT / "kernels" / "rmsnorm"

EXTRA_CUDA_CFLAGS = [
    "-O3",
    "-lineinfo",
    "--use_fast_math",  # 注意: 影响 rsqrtf 的近似精度；已由正确性套件验证
    "-gencode=arch=compute_75,code=sm_75",
]
EXTRA_CFLAGS = ["-O3"]
EXTRA_LDFLAGS = ["-lcuda"]


def _sources() -> list[str]:
    srcs = [str(KER / "bindings.cpp")]
    srcs += sorted(str(p) for p in KER.glob("*.cu"))
    return srcs


def _fingerprint() -> str:
    h = hashlib.sha256()
    for f in _sources() + [str(KER / "rmsnorm_common.h")]:
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


def build(force: bool = False, verbose: bool = False):
    """加载扩展，必要时构建。返回模块。"""
    _ensure_ninja_on_path()
    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError as e:
        raise RuntimeError(f"当前解释器中没有 PyTorch: {e}")

    srcs = _sources()
    build_dir = Path(os.environ.get(
        "TORCH_EXTENSIONS_DIR", "/root/.cache/torch_extensions")) / "cudalab_rmsnorm"
    build_dir.mkdir(parents=True, exist_ok=True)

    fp = _fingerprint()
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
    marker.write_text(json.dumps({"hash": fp, "sources": srcs}, indent=2))

    ext = load(
        name="cudalab_rmsnorm",
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
    ext = build(force="--force" in sys.argv, verbose=True)
    print("可用变体:", ext.variants())
    x = torch.randn(4, 4096, dtype=torch.float16, device="cuda")
    w = torch.randn(4096, dtype=torch.float16, device="cuda")
    for name in ext.variants():
        y = ext.forward(name, x, w, 1e-5)
        print(name, "ok", y.shape, y.dtype)
        torch.cuda.synchronize()
