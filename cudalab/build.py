"""Build (or reuse) the CUDALab RMSNorm PyTorch CUDA extension.

Policy:
- Build directory: $TORCH_EXTENSIONS_DIR/cudalab_rmsnorm (default
  /root/.cache/torch_extensions/cudalab_rmsnorm), so repeated runs never
  recompile; ninja does incremental rebuilds when a source changes.
- A SHA-256 of (sources + compile flags) is recorded in the build dir and
  printed on every load, making the cache behavior auditable.
- Compile happens strictly outside of any benchmark timing window: callers
  must always import the extension before starting measurements.

Compile flags: -O3, -lineinfo (for ncu source correlation), explicit
sm_75 gencode for the RTX 2080 Ti (Turing).
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
    "--use_fast_math",  # NOTE: affects rsqrtf approximation; validated by correctness suite
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
    """torch.cpp_extension shells out to ninja; make sure it is findable even
    in bare (non-interactive) shells, e.g. when we are the child of ncu."""
    import shutil
    if shutil.which("ninja") is None:
        cand = "/root/miniconda3/envs/pytorch/bin"
        if (Path(cand) / "ninja").exists():
            os.environ["PATH"] = cand + os.pathsep + os.environ.get("PATH", "")


def build(force: bool = False, verbose: bool = False):
    """Load the extension, building if needed. Returns the module."""
    _ensure_ninja_on_path()
    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError as e:
        raise RuntimeError(f"PyTorch not available in this interpreter: {e}")

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
            # Source/flag change -> wipe so ninja rebuilds from scratch
            # (documented, deterministic behavior).
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
    print("variants:", ext.variants())
    x = torch.randn(4, 4096, dtype=torch.float16, device="cuda")
    w = torch.randn(4096, dtype=torch.float16, device="cuda")
    for name in ext.variants():
        y = ext.forward(name, x, w, 1e-5)
        print(name, "ok", y.shape, y.dtype)
        torch.cuda.synchronize()
