# CUDALab v0.1 — Project Plan

Autonomous CUDA Kernel Optimization Laboratory. v0.1 scope: **RMSNorm only**,
on 2× NVIDIA RTX 2080 Ti (Turing, sm_75), CUDA 11.8, PyTorch 2.4.1+cu118.

## Pipeline

```
reference → correctness → benchmark → GPU profiling → bottleneck analysis
→ optimization hypothesis → kernel modification → compile → correctness
→ benchmark → accept/reject → experiment record
```

## Phases

### Phase 0 — Repository initialization (done by definition of start)
- Clean `/root/code/cuda`, `git init -b main`, repo-local git identity.
- `.gitignore` (exclude builds/`.so`/raw profiler dumps; keep benchmark
  CSV/JSON, experiment metadata, reports).
- `tools/env.sh` as the single source of environment truth
  (`CUDA_HOME=/usr/local/cuda`, `PYTHON=/root/miniconda3/envs/pytorch/bin/python`,
  `TORCH_CUDA_ARCH_LIST=7.5`, `CUDA_VISIBLE_DEVICES=0`).
- `PROJECT_PLAN.md`, `STATUS.md`.
- ✅ Complete

### Phase 1 — RMSNorm reference implementation
- `cudalab/reference.py`: explicit FP32-accumulation formula
  `y = x * rsqrt(mean(x², -1) + eps) * w`, default `eps=1e-5`.
- ✅ Complete

### Phase 2 — Baseline CUDA kernel
- `kernels/rmsnorm/rmsnorm_baseline.cu`: one block per row,
  warp-reduce + shared-memory reduction, FP16 in/out, FP32 accumulation.
- `kernels/rmsnorm/bindings.cpp` + `cudalab/build.py` using
  `torch.utils.cpp_extension.load()` with explicit `sm_75` and a
  content-hashed build dir so unchanged sources never recompile and each
  candidate variant has its own cache.
- ✅ Complete

### Phase 3 — Correctness harness
- `cudalab/correctness.py`: max_abs_error, max_rel_error, NaN/Inf, allclose.
  Fixed recorded tolerance (fp16: atol=1e-2, rtol=2e-2 — fixed for all
  candidates, never relaxed per-candidate).
- Matrix: M ∈ {1,16,128,1024} × H ∈ {1024,2048,4096,8192} subset + edge cases
  (zeros, tiny values, multiple scales, multiple seeds).
- JSON + human-readable output.
- ✅ Complete

### Phase 4 — Benchmark harness
- `cudalab/benchmark.py`: `torch.cuda.Event` timing, warmup ≥ 100,
  iterations ≥ 200, ≥ 5 independent rounds, median primary metric,
  p50/p95/min/max, effective DRAM bandwidth (read x + weight, write y).
- GPU state snapshot (nvidia-smi: clocks, temp, power, utilization)
  before/after each suite run.
- Fixed input tensors shared by all variants per (shape, dtype).
- Benchmark matrix for all variants, CSV + JSON.
- ✅ Complete

### Phase 5 — Profiler integration
- `cudalab/profiler.py`: run `ncu` on a small profile-only driver script,
  map Nsight-Compute 2022.3 metric names for sm_75 (discovered via
  `ncu --query-metrics`), emit structured JSON summary
  (kernel duration, DRAM throughput, SM throughput, occupancy,
  registers/thread, shared memory, warp stalls). Nulls when unavailable.
- Fallback: save real ncu error, record in STATUS.md, use nsys /
  torch.profiler as alternative.
- ✅ Complete (or fallback recorded)

### Phase 6 — Experiment tracking
- `cudalab/experiment.py`: `experiments/rmsnorm/EXP-NNNN.json` records with
  hypothesis, changes, correctness, benchmark vs current-best, profile
  observation, decision (KEEP/REJECT/NEUTRAL).
- Decision rules (v0.1):
  - correctness FAIL → REJECT (unconditional);
  - ≥ 5% faster (median of rounds) AND majority of rounds faster → KEEP;
  - within ±5% → NEUTRAL; clearly slower → REJECT.
  - Full benchmark matrix always saved; no cherry-picked shapes.
- `scripts/optimize_rmsnorm.py`: build → correctness → benchmark → profile
  → evaluate candidate vs current best → decision → record.
- ✅ Complete

### Phase 7 — First optimization loop
- ≥ 3 real, data-driven optimization experiments against the baseline:
  1. vectorized loads (float4/half2) — memory-bandwidth bound hypothesis;
  2. half2 + reduced synchronization / single-pass restructure (if justified
     by profile);
  3. one deliberate "expected to fail" hypothesis to validate the REJECT
     branch (e.g., larger grid-stride / different block size) — only if
     genuinely motivated by data.
- Primary target shape: **M=128, H=4096, FP16**; full matrix still reported.
- ✅ Complete

### Phase 8 — Documentation and final validation
- Re-run full correctness suite + full benchmark matrix on the final best
  kernel.
- `README.md` (GitHub-quality, real numbers only), `docs/design.md`,
  `STATUS.md` updated, PROJECT_PLAN finalized.
- Git commits at each phase; final clean tree + commit.
- ✅ Complete

## Non-negotiable rules
- No fabricated numbers; every reported metric comes from an executed run.
- Compile time never counts in kernel timing; benchmark before profile.
- Same input for all variants; weight multiplication never skipped.
- Failed experiments are kept and reported.

## Status
See `STATUS.md` for live progress.
