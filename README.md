# CUDALab

**Autonomous CUDA Kernel Optimization Laboratory.**

CUDALab closes the loop of automated kernel optimization:

```
reference → correctness → benchmark → GPU profiling → bottleneck analysis
→ optimization hypothesis → kernel modification → compile → correctness
→ benchmark → accept / reject → experiment record
```

An outer LLM agent (the developer's coding agent) supplies hypotheses and
kernel code; an **objective, non-LLM evaluation layer** — correctness
harness, CUDA-event benchmark harness, Nsight Compute integration, and a
fixed KEEP/REJECT/NEUTRAL decision rule — supplies the evidence. The agent
cannot declare a win; only the harness's numbers can. See
[docs/design.md](docs/design.md).

## Current v0.1 scope

- **One kernel: RMSNorm** (`y = x * rsqrt(mean(x², dim=-1) + eps) * w`,
  default `eps=1e-5`, FP32 accumulation).
- Hardware: NVIDIA RTX 2080 Ti (Turing, **sm_75**), CUDA 11.8,
  PyTorch 2.4.1+cu118. No sm_80+/BF16/FP8 features are used.
- dtypes: **fp16 primary**, fp32 supported. Contiguous inputs.
- Supported H: any multiple of 256 (per-thread slice layouts of the
  variants require H ∈ {512, 1024, 2048, 4096, 8192} for v2/v4; baseline
  and v3 accept any H ≥ 256, v1 requires H % 8 == 0 / H % 4 == 0).
- Primary optimization target shape: **M=128, H=4096, fp16**.
- The full benchmark matrix is always measured and stored — no
  cherry-picked shapes.

## Results (real, from executed runs)

Environment: 1× RTX 2080 Ti (GPU 0 via `CUDA_VISIBLE_DEVICES=0`),
GPU clocks not locked (no container permission) → expect ~±10% run-to-run
variance at the ~5 µs scale. All variants in one matrix run share the same
inputs, GPU, and harness settings.

### Benchmark matrix (final run, fp16, batched cuda-event harness)

Median latency in µs (`speedup` is vs `baseline`, same run):

| shape (M×H) | baseline | v1_vec | v2_reg | v3_wideblock | **v4_vec_reg** |
|---|---|---|---|---|---|
| 1×4096   | 9.08  | 5.12  | 5.44  | 6.33  | **5.25** |
| 16×4096  | 6.84  | 5.05  | 5.35  | 6.34  | 5.14 |
| 128×4096 | 7.33  | 5.82  | 6.00  | 6.46  | **5.27** (1.39×) |
| 1024×4096| 42.35 | 34.75 | 33.09 | 41.14 | **32.77** (1.29×) |
| 128×8192 | 12.20 | **5.12** | 6.27  | 7.74  | 7.36 |
| 1×1024   | 5.66  | 5.12  | 5.76  | 5.96  | **5.05** |
| 128×1024 | 6.21  | **5.17** | 6.41  | 6.44  | 5.24 |

Best kernel: **`v4_vec_reg`** (primary target M=128×H=4096: **7.33 µs →
5.27 µs, 1.39× vs baseline**). Disclosed trade-off: at M=128×H=8192
`v1_vec` is faster than `v4_vec_reg` (5.12 vs 7.36 µs) — v4's register
residency costs more at PER=32. A shape-dispatched kernel combining v1/v4
is the obvious next step (roadmap).

Effective bandwidth column (logical traffic = read x + read w + write y,
÷ median time) reaches 512 GB/s at 1024×4096; the 2080 Ti's DRAM peak is
550 GB/s, and numbers above ~550 GB/s (e.g. 823 GB/s at 128×8192) are
L2-cache effects (2 MB working set inside the 5.5 MB L2), not DRAM.

### Correctness (final re-validation of best kernel)

`v4_vec_reg`: **76/76 PASS** across shapes {1,16,128,1024} ×
{1024,2048,4096,8192} × seeds {0,1,42} × dtypes {fp16, fp32} plus edge
cases (all-zero, 1e-4-scale, ×10 scale, +3 bias).
max_abs_error = 3.91e-3, max_rel_error = 9.7e-4 (fp16).
Fixed tolerance for **all** variants: fp16 atol=2e-3, rtol=5e-3
(recorded in `cudalab/correctness.py`, never relaxed per candidate).
All five variants pass the full suite.

### Profiling (ncu 2022.3, M=128×H=4096 fp16, cold L2, 4 launches)

| variant | kernel µs | DRAM % | SM % | occ % | regs | top stalls |
|---|---|---|---|---|---|---|
| baseline | 14.36 | 14.1 | 11.8 | 46.6 | 16 | long_scoreboard 80.1% |
| v1_vec | 6.13 | 30.1 | 10.2 | 44.6 | 22 | long_scoreboard 66.4% |
| v2_reg | 6.01 | 33.3 | 22.7 | 45.4 | 50 | long_scoreboard 55.5% |
| v3_wideblock | 9.48 | 22.0 | 20.5 | 90.7 | 16 | long_scoreboard 75.1% |
| v4_vec_reg | 6.75 | 30.9 | 7.7 | 41.5 | 30 | long_scoreboard 58.1% |

Findings that drove the loop: the baseline is **memory-latency bound, not
bandwidth bound** (14% DRAM, 80% long-scoreboard stalls, 2-pass scalar
loads). Vectorization cut instruction count and stall share (v1).
Register residency removed the second read (v2 NEUTRAL — its scalar loads
masked the win); combining both (v4) won the primary target. v3 shows
that occupancy alone (90.7%) does not compensate for non-vectorized
accesses. Note: ncu uses cold L2, the decision harness measures the
warm-L2 steady state; both are stored per experiment.

## Optimization experiments

| exp | variant | hypothesis (abridged) | decision | evidence |
|---|---|---|---|---|
| EXP-0001 | baseline | reference implementation | KEEP (baseline) | old harness, superseded |
| EXP-0002 | v1_vec | 16B vectorized loads | REJECT | **superseded** — single-launch event timing was too noisy (see below) |
| EXP-0003 | baseline | re-baseline with fixed harness | KEEP (baseline) | cuda-event-batched-v1 |
| EXP-0004 | v1_vec | 16B vectorized loads (re-measured) | **KEEP** | 1.212×, faster 5/5 rounds |
| EXP-0005 | v2_reg | single-pass register residency (scalar) | **NEUTRAL** | 0.989× (±5% band) |
| EXP-0006 | v3_wideblock | 512-thread blocks hide latency | **REJECT** | 0.768×, slower 5/5 rounds |
| EXP-0007 | v4_vec_reg | vectorized + register-resident | **KEEP** | 1.231×, faster 5/5 rounds |

**Methodology incident (kept, not hidden):** EXP-0002's single-launch
cuda-event harness added ~6 µs of launch noise and reported v1 as *slower*
than baseline (0.844×), while ncu simultaneously showed the v1 kernel was
**2.35× faster** (6.13 vs 14.36 µs). The discrepancy led to the
`cuda-event-batched-v1` harness (32 launches per sample, synchronized),
and every variant was re-measured under it (EXP-0003 onward). This is the
core reason CUDALab keeps two independent instruments (event timing + ncu)
and the full record of rejected/superseded experiments.

Full records: [`experiments/rmsnorm/`](experiments/rmsnorm/).

## Architecture

```
cudalab/
  reference.py      explicit FP32-accumulation RMSNorm reference
  build.py          extension build + content-hash cache control
  correctness.py    fixed-tolerance correctness harness (76-case suite)
  benchmark.py      batched cuda-event benchmark harness + GPU state
  profiler.py       ncu --csv integration → structured JSON summaries
  experiment.py     experiment records + KEEP/REJECT/NEUTRAL rules
kernels/rmsnorm/
  rmsnorm_common.h  self-registering variant registry + device helpers
  bindings.cpp      PyTorch extension entry points (pure C++)
  rmsnorm_baseline.cu  one block/row, scalar, 2-pass
  rmsnorm_v1.cu        16B vectorized, 2-pass
  rmsnorm_v2.cu        single-pass register-resident (scalar)
  rmsnorm_v3.cu        512-thread blocks (scalar)
  rmsnorm_v4.cu        16B vectorized + register-resident  ← best
scripts/
  test_rmsnorm.py         correctness entry point
  benchmark_rmsnorm.py    benchmark entry point
  profile_rmsnorm.py      ncu profile entry point
  optimize_rmsnorm.py     build→correctness→bench→profile→decide→record
tools/env.sh        single source of environment truth
experiments/        EXP-*.json records + correctness JSON + best.json
benchmarks/         bench_*.json / bench_*.csv matrices
profiles/rmsnorm/   structured profile JSON + raw ncu output (raw/ git-ignored)
```

Adding a kernel variant = one new `.cu` file (self-registers; no binding
edits); the next build picks it up automatically.

## Environment

| item | value |
|---|---|
| GPU | 2× NVIDIA RTX 2080 Ti (Turing, CC 7.5); benchmark uses GPU 0 |
| CUDA toolkit | 11.8 (`/usr/local/cuda`) |
| Python | `/root/miniconda3/envs/pytorch/bin/python` (3.10) |
| PyTorch | 2.4.1+cu118 |
| Profiler | Nsight Compute 2022.3 (`/usr/local/bin/ncu`, profiling perms OK) |
| Compile | `-O3 -lineinfo --use_fast_math -gencode=arch=compute_75,code=sm_75` |

`tools/env.sh` sets `CUDA_HOME`, `PATH` (conda bin + CUDA bin), `PYTHON`,
`TORCH_CUDA_ARCH_LIST=7.5`, `CUDA_VISIBLE_DEVICES=0`.

## Correctness methodology

- Primary reference: explicit formula
  `y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)`
  `* w.float()`, cast back to input dtype (FP32 accumulation, version-
  independent).
- Metrics per case: max_abs_error, max_rel_error (denominator clamped at
  1e-3), NaN, Inf, allclose.
- Fixed tolerance (all variants, recorded): fp16 atol=2e-3 / rtol=5e-3;
  fp32 atol=1e-5 / rtol=1e-4.
- 76 cases per variant: 11 shapes × 3 seeds × 2 dtypes + 5 edge cases × 2
  dtypes (all-zero, 1e-4 scale, ×10 scale, +3 bias).
- A FAILing variant is unconditionally REJECTed and never becomes "best".

## Benchmark methodology

- `torch.cuda.Event` pairs; **batched**: each sample = 32 consecutive
  kernel launches (same input/output tensors), one synchronize per sample;
  sample time = elapsed/32. Warmup 150 launches per round; 100 samples ×
  5 independent rounds per (variant, shape, dtype) = 500 samples.
- Primary metric: **median** over all samples; p95/min/max and per-round
  medians also stored. Compile time is strictly outside the timed region
  (build first, content-hash cache).
- Same input tensors for every variant of a shape; same GPU (GPU 0);
  nvidia-smi state (temp, clocks, power, utilization) snapshotted
  before/after each variant.
- Decision (primary shape 128×4096 fp16, vs current best, same run):
  ≥5% faster AND ≥3/5 rounds faster → KEEP; ≤5% slower AND ≤2/5 rounds
  faster → REJECT; else NEUTRAL. Full matrix always stored.
- Known limitation: GPU clocks cannot be locked in this container;
  ~±10% run-to-run variance at ~5 µs. Cross-run comparisons are invalid;
  only same-matrix (same-run) comparisons are used for decisions.

## Profiling

`cudalab/profiler.py` runs `ncu --csv -k regex:rmsnorm --launch-skip 2
--launch-count 4 --metrics …` over a dedicated driver and parses the CSV
into JSON (`profiles/rmsnorm/<variant>_M<M>_H<H>.json`): kernel duration,
DRAM/SM throughput %, achieved occupancy, registers/thread, shared memory,
and the warp-stall distribution (stalled warp-cycles per issue-active
cycle, with % of total stalls). Metric names were verified with
`ncu --query-metrics` for NCU 2022.3 on sm_75. Raw ncu stdout/stderr is
kept under `profiles/rmsnorm/raw/` (git-ignored). No fabricated numbers:
unavailable fields are `null`.

## Repository structure

See the Architecture section; top level: `cudalab/`, `kernels/`,
`scripts/`, `tools/`, `experiments/`, `benchmarks/`, `profiles/`, `docs/`.

## How to reproduce

```bash
cd /root/code/cuda
source tools/env.sh          # sets CUDA_HOME, PATH, PYTHON, arch list

# build (cached; ~1 min cold, near-instant warm)
$PYTHON cudalab/build.py

# correctness (all variants)
$PYTHON scripts/test_rmsnorm.py

# benchmark matrix (all variants)
$PYTHON scripts/benchmark_rmsnorm.py --tag myrun

# profile one variant
$PYTHON scripts/profile_rmsnorm.py --variant v4_vec_reg --M 128 --H 4096

# establish / update the baseline, or run one optimization experiment
$PYTHON scripts/optimize_rmsnorm.py baseline
$PYTHON scripts/optimize_rmsnorm.py evaluate --variant v4_vec_reg \
    --parent v1_vec --hypothesis "..." --changes "..."
```

Artifacts: `experiments/rmsnorm/EXP-*.json`,
`experiments/rmsnorm/correctness/*.json`, `benchmarks/bench_*.json|csv`,
`profiles/rmsnorm/*.json`.

## Limitations

- RMSNorm only; single GPU (GPU 0); contiguous inputs.
- v2/v4 require H/256 ∈ {2,4,8,16,32} (H up to 8192); v1 needs H%8==0
  (fp16); v3 needs H%512==0. baseline is the only fully general variant.
- No GPU clock locking (container) → ±10% variance at 5 µs scale;
  NEUTRAL calls near the ±5% boundary are not stable across runs.
- Batched warm-L2 decision harness vs cold-L2 ncu can rank close variants
  differently (both numbers stored; decision uses steady-state harness).
- `effective_bw_gbps` is logical traffic ÷ time, not measured DRAM
  throughput; >550 GB/s values indicate L2 residency.
- `speedup_vs_pytorch_reference` column is reserved (null) in v0.1.

## Roadmap

- v0.2: shape-dispatched best kernel (v1 for large-H, v4 for the rest);
  fp16 GEMV-like micro-optimizations; clock-lock support where permitted.
- v0.3: more kernels (Softmax, RoPE), per-kernel reference libraries.
- v1.0: agent-side hypothesis search over a structured kernel DSL with the
  same objective layer (the objective layer is intentionally already
  kernel-agnostic).
