# CUDALab — Status

**Date:** 2026-07-16
**Phase:** 8 complete — v0.1 finished
**Status:** All stop conditions met (see checklist below).

## Current state

| item | value |
|---|---|
| current best kernel | `v4_vec_reg` (vectorized + register-resident, 1 block/row) |
| primary target (M=128, H=4096, fp16) | baseline 7.33 µs → **5.27 µs (1.39×)**, re-validated 76/76 PASS |
| experiments | 7 (EXP-0001…EXP-0007): 3 valid KEEPs, 1 NEUTRAL, 1 REJECT, 2 superseded |
| correctness | 5/5 variants pass the fixed-tolerance 76-case suite (fp16 + fp32, edge cases) |
| benchmark harness | `cuda-event-batched-v1` (32-launch batched samples, 5×100, median primary) |
| profiler | ncu 2022.3.0 **working** (permissions OK); structured JSON + raw output |
| git | local repo, `main`, repo-local identity, no remote; working tree clean |

## Stop-condition checklist

- [x] Clean git repository created (local `main`, repo-local identity, no remote)
- [x] Reference implementation working
- [x] Correctness harness passing (fixed tolerance, 76 cases × 5 variants, no relaxation)
- [x] Baseline kernel + extension build working (sm_75, CUDA 11.8, torch 2.4.1+cu118)
- [x] Benchmark harness with structured JSON/CSV results (full matrix, GPU state, warmup, batched timing)
- [x] Real GPU profiler attempt: **ncu succeeded** — real profile JSON + real raw output saved
- [x] Working experiment tracking (EXP-*.json + decision rule + best.json)
- [x] ≥3 real optimization experiments with real KEEP/REJECT/NEUTRAL (v1 KEEP, v2 NEUTRAL, v3 REJECT, v4 KEEP)
- [x] Best kernel determined (v4_vec_reg) and re-validated (full correctness 76/76 + full benchmark matrix)
- [x] README.md with real numbers only
- [x] STATUS.md (this file)
- [x] PROJECT_PLAN.md with phase progress
- [x] Final local git commit, clean tree

## Known limitations (also in README)

- GPU clocks not lockable in container → ±10% run-to-run variance at ~5 µs.
- warm-L2 batched harness vs cold-L2 ncu can rank close variants differently.
- `effective_bw_gbps` is logical traffic ÷ time (L2 effects at small working sets).
- v2/v4 limited to H ≤ 8192 (H/256 ∈ {2,4,8,16,32}).

## Notable incidents (recorded, not hidden)

- EXP-0002 false REJECT: single-launch event timing (~6 µs launch noise)
  contradicted ncu (v1 was 2.35× faster). Fixed harness →
  `cuda-event-batched-v1`; EXP-0001/EXP-0002 marked `superseded`; all
  variants re-measured.
