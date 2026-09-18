# CUDALab — Core Design

## The central principle

**The LLM / agent is the optimizer; it is NOT the objective evaluator.**

```
        hypothesis                code change
  ┌──────────────┐   ──────────►  ┌──────────────┐
  │              │                │              │
  │  Agent/LLM   │                │  CUDA source │
  │  (proposes)  │   ◄──────────  │  (modified)  │
  │              │  evidence      └──────┬───────┘
  └──────▲───────┘                       │ compile
         │                               ▼
         │                     ┌──────────────────┐
         │                     │  Objective layer │
         │   structured        │  (no LLM inside) │
         └─────────────────────┤                  │
               results         │ 1. correctness   │  pass/fail + error metrics
                               │ 2. benchmark     │  median/p95 per round, matrix
                               │ 3. ncu profiler  │  stalls, DRAM/SM %, occupancy
                               └──────────────────┘
```

The agent's job: read the structured evidence, form a **hypothesis**
("the kernel is long-scoreboard bound, so vectorizing loads should reduce
stall cycles"), write the kernel change, and record the experiment.

The objective layer's job: produce numbers the agent cannot argue with.
It has no natural-language output, no opinion, no "rounding up" of weak
results, and no access to the hypothesis.

## Why this is more reliable than "let the LLM look at CUDA code and guess"

1. **Honesty of the feedback channel.** An LLM judging its own kernel
   change is a self-grading exam: it will describe a 0.99x result as a
   "modest improvement" and skip the shapes where it lost. A benchmark
   harness that emits `median_us`, per-round medians, and the full shape
   matrix cannot flatter. The acceptance rule (KEEP requires >=5% AND a
   round majority) is applied by code, not by interpretation.

2. **Failure is data, not embarrassment.** In this project the first
   timing harness (single-launch cuda events) *falsely rejected a real
   2.35x kernel-time improvement* (EXP-0002, superseded). Only because the
   profiler (ncu) was an independent objective instrument did the
   discrepancy surface, the harness get fixed, and every variant get
   re-measured under the corrected method. A pure "LLM guesses" workflow
   has no second instrument to contradict the first number.

3. **Reproducibility of every decision.** Each experiment JSON stores:
   the hypothesis, the exact source changes, the correctness summary, the
   full benchmark matrix (parent and candidate, same run), the profile
   observation, the decision, and the decision rule that fired. Any human
   (or later agent) can re-derive the KEEP/REJECT/NEUTRAL verdict from the
   stored numbers alone.

4. **Scope discipline.** Because the objective layer is fixed (shapes,
   dtypes, tolerances, rounds, target shape), the agent cannot quietly
   narrow the test to where it wins. The full matrix is always saved and
   always reported — including the shapes where the current best loses
   (e.g. v4 vs v1 at M=128/H=8192).

5. **The agent still does what LLMs are good at.** Hypothesis generation
   from profiler evidence, writing the kernel, diagnosing compile/runtime
   errors (alignment faults, macro/template errors), and interpreting
   cross-instrument discrepancies (cold-L2 ncu vs warm-L2 benchmark
   rankings) are exactly the open-ended work where structured automation
   alone fails.

## Concretization in v0.1

| Concern            | Owner                              | Mechanism |
|--------------------|------------------------------------|-----------|
| Hypothesis         | Agent                              | reads profile JSON + bench matrix |
| Kernel code        | Agent                              | new `kernels/rmsnorm/rmsnorm_*.cu`, self-registering |
| Build determinism  | Objective                          | `cudalab/build.py` content-hash build cache |
| Correctness        | Objective                          | `cudalab/correctness.py`, fixed tolerances |
| Timing             | Objective                          | `cudalab/benchmark.py`, batched cuda events |
| GPU analysis       | Objective                          | `cudalab/profiler.py` (ncu --csv -> JSON) |
| Decision           | Objective (rule), Agent (invocation) | `cudalab/experiment.py` |
| Record             | Objective                          | `experiments/rmsnorm/EXP-*.json`, all kept |

## Known tensions (documented, not hidden)

- **Warm vs cold L2:** ncu profiles with cold L2 (its default cache
  control), while the decision benchmark measures the steady state of
  back-to-back launches (L2-warm). At this data size (1MB x at 128x4096
  fits in the 5.5MB L2) the two can rank close variants differently
  (v1 vs v4). The decision uses the steady-state harness because it
  matches deployment; both numbers are stored.
- **Run-to-run variance** at ~5us kernels is ~±10% (GPU clocks are not
  lockable in this container). KEEP margins in this project (1.21-1.39x)
  are far outside that band; a NEUTRAL call (0.989x) could in principle
  flip across runs — a real limitation, noted per-record.
