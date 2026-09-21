# experiments/regression/v0.5/ — v0.5 回归验证记录（append-only）

v0.5 merge review（2026-09-21）确立的新验证输出目录。

## 不可变约定（artifact immutability）

历史 experiment artifact —— `experiments/` 下 main（v0.4.1,
4eb520b）已发布的记录 —— **不可变**: 任何重跑都不得覆盖原文件。
v0.5 起因默认输出目录仍指向历史目录, 3 个遗留算子的 v0.5 smoke
验证覆盖了 7 个 main 已发布文件（merge review 发现）。处置:

1. 被覆盖的 7 个文件已 `git restore --source=4eb520b` 恢复为 main
   版本（与 4eb520b 逐字节一致）;
2. v0.5 的覆盖结果先迁移到本目录（见下表, 文件名加 `_rerun_v0.5`
   后缀, 避免与新跑记录的规范文件名冲突）;
3. rmsnorm / softmax / rope 三个算子的 `run_correctness` /
   `run_negative` 默认输出目录改指本目录对应子目录
   （GEMV 算子本轮通过 `out_dir` 参数显式指向本目录）——
   新验证一律只**追加**到本目录, 绝不回写历史目录。

## 迁移记录（`*_rerun_v0.5.json`）

v0.5 smoke 重跑历史用例的产物, 迁移于恢复 main 版本之前:

| 迁移文件 | 被覆盖的原始文件（已恢复） |
|---|---|
| `rmsnorm/baseline_rerun_v0.5.json` | `experiments/rmsnorm/correctness/v0.3_regression/baseline.json` |
| `rmsnorm/v4_vec_reg_rerun_v0.5.json` | `experiments/rmsnorm/correctness/v0.3_regression/v4_vec_reg.json` |
| `rmsnorm/invalid_inputs_rerun_v0.5.json` | `experiments/rmsnorm/correctness/v0.3_regression/invalid_inputs.json` |
| `softmax/softmax_baseline_rerun_v0.5.json` | `experiments/softmax/correctness/v0.3/softmax_baseline.json` |
| `softmax/softmax_vec4_rerun_v0.5.json` | `experiments/softmax/correctness/v0.3/softmax_vec4.json` |
| `softmax/invalid_inputs_rerun_v0.5.json` | `experiments/softmax/correctness/v0.3/invalid_inputs.json` |
| `rope/invalid_inputs_rerun_v0.5.json` | `experiments/rope/correctness/v0.4/invalid_inputs.json` |

迁移内容与原文件**数值逐项一致**（确定性 seed, 同版本内核）;
差异只有: ① `generated` 时间戳; ② correctness 文件的 JSON 浮点
序列化格式（main 上的 v0.3 时代文件经 `round(x, 8)` 舍入保存,
现行为 evaluator 核心统一的全精度浮点）—— 非数值差异。
迁移内容是 v0.5 行为证据, 不属于历史记录本身。

## v0.5 merge review 轮验证记录（本轮新跑）

- `gemv/`: splitk4 隔离后, 4 个正常变体（gemv_baseline /
  gemv_vec4_row / gemv_warp_vec4_b256 / gemv_warp_vec4_b512）
  重验 correctness（各 100/100）+ negative（各 24/24, per-variant）;
  官方记录 `experiments/gemv/correctness/v0.5/` 原样未动。
- `softmax/`: incumbent softmax_vec4 的 negative 套件**首次按
  per-variant 语义**运行（15 例, 1 环境性 skip）+ 对应
  correctness 重跑。
- `rope/`: incumbent rope_v3_half2 的 negative 套件（37 例,
  per-variant）+ 对应 correctness 重跑。
- `rmsnorm/`: cross-variant negative 套件单跑
  （`negative_suite_scope = "cross-variant"`, 覆盖全部变体,
  30 例, 1 环境性 skip）+ baseline correctness 重跑。

本目录只追加: 新验证新增文件, 不修改既有文件。
