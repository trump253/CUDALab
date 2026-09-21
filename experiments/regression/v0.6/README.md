# experiments/regression/v0.6/ — v0.6 回归验证记录（append-only）

v0.6（INT8 Weight-Only GEMV, 分支 `v0.6-qgemv`）的 smoke 回归输出目录。
沿用 v0.5 merge review 确立的**不可变约定**（见
`experiments/regression/v0.5/README.md`）:

- 历史 experiment artifact（`experiments/` 下 main 已发布记录）**不可变**;
- 新验证一律只**追加**到本目录对应算子子目录, 绝不回写历史目录;
- 各算子 `run_correctness(ext, v, out_dir=本目录/<op>/)` /
  `run_negative(ext, v, out_dir=本目录/<op>/)` 显式指向本目录
  （rmsnorm/softmax/rope 的**默认**输出仍指向 `regression/v0.5/<op>/` ——
  那些目录是 v0.5 的 append-only 目录, v0.6 不写入; 本目录只接受显式
  `out_dir` 指向）。

## 内容（2026-09-21, GPU 0, 分支 v0.6-qgemv 最终态）

| 子目录 | 覆盖 |
|---|---|
| `gemv/` | 4 正常变体（baseline / vec4_row / warp_vec4_b256 / warp_vec4_b512）correctness + per-variant negative; `gemv_splitk4` **被隔离, 不在正常列表, 未运行** |
| `rmsnorm/` | 正常变体 correctness + negative |
| `softmax/` | 正常变体 correctness + per-variant negative; `softmax_hsplit2` **被隔离, 不在正常列表, 未运行** |
| `rope/` | 正常变体 correctness + negative |
| `qgemv/` | 5 变体（baseline / vec16_row / vec16_scale / warp_vec16 / warp_vec16_ilp4）correctness（50 项/变体 + 保真度套件）+ per-variant negative（29 项/变体） |
| `quarantine_audit.json` | 全 5 算子的 `variants()` / `all_variants()` / `quarantined_variants()` 绑定列表审计: `gemv_splitk4` 与 `softmax_hsplit2` 不在任何正常变体列表（quarantine_leaked_into_normal 全空）、仍在 all_variants（历史审计入口保留）; qgemv quarantine 集为空（机制保留, 当前无隔离变体） |

## 生成命令（可复现）

```python
# 见 git 历史 v0.6 分支最终 commit; 核心循环:
# for op in (gemv, rmsnorm, softmax, rope, qgemv):
#     for v in op.variants(ext):
#         op.run_correctness(ext, v, out_dir=REG/<op>/)
#         op.run_negative(ext, v, out_dir=REG/<op>/)
```
