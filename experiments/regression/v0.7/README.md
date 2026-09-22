# experiments/regression/v0.7/ — v0.7 回归验证记录（append-only）

v0.7（W4A16 Group-wise INT4 GEMV, 分支 `v0.7-int4-gemv`）的 smoke 回归输出
目录。沿用 v0.5/v0.6 确立的**不可变约定**（见
`experiments/regression/v0.5/README.md` / `v0.6/README.md`）:

- 历史 experiment artifact（`experiments/` 下 main 已发布记录）**不可变**;
- 新验证一律只**追加**到本目录对应算子子目录, 绝不回写历史目录;
- 各算子 `run_correctness(ext, v, out_dir=本目录/<op>/)` /
  `run_negative(ext, v, out_dir=本目录/<op>/)` 显式指向本目录。

## 内容（2026-09-22, GPU 0, 分支 v0.7-int4-gemv, 最终态）

| 子目录 | 覆盖 |
|---|---|
| `int4gemv/` | 5 正常变体（baseline / vec16_row / rowtile4 / rowtile4_hx / rowtile8）correctness（50 项/变体）+ per-variant negative（30 项/变体, 含 2 例回退 bit-identical 钉死）; int4gemv quarantine 集为空（机制保留, 当前无隔离变体） |
| `gemv/` | 4 正常变体（baseline / vec4_row / warp_vec4_b256 / warp_vec4_b512）correctness（100 项/变体）+ per-variant negative（24 项/变体）; `gemv_splitk4` **被隔离, 不在正常列表, 未运行**（历史隔离, v0.5 起） |
| `qgemv/` | 5 变体（baseline / vec16_row / vec16_scale / warp_vec16 / warp_vec16_ilp4）correctness（50 项/变体）+ per-variant negative（29 项/变体） |
| `quarantine_audit.json` | int4gemv/gemv/qgemv 的 `variants()` / `all_variants()` / `quarantined_variants()` 绑定列表审计: `gemv_splitk4` 不在正常变体列表（quarantine_leaked_into_normal 空）、仍在 all_variants（历史审计入口保留） |
| `summary.json` | 全变体 pass 汇总 |

历史隔离备注（本分支未构建/未运行, 状态记录）: `softmax_hsplit2`
（v0.4 起隔离）与本分支算子集无关, 其隔离状态以 main 上 v0.5/v0.6
回归审计为准。

## 生成命令（可复现）

```
source tools/env.sh
$PYTHON scripts/regression_v07.py
```

## 结果

全部通过: 14 个变体运行（int4gemv 5 + gemv 4 + qgemv 5）correctness
与 negative 全绿（`summary.json`）。
