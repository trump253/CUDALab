# profiles/gen3_v0.7/ — v0.7 三代对比 NCU fresh 测量（append-only）

v0.7 session（2026-09-22, GPU 0: 2080 Ti, 616 GB/s 规格峰值, sm_75,
NCU 2022.3, cc=all @base 1350 MHz, 4096×4096 fp16）三代对比
（FP16 `gemv_vec4_row` / INT8 `qgemv_vec16_row` / INT4
`int4gemv_rowtile4_hx`）的 NCU 主 pass（ccall clkbase / ccnone clkbase）
fresh 测量归档。**append-only：本目录与文件不可变，未来复测进新文件/新目录。**

## 规则（v0.7.1 merge 修复起生效）

- **historical artifacts = immutable**：main（v0.5/v0.6 发布状态）已存在的
  `profiles/gemv/`、`profiles/qgemv/` 等路径不得被新版本就地重写。
- **fresh cross-generation measurements = append-only**：新版本的 fresh
  跨代参照测量一律进新目录（本目录），不覆盖历史路径。

## 背景

v0.7 三代对比（commit b2580d6）当时按「fresh 测量不沿用历史」协议**就地
覆盖**了 3 个历史 profile JSON。v0.7.1 merge 修复（外部 review MAJOR
级 blocker）：3 个历史文件已恢复为 main 原版本，v0.7 fresh 数据迁移至
本目录。两个版本在 git 历史中均可追溯（fresh 值原路径见下表；恢复后
历史路径只含 main 原值）。

## 文件清单

| 本目录文件 | 来源路径（v0.7 pre-fix 状态） | fresh 值 | 说明 |
|---|---|---|---|
| `gemv_vec4_row_4096_ccall.json` | `profiles/gemv/gemv_vec4_row_M4096_H4096_ccall_clkbase.json` | 64.24 µs, DRAM 89.3% | 曾被就地覆盖，历史路径已恢复 main 原值 64.192 µs / 87.89% |
| `gemv_vec4_row_4096_ccnone.json` | `profiles/gemv/gemv_vec4_row_M4096_H4096_ccnone_clkbase.json` | 64.008 µs, DRAM 89.1% | 曾被就地覆盖，历史路径已恢复 main 原值 63.808 µs / 87.91% |
| `qgemv_vec16_row_4096_ccall.json` | `profiles/qgemv/qgemv_vec16_row_M4096_H4096_ccall_clkbase.json` | 36.432 µs, DRAM 86.55% | 曾被就地覆盖，历史路径已恢复 main 原值 36.368 µs / 86.02% |
| `qgemv_vec16_row_4096_ccnone.json` | `profiles/qgemv/qgemv_vec16_row_M4096_H4096_ccnone_clkbase.json`（v0.7 新增，无历史版本，自原路径迁移至此） | 36.2 µs, DRAM 85.98% | 无任何实验记录引用 |
| `int4gemv_rowtile4_hx_4096_ccall.json` | `profiles/int4gemv/int4gemv_rowtile4_hx_M4096_H4096_ccall_clkbase.json`（v0.7 新增；原路径保留，`INT4GEMV-0004.json` 引用之） | 25.176 µs, DRAM 68.84% | 复制（非移动），保持算子目录完整 |

补充：三代 NCU 指令构成/物理流量补充 pass 的原始 CSV 在
`profiles/int4gemv/gen3_pipe_{gemv,qgemv,int4}.csv`（12 指标 × 4 launch，
ncu CSV export；4-launch 均值：INT8 物理流量 18,435,144 B、INT4
10,262,584 B、FP16 35,192,840 B；duration 均值 36.448 / 25.496 /
64.384 µs）。本目录 5 个 JSON 为主 pass（ccall/ccnone clkbase）结果，
与 CSV 补充 pass 是**不同 NCU pass**（duration 差 <1%，量级一致）。
