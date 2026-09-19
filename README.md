# CUDALab

**自主 CUDA 内核优化实验室。**

CUDALab 闭环自动化内核优化：

```
参考实现 → 正确性 → 基准测试 → GPU 剖析 → 瓶颈分析
→ 优化假设 → 内核修改 → 编译 → 正确性
→ 基准测试 → 采纳 / 拒绝 → 实验记录
```

外层 LLM 智能体（开发者的编码代理）提供优化假设与内核代码；**客观、非 LLM 的评估层** —— 正确性校验框架、CUDA 事件基准框架、Nsight Compute 集成、以及固定的 KEEP/REJECT/NEUTRAL 判定规则 —— 提供证据。智能体不能自封胜者；只有框架的数字才算数。见 [docs/design.md](docs/design.md)。

## 当前 v0.1 范围

- **一个内核：RMSNorm**（`y = x * rsqrt(mean(x², dim=-1) + eps) * w`，默认 `eps=1e-5`，FP32 累加）。
- 硬件：NVIDIA RTX 2080 Ti（Turing，**sm_75**），CUDA 11.8，PyTorch 2.4.1+cu118。不使用任何 sm_80+/BF16/FP8 特性。
- dtype：**fp16 为主**，支持 fp32。连续（contiguous）输入。
- 支持的 H：任意 256 的倍数（v2/v4 的每线程切片布局要求 H ∈ {512, 1024, 2048, 4096, 8192}；baseline 与 v3 接受任意 H ≥ 256，v1 要求 H%8==0（fp16）/ H%4==0（fp32））。
- 主要优化目标形状：**M=128, H=4096, fp16**。
- 完整基准矩阵始终测量并保存 —— 不做形状挑拣。

## 结果（全部为真实执行数据）

环境：1× RTX 2080 Ti（经 `CUDA_VISIBLE_DEVICES=0` 使用 GPU 0），GPU 时钟未锁定（容器内无权限）→ ~5 µs 尺度预期约 ±10% 的逐次运行方差。同一次矩阵运行内的所有变体共享相同的输入、相同的 GPU、相同的框架参数。

### 基准矩阵（最终运行，fp16，批量 cuda-event 框架）

中位延迟（µs），`加速比` 相对同次运行的 `baseline`：

| 形状 (M×H) | baseline | v1_vec | v2_reg | v3_wideblock | **v4_vec_reg** |
|---|---|---|---|---|---|
| 1×4096   | 9.08  | 5.12  | 5.44  | 6.33  | **5.25** |
| 16×4096  | 6.84  | 5.05  | 5.35  | 6.34  | 5.14 |
| 128×4096 | 7.33  | 5.82  | 6.00  | 6.46  | **5.27**（1.39×） |
| 1024×4096| 42.35 | 34.75 | 33.09 | 41.14 | **32.77**（1.29×） |
| 128×8192 | 12.20 | **5.12** | 6.27  | 7.74  | 7.36 |
| 1×1024   | 5.66  | 5.12  | 5.76  | 5.96  | **5.05** |
| 128×1024 | 6.21  | **5.17** | 6.41  | 6.44  | 5.24 |

最佳内核：**`v4_vec_reg`**（主目标 M=128×H=4096：**7.33 µs → 5.27 µs，较 baseline 1.39×**）。如实披露的取舍：在 M=128×H=8192 上 `v1_vec` 快于 `v4_vec_reg`（5.12 vs 7.36 µs）—— v4 的寄存器驻留在 PER=32 时代价更高。把 v1/v4 组合成按形状分派的内核是最明显的下一步（路线图）。

有效带宽列（逻辑流量 = 读 x + 读 w + 写 y，÷ 中位时间）在 1024×4096 达到 512 GB/s；2080 Ti 的 DRAM 峰值为 550 GB/s，高于 ~550 GB/s 的数值（如 128×8192 的 823 GB/s）是 L2 缓存效应（2 MB 工作集落在 5.5 MB L2 内），并非真实 DRAM 带宽。

### 正确性（对最终最佳内核的复验）

`v4_vec_reg`：**76/76 PASS**，覆盖形状 {1,16,128,1024} × {1024,2048,4096,8192} × 种子 {0,1,42} × dtype {fp16, fp32}，外加边界用例（全零、1e-4 幅度、×10 幅度、+3 偏置）。
max_abs_error = 3.91e-3，max_rel_error = 9.7e-4（fp16）。
固定容差（**所有变体一致**）：fp16 atol=2e-3 / rtol=5e-3（记录于 `cudalab/correctness.py`，绝不针对候选放宽）。
五个变体全部通过完整套件。

### Profiling（ncu 2022.3，M=128×H=4096 fp16，冷 L2，4 次启动）

| 变体 | 内核 µs | DRAM % | SM % | 占用率 % | 寄存器 | 主要停顿 |
|---|---|---|---|---|---|---|
| baseline | 14.36 | 14.1 | 11.8 | 46.6 | 16 | long_scoreboard 80.1% |
| v1_vec | 6.13 | 30.1 | 10.2 | 44.6 | 22 | long_scoreboard 66.4% |
| v2_reg | 6.01 | 33.3 | 22.7 | 45.4 | 50 | long_scoreboard 55.5% |
| v3_wideblock | 9.48 | 22.0 | 20.5 | 90.7 | 16 | long_scoreboard 75.1% |
| v4_vec_reg | 6.75 | 30.9 | 7.7 | 41.5 | 30 | long_scoreboard 58.1% |

驱动本优化循环的发现：baseline 是**访存延迟瓶颈而非带宽瓶颈**（DRAM 仅 14%，80% long_scoreboard 停顿，两遍标量加载）。向量化降低了指令数与停顿占比（v1）。寄存器驻留消除了第二次读取（v2 NEUTRAL —— 其标量访问掩盖了收益）；两者结合（v4）赢得主目标。v3 表明仅提高占用率（90.7%）无法弥补非向量化访问。注意：ncu 使用冷 L2，判定框架测量的是 warm-L2 稳态；两套数字均按实验存档。

## 优化实验

| 实验 | 变体 | 假设（摘要） | 判定 | 证据 |
|---|---|---|---|---|
| EXP-0001 | baseline | 参考实现 | KEEP（基线） | 旧框架，已作废（superseded） |
| EXP-0002 | v1_vec | 16B 向量化加载 | REJECT | **已作废** —— 单发事件计时噪声过大（见下） |
| EXP-0003 | baseline | 用修复后的框架重定基线 | KEEP（基线） | cuda-event-batched-v1 |
| EXP-0004 | v1_vec | 16B 向量化加载（重测） | **KEEP** | 1.212×，5/5 轮更快 |
| EXP-0005 | v2_reg | 单遍寄存器驻留（标量） | **NEUTRAL** | 0.989×（±5% 带内） |
| EXP-0006 | v3_wideblock | 512 线程块隐藏延迟 | **REJECT** | 0.768×，5/5 轮更慢 |
| EXP-0007 | v4_vec_reg | 向量化 + 寄存器驻留 | **KEEP** | 1.231×，5/5 轮更快 |

**方法论事故（保留在案，未隐藏）：** EXP-0002 的单发 cuda-event 框架引入了约 6 µs 的启动噪声，把 v1 判为比 baseline *更慢*（0.844×），而 ncu 同时显示 v1 内核**快 2.35×**（6.13 vs 14.36 µs）。这一矛盾促使我们改用 `cuda-event-batched-v1` 框架（每样本 32 次连发 + 同步），并对所有变体重测（EXP-0003 起）。这正是 CUDALab 同时保留两件独立仪器（事件计时 + ncu）、并保留全部被拒/已作废实验完整记录的核心原因。

完整记录：[`experiments/rmsnorm/`](experiments/rmsnorm/)。

## 架构

```
cudalab/
  reference.py      显式 FP32 累加的 RMSNorm 参考实现
  build.py          扩展构建 + 内容哈希缓存管理
  correctness.py    固定容差正确性框架（76 例套件）
  benchmark.py      批量 cuda-event 基准框架 + GPU 状态
  profiler.py       ncu --csv 集成 → 结构化 JSON 摘要
  experiment.py     实验记录 + KEEP/REJECT/NEUTRAL 判定规则
kernels/rmsnorm/
  rmsnorm_common.h  自注册变体注册表 + 设备端辅助函数
  bindings.cpp      PyTorch 扩展入口（纯 C++）
  rmsnorm_baseline.cu  每行一个 block，标量，两遍
  rmsnorm_v1.cu        16B 向量化，两遍
  rmsnorm_v2.cu        单遍寄存器驻留（标量）
  rmsnorm_v3.cu        512 线程块（标量）
  rmsnorm_v4.cu        16B 向量化 + 寄存器驻留  ← 最佳
scripts/
  test_rmsnorm.py         正确性入口
  benchmark_rmsnorm.py    基准入口
  profile_rmsnorm.py      ncu 剖析入口
  optimize_rmsnorm.py     构建→正确性→基准→剖析→判定→记录
tools/env.sh        环境变量的唯一事实来源
experiments/        EXP-*.json 记录 + 正确性 JSON + best.json
benchmarks/         bench_*.json / bench_*.csv 矩阵
profiles/rmsnorm/   结构化剖析 JSON + ncu 原始输出（raw/ 被 git 忽略）
```

新增内核变体 = 新增一个 `.cu` 文件（自注册；无需改动绑定层）；下一次构建自动收录。

## 环境

| 项目 | 取值 |
|---|---|
| GPU | 2× NVIDIA RTX 2080 Ti（Turing，CC 7.5）；基准使用 GPU 0 |
| CUDA 工具链 | 11.8（`/usr/local/cuda`） |
| Python | `/root/miniconda3/envs/pytorch/bin/python`（3.10） |
| PyTorch | 2.4.1+cu118 |
| 剖析器 | Nsight Compute 2022.3（`/usr/local/bin/ncu`，剖析权限正常） |
| 编译参数 | `-O3 -lineinfo --use_fast_math -gencode=arch=compute_75,code=sm_75` |

`tools/env.sh` 设置 `CUDA_HOME`、`PATH`（conda bin + CUDA bin）、`PYTHON`、`TORCH_CUDA_ARCH_LIST=7.5`、`CUDA_VISIBLE_DEVICES=0`。

## 正确性方法论

- 主参考：显式公式
  `y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)`
  `* w.float()`，再转回输入 dtype（FP32 累加，与 PyTorch 版本无关）。
- 每个用例的指标：max_abs_error、max_rel_error（分母下限钳制 1e-3）、NaN、Inf、allclose。
- 固定容差（所有变体一致，已记录）：fp16 atol=2e-3 / rtol=5e-3；fp32 atol=1e-5 / rtol=1e-4。
- 每变体 76 例：11 形状 × 3 种子 × 2 dtype + 5 个边界用例 × 2 dtype（全零、1e-4 幅度、×10 幅度、+3 偏置）。
- 正确性 FAIL 的变体无条件 REJECT，永远不可能成为"最佳"。

## 基准方法论

- `torch.cuda.Event` 事件对；**批量**：每样本 = 32 次连续内核启动（相同输入/输出张量），每样本一次同步；样本时间 = 耗时/32。每轮 150 次不计时的预热启动；每个（变体, 形状, dtype）100 样本 × 5 个独立轮次 = 500 样本。
- 主指标：所有样本的**中位数**；同时保存 p95/min/max 与每轮中位数。编译严格位于计时区域之外（先构建，内容哈希缓存）。
- 同一形状的每个变体使用相同输入张量；同一 GPU（GPU 0）；每个变体运行前后各做一次 nvidia-smi 状态快照（温度、时钟、功率、利用率）。
- 判定（主形状 128×4096 fp16，相对当前最佳，同次运行）：
  ≥5% 更快且 ≥3/5 轮更快 → KEEP；≤5% 更慢且 ≤2/5 轮更快 → REJECT；其余 → NEUTRAL。完整矩阵始终保存。
- 已知局限：本容器内无法锁定 GPU 时钟；~5 µs 尺度约 ±10% 逐次运行方差。跨运行对比无效；判定只使用同矩阵（同次运行）对比。

## Profiling

`cudalab/profiler.py` 在一个专用驱动程序上运行 `ncu --csv -k regex:rmsnorm --launch-skip 2 --launch-count 4 --metrics …`，并把 CSV 解析为 JSON（`profiles/rmsnorm/<variant>_M<M>_H<H>.json`）：内核时长、DRAM/SM 吞吐率 %、实测占用率、每线程寄存器、共享内存、以及 warp 停顿分布（每 issue-active 周期的停顿 warp 周期数，以及占全部停顿的百分比）。指标名已在 NCU 2022.3 / sm_75 上用 `ncu --query-metrics` 核实。ncu 原始 stdout/stderr 保留在 `profiles/rmsnorm/raw/`（git 忽略）。没有伪造数字：取不到的字段一律为 `null`。

## 如何复现

```bash
cd /root/code/cuda
source tools/env.sh          # 设置 CUDA_HOME、PATH、PYTHON、架构列表

# 构建（有缓存；冷启动约 1 分钟，热启动几乎瞬时）
$PYTHON cudalab/build.py

# 正确性（全部变体）
$PYTHON scripts/test_rmsnorm.py

# 基准矩阵（全部变体）
$PYTHON scripts/benchmark_rmsnorm.py --tag myrun

# 剖析单个变体
$PYTHON scripts/profile_rmsnorm.py --variant v4_vec_reg --M 128 --H 4096

# 建立/更新基线，或执行一个优化实验
$PYTHON scripts/optimize_rmsnorm.py baseline
$PYTHON scripts/optimize_rmsnorm.py evaluate --variant v4_vec_reg \
    --parent v1_vec --hypothesis "..." --changes "..."
```

产物：`experiments/rmsnorm/EXP-*.json`、`experiments/rmsnorm/correctness/*.json`、`benchmarks/bench_*.json|csv`、`profiles/rmsnorm/*.json`。

## 局限

- 仅 RMSNorm；单 GPU（GPU 0）；仅连续输入。
- v2/v4 要求 H/256 ∈ {2,4,8,16,32}（H ≤ 8192）；v1 要求 H%8==0（fp16）；v3 要求 H%512==0。baseline 是唯一完全通用的变体。
- 容器内无法锁 GPU 时钟 → 5 µs 尺度 ±10% 方差；NEUTRAL 判定在 ±5% 边界附近跨运行不稳定。
- 批量 warm-L2 判定框架与冷 L2 的 ncu 对接近的变体可能给出不同排序（两套数字均存档；判定使用稳态框架）。
- `effective_bw_gbps` 是逻辑流量 ÷ 时间，不是实测 DRAM 吞吐；>550 GB/s 的数值表示 L2 驻留效应。
- `speedup_vs_pytorch_reference` 列在 v0.1 中为预留字段（null）。

## 路线图

- v0.2：按形状分派的最佳内核（大 H 用 v1、其余用 v4）；fp16 微优化；在允许的环境支持锁频。
- v0.3：更多内核（Softmax、RoPE），按内核组织的参考实现库。
- v1.0：在结构化内核 DSL 上由智能体做假设搜索，复用同一客观层（客观层已刻意做成与内核无关）。
