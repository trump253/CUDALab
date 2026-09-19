# CUDALab v0.1 — 项目计划

自主 CUDA 内核优化实验室。v0.1 范围：**仅 RMSNorm**，
硬件为 2× NVIDIA RTX 2080 Ti（Turing，sm_75），CUDA 11.8，PyTorch 2.4.1+cu118。

## 流水线

```
参考实现 → 正确性 → 基准测试 → GPU 剖析 → 瓶颈分析
→ 优化假设 → 内核修改 → 编译 → 正确性
→ 基准测试 → 采纳/拒绝 → 实验记录
```

## 阶段

### 阶段 0 — 仓库初始化（按定义即起点）
- 清理 `/root/code/cuda`，`git init -b main`，repo-local git 身份。
- `.gitignore`（排除构建产物/`.so`/剖析原始输出；保留基准 CSV/JSON、
  实验元数据、报告）。
- `tools/env.sh` 作为环境变量的唯一事实来源
  （`CUDA_HOME=/usr/local/cuda`、`PYTHON=/root/miniconda3/envs/pytorch/bin/python`、
  `TORCH_CUDA_ARCH_LIST=7.5`、`CUDA_VISIBLE_DEVICES=0`）。
- `PROJECT_PLAN.md`、`STATUS.md`。
- ✅ 已完成

### 阶段 1 — RMSNorm 参考实现
- `cudalab/reference.py`：显式 FP32 累加公式
  `y = x * rsqrt(mean(x², -1) + eps)`，默认 `eps=1e-5`。
- ✅ 已完成

### 阶段 2 — 基线 CUDA 内核
- `kernels/rmsnorm/rmsnorm_baseline.cu`：每行一个 block，
  warp 归约 + 共享内存归约，FP16 进出，FP32 累加。
- `kernels/rmsnorm/bindings.cpp` + `cudalab/build.py`，使用
  `torch.utils.cpp_extension.load()`，显式 `sm_75`，并采用内容哈希
  构建目录：源文件不变绝不重编译，每个候选变体有独立缓存。
- ✅ 已完成

### 阶段 3 — 正确性框架
- `cudalab/correctness.py`：max_abs_error、max_rel_error、NaN/Inf、allclose。
  固定且已记录的容差（fp16: atol=2e-3, rtol=5e-3；fp32: atol=1e-5,
  rtol=1e-4 —— 对所有候选一致，绝不按候选放宽）。
- 矩阵：M ∈ {1,16,128,1024} × H ∈ {1024,2048,4096,8192} 子集 + 边界用例
  （全零、微小值、多尺度、多种子）。
- JSON + 人类可读输出。
- ✅ 已完成

### 阶段 4 — 基准框架
- `cudalab/benchmark.py`：`torch.cuda.Event` 计时，预热 ≥ 100，
  迭代 ≥ 200，≥ 5 个独立轮次，中位数为主指标，
  p50/p95/min/max，有效 DRAM 带宽（读 x + 权重、写 y）。
- 每套运行前后做 GPU 状态快照（nvidia-smi：时钟、温度、功率、利用率）。
- 每个（形状, dtype）的输入张量固定，所有变体共享。
- 所有变体的基准矩阵，CSV + JSON。
- ✅ 已完成

### 阶段 5 — 剖析器集成
- `cudalab/profiler.py`：在小型专用剖析驱动程序上运行 `ncu`，
  映射 Nsight-Compute 2022.3 在 sm_75 上的指标名（通过
  `ncu --query-metrics` 发现），输出结构化 JSON 摘要
  （内核时长、DRAM 吞吐、SM 吞吐、占用率、
  每线程寄存器、共享内存、warp 停顿）。取不到的字段为 null。
- 兜底方案：保存真实的 ncu 错误，记录到 STATUS.md，改用
  nsys / torch.profiler 作为替代。
- ✅ 已完成（或兜底已记录）

### 阶段 6 — 实验追踪
- `cudalab/experiment.py`：`experiments/rmsnorm/EXP-NNNN.json` 记录，
  含假设、改动、正确性、相对当前最佳的基准、剖析观察、
  判定（KEEP/REJECT/NEUTRAL）。
- 判定规则（v0.1）：
  - 正确性 FAIL → REJECT（无条件）；
  - 中位数加速 ≥ 5% 且多数轮次更快 → KEEP；
  - ±5% 以内 → NEUTRAL；明显更慢 → REJECT。
  - 完整基准矩阵始终保存；不做形状挑拣。
- `scripts/optimize_rmsnorm.py`：构建 → 正确性 → 基准 → 剖析
  → 候选 vs 当前最佳评估 → 判定 → 记录。
- ✅ 已完成

### 阶段 7 — 第一轮优化循环
- ≥ 3 个真实、数据驱动的优化实验（对照 baseline）：
  1. 向量化加载（float4/half2）—— 内存带宽瓶颈假设；
  2. half2 + 减少同步 / 单遍重构（若剖析数据支持）；
  3. 一个"预期会失败"的假设以验证 REJECT 分支
     （如更大的 grid-stride / 不同块大小）—— 仅在数据真正支持时做。
- 主要目标形状：**M=128, H=4096, FP16**；完整矩阵仍全部报告。
- ✅ 已完成

### 阶段 8 — 文档与最终验证
- 对最终最佳内核重跑完整正确性套件 + 完整基准矩阵。
- `README.md`（GitHub 质量、只含真实数字）、`docs/design.md`、
  更新 `STATUS.md`、定稿 PROJECT_PLAN。
- 每个阶段一个 git 提交；最终工作树干净 + 提交。
- ✅ 已完成

## 不可妥协的规则
- 不伪造任何数字；每个报告的指标都来自真实执行。
- 编译时间绝不计入内核计时；先基准、后剖析。
- 所有变体使用相同输入；权重乘法绝不跳过。
- 失败的实验保留并报告。

## 状态
实时进展见 `STATUS.md`。
