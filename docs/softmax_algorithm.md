# Softmax 算法推导 — 行内 softmax 与 online (m, l) 合并恒等

本文档是 v0.3 Softmax 优化实验的**前置推导**（任务要求：online softmax
在 CUDA 实现之前，必须先完成本推导 + CPU merge 恒等测试）。

参考定义（行内，dim = -1）：

```
M  = max_j  x_j
L  = Σ_j  exp(x_j − M)
y_i = exp(x_i − M) / L
```

减 M 是数值稳定的标准技巧（保证 exp 参数 ≤ 0，无溢出；L ≥ 1，因为
最大元贡献 exp(0) = 1）。baseline（3 遍）按字面实现：遍 1 求 M，
遍 2 求 L，遍 3 重读 x 计算 y。

## 1. 部分 (m, l) 的状态量

把一行的元素划分为若干**段**（segment）。对任意段 S 定义：

```
m_S = max_{j∈S} x_j
l_S = Σ_{j∈S} exp(x_j − m_S)
```

(m_S, l_S) 是该段关于行内 softmax 的**充分统计量**：任何段的 (m, l)
都能与另一段合并出并集的 (m, l)。

## 2. 合并恒等（merge identity）

对不相交段 A、B，设 m_A, l_A, m_B, l_B，并集 S = A ∪ B：

**命题**: m_S = max(m_A, m_B)，且

```
l_S = l_A · exp(m_A − m_S) + l_B · exp(m_B − m_S)
```

**证明**: max 显然。对 l：

```
Σ_{j∈S} exp(x_j − m_S)
  = Σ_{j∈A} exp(x_j − m_S) + Σ_{j∈B} exp(x_j − m_S)
  = Σ_{j∈A} exp(x_j − m_A)·exp(m_A − m_S)
    + Σ_{j∈B} exp(x_j − m_B)·exp(m_B − m_S)
  = l_A·exp(m_A − m_S) + l_B·exp(m_B − m_S).
```

（用了 exp(x − m_S) = exp(x − m_A)·exp(m_A − m_S)，标量指数律。）∎

**数值安全**: m_A ≤ m_S 恒成立 ⇒ exp(m_A − m_S) ∈ (0, 1]，合并因子
只会缩小、永不放大 —— 无溢出路径。l_S ≤ |A| + |B|（每项 ≤ 1），
线性有界。

**退化情形**: 若 A 为空，取 m_A = −∞, l_A = 0，则 exp(m_A − m_S) = 0，
l_S = l_B —— 与"空段不贡献"一致（实现中 −∞ 用 float('-inf') 或
足够小的有限值；`-inf` 经 `exp(-inf - m) = 0` 同样安全）。

## 3. 逐元素在线更新（segment 退化为单元素）

把 merge 用于单元素 {x}（m_x = x, l_x = 1）与当前状态 (m, l)：

```
m' = max(m, x)
l' = l · exp(m − m') + exp(x − m')
```

逐元素扫描即得整行的 (m, l)，与"先求 M 再求 L"在**精确算术**下
等价；浮点下两者是**不同的求和顺序**（在线版把"减 M"推迟到扫描
完成，部分和反复重标度），差异在 ε 量级 —— 由固定容差（fp16
2e-3 / fp32 1e-5 abs）与 row_sum 检查覆盖，不需要逐位一致。

## 4. 并行归约下的合并

一行一个 block（block = B 线程）：

1. 每个线程 t 对自己的 stride 下标集 S_t 逐元素累积局部 (m_t, l_t)
   （§3）——**一次读遍 x**；
2. block 归约：把 (m_t, l_t) 按 merge 恒等合并（warp shuffle 阶段
   两两合并 + shared memory 跨 warp 合并），得全行 (M, L)；
3. 每个线程重读自己下标的 x，写 y_i = exp(x_i − M)/L —— **第二次读
   x + 一次写 y**。

总内部流量 **2 读 1 写 = 3× 算法字节数**，对比 baseline 的 3 读 1 写
= 4× —— 减 25%；且 exp 在归约后只算一次（baseline 在遍 2、遍 3 各
算一遍），ALU 侧同步受益。

归约实现注意：merge 不是结合律友好的"加法"（重标度因子依赖于两个
操作数的 m），两两合并必须**成对**应用 §2 的公式（a 合并 b 的完整
(m,l) 对），不能拆成 max 归约 + sum 归约各走各的。shuffle 阶段：
每个线程持有 (m, l) 对，offset 步长内线程 i 与 i+offset 交换后各自
计算并集 —— 标准 "reduce 任意可交换二元运算" 的骨架，这里运算
`⊕: (m,l) ⊕ (m',l') = (max(m,m'), l·exp(m−M) + l'·exp(m'−M))`，
M = max(m, m')。⊕ 是可交换、可结合的（精确算术下；浮点下与顺序
相关但结构确定 ⇒ 结果确定性不变）。

## 5. 与 baseline 的数值关系

- 相同输入下，两实现的 y 在容差内一致（固定容差不放宽）；
- 两者都不需要 x 的有限性假设之外的处理（NaN/Inf 行为与 baseline
  一致：baseline 对 +inf 输入 exp(+inf−M)=exp(0)=1，多 +inf 时
  L = n_inf，y 均分 —— online 版经同样的 exp 语义得到相同结果；
  本套件不测非有限输入，negative 套件覆盖的是**输入契约**而非
  数值边界）；
- 行和误差: Σ y_i = (Σ exp(x_i − M))/L = 1（精确算术）；浮点下
  误差与归约顺序相关，由 ROW_SUM_TOL（fp16 5e-3 / fp32 1e-4）
  把关。

## 6. CPU merge 恒等测试（CUDA 实现前的门禁）

`tests/test_softmax_cpu.py` 中的 online 测试（本节推导的机器验证）：

1. **单段对拍**: 随机向量 x（含极端 ±80、常数、单主元），逐元素
   在线累积 (m, l) 后归一化，对 torch.softmax（fp32 参考）
   max_abs ≤ 1e-5；
2. **多段合并**: 随机切分点（1/2/k 段），每段独立 (m, l)，按任意
   二分树顺序两两 merge，最终 (M, L) 与直接计算
   (max(x), Σexp(x−max(x))) 的相对误差 ≤ 1e-6（fp32）；
3. **退化段**: 空段 (−inf, 0) 合并不影响结果；
4. **确定性**: 同一输入、同一切分、重复 3 次 → 逐位相同。

以上全部通过之前，不允许任何 online/单遍 CUDA 变体进入仓库。
