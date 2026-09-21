// CUDALab QGEMV — 公共变体注册表 + 设备端辅助函数。
//
// QGEMV（INT8 weight-only GEMV）算子定义（v0.6 主路径, 用户指定合同）:
//
//   W_q   (N, K)  连续, row-major, **int8**   （量化权重）
//   scale (N,)    连续, **float32**           （每行 scale）
//   x     (K,)    连续, **float16**           （activation）
//   y     (N,)    连续, **float16**           （输出）
//
//   W_dequant[n,k] = scale[n] · W_q[n,k]
//   y[n] = Σ_k W_dequant[n,k] · x[k]   （FP32 累加, 最后 cast 到 fp16）
//
// 量化合同（对称 per-row, zero_point = 0; 实现见 cudalab/qgemv_quantize.py,
// 量化在 benchmark 计时区**外**预先完成）:
//   scale[n]  = max_k |W[n,k]| / 127        （W 为原始 FP16 权重, fp32 计算）
//   W_q[n,k]  = clamp(round(W[n,k]/scale[n]), -127, 127)
//   scale = 0 的行必须安全: W_q[n,:] ≡ 0（不除零、无 NaN, 输出 y[n] = 0）。
//
// 两层正确性（用户指定, 见 cudalab/qgemv_correctness.py; 两种容差**不混用**）:
//   (a) kernel 正确性: y vs ref = (W_q.float()·scale[:,None]) @ x.float()
//       cast FP16 —— 固定累加误差界（对精确 fp64 反量化 GEMV, 见
//       qgemv_correctness.py (1); 本算子每 term 最多 3 次 FP32 舍入
//       （q→fp32 ×scale ×x + 加法）, 界系数 3K·2^-24）+ 有限性门;
//   (b) 量化保真度: 量化后的 ref vs 原始 FP16 W @ x —— max_abs / max_rel /
//       RMSE / cosine similarity, **只报告不判定**（量化误差不是 kernel
//       bug, 不用 kernel 容差衡量）。
//
// 每个内核变体位于独立的 .cu 文件中，并通过静态注册器自注册:
//
//   static void my_fwd(const at::Tensor& Wq, const at::Tensor& scale,
//                      const at::Tensor& x, at::Tensor& out) { ... }
//   static struct MyRegistrar {
//     MyRegistrar() { register_qgemv_variant("my_variant", my_fwd); }
//   } my_registrar;
//
// 所有输入验证（TORCH_CHECK）都在 kernel launch 之前完成（bindings.cpp
// 的 validate_*）; 内核自身不含设备端断言。与 GEMV 相同: **没有任何
// 数据依赖检查**（无 D2H 同步）, 全部是 host 侧元数据检查
// （dim/shape/dtype/连续/设备）, 逐 launch 开销可忽略, 不需要 RoPE 式
// validate=false 基准池开关 —— forward_into 每次调用都执行完整验证。
// scale 中的 NaN/Inf 属于**值域**（metadata 检查覆盖不到, 也不做 D2H
// 值检查）: 与 v0.5 GEMV 的 NaN/Inf W 语义一致 —— 垃圾进垃圾出, 量化
// 器（离线路径, cudalab/qgemv_quantize.py）对非有限 W 显式拒绝。
//
// 对齐契约总则（v0.6, 沿用 v0.5 向量化契约模式）: **任何 16B 向量化
// load 变体必须对其输入基指针声明明确的对齐契约, 并在 host 侧 launch
// 前检查; 不满足时回退标量路径（合法输入不得被拒）。** 本算子的 16B
// 单位在两个张量上元素数不同（int8: 16 个, fp16 x: 8 个）, 因此契约
// 为: W_q 基指针 16B 对齐 ∧ x 基指针 16B 对齐 ∧ K % 16 == 0
// （K%16==0 同时保证 x 的 16B 单位数 K/8 为偶数, 每 W 向量配对的
// 两个 x 向量完整落在行内）。`is_contiguous()==true` **不保证**
// 16B 对齐 —— 1 字节 storage offset 的连续 int8 视图基指针偏移 1B。
// qgemv_baseline 是纯标量访存（每 thread 1B int8 load + 2B fp16 load）,
// **无对齐契约**, 任何连续合法输入都必须成功。

#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <string>
#include <vector>

// 变体入口: W_q (N,K) int8 连续 / scale (N,) fp32 连续 / x (K,) fp16 连续;
// out (N,) fp16（预分配）。
using qgemv_fn_t = void (*)(const at::Tensor& Wq,
                            const at::Tensor& scale,
                            const at::Tensor& x,
                            at::Tensor& out);

void register_qgemv_variant(const std::string& name, qgemv_fn_t fn);

// 注: qgemv_forward / qgemv_forward_into / qgemv_native_timing 的入口
// 声明只存在于 bindings.cpp（pybind 入口）, 不在公共头里（与
// gemv_common.h 相同的教训: 头里旧签名前向声明会与 bindings.cpp 的
// 新签名构成重载, 使取址时模板推导失败）。

std::vector<std::string> qgemv_variant_list();
std::vector<std::string> qgemv_all_variant_list();
// v0.6: 当前无隔离变体; 本函数保留以维持与 v0.5 GEMV 相同的注册表
// 协议（若未来某候选被 REJECTED/隔离, 走同一机制）。
std::vector<std::string> qgemv_quarantined_variant_list();

// ---- 16B 向量化变体的对齐契约（host 侧, launch 前检查）-----------------
// 任何 16B 向量 load 变体必须满足:
//   (1) W_q 基指针 16B 对齐;  (2) x 基指针 16B 对齐;
//   (3) K 可被 16 整除（16B 单位的 int8 元素数; 见头部推导, 同时保证
//       x 侧 16B 单位 K/8 为偶数）。
// 不满足时变体**必须回退标量路径**（qgemv_scalar_kernel）—— 合法输入
// （包括 1 字节 storage offset 的连续视图）不得被拒绝。scale / out 是
// 每行 1 个标量（4B / 2B）读写, 无对齐要求（不检查）。
inline bool qgemv_vec_contract_ok(const at::Tensor& Wq,
                                  const at::Tensor& x) {
    if (Wq.size(1) % 16 != 0) return false;
    return (reinterpret_cast<uintptr_t>(Wq.const_data_ptr()) & 15u) == 0 &&
           (reinterpret_cast<uintptr_t>(x.const_data_ptr()) & 15u) == 0;
}

// ---- 共享设备端辅助函数（仅限 CUDA 翻译单元）---------------------------

#ifdef __CUDACC__

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

// ---- 共享标量 kernel（QGEMV-0000 计算本体 + 所有向量化变体的回退）-------
// **单一来源**: qgemv_baseline 与全部向量化变体的标量回退都调用这里,
// 保证 "回退输出与 qgemv_baseline 在同一输入上逐位一致"（negative
// 套件的 per-variant 回退回归用例以此为参照）。形态（用户指定的
// baseline, 逐字执行 §4）: 每输出行一个 block, 256 线程, 标量 int8
// load → FP32 → ×scale → ×FP16 activation, FP32 归约, FP16 输出。
static constexpr int kQgemvScalarBlock = 256;
static constexpr int kQgemvScalarWarps = kQgemvScalarBlock / 32;  // 8

static __global__ void qgemv_scalar_kernel(const int8_t* __restrict__ Wq,
                                           const float* __restrict__ scale,
                                           const __half* __restrict__ x,
                                           __half* __restrict__ out,
                                           int64_t N, int64_t K) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int8_t* __restrict__ wrow = Wq + row * K;
    const float s = scale[row];
    // 每元素反量化（spec 逐字）: int8→FP32, ×scale, ×FP16 activation。
    // scale = 0 的行: 全部乘积为 0, y = 0（安全, 无除零）。
    float acc = 0.f;
    for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
        acc += (static_cast<float>(wrow[k]) * s) * __half2float(x[k]);
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    __shared__ float warp_sums[kQgemvScalarWarps];
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        acc = (lane < kQgemvScalarWarps) ? warp_sums[lane] : 0.f;
#pragma unroll
        for (int off = kQgemvScalarWarps / 2; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) out[row] = __float2half_rn(acc);
    }
}

static inline void launch_qgemv_scalar(const at::Tensor& Wq,
                                       const at::Tensor& scale,
                                       const at::Tensor& x,
                                       at::Tensor& out) {
    const int64_t N = Wq.size(0);
    const int64_t K = Wq.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    qgemv_scalar_kernel<<<static_cast<int>(N), kQgemvScalarBlock, 0,
                          stream>>>(
        static_cast<const int8_t*>(Wq.const_data_ptr()),
        static_cast<const float*>(scale.const_data_ptr()),
        reinterpret_cast<const __half*>(x.const_data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- 16B 打包 load 辅助（QGEMV-0001 起向量化变体共用）-------------------
// 16B 单位: W_q 侧 16 × int8, x 侧 8 × __half。每个 W 向量（16 个
// int8）需要 16 个 x 值 = 两个 16B x 向量（xv0 + xv1）。以 uint4 载入
// 后按元素解释 —— 与 v0.5 gemv U16 同一"16B 打包 load"硬件原语,
// 基线不抄任何成熟 INT8 GEMV 实现。
union U16Q {
    uint4 v;
    int8_t q[16];
    __half h[8];
};

// 每元素反量化（baseline 逐字语义）: q→fp32, ×s, ×x（FP32 累加,
// 允许 nvcc 收缩）。每 W 向量 16 个 term。
__device__ __forceinline__ void qgemv_vec_acc_dequant(const U16Q& w,
                                                      const U16Q& xv0,
                                                      const U16Q& xv1,
                                                      float s, float& acc) {
    const __half* xh = xv0.h;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        acc += (static_cast<float>(w.q[j]) * s) * __half2float(xh[j]);
    xh = xv1.h;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        acc += (static_cast<float>(w.q[j + 8]) * s) * __half2float(xh[j]);
}

// scale 提升（数学恒等: scale 对整行是常数, y = s·Σ_k q[k]·x[k]）:
// 每行只有 1 次 ×scale（row 末尾）, 每 term 只 1 次乘 + 1 次加
// （q·x 在 FP32 中精确: 7 位 × 11 位 ≤ 24 位尾数, 无舍入）。
// 与每元素反量化是**不同**的合法 FP32 累加顺序（固定 arith 界覆盖,
// 两者不要求逐位一致）。
__device__ __forceinline__ void qgemv_vec_acc_hoist(const U16Q& w,
                                                    const U16Q& xv0,
                                                    const U16Q& xv1,
                                                    float& acc) {
    const __half* xh = xv0.h;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        acc += static_cast<float>(w.q[j]) * __half2float(xh[j]);
    xh = xv1.h;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        acc += static_cast<float>(w.q[j + 8]) * __half2float(xh[j]);
}

#endif  // __CUDACC__
