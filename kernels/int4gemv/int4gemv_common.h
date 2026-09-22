// CUDALab INT4GEMV — 公共变体注册表 + 设备端辅助函数。
//
// INT4GEMV（W4A16: INT4 weight-only + FP16 activation GEMV）算子
// 定义（v0.7 主路径, 用户指定合同）:
//
//   W_packed  (N, K/2)  连续, row-major, **uint8**   （两个 INT4 / byte）
//   scale     (N, K/G)  连续, **float16**            （group-wise scale,
//                                                     G = 128 固定）
//   x         (K,)      连续, **float16**            （activation）
//   y         (N,)      连续, **float16**            （输出）
//
//   W_dequant[n,k] = scale[n, k/128] · unpack(W_packed)[n,k]
//   y[n] = Σ_k W_dequant[n,k] · x[k]   （FP32 累加, 最后 cast 到 fp16）
//
// 量化合同（对称 group-wise, zero_point = 0; 实现见
// cudalab/int4gemv_quantize.py, 量化与 packing 在 benchmark 计时区
// **外**预先完成）:
//   q ∈ [-7, 7],  zero_point = 0
//   scale[n,g]  = max_k∈group |W[n,k]| / 7          （W 为原始 FP16,
//                                                    fp32 计算）
//   q[n,k]      = clamp(round(W[n,k]/scale), -7, 7) （round-half-to-even）
//   零 group 必须安全: scale=0 → q≡0（不除零、无 NaN, 该行该 group
//   对 y 贡献 0）。
//   **scale 以 FP16 存储**（合同指定）: 量化器 fp32 计算后 cast
//   fp16; 内核与参考层 A 都读存储的 fp16 scale —— fp16 scale 量化
//   是算子数据合同的一部分。
//   K 要求: K % 128 == 0（量化器与内核 host 验证都显式拒绝）。
//
// Nibble 打包合同（与 cudalab/int4gemv_quantize.py pack_q/unpack_w
// 逐元素一致; 全字节双射与符号扩展由 tests/test_int4gemv_cpu.py
// 第 [1][2] 层钉死, GPU 侧由 kernel 正确性层 A 钉死）:
//   W_packed[n, b] 低 4 bit  = 元素 k = 2b   （low nibble）
//   W_packed[n, b] 高 4 bit  = 元素 k = 2b+1  （high nibble）
//   编码 = 4-bit two's complement（值域 [-8, 7], 量化域 [-7, 7]）。
//   1 byte = 2 个 INT4 = 2 个连续 k; group 边界（128 元素 = 64 byte）
//   永远落在 byte 边界上（128 % 2 == 0）, 因此**一个 byte 的两个元素
//   恒属同一 group** —— 每 byte 只需 1 次 scale lookup（b>>6）。
//
// 三层正确性（用户 §2, 见 cudalab/int4gemv_correctness.py, 容差
// **不混用**）:
//   (a) kernel 正确性（判定门, per-variant）: y vs
//       ref = unpack(W_packed).float()·scale_fp16_expanded @ x.float()
//       cast FP16 —— 固定累加误差界（对精确 fp64 反量化 GEMV, 每
//       term 最多 3 次 FP32 舍入, 界系数 3K·2^-24）+ 有限性门;
//   (b) pack/unpack correctness（CPU 独立层, tests/test_int4gemv_cpu.py）:
//       nibble 编码 -7..7（+ 全域 -8..7）pack → unpack 逐元素一致,
//       负数符号扩展 / high-low nibble / 边界 -7/0/+7;
//   (c) 量化保真度（只报告, 不判定, variant 无关）: INT4 量化后
//       reference vs 原 FP16 W @ x —— max_abs / RMSE / cosine
//       similarity（量化误差不是 kernel bug）。
//
// 每个内核变体位于独立的 .cu 文件中，并通过静态注册器自注册:
//
//   static void my_fwd(const at::Tensor& Wp, const at::Tensor& scale,
//                      const at::Tensor& x, at::Tensor& out) { ... }
//   static struct MyRegistrar {
//     MyRegistrar() { register_int4gemv_variant("my_variant", my_fwd); }
//   } my_registrar;
//
// 所有输入验证（TORCH_CHECK）都在 kernel launch 之前完成（bindings.cpp
// 的 validate_*）; 内核自身不含设备端断言。与 GEMV/QGEMV 相同:
// **没有任何数据依赖检查**（无 D2H 同步）, 全部是 host 侧元数据
// 检查（dim/shape/dtype/连续/设备/K%128）, 逐 launch 开销可忽略 ——
// forward_into 每次调用都执行完整验证。scale 中的 NaN/Inf 属于
// **值域**（metadata 检查覆盖不到, 也不做 D2H 值检查）: 与 v0.5
// GEMV / v0.6 QGEMV 语义一致 —— 垃圾进垃圾出, 量化器（离线路径）
// 对非有限 W 显式拒绝。
//
// 对齐契约总则（v0.7, 沿用 v0.5/v0.6 向量化契约模式）: **任何 16B
// 向量化 packed load 变体必须对其输入基指针声明明确的对齐契约, 并在
// host 侧 launch 前检查; 不满足时回退标量路径（合法输入不得被拒）。**
// 本算子的 16B 单位: W_packed 侧 16 byte = 32 个 INT4; x 侧 16B =
// 8 个 fp16 —— 注意**一个 16B packed 向量（32 个 INT4）需要 32 个
// x 值 = 4 个 16B x 向量**。因此契约为: W_packed 基指针 16B 对齐
// ∧ x 基指针 16B 对齐 ∧ K % 32 == 0（K%128==0 已保证, 保留在检查里
// 作为自文档化）, 即 `int4gemv_vec_contract_ok`。`is_contiguous()==true`
// **不保证** 16B 对齐 —— 1 字节 storage offset 的连续 uint8 视图基指针
// 偏移 1B。int4gemv_baseline 是纯标量访存（每 thread 1B packed load
// + 2B fp16 x load）, **无对齐契约**, 任何连续合法输入都必须成功。

#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <string>
#include <vector>

// 变体入口: W_packed (N,K/2) uint8 连续 / scale (N,K/128) fp16 连续
// / x (K,) fp16 连续; out (N,) fp16（预分配）。
using int4gemv_fn_t = void (*)(const at::Tensor& Wp,
                               const at::Tensor& scale,
                               const at::Tensor& x,
                               at::Tensor& out);

void register_int4gemv_variant(const std::string& name, int4gemv_fn_t fn);

// 注: int4gemv_forward / int4gemv_forward_into / int4gemv_native_timing
// 的入口声明只存在于 bindings.cpp（pybind 入口）, 不在公共头里（与
// gemv_common.h / qgemv_common.h 相同的教训: 头里旧签名前向声明会与
// bindings.cpp 的新签名构成重载, 使取址时模板推导失败）。

std::vector<std::string> int4gemv_variant_list();
std::vector<std::string> int4gemv_all_variant_list();
// v0.7: 当前无隔离变体; 本函数保留以维持与 v0.5/v0.6 相同的注册表
// 协议（若未来某候选被 REJECTED/隔离, 走同一机制）。
std::vector<std::string> int4gemv_quarantined_variant_list();

// ---- 16B 向量化变体的对齐契约（host 侧, launch 前检查）-----------------
// 任何 16B 向量 load 变体必须满足:
//   (1) W_packed 基指针 16B 对齐;  (2) x 基指针 16B 对齐;
//   (3) K % 32 == 0（16B packed 单位 = 32 个 INT4, 需要 32 个连续
//       x 值完整落在向量内; K%128==0 已保证, 保留为自文档化）。
// 不满足时变体**必须回退标量路径**（int4gemv_scalar_kernel）——
// 合法输入（包括 1 字节 storage offset 的连续视图）不得被拒绝。
// scale / out 是每行每 group 2B / 每行 2B 读写, 无对齐要求（不检查）。
inline bool int4gemv_vec_contract_ok(const at::Tensor& Wp,
                                     const at::Tensor& x) {
    const int64_t K = 2 * Wp.size(1);
    if (K % 32 != 0) return false;
    return (reinterpret_cast<uintptr_t>(Wp.const_data_ptr()) & 15u) == 0 &&
           (reinterpret_cast<uintptr_t>(x.const_data_ptr()) & 15u) == 0;
}

// ---- 共享设备端辅助函数（仅限 CUDA 翻译单元）---------------------------

#ifdef __CUDACC__

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

// ---- nibble unpack（合同见文件头; CPU 镜像 = pack_q/unpack_w）---------
// 低 nibble = 元素 2b, 高 nibble = 元素 2b+1; 4-bit two's complement,
// 两个返回值都做符号扩展（-8..7）。
//
// 实现注意（v0.7 首跑 kernel bug, 已由层 A 正确性套件钉死回归）:
// 低 nibble **不能**直接 cast int8_t（b & 0xF ∈ 0..15 在 int8_t 里
// 仍是正的 0..15, 必须把符号位放到 bit 7 再算术右移）; 高 nibble
// **不能**用 (b << 4)（那会把 hi 推出字节、把 lo 送进符号位, 得到
// sign_ext(lo) 而非 hi）—— byte 自己的 bit 7 就是 hi nibble 的符号
// 位, 直接 cast int8_t 再算术右移 4 即可。两条都是无分支单移位。
__device__ __forceinline__ void int4gemv_unpack_byte(uint8_t b,
                                                     int& lo, int& hi) {
    // lo: (b & 0xF) << 4 把 4 位值放到 bit 4-7, bit 7 = lo 的符号位;
    //     int8_t cast 后 8..15 → 负, 算术右移 4 完成符号扩展
    lo = static_cast<int8_t>((b & 0x0Fu) << 4) >> 4;
    // hi: byte 的 bit 7 就是 hi nibble 的符号位; int8_t cast 从 bit 7
    //     符号扩展, 算术右移 4 取出 hi
    hi = static_cast<int8_t>(b) >> 4;
}

// ---- 共享标量 kernel（INT4GEMV-0000 计算本体 + 所有向量化变体的回退）----
// **单一来源**: int4gemv_baseline 与全部向量化变体的标量回退都调用
// 这里, 保证 "回退输出与 int4gemv_baseline 在同一输入上逐位一致"
// （negative 套件的 per-variant 回退回归用例以此为参照）。形态（用户
// 指定的 baseline, 逐字执行 §4）: 每输出行一个 block, 256 线程, 每
// thread 步长读 packed byte → unpack 两个 signed INT4 → 按 k/128 读
// group scale（fp16→fp32）→ FP32 dequant × activation, FP32 归约,
// FP16 输出。
static constexpr int kInt4ScalarBlock = 256;
static constexpr int kInt4ScalarWarps = kInt4ScalarBlock / 32;  // 8

static __global__ void int4gemv_scalar_kernel(const uint8_t* __restrict__ Wp,
                                              const __half* __restrict__ scale,
                                              const __half* __restrict__ x,
                                              __half* __restrict__ out,
                                              int64_t N, int64_t K) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const uint8_t* __restrict__ wrow = Wp + row * (K / 2);
    const __half* __restrict__ srow = scale + row * (K / 128);
    float acc = 0.f;
    for (int64_t b = threadIdx.x; b < K / 2; b += blockDim.x) {
        const uint8_t byte = wrow[b];
        // 1 byte = 2 个连续 k, 恒属同一 group: g = (2b)/128 = b/64
        const float s = __half2float(srow[b >> 6]);
        int lo, hi;
        int4gemv_unpack_byte(byte, lo, hi);
        // 每元素反量化（spec 逐字）: q→fp32, ×scale, ×FP16 activation
        acc += (static_cast<float>(lo) * s) * __half2float(x[2 * b]);
        acc += (static_cast<float>(hi) * s) * __half2float(x[2 * b + 1]);
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    __shared__ float warp_sums[kInt4ScalarWarps];
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        acc = (lane < kInt4ScalarWarps) ? warp_sums[lane] : 0.f;
#pragma unroll
        for (int off = kInt4ScalarWarps / 2; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) out[row] = __float2half_rn(acc);
    }
}

static inline void launch_int4gemv_scalar(const at::Tensor& Wp,
                                          const at::Tensor& scale,
                                          const at::Tensor& x,
                                          at::Tensor& out) {
    const int64_t N = Wp.size(0);
    const int64_t K = 2 * Wp.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int4gemv_scalar_kernel<<<static_cast<int>(N), kInt4ScalarBlock, 0,
                             stream>>>(
        static_cast<const uint8_t*>(Wp.const_data_ptr()),
        reinterpret_cast<const __half*>(scale.const_data_ptr()),
        reinterpret_cast<const __half*>(x.const_data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- 16B 打包 load 辅助（向量化变体共用）-------------------------------
// 16B 单位: W_packed 侧 16 byte = 32 个 INT4（uint4）; x 侧 8 个
// __half（16B）。每个 W 向量需要 32 个 x 值 = 4 个连续 16B x 向量。
union U32I4 {
    uint4 v;
    uint8_t b[16];
    __half h[8];
};

// 每元素反量化（baseline 逐字语义）: q→fp32, ×s, ×x（FP32 累加,
// 允许 nvcc 收缩）。每 W 向量 32 个 term; scale 逐 group 更新
// （16 byte = 32 个连续 k = 恰好 1 个 group 的 1/4; 向量内 4 段
// group: 元素 0-31 → group (b0/64) .. 跨越最多 2 个 group —— 见
// 调用侧的逐 group 处理）。
__device__ __forceinline__ void int4gemv_vec_acc_unpack(const U32I4& w,
                                                        int (&q)[32]) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int lo, hi;
        int4gemv_unpack_byte(w.b[j], lo, hi);
        q[2 * j] = lo;
        q[2 * j + 1] = hi;
    }
}

#endif  // __CUDACC__
