// CUDALab RoPE — ROPE-0003 候选: __half2 打包访存 + 转换指令削减。
//
// 相对 rope_baseline 的改动（仅 fp16 路径有实质差异; grid/每线程
// 工作量与 baseline 完全相同 —— 1 thread -> 1 pair）:
//   * x 的一对 (a,b) 用一次 4B __half2 load 载入（baseline: 2 次 2B）
//   * 输出 (y0,y1) 用一次 __floats2half2_rn（2 值 1 指令）+ 一次
//     4B __half2 store（baseline: 2 次 el_from_float + 2 次 2B store）
//   * cos/sin 各 1 个 2B 标量 load（与 baseline 相同）
// 旋转数学保持在 FP32（4 mul + 1 sub + 1 add）—— 本变体不改变算术,
// 只改访存/转换指令结构。逐位一致性按路径区分（v0.4 review F5 更正,
// 原"与 baseline 逐位一致"的表述被记录数据推翻）:
//   * fp16 路径（本变体唯一实质改动处）: 与 baseline 逐位一致 ——
//     384 项套件中 192/192 个 fp16 用例在 5 个变体间 max_abs 完全相同;
//   * fp32 路径: 源码与 baseline 相同, 但记录显示 110/192 个 fp32 用例
//     的误差值在变体/构建间不同（如 (1024,64) fp32 pos_max_seq_len-1:
//     baseline 4.8e-7 vs 候选 2.4e-7, arith ratio 0.4438 vs 0.4831,
//     均在界内全过）—— nvcc 逐函数 codegen/FMA 收缩漂移, 故 fp32
//     路径**不作逐位一致声明**, 只声明在固定 arith 界内。
// fp32 路径无转换可削, 退化为与 baseline 相同的标量代码（预期
// NEUTRAL, 在矩阵中如实记录）。
//
// 依据（baseline NCU, M=1024 D=128 fp16, cc=all clkbase）:
//   主 stall 是 long_scoreboard（69.3%, 延迟）, 而 sm 吞吐 11.61% /
//   math_pipe_throttle 1.9% / no_instruction 1.4% —— 发射端不紧。
//   因此**预测本实验 NEUTRAL/REJECT**: 这是"不同杠杆"的对照实验,
//   同时检验 v2.3 评测器能否正确拒绝一个（按证据预期）不改进的
//   候选（评测器灵敏度/负对照检查）。
//
// 对齐契约（v0.4.1 修复; 原"无对齐契约"注释错误）: fp16 路径的
// __half2 load/store 要求 **x 与 out 的基指针** 4B 对齐。线程内偏移
// (2*i 元素 = 4i B) 与行距 (2*d2 元素 = 4*d2 B) 在 D 偶数时均为 4B
// 倍数, 因此只有基指针对齐是前提; 而 `is_contiguous()==true` **不
// 保证** 4B 对齐 —— storage offset 为奇数元素（奇 × 2B）的连续视图
// （如 big[1:].view(M,D)）基指针偏移 2B, 进入 half2 路径是未定义
// 行为。新分配的 PyTorch 张量按 256B 对齐, 故该风险只出现在视图上。
// 处理（rope_v3_half2_fwd, host 侧, launch 前）:
//   x/out 基指针均 4B 对齐 -> 现有 __half2 打包 kernel;
//   任一未对齐            -> 回退 rope_v3_half2_scalar_kernel
//     （fp16 标量, 数学与 rope_baseline fp16 路径逐语句一致:
//     el_to_float 提升 + FP32 旋转 + el_from_float 逐值 RN 舍入 ——
//     与 half2 路径位级一致, 同 fp16 路径对 baseline 的位级声明）。
// 选择回退而非拒绝: 合法输入不应被拒, 且回退路径与 baseline 同形态
// （标量 2B 访存）, 无正确性损失（性能回退仅限该罕见输入形态）。
// fp32 路径无 __half2, 标量 4B load 在连续契约下天然满足, 无对齐检查。

#include "rope_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

static constexpr int kBlock = 128;

// v0.4.1 回退 kernel: x/out 基指针未 4B 对齐时使用的 fp16 标量路径。
// 数学与 rope_baseline fp16 路径逐语句一致（el_to_float 提升 + FP32
// 旋转 + el_from_float 逐值 RN 舍入）——与 half2 路径位级一致（同为
// 逐值 RN, __floats2half2_rn == 两次 __float2half_rn）。
__global__ void rope_v3_half2_scalar_kernel(const __half* __restrict__ x,
                                            const int64_t* __restrict__ positions,
                                            const __half* __restrict__ cos_t,
                                            const __half* __restrict__ sin_t,
                                            __half* __restrict__ out,
                                            int64_t n_pairs, int64_t d2) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= n_pairs) return;
    int64_t m = t / d2;
    int64_t i = t - m * d2;
    int64_t pos = positions[m];

    const __half* xp = x + m * (2 * d2) + 2 * i;
    float a = el_to_float(xp[0]);
    float b = el_to_float(xp[1]);
    float c = el_to_float(cos_t[pos * d2 + i]);
    float s = el_to_float(sin_t[pos * d2 + i]);

    __half* yp = out + m * (2 * d2) + 2 * i;
    yp[0] = el_from_float<__half>(a * c - b * s);
    yp[1] = el_from_float<__half>(a * s + b * c);
}

template <typename T>
__global__ void rope_v3_half2_kernel(const T* __restrict__ x,
                                     const int64_t* __restrict__ positions,
                                     const T* __restrict__ cos_t,
                                     const T* __restrict__ sin_t,
                                     T* __restrict__ out,
                                     int64_t n_pairs, int64_t d2) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= n_pairs) return;
    int64_t m = t / d2;
    int64_t i = t - m * d2;
    int64_t pos = positions[m];

    const T* xp = x + m * (2 * d2) + 2 * i;
    float c = el_to_float(cos_t[pos * d2 + i]);
    float s = el_to_float(sin_t[pos * d2 + i]);

    T* yp = out + m * (2 * d2) + 2 * i;
    if constexpr (std::is_same<T, __half>::value) {
        // 一次 4B __half2 load; FP32 旋转; 一次打包 store
        __half2 ab = *reinterpret_cast<const __half2*>(xp);
        float a = __half2float(ab.x);
        float b = __half2float(ab.y);
        __half2 y = __floats2half2_rn(a * c - b * s, a * s + b * c);
        *reinterpret_cast<__half2*>(yp) = y;
    } else {
        // fp32: 与 baseline 相同的标量路径（无转换可削）
        float a = xp[0];
        float b = xp[1];
        yp[0] = a * c - b * s;
        yp[1] = a * s + b * c;
    }
}

template <typename T>
void launch_v3_half2(const at::Tensor& x, const at::Tensor& positions,
                     const at::Tensor& cos_t, const at::Tensor& sin_t,
                     at::Tensor& out) {
    const int64_t M = x.size(0);
    const int64_t d2 = x.size(1) / 2;
    const int64_t n_pairs = M * d2;
    const int grid = static_cast<int>((n_pairs + kBlock - 1) / kBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rope_v3_half2_kernel<T><<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<const T*>(x.const_data_ptr()),
        positions.const_data_ptr<int64_t>(),
        reinterpret_cast<const T*>(cos_t.const_data_ptr()),
        reinterpret_cast<const T*>(sin_t.const_data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()),
        n_pairs, d2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rope_v3_half2_fwd(const at::Tensor& x, const at::Tensor& positions,
                       const at::Tensor& cos_t, const at::Tensor& sin_t,
                       at::Tensor& out) {
    if (x.scalar_type() == at::kHalf) {
        // v0.4.1: fp16 路径做 __half2 4B load/store, 要求 x 与 out 的
        // 基指针 4B 对齐（见文件头"对齐契约"）。host 侧 launch 前检查:
        // 对齐 -> half2 打包 kernel; 任一未对齐 -> 标量回退 kernel
        // （baseline 兼容数学, 位级一致）。
        const bool x_aligned =
            (reinterpret_cast<uintptr_t>(x.const_data_ptr()) & 3u) == 0;
        const bool out_aligned =
            (reinterpret_cast<uintptr_t>(out.data_ptr()) & 3u) == 0;
        if (x_aligned && out_aligned) {
            launch_v3_half2<__half>(x, positions, cos_t, sin_t, out);
            return;
        }
        // 回退: 标量 fp16 路径（baseline 兼容数学, 位级一致）
        const int64_t M = x.size(0);
        const int64_t d2 = x.size(1) / 2;
        const int64_t n_pairs = M * d2;
        const int grid = static_cast<int>((n_pairs + kBlock - 1) / kBlock);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        rope_v3_half2_scalar_kernel<<<grid, kBlock, 0, stream>>>(
            reinterpret_cast<const __half*>(x.const_data_ptr()),
            positions.const_data_ptr<int64_t>(),
            reinterpret_cast<const __half*>(cos_t.const_data_ptr()),
            reinterpret_cast<const __half*>(sin_t.const_data_ptr()),
            reinterpret_cast<__half*>(out.data_ptr()),
            n_pairs, d2);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        // fp32: 标量 4B load/store, 连续契约下天然 4B 对齐, 无需检查
        launch_v3_half2<float>(x, positions, cos_t, sin_t, out);
    }
}

static struct Registrar {
    Registrar() { register_rope_variant("rope_v3_half2", rope_v3_half2_fwd); }
} registrar;

}  // namespace
