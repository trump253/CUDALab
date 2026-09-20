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
// 无对齐契约: __half2 load/store 要求 4B 对齐 —— x 行起点
// (M*D*2B, D%2==0 => 4B 倍数) 与线程内偏移 2*i*2B 均为 4B 倍数,
// 由 x 连续 + D 偶数保证。

#include "rope_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

namespace {

static constexpr int kBlock = 128;

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
        launch_v3_half2<__half>(x, positions, cos_t, sin_t, out);
    } else {
        launch_v3_half2<float>(x, positions, cos_t, sin_t, out);
    }
}

static struct Registrar {
    Registrar() { register_rope_variant("rope_v3_half2", rope_v3_half2_fwd); }
} registrar;

}  // namespace
