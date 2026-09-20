// CUDALab RoPE — 公共变体注册表 + 设备端辅助函数。
//
// RoPE（Rotary Position Embedding）算子定义（interleaved pair 约定）:
//
//   x        (M, D)   连续, D 为偶数; M = token-head 行数
//   positions (M,)    int64, 每行的旋转位置
//   cos/sin  (L, D/2) 连续, L = max_seq_len; 表常驻 GPU（计时区外预计算）
//
//   对每行 m、每对 i ∈ [0, D/2):
//     a = x[m, 2i]          b = x[m, 2i+1]
//     c = cos[positions[m], i]   s = sin[positions[m], i]
//     y[m, 2i]   = a*c - b*s
//     y[m, 2i+1] = a*s + b*c
//
// FP32 中间运算，输出 dtype 与 x 相同。**不使用 NeoX half-split 约定**
// （README 与算子文档均标注 "interleaved RoPE"）。
//
// 每个内核变体位于独立的 .cu 文件中，并通过静态注册器自注册:
//
//   static void my_fwd(const at::Tensor& x, const at::Tensor& positions,
//                      const at::Tensor& cos_t, const at::Tensor& sin_t,
//                      at::Tensor& out) { ... }
//   static struct MyRegistrar {
//     MyRegistrar() { register_rope_variant("my_variant", my_fwd); }
//   } my_registrar;
//
// 内核以原生 CUDA 元素类型（`__half`（fp16）或 `float`（fp32））为
// 模板参数。启动器传原始数据指针，不依赖 c10::Half 与 __half 之间的
// 头文件互操作。
//
// 所有输入验证（TORCH_CHECK）都在 kernel launch 之前完成（bindings.cpp
// 的 validate_rope_inputs）; 内核自身不含设备端断言。

#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <string>
#include <vector>

// 变体入口: x (M,D) / positions (M,) / cos_t & sin_t (L, D/2) 均连续;
// out (M,D)（预分配, 布局与 x 相同）。
using rope_fn_t = void (*)(const at::Tensor& x,
                           const at::Tensor& positions,
                           const at::Tensor& cos_t,
                           const at::Tensor& sin_t,
                           at::Tensor& out);

void register_rope_variant(const std::string& name, rope_fn_t fn);

at::Tensor rope_forward(const std::string& name, const at::Tensor& x,
                        const at::Tensor& positions,
                        const at::Tensor& cos_t, const at::Tensor& sin_t);

void rope_forward_into(const std::string& name, const at::Tensor& x,
                       const at::Tensor& positions,
                       const at::Tensor& cos_t, const at::Tensor& sin_t,
                       at::Tensor& out);

std::vector<std::string> rope_variant_list();
std::vector<std::string> rope_all_variant_list();

// ---- 共享设备端辅助函数（仅限 CUDA 翻译单元）---------------------------

#ifdef __CUDACC__

__device__ __forceinline__ float el_to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float el_to_float(float v) { return v; }

template <typename T>
__device__ __forceinline__ T el_from_float(float v) {
    return T(v);  // float 恒等
}

// 显式特化: torch 以 -D__CUDA_NO_HALF_CONVERSIONS__ 编译，
// 因此使用内建函数而不是（被禁用的）__half(float) 构造器。
template <>
__device__ __forceinline__ __half el_from_float<__half>(float v) {
    return __float2half_rn(v);
}

#endif  // __CUDACC__
