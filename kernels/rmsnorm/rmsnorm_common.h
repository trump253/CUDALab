// CUDALab RMSNorm — 公共变体注册表 + 设备端辅助函数。
//
// 每个内核变体位于独立的 .cu 文件中，并通过静态注册器自注册。
// 新增变体永远不需要改动 bindings.cpp。
//
//   static void my_fwd(const at::Tensor& x, const at::Tensor& w,
//                      at::Tensor& out, double eps) { ... }
//   static struct MyRegistrar {
//     MyRegistrar() { register_rmsnorm_variant("my_variant", my_fwd); }
//   } my_registrar;
//
// 内核以原生 CUDA 元素类型（`__half`（fp16）或 `float`（fp32））为
// 模板参数。启动器传原始数据指针（void* / __half* / float*），因此
// 不依赖 c10::Half 与 __half 之间的头文件互操作。

#pragma once
#include <torch/extension.h>
#include <string>
#include <vector>

// 变体入口: x (M,H) 连续, w (H,) 连续, out (M,H)
// （预分配，布局与 x 相同）, eps。
using rmsnorm_fn_t = void (*)(const at::Tensor& x, const at::Tensor& w,
                              at::Tensor& out, double eps);

void register_rmsnorm_variant(const std::string& name, rmsnorm_fn_t fn);

at::Tensor rmsnorm_forward(const std::string& name, const at::Tensor& x,
                           const at::Tensor& w, double eps);

std::vector<std::string> rmsnorm_variant_list();

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
