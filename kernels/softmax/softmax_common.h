// CUDALab Softmax — 公共变体注册表 + 设备端辅助函数。
//
// 每个内核变体位于独立的 .cu 文件中，并通过静态注册器自注册。
// 新增变体永远不需要改动 bindings.cpp。
//
//   static void my_fwd(const at::Tensor& x, at::Tensor& out) { ... }
//   static struct MyRegistrar {
//     MyRegistrar() { register_softmax_variant("my_variant", my_fwd); }
//   } my_registrar;
//
// 内核以原生 CUDA 元素类型（`__half`（fp16）或 `float`（fp32））为
// 模板参数。启动器传原始数据指针（void* / __half* / float*），因此
// 不依赖 c10::Half 与 __half 之间的头文件互操作。
//
// Softmax 无辅助张量: 入口只有 x 与预分配的 out（行-wise，dim=-1）。

#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <string>
#include <vector>

// 变体入口: x (M,H) 连续, out (M,H)（预分配，布局与 x 相同）。
using softmax_fn_t = void (*)(const at::Tensor& x, at::Tensor& out);

// ---- 主机端对齐 helper（与 rmsnorm 相同的策略 1: 显式 validation）----
// 向量化内核发出宽加载，要求基指针按对应字节数对齐。PyTorch 分配器的
// 普通分配满足 512B 对齐，但带 storage offset 的视图（切片、拼接等）
// 可能破坏对齐。向量化内核必须在 launch 前显式检查并清晰报错，
// 绝不静默执行未对齐加载。
inline bool ptr_aligned(const void* p, size_t bytes) {
    return (reinterpret_cast<std::uintptr_t>(p) & (bytes - 1)) == 0;
}

void register_softmax_variant(const std::string& name, softmax_fn_t fn);

at::Tensor softmax_forward(const std::string& name, const at::Tensor& x);

// 正常（可 dispatch）变体列表：不含被隔离变体。
// v0.3.1 起 softmax_hsplit2 被隔离（UNSAFE_HISTORICAL_EXPERIMENT /
// REJECTED / NOT_FOR_NORMAL_DISPATCH；隔离策略见 bindings.cpp 头部
// 注释，实验记录见 experiments/softmax/SFM-0004.md）。
std::vector<std::string> softmax_variant_list();

// 全部已注册变体（含被隔离者）；仅供显式历史审计入口使用。
std::vector<std::string> softmax_all_variant_list();

// 被隔离变体列表（仅实际已注册者）。
std::vector<std::string> softmax_quarantined_variant_list();

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
