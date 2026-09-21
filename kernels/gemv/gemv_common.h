// CUDALab GEMV — 公共变体注册表 + 设备端辅助函数。
//
// GEMV（广义矩阵-向量乘）算子定义（v0.5 主路径: FP16 输入/输出,
// FP32 累加; FP32 dtype 同样支持）:
//
//   W  (N, K)   连续, row-major; N = 输出行数, K = 归约维
//   x  (K,)     连续
//   y  (N,)     连续（预分配, 输出 dtype = W dtype）
//
//   对每行 n:  y[n] = ( Σ_k W[n,k] · x[k]  在 FP32 中累加 )
//              最后 cast 到输出 dtype。
//
// 参考实现（用户指定, 与内核隔离验证）:
//   fp16:  torch.mv(W.float(), x.float()).to(torch.float16)
//   fp32:  torch.mv(W, x)
//
// 每个内核变体位于独立的 .cu 文件中，并通过静态注册器自注册:
//
//   static void my_fwd(const at::Tensor& W, const at::Tensor& x,
//                      at::Tensor& out) { ... }
//   static struct MyRegistrar {
//     MyRegistrar() { register_gemv_variant("my_variant", my_fwd); }
//   } my_registrar;
//
// 内核以原生 CUDA 元素类型（`__half`（fp16）或 `float`（fp32））为
// 模板参数。启动器传原始数据指针，不依赖 c10::Half 与 __half 之间的
// 头文件互操作。
//
// 所有输入验证（TORCH_CHECK）都在 kernel launch 之前完成（bindings.cpp
// 的 validate_gemv_inputs）; 内核自身不含设备端断言。
//
// GEMV 的验证与 RoPE 不同: **没有任何数据依赖检查**（无 positions 值域
// 之类的 D2H 同步需求）, 全部是 host 侧元数据检查（dim/shape/dtype/
// 连续/设备）, 因此逐 launch 开销可忽略, 不需要 RoPE 式的
// validate=false 基准池开关 —— forward_into 每次调用都执行完整验证。
//
// 对齐契约总则（v0.5, 沿用 v0.4.1 RoPE 教训）: **任何向量化
// load/store 变体（__half2 / 4×__half / float4 等）必须对其输入基
// 指针声明明确的对齐契约, 并在 host 侧 launch 前检查; 不满足时回退
// 标量路径（合法输入不得被拒）或给出 launch 前的明确 TORCH_CHECK。
// `is_contiguous()==true` **不保证** 向量化对齐 —— 奇数元素 storage
// offset 的连续视图（如 big[1:].view(N,K)）基指针偏移 2B。
// gemv_baseline 是纯标量访存（每 thread 2B/4B 元素 load）, **无对齐
// 契约**, 任何连续合法输入都必须成功。

#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <string>
#include <vector>

// 变体入口: W (N,K) 连续 / x (K,) 连续; out (N,)（预分配, dtype = W）。
using gemv_fn_t = void (*)(const at::Tensor& W,
                           const at::Tensor& x,
                           at::Tensor& out);

void register_gemv_variant(const std::string& name, gemv_fn_t fn);

// 注: gemv_forward / gemv_forward_into / gemv_native_timing 的入口声明
// 只存在于 bindings.cpp（pybind 入口）, 不在公共头里——头里的旧签名
// 前向声明会与 bindings.cpp 的新签名构成重载, 使取址时模板推导失败
// （与 rope_common.h 相同的教训）。

std::vector<std::string> gemv_variant_list();
std::vector<std::string> gemv_all_variant_list();

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
