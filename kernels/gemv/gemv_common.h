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

// ---- 16B 向量化变体的对齐契约（host 侧, launch 前检查）-----------------
// 任何 16B 向量 load 变体（8×__half 或 4×float）必须满足:
//   (1) W 基指针 16B 对齐;  (2) x 基指针 16B 对齐;
//   (3) K 可被 每 16B 的元素数（fp16: 8, fp32: 4）整除。
// 不满足时变体**必须回退标量路径**（launch_gemv_scalar）—— 合法输入
// （包括奇数元素 storage offset 的连续视图）不得被拒绝。
// `is_contiguous()==true` 不保证 (1)/(2)。out 是每行 1 个标量写,
// 无对齐要求（不检查）。
template <typename T>
inline bool gemv_vec_contract_ok(const at::Tensor& W, const at::Tensor& x) {
    constexpr int64_t epv = 16 / static_cast<int64_t>(sizeof(T));
    if (W.size(1) % epv != 0) return false;
    return (reinterpret_cast<uintptr_t>(W.const_data_ptr()) & 15u) == 0 &&
           (reinterpret_cast<uintptr_t>(x.const_data_ptr()) & 15u) == 0;
}

// ---- 共享设备端辅助函数（仅限 CUDA 翻译单元）---------------------------

#ifdef __CUDACC__

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

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

// ---- 16B 向量 load 辅助（GEMV-0001..0003 向量化变体共用）---------------
// 16B = 8 × __half 或 4 × float（epv = 16/sizeof(T)）。以 uint4 载入后
// 按元素解释 —— 标准 16B 打包 load（与 llama.cpp 的 float4-as-half8
// 同一手法, 但基线不抄任何成熟实现, 这里只共用"16B 对齐 load"这一
// 硬件原语）。
union U16 {
    uint4 v;
    __half h[8];
    float  f[4];
};

// 把 (w 向量, x 向量) 的 epv 个乘积累加进 acc（逐元素 FP32 FMA,
// 允许 nvcc 收缩）。模板分发 fp16 / fp32 的元素数组。
template <typename T>
__device__ __forceinline__ void vec_acc(const U16& w, const U16& xv,
                                        float& acc) {
    static_assert(sizeof(T) == 2 || sizeof(T) == 4, "only fp16/fp32");
    if (sizeof(T) == 2) {
        const __half* wh = w.h;
        const __half* xh = xv.h;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc += el_to_float(wh[j]) * el_to_float(xh[j]);
    } else {
        const float* wf = w.f;
        const float* xf = xv.f;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            acc += el_to_float(wf[j]) * el_to_float(xf[j]);
    }
}

// ---- 共享标量 kernel（GEMV-0000 计算本体 + 所有向量化变体的回退）-------
// **单一来源**: gemv_baseline 与全部向量化变体的标量回退都调用这里,
// 保证 "回退输出与 gemv_baseline 在同一输入上逐位一致"（negative
// 套件的 per-variant 回退回归用例以此为参照）。形态: 每输出行一个
// block, 256 线程步长 FP32 累加, 两级 warp 归约。
static constexpr int kGemvScalarBlock = 256;
static constexpr int kGemvScalarWarps = kGemvScalarBlock / 32;  // 8

template <typename T>
static __global__ void gemv_scalar_kernel(const T* __restrict__ W,
                                          const T* __restrict__ x,
                                          T* __restrict__ out,
                                          int64_t N, int64_t K) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const T* __restrict__ wrow = W + row * K;

    float acc = 0.f;
    for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
        acc += el_to_float(wrow[k]) * el_to_float(x[k]);
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    __shared__ float warp_sums[kGemvScalarWarps];
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        acc = (lane < kGemvScalarWarps) ? warp_sums[lane] : 0.f;
#pragma unroll
        for (int off = kGemvScalarWarps / 2; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) out[row] = el_from_float<T>(acc);
    }
}

template <typename T>
static inline void launch_gemv_scalar(const at::Tensor& W,
                                      const at::Tensor& x,
                                      at::Tensor& out) {
    const int64_t N = W.size(0);
    const int64_t K = W.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gemv_scalar_kernel<T><<<static_cast<int>(N), kGemvScalarBlock, 0,
                            stream>>>(
        reinterpret_cast<const T*>(W.const_data_ptr()),
        reinterpret_cast<const T*>(x.const_data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()),
        N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dtype 分发入口（向量化变体 host 侧回退调用）。
static inline void launch_gemv_scalar_dispatch(const at::Tensor& W,
                                               const at::Tensor& x,
                                               at::Tensor& out) {
    if (W.scalar_type() == at::kHalf) {
        launch_gemv_scalar<__half>(W, x, out);
    } else {
        launch_gemv_scalar<float>(W, x, out);
    }
}

#endif  // __CUDACC__
