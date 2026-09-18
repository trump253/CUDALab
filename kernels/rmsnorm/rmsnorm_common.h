// CUDALab RMSNorm — common variant registry + device helpers.
//
// Each kernel variant lives in its own .cu file and registers itself via a
// static registrar. Adding a new variant never requires touching bindings.cpp.
//
//   static void my_fwd(const at::Tensor& x, const at::Tensor& w,
//                      at::Tensor& out, double eps) { ... }
//   static struct MyRegistrar {
//     MyRegistrar() { register_rmsnorm_variant("my_variant", my_fwd); }
//   } my_registrar;
//
// Kernels are templated on a native CUDA element type: `__half` (fp16) or
// `float` (fp32). Launchers pass raw data pointers (void* / __half* / float*)
// so we never depend on c10::Half <-> __half header interop.

#pragma once
#include <torch/extension.h>
#include <string>
#include <vector>

// variant entry point: x (M,H) contiguous, w (H,) contiguous, out (M,H)
// (pre-allocated, same layout as x), eps.
using rmsnorm_fn_t = void (*)(const at::Tensor& x, const at::Tensor& w,
                              at::Tensor& out, double eps);

void register_rmsnorm_variant(const std::string& name, rmsnorm_fn_t fn);

at::Tensor rmsnorm_forward(const std::string& name, const at::Tensor& x,
                           const at::Tensor& w, double eps);

std::vector<std::string> rmsnorm_variant_list();

// ---- shared device helpers (CUDA translation units only) -------------------

#ifdef __CUDACC__

__device__ __forceinline__ float el_to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float el_to_float(float v) { return v; }

template <typename T>
__device__ __forceinline__ T el_from_float(float v) {
    return T(v);  // float identity
}

// Explicit specialization: torch builds with -D__CUDA_NO_HALF_CONVERSIONS__,
// so use the intrinsic instead of the (disabled) __half(float) constructor.
template <>
__device__ __forceinline__ __half el_from_float<__half>(float v) {
    return __float2half_rn(v);
}

#endif  // __CUDACC__
