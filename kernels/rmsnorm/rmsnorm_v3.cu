// CUDALab RMSNorm — v3: 更宽的 block（512 线程），标量访问。
//
// 假设（待证伪）: baseline 的 256 线程 block 每个 SM-block 只有 ~8 个
// warp；把 block 加大到 512 线程可使驻留 warp 翻倍，从而在归约遍内
// 更好地隐藏全局访存延迟。其他一切不变（仍是标量加载、两遍），
// 以隔离块大小的效果。
//
// 预期风险: block 级归约屏障现在跨越 16 个 warp，且总网格并行度
// （M 个 block）不变 —— 对 REJECT/NEUTRAL 判定分支的一次真实检验。
//
// 要求 H % 512 == 0。

#include "rmsnorm_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>

namespace {

constexpr int V3_BLOCK = 512;

template <typename T>
__global__ void rmsnorm_v3_kernel(const T* __restrict__ x,
                                  const T* __restrict__ w,
                                  T* __restrict__ y,
                                  int H, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const T* __restrict__ xrow = x + (size_t)row * H;
    T* __restrict__ yrow = y + (size_t)row * H;

    float ss = 0.f;
    for (int i = tid; i < H; i += nthreads) {
        float v = el_to_float(xrow[i]);
        ss += v * v;
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        ss += __shfl_down_sync(0xffffffffu, ss, offset);

    const int nwarp = (nthreads + 31) >> 5;
    __shared__ float warp_sums[32];
    __shared__ float s_inv_rms;
    if ((tid & 31) == 0) warp_sums[tid >> 5] = ss;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < nwarp) ? warp_sums[tid] : 0.f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, offset);
        if (tid == 0) s_inv_rms = rsqrtf(v / (float)H + eps);
    }
    __syncthreads();
    const float inv_rms = s_inv_rms;

    for (int i = tid; i < H; i += nthreads) {
        float v = el_to_float(xrow[i]);
        float wv = el_to_float(w[i]);
        yrow[i] = el_from_float<T>(v * inv_rms * wv);
    }
}

template <typename T>
void launch(const at::Tensor& x, const at::Tensor& w, at::Tensor& out,
            double eps) {
    const int M = x.size(0);
    const int H = x.size(1);
    TORCH_CHECK(H % V3_BLOCK == 0, "v3 要求 H % 512 == 0；实际 H=", H);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_v3_kernel<T><<<dim3(M), V3_BLOCK, 0, stream>>>(
        reinterpret_cast<const T*>(x.data_ptr()),
        reinterpret_cast<const T*>(w.data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()), H, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rmsnorm_v3_fwd(const at::Tensor& x, const at::Tensor& w,
                    at::Tensor& out, double eps) {
    c10::cuda::CUDAGuard guard(x.device());
    switch (x.scalar_type()) {
        case at::kHalf:
            launch<__half>(x, w, out, eps);
            break;
        case at::kFloat:
            launch<float>(x, w, out, eps);
            break;
        default:
            TORCH_CHECK(false, "rmsnorm v3 不支持该 dtype");
    }
}

}  // namespace

static struct RmsnormV3Registrar {
    RmsnormV3Registrar() { register_rmsnorm_variant("v3_wideblock", rmsnorm_v3_fwd); }
} rmsnorm_v3_registrar;
