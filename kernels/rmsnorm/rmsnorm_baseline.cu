// CUDALab RMSNorm — baseline variant.
//
// Strategy: one CUDA block per row; all threads cooperatively reduce
// sum(x^2) with FP32 accumulation (warp shuffle + one shared-memory
// step), then a second pass recomputes each element as
//   y_i = x_i * rsqrt(ss/H + eps) * w_i.
//
// Deliberately simple and readable: scalar (non-vectorized) loads,
// fixed 256-thread blocks, no register residency across passes.
// This is the reference point for all optimization experiments.

#include "rmsnorm_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>

namespace {

constexpr int BASELINE_BLOCK = 256;

template <typename T>
__global__ void rmsnorm_baseline_kernel(const T* __restrict__ x,
                                        const T* __restrict__ w,
                                        T* __restrict__ y,
                                        int H, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const T* __restrict__ xrow = x + (size_t)row * H;
    T* __restrict__ yrow = y + (size_t)row * H;

    // ---- pass 1: sum of squares, FP32 accumulation ----
    float ss = 0.f;
    for (int i = tid; i < H; i += nthreads) {
        float v = el_to_float(xrow[i]);
        ss += v * v;
    }

    // intra-warp reduction
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        ss += __shfl_down_sync(0xffffffffu, ss, offset);

    const int nwarp = (nthreads + 31) >> 5;
    __shared__ float warp_sums[32];   // max 32 warps (block <= 1024)
    __shared__ float s_inv_rms;
    if ((tid & 31) == 0) warp_sums[tid >> 5] = ss;
    __syncthreads();

    // cross-warp reduction in warp 0
    if (tid < 32) {
        float v = (tid < nwarp) ? warp_sums[tid] : 0.f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, offset);
        if (tid == 0) s_inv_rms = rsqrtf(v / (float)H + eps);
    }
    __syncthreads();
    const float inv_rms = s_inv_rms;

    // ---- pass 2: normalize + scale by weight ----
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(M), block(BASELINE_BLOCK);
    rmsnorm_baseline_kernel<T><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(x.data_ptr()),
        reinterpret_cast<const T*>(w.data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()), H, (float)eps);
}

void rmsnorm_baseline_fwd(const at::Tensor& x, const at::Tensor& w,
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
            TORCH_CHECK(false, "unsupported dtype for rmsnorm baseline");
    }
}

}  // namespace

static struct RmsnormBaselineRegistrar {
    RmsnormBaselineRegistrar() {
        register_rmsnorm_variant("baseline", rmsnorm_baseline_fwd);
    }
} rmsnorm_baseline_registrar;
