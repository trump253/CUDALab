// CUDALab RMSNorm — v2: single-pass, register-resident x.
//
// Hypothesis (from baseline ncu profile): 80% of stall cycles are
// long_scoreboard (global memory latency) and the kernel reads x TWICE
// (pass 1 sum-of-squares, pass 2 normalize). With H/256 <= 32 elements per
// thread, the whole row slice fits in registers (<=16 half2 / <=32 floats).
// Loading x once into registers and reusing it for the output pass removes
// the second global read entirely: less DRAM traffic, and the output pass
// becomes pure register work with no memory dependency after the reduction.
//
// Weights are still streamed from L1/L2 (H*2 bytes, shared across rows).
//
// Requires H/256 (elements/thread) to be in {2,4,8,16,32}, i.e. for the
// 256-thread block: H in {512, 1024, 2048, 4096, 8192}.
// fp16: loads/stores as half2 (4B); fp32: as float4 (16B).

#include "rmsnorm_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>

namespace {

constexpr int V2_BLOCK = 256;

template <typename T, int PER>
__global__ void rmsnorm_v2_kernel(const T* __restrict__ x,
                                  const T* __restrict__ w,
                                  T* __restrict__ y,
                                  int H, float eps) {
    static_assert(PER == 2 || PER == 4 || PER == 8 || PER == 16 || PER == 32,
                  "unsupported PER");
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const T* __restrict__ xrow = x + (size_t)row * H;
    T* __restrict__ yrow = y + (size_t)row * H;

    // ---- load x once into registers, accumulate ss in FP32 ----
    float xf[PER];
    float ss = 0.f;
    {
        const T* p = xrow + tid;
#pragma unroll
        for (int i = 0; i < PER; i++)
            xf[i] = el_to_float(p[i * nthreads]);
#pragma unroll
        for (int i = 0; i < PER; i++)
            ss += xf[i] * xf[i];
    }

    // ---- block reduction (same as baseline) ----
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

    // ---- output pass: pure register x + streamed weight ----
    {
        T* q = yrow + tid;
        const T* wp = w + tid;
#pragma unroll
        for (int i = 0; i < PER; i++) {
            float wv = el_to_float(wp[i * nthreads]);
            q[i * nthreads] = el_from_float<T>(xf[i] * inv_rms * wv);
        }
    }
}

template <typename T>
void launch(const at::Tensor& x, const at::Tensor& w, at::Tensor& out,
            double eps) {
    const int M = x.size(0);
    const int H = x.size(1);
    const int per = H / V2_BLOCK;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const T* xp = reinterpret_cast<const T*>(x.data_ptr());
    const T* wp = reinterpret_cast<const T*>(w.data_ptr());
    T* yp = reinterpret_cast<T*>(out.data_ptr());
    dim3 grid(M), block(V2_BLOCK);
    switch (per) {
        case 2:  rmsnorm_v2_kernel<T, 2><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); return;
        case 4:  rmsnorm_v2_kernel<T, 4><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); return;
        case 8:  rmsnorm_v2_kernel<T, 8><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); return;
        case 16: rmsnorm_v2_kernel<T, 16><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); return;
        case 32: rmsnorm_v2_kernel<T, 32><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); return;
        default:
            TORCH_CHECK(false,
                        "v2 requires H/256 in {2,4,8,16,32}; got H=", H);
    }
}

void rmsnorm_v2_fwd(const at::Tensor& x, const at::Tensor& w,
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
            TORCH_CHECK(false, "unsupported dtype for rmsnorm v2");
    }
}

}  // namespace

static struct RmsnormV2Registrar {
    RmsnormV2Registrar() { register_rmsnorm_variant("v2_reg", rmsnorm_v2_fwd); }
} rmsnorm_v2_registrar;
