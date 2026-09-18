// CUDALab RMSNorm — v4: vectorized + register-resident single pass.
//
// Combines the two validated directions:
//   - v1 (KEEP): 16B vectorized accesses cut instruction count (EXP-0004,
//     1.21x on primary target);
//   - v2 (NEUTRAL): register-resident x removes the second global read,
//     but v2 loaded with SCALAR 2B/4B accesses, which likely masked the
//     benefit (EXP-0005: 0.989x, 50 regs).
//
// Hypothesis: a 16B-aligned vectorized initial load that keeps x in
// registers gives both effects: minimum instruction count AND one global
// read of x.
//
// Memory layout: thread t owns the CONTIGUOUS slice [t*PER, (t+1)*PER) of
// the row (slice-major, not strided), so 16B vector loads are naturally
// aligned (PER is a multiple of 8 for fp16 / 4 for fp32) and a warp reads
// a contiguous 512B span (perfectly coalesced).
//
// Register budget: fp16 keeps x as __half2 (PER/2 regs); fp32 as float
// (PER regs). PER = H/256 in {4,8,16,32}.

#include "rmsnorm_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>

namespace {

constexpr int V4_BLOCK = 256;

// fp16 specialization ------------------------------------------------------
template <int PER>
__global__ void rmsnorm_v4_half_kernel(const __half* __restrict__ x,
                                       const __half* __restrict__ w,
                                       __half* __restrict__ y,
                                       int H, float eps) {
    static_assert(PER == 4 || PER == 8 || PER == 16 || PER == 32, "PER");
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const __half* __restrict__ xrow = x + (size_t)row * H;
    __half* __restrict__ yrow = y + (size_t)row * H;
    const int base = tid * PER;   // first element owned by this thread

    __half2 buf[PER / 2];
    float ss = 0.f;

    if (PER % 8 == 0) {
        // 8 contiguous halves = one 16B float4
        const float4* p = reinterpret_cast<const float4*>(xrow + base);
        const int nvec = PER / 8;
#pragma unroll
        for (int i = 0; i < nvec; i++) {
            float4 v = p[i];
            const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
            for (int k = 0; k < 4; k++) {
                buf[i * 4 + k] = h[k];
                float2 f = __half22float2(h[k]);
                ss += f.x * f.x + f.y * f.y;
            }
        }
    } else {
        // PER == 4: two 4B half2 loads
        const __half2* p = reinterpret_cast<const __half2*>(xrow + base);
#pragma unroll
        for (int i = 0; i < PER / 2; i++) {
            __half2 h = p[i];
            buf[i] = h;
            float2 f = __half22float2(h);
            ss += f.x * f.x + f.y * f.y;
        }
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

    if (PER % 8 == 0) {
        const int nvec = PER / 8;
        const float4* wp = reinterpret_cast<const float4*>(w + base);
        float4* q = reinterpret_cast<float4*>(yrow + base);
#pragma unroll
        for (int i = 0; i < nvec; i++) {
            float4 wv = wp[i];
            const __half2* hw = reinterpret_cast<const __half2*>(&wv);
            float4 o;
            __half2* ho = reinterpret_cast<__half2*>(&o);
#pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 f = __half22float2(buf[i * 4 + k]);
                float2 fw = __half22float2(hw[k]);
                ho[k] = __floats2half2_rn(f.x * inv_rms * fw.x,
                                          f.y * inv_rms * fw.y);
            }
            q[i] = o;
        }
    } else {
        const __half2* wp = reinterpret_cast<const __half2*>(w + base);
        __half2* q = reinterpret_cast<__half2*>(yrow + base);
#pragma unroll
        for (int i = 0; i < PER / 2; i++) {
            float2 f = __half22float2(buf[i]);
            float2 fw = __half22float2(wp[i]);
            q[i] = __floats2half2_rn(f.x * inv_rms * fw.x,
                                                f.y * inv_rms * fw.y);
        }
    }
}

// fp32 specialization -------------------------------------------------------
template <int PER>
__global__ void rmsnorm_v4_float_kernel(const float* __restrict__ x,
                                        const float* __restrict__ w,
                                        float* __restrict__ y,
                                        int H, float eps) {
    static_assert(PER == 4 || PER == 8 || PER == 16 || PER == 32, "PER");
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const float* __restrict__ xrow = x + (size_t)row * H;
    float* __restrict__ yrow = y + (size_t)row * H;
    const int base = tid * PER;

    float buf[PER];
    float ss = 0.f;
    {
        const float4* p = reinterpret_cast<const float4*>(xrow + base);
        const int nvec = PER / 4;
#pragma unroll
        for (int i = 0; i < nvec; i++) {
            float4 v = p[i];
            buf[i * 4] = v.x; buf[i * 4 + 1] = v.y;
            buf[i * 4 + 2] = v.z; buf[i * 4 + 3] = v.w;
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
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

    {
        const int nvec = PER / 4;
        const float4* wp = reinterpret_cast<const float4*>(w + base);
        float4* q = reinterpret_cast<float4*>(yrow + base);
#pragma unroll
        for (int i = 0; i < nvec; i++) {
            float4 wv = wp[i];
            q[i] = make_float4(buf[i * 4] * inv_rms * wv.x,
                                          buf[i * 4 + 1] * inv_rms * wv.y,
                                          buf[i * 4 + 2] * inv_rms * wv.z,
                                          buf[i * 4 + 3] * inv_rms * wv.w);
        }
    }
}

void launch_half(const at::Tensor& x, const at::Tensor& w, at::Tensor& out,
                 double eps) {
    const int M = x.size(0);
    const int H = x.size(1);
    const int per = H / V4_BLOCK;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr());
    const __half* wp = reinterpret_cast<const __half*>(w.data_ptr());
    __half* yp = reinterpret_cast<__half*>(out.data_ptr());
    dim3 grid(M), block(V4_BLOCK);
    switch (per) {
        case 4:  rmsnorm_v4_half_kernel<4><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); return;
        case 8:  rmsnorm_v4_half_kernel<8><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); return;
        case 16: rmsnorm_v4_half_kernel<16><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); return;
        case 32: rmsnorm_v4_half_kernel<32><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); return;
        default:
            TORCH_CHECK(false, "v4 requires H/256 in {4,8,16,32}; got H=", H);
    }
}

void launch_float(const at::Tensor& x, const at::Tensor& w, at::Tensor& out,
                  double eps) {
    const int M = x.size(0);
    const int H = x.size(1);
    const int per = H / V4_BLOCK;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* xp = reinterpret_cast<const float*>(x.data_ptr());
    const float* wp = reinterpret_cast<const float*>(w.data_ptr());
    float* yp = reinterpret_cast<float*>(out.data_ptr());
    dim3 grid(M), block(V4_BLOCK);
    switch (per) {
        case 4:  rmsnorm_v4_float_kernel<4><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); return;
        case 8:  rmsnorm_v4_float_kernel<8><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); return;
        case 16: rmsnorm_v4_float_kernel<16><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); return;
        case 32: rmsnorm_v4_float_kernel<32><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); return;
        default:
            TORCH_CHECK(false, "v4 requires H/256 in {4,8,16,32}; got H=", H);
    }
}

void rmsnorm_v4_fwd(const at::Tensor& x, const at::Tensor& w,
                    at::Tensor& out, double eps) {
    c10::cuda::CUDAGuard guard(x.device());
    switch (x.scalar_type()) {
        case at::kHalf:
            launch_half(x, w, out, eps);
            break;
        case at::kFloat:
            launch_float(x, w, out, eps);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype for rmsnorm v4");
    }
}

}  // namespace

static struct RmsnormV4Registrar {
    RmsnormV4Registrar() { register_rmsnorm_variant("v4_vec_reg", rmsnorm_v4_fwd); }
} rmsnorm_v4_registrar;
