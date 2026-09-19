// CUDALab RMSNorm — v1: 向量化加载/存储。
//
// 假设（来自 baseline 的 ncu 剖析）: baseline 是访存延迟瓶颈
// （80% long_scoreboard 停顿，DRAM 仅 ~14%），且发出 2 字节标量
// 加载（每个 warp 指令 64B = 4 个 sector 中的 2 个）。每线程
// 加载/存储 16 字节（float4 装 8 个 fp16）把两遍的指令数减半，
// 并让每个 warp 加载 512B（16 个完整 sector），提高每条已发射
// 指令的访存级并行度。
//
// 仍是两遍（第二遍重读 x）、同样 256 线程 block、同样的归约。
// 只改访存宽度，以隔离其效果。
//
// 要求: fp16 需 H % 8 == 0（fp32 需 H % 4 == 0）。

#include "rmsnorm_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>

namespace {

constexpr int V1_BLOCK = 256;

// 向量辅助 --------------------------------------------------------------
template <typename T> struct Vec;

template <>
struct Vec<__half> {
    using V = float4;
    static constexpr int N = 8;  // 8 x fp16 = 16 B
    __device__ static float sqsum_and_unpack(V v, float* out) {
        const __half2* h = reinterpret_cast<const __half2*>(&v);
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 f = __half22float2(h[i]);
            out[2 * i] = f.x;
            out[2 * i + 1] = f.y;
            s += f.x * f.x + f.y * f.y;
        }
        return s;
    }
    __device__ static V pack(const float* f) {
        V r;
        __half2* h = reinterpret_cast<__half2*>(&r);
#pragma unroll
        for (int i = 0; i < 4; i++)
            h[i] = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
        return r;
    }
};

template <>
struct Vec<float> {
    using V = float4;
    static constexpr int N = 4;
    __device__ static float sqsum_and_unpack(V v, float* out) {
        out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
        return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    __device__ static V pack(const float* f) {
        return make_float4(f[0], f[1], f[2], f[3]);
    }
};

template <typename T>
__global__ void rmsnorm_v1_kernel(const T* __restrict__ x,
                                  const T* __restrict__ w,
                                  T* __restrict__ y,
                                  int Hv, float eps) {
    using VecT = Vec<T>;
    using V = typename VecT::V;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const V* __restrict__ xv = reinterpret_cast<const V*>(x + (size_t)row * Hv * VecT::N);
    const V* __restrict__ wv = reinterpret_cast<const V*>(w);
    V* __restrict__ yv = reinterpret_cast<V*>(y + (size_t)row * Hv * VecT::N);

    // 第一遍: 平方和（FP32），向量化
    float ss = 0.f;
    for (int i = tid; i < Hv; i += nthreads) {
        float f[VecT::N];
        ss += VecT::sqsum_and_unpack(xv[i], f);
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
        if (tid == 0) s_inv_rms = rsqrtf(v / ((float)Hv * VecT::N) + eps);
    }
    __syncthreads();
    const float inv_rms = s_inv_rms;

    // 第二遍: 归一化 + 乘权重，向量化
    for (int i = tid; i < Hv; i += nthreads) {
        float f[VecT::N];
        float wv_f[VecT::N];
        VecT::sqsum_and_unpack(xv[i], f);      // 复用解包（忽略求和结果）
        VecT::sqsum_and_unpack(wv[i], wv_f);
#pragma unroll
        for (int k = 0; k < VecT::N; k++)
            f[k] *= inv_rms * wv_f[k];
        yv[i] = VecT::pack(f);
    }
}

template <typename T>
void launch(const at::Tensor& x, const at::Tensor& w, at::Tensor& out,
            double eps) {
    const int M = x.size(0);
    const int H = x.size(1);
    using VecT = Vec<T>;
    TORCH_CHECK(H % VecT::N == 0, "v1 要求 H % ", VecT::N, " == 0");
    const int Hv = H / VecT::N;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_v1_kernel<T><<<dim3(M), V1_BLOCK, 0, stream>>>(
        reinterpret_cast<const T*>(x.data_ptr()),
        reinterpret_cast<const T*>(w.data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()), Hv, (float)eps);
}

void rmsnorm_v1_fwd(const at::Tensor& x, const at::Tensor& w,
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
            TORCH_CHECK(false, "rmsnorm v1 不支持该 dtype");
    }
}

}  // namespace

static struct RmsnormV1Registrar {
    RmsnormV1Registrar() { register_rmsnorm_variant("v1_vec", rmsnorm_v1_fwd); }
} rmsnorm_v1_registrar;
