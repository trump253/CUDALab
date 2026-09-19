// CUDALab RMSNorm — baseline 变体。
//
// 策略: 每行一个 CUDA block；所有线程协作归约 sum(x^2)（FP32 累加，
// warp shuffle + 一步共享内存），然后第二遍把每个元素重算为
//   y_i = x_i * rsqrt(ss/H + eps) * w_i。
//
// 刻意保持简单易读: 标量（非向量化）加载、固定 256 线程 block、
// 两遍之间无寄存器驻留。这是所有优化实验的参考点。

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

    // ---- 第一遍: 平方和，FP32 累加 ----
    float ss = 0.f;
    for (int i = tid; i < H; i += nthreads) {
        float v = el_to_float(xrow[i]);
        ss += v * v;
    }

    // warp 内归约
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        ss += __shfl_down_sync(0xffffffffu, ss, offset);

    const int nwarp = (nthreads + 31) >> 5;
    __shared__ float warp_sums[32];   // 最多 32 个 warp（block <= 1024）
    __shared__ float s_inv_rms;
    if ((tid & 31) == 0) warp_sums[tid >> 5] = ss;
    __syncthreads();

    // warp 0 做跨 warp 归约
    if (tid < 32) {
        float v = (tid < nwarp) ? warp_sums[tid] : 0.f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, offset);
        if (tid == 0) s_inv_rms = rsqrtf(v / (float)H + eps);
    }
    __syncthreads();
    const float inv_rms = s_inv_rms;

    // ---- 第二遍: 归一化 + 乘权重 ----
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
            TORCH_CHECK(false, "rmsnorm baseline 不支持该 dtype");
    }
}

}  // namespace

static struct RmsnormBaselineRegistrar {
    RmsnormBaselineRegistrar() {
        register_rmsnorm_variant("baseline", rmsnorm_baseline_fwd);
    }
} rmsnorm_baseline_registrar;
