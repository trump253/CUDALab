// CUDALab RMSNorm — v2: 单遍、x 寄存器驻留。
//
// 假设（来自 baseline 的 ncu 剖析）: 80% 的停顿周期是 long_scoreboard
// （全局访存延迟），且内核读取了 x 两次（第一遍平方和、第二遍归一化）。
// 当 H/256 <= 32 时，每线程切片可完整装入寄存器（<=16 个 half2 /
// <=32 个 float）。把 x 一次性读进寄存器并在输出遍复用，完全消除第二
// 次全局读取: DRAM 流量更少，且归约之后输出遍是纯寄存器操作，没有
// 访存依赖。
//
// 权重仍从 L1/L2 流式读取（H*2 字节，跨行共享）。
//
// 要求 H/256（每线程元素数）∈ {2,4,8,16,32}，即对 256 线程 block:
// H ∈ {512, 1024, 2048, 4096, 8192}。
// 注意: 加载/存储刻意采用标量（逐元素 2B fp16 / 4B fp32）访问
// （步长 nthreads），以便把寄存器驻留的效果与向量化隔离开来。
// （见 EXP-0005: NEUTRAL —— 标量访问掩盖了单遍的收益；v4 把两者结合。）

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
                  "不支持的 PER");
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const T* __restrict__ xrow = x + (size_t)row * H;
    T* __restrict__ yrow = y + (size_t)row * H;

    // ---- 把 x 一次性读入寄存器，FP32 累加 ss ----
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

    // ---- block 归约（与 baseline 相同）----
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

    // ---- 输出遍: 纯寄存器 x + 流式权重 ----
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
                        "v2 要求 H/256 ∈ {2,4,8,16,32}；实际 H=", H);
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
            TORCH_CHECK(false, "rmsnorm v2 不支持该 dtype");
    }
}

}  // namespace

static struct RmsnormV2Registrar {
    RmsnormV2Registrar() { register_rmsnorm_variant("v2_reg", rmsnorm_v2_fwd); }
} rmsnorm_v2_registrar;
