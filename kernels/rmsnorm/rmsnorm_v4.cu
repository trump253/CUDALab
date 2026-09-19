// CUDALab RMSNorm — v4: 向量化 + 寄存器驻留单遍。
//
// 结合两个已验证的方向:
//   - v1（KEEP）: 16B 向量化访存减少指令数（EXP-0004，主目标 1.21x）；
//   - v2（NEUTRAL）: x 寄存器驻留消除第二次全局读取，但 v2 用的是
//     标量 2B/4B 访问，可能掩盖了收益（EXP-0005: 0.989x，50 寄存器）。
//
// 假设: 用 16B 对齐的向量化初始加载把 x 装入寄存器，可同时获得
// 两个效果: 最少指令数 + 只读一次 x。
//
// 内存布局: 线程 t 拥有行内连续切片 [t*PER, (t+1)*PER)（切片主序，
// 非跨步），因此 16B 向量加载天然对齐（fp16 时 PER 是 8 的倍数 /
// fp32 时是 4 的倍数），一个 warp 读取连续 512B 跨度（完美合并访存）。
//
// 寄存器预算: fp16 以 __half2 保存 x（PER/2 个寄存器）；fp32 以
// float 保存（PER 个寄存器）。PER = H/256 ∈ {4,8,16,32}。

#include "rmsnorm_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>

namespace {

constexpr int V4_BLOCK = 256;

// fp16 特化 --------------------------------------------------------------
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
    const int base = tid * PER;   // 本线程拥有的首个元素

    __half2 buf[PER / 2];
    float ss = 0.f;

    if (PER % 8 == 0) {
        // 8 个连续 half = 一个 16B float4
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
        // PER == 4: 两个 4B half2 加载
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

// fp32 特化 ---------------------------------------------------------------
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

// v0.2 公共 launch 前置检查（Finding A + D）:
// - 整除性: H % 256 == 0（旧代码直接取整除商，H=4100 会误入 PER=16
//   而只覆盖 4096 个元素，产生静默错误输出）；
// - 对齐契约: float4 路径（PER % 8 == 0）要求 x/w/out 基指针 16B 对齐；
//   half2 路径（PER == 4）要求 4B 对齐。行步长 = H * 2 字节，在上述
//   PER 取值下都是检查值的整数倍，因此基指针对齐即全部行首对齐。
//   PyTorch 分配器普通分配满足 512B 对齐；storage offset 视图可能
//   破坏（策略 1: 显式报错，不静默执行未对齐加载）。
void v4_precheck(const at::Tensor& x, const at::Tensor& w, at::Tensor& out,
                 int H) {
    const int per = H / V4_BLOCK;
    TORCH_CHECK(H % V4_BLOCK == 0,
                "v4 要求 H % 256 == 0；实际 H=", H);
    const size_t align = (per % 8 == 0) ? 16 : 4;
    TORCH_CHECK(ptr_aligned(x.data_ptr(), align) &&
                ptr_aligned(w.data_ptr(), align) &&
                ptr_aligned(out.data_ptr(), align),
                "v4 对齐契约不满足: 需要 x/w/out 基指针 ", align,
                "B 对齐（H=", H, "）；普通分配满足，storage offset "
                "视图可能破坏对齐");
}

void launch_half(const at::Tensor& x, const at::Tensor& w, at::Tensor& out,
                 double eps) {
    const int M = x.size(0);
    const int H = x.size(1);
    v4_precheck(x, w, out, H);
    const int per = H / V4_BLOCK;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr());
    const __half* wp = reinterpret_cast<const __half*>(w.data_ptr());
    __half* yp = reinterpret_cast<__half*>(out.data_ptr());
    dim3 grid(M), block(V4_BLOCK);
    switch (per) {
        case 4:  rmsnorm_v4_half_kernel<4><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); C10_CUDA_KERNEL_LAUNCH_CHECK(); return;
        case 8:  rmsnorm_v4_half_kernel<8><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); C10_CUDA_KERNEL_LAUNCH_CHECK(); return;
        case 16: rmsnorm_v4_half_kernel<16><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); C10_CUDA_KERNEL_LAUNCH_CHECK(); return;
        case 32: rmsnorm_v4_half_kernel<32><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); C10_CUDA_KERNEL_LAUNCH_CHECK(); return;
        default:
            TORCH_CHECK(false, "v4 要求 H/256 ∈ {4,8,16,32}（即 H ∈ "
                        "{1024,2048,4096,8192}）；实际 H=", H);
    }
}

void launch_float(const at::Tensor& x, const at::Tensor& w, at::Tensor& out,
                  double eps) {
    const int M = x.size(0);
    const int H = x.size(1);
    v4_precheck(x, w, out, H);
    const int per = H / V4_BLOCK;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* xp = reinterpret_cast<const float*>(x.data_ptr());
    const float* wp = reinterpret_cast<const float*>(w.data_ptr());
    float* yp = reinterpret_cast<float*>(out.data_ptr());
    dim3 grid(M), block(V4_BLOCK);
    switch (per) {
        case 4:  rmsnorm_v4_float_kernel<4><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); C10_CUDA_KERNEL_LAUNCH_CHECK(); return;
        case 8:  rmsnorm_v4_float_kernel<8><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); C10_CUDA_KERNEL_LAUNCH_CHECK(); return;
        case 16: rmsnorm_v4_float_kernel<16><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); C10_CUDA_KERNEL_LAUNCH_CHECK(); return;
        case 32: rmsnorm_v4_float_kernel<32><<<grid, block, 0, stream>>>(xp, wp, yp, H, (float)eps); C10_CUDA_KERNEL_LAUNCH_CHECK(); return;
        default:
            TORCH_CHECK(false, "v4 要求 H/256 ∈ {4,8,16,32}（即 H ∈ "
                        "{1024,2048,4096,8192}）；实际 H=", H);
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
            TORCH_CHECK(false, "rmsnorm v4 不支持该 dtype");
    }
}

}  // namespace

static struct RmsnormV4Registrar {
    RmsnormV4Registrar() { register_rmsnorm_variant("v4_vec_reg", rmsnorm_v4_fwd); }
} rmsnorm_v4_registrar;
