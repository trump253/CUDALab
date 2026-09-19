// CUDALab Softmax — SFM-0004: H 对半分, 每行 2 个 block, (m,l) 跨块合并。
//
// 动机（experiments/softmax/SFM-0004.md）:
//   前三个正交维度已测 —— 交易宽度（vec4, KEEP）, 流量（online, NEUTRAL）,
//   per-thread ILP（ilp2, NEUTRAL）。剩余未测维度是 block 级并行度 /
//   occupancy: vec4 在 (128,4096) 上 128 块 / 30 SM ≈ 4.27 块/SM（上限
//   8 块/SM）, 实测 occupancy 44.5%, 每个 SM 有一半 warp 槽位空闲。
//   本变体把每行拆成两个半行 block: 块数 128 -> 256（≈8.5 块/SM,
//   接近满 occupancy）, 每块串行链减半。
//
// 跨块合并算法（推导与 CPU 恒等门禁见 docs/softmax_algorithm.md）:
//   每个半行 block 在线累积自己段的 (m, l) = (段内 max, Σexp(x−m)),
//   block 内用 merge 恒等归约, 然后把半行 (m,l) 写入 scratch 并用
//   atomicAdd + spin-wait 与另一半会合:
//       (m,l) ⊕ (m',l') = (M, l·exp(m−M) + l'·exp(m'−M)), M = max(m,m')
//   两个 block 各自独立地从同一对分块值计算 (M, L) —— 运算顺序完全
//   相同 ⇒ 结果逐位一致, 不需要 ready flag, 不需要内核内重置。
//   合并后各半行 block 重读本半行写 y = exp(x−M)/L（归一化必须用整行
//   的 M, L —— 这正是不能简单"每半行各自 softmax"的原因）。
//
// 同步 / 活性:
//   grid = (2, M): blockIdx.x = half, blockIdx.y = row。线性块序号
//   x + y*2, 因此同一行的两个半块序号相邻（2r, 2r+1）。GPU 按线性
//   序号分波次调度; 波次容量 = SM 数 × 每 SM 块上限（256 线程块 →
//   2048/256 = 8 块/SM, 30 SM → 240, 偶数）。相邻对不会跨波次,
//   因此 spin-wait 的另一半必然与自己在同一波次内 ⇒ 无死锁。
//   主机端启动前用设备属性校验波次容量为偶数, 否则回退 vec4。
//   释放 / 获取遵循标准 fence 模式: 写分块值 -> __threadfence() ->
//   atomicAdd(cnt); 轮询 volatile cnt 直到 >= 2 且为偶数 ->
//   __threadfence() -> 读分块值（sm_75, PTX fence.sc.gpu 语义）。
//   偶数判定使 NCU kernel-replay（计数器跨回放累积）仍然正确。
//
// scratch:
//   m_part[2M] + l_part[2M] + cnt[M]（int32）= 5M float。
//   进程级静态分配一次（最大 M = 8192 → 163.8KB）, 每次启动前
//   cudaMemsetAsync 只清 cnt（m_part/l_part 在读取前必然已被写,
//   因为读发生在 cnt==2 之后）。memset 在计时流上, 属于 launch 路径
//   的一部分 —— 基准计时如实包含它（几 KB 的 memset, 与 kernel
//   异步流水）。计时区域内无 cudaMalloc。
//
// 访存宽度: 半行内 4 元素（与 vec4 相同的 8B/16B 事务）。
// 回退: H % 8 != 0（半行不能对齐到 4 元素）或基址未按向量宽度对齐
//   或 M 超出 scratch 容量 或 波次容量为奇数（理论上不可达）→
//   与 vec4 完全相同的回退链（vec4 3 遍向量化 → 标量 3 遍）,
//   不拒绝输入。
//
// 数值: FP32 中间量, 与 baseline 相同的 FP32 expf。每半行的在线
//   累积顺序与 online 变体相同（docs/softmax_algorithm.md §5 给出
//   了与 3-pass 的误差关系）。

#include "softmax_common.h"
#include "softmax_scalar.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>

namespace {

constexpr int kMaxScratchM = 8192;   // scratch 静态分配上限（163.8KB @ M=8192）

struct __align__(8) Half4Hs {
    __half2 lo;
    __half2 hi;
};

template <typename T>
struct V4Hs {
    static constexpr size_t vec_align_bytes = 0;
};

template <>
struct V4Hs<__half> {
    static constexpr size_t vec_align_bytes = 8;
    static __device__ float4 load(const __half* p, int v) {
        const Half4Hs h = reinterpret_cast<const Half4Hs*>(p)[v];
        return make_float4(el_to_float(h.lo.x), el_to_float(h.lo.y),
                           el_to_float(h.hi.x), el_to_float(h.hi.y));
    }
    static __device__ void store(__half* p, int v, float4 f) {
        Half4Hs h;
        h.lo = __floats2half2_rn(f.x, f.y);
        h.hi = __floats2half2_rn(f.z, f.w);
        reinterpret_cast<Half4Hs*>(p)[v] = h;
    }
};

template <>
struct V4Hs<float> {
    static constexpr size_t vec_align_bytes = 16;
    static __device__ float4 load(const float* p, int v) {
        return reinterpret_cast<const float4*>(p)[v];
    }
    static __device__ void store(float* p, int v, float4 f) {
        reinterpret_cast<float4*>(p)[v] = f;
    }
};

// block 内 (m, l) 对的 merge 归约（与 SFM-0002 softmax_online 相同）。
template <int NT>
__device__ void block_merge_hs(float m, float l, float* s_m, float* s_l) {
    const int tid = threadIdx.x;
    const unsigned mask = 0xffffffffu;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float om = __shfl_down_sync(mask, m, offset);
        const float ol = __shfl_down_sync(mask, l, offset);
        const float M = fmaxf(m, om);
        l = l * expf(m - M) + ol * expf(om - M);
        m = M;
    }
    const int nwarp = NT / 32;
    if ((tid & 31) == 0) {
        s_m[tid >> 5] = m;
        s_l[tid >> 5] = l;
    }
    __syncthreads();
    if (tid == 0) {
        float rm = s_m[0];
        float rl = s_l[0];
        for (int i = 1; i < nwarp; i++) {
            const float M = fmaxf(rm, s_m[i]);
            rl = rl * expf(rm - M) + s_l[i] * expf(s_m[i] - M);
            rm = M;
        }
        s_m[0] = rm;
        s_l[0] = rl;
    }
    __syncthreads();
}

__device__ __forceinline__ int volatile_read_i(const int* p) {
    return *reinterpret_cast<const volatile int*>(p);
}

// ---- hsplit2 向量化内核: grid (2, M), 每块处理半行 ----
template <typename T>
__global__ void softmax_hsplit2_vec4_kernel(const T* __restrict__ x,
                                            T* __restrict__ y,
                                            int H,
                                            float* __restrict__ m_part,
                                            float* __restrict__ l_part,
                                            int* __restrict__ cnt) {
    const int half = blockIdx.x;      // 0 = 前半行, 1 = 后半行
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    __shared__ float s_m[SB_BLOCK / 32];
    __shared__ float s_l[SB_BLOCK / 32];

    const int halfH = H / 2;                          // 半行元素数
    const int nvec_half = H / 8;                      // 半行内 4 元素向量数
    const T* __restrict__ xhalf = x + (size_t)row * H + half * halfH;
    T* __restrict__ yhalf = y + (size_t)row * H + half * halfH;

    // ---- 遍 1: 半行内在线累积局部 (m, l) ----
    float m = -FLT_MAX;
    float l = 0.f;
    for (int v = tid; v < nvec_half; v += nthreads) {
        const float4 f = V4Hs<T>::load(xhalf, v);
        const float m2 = fmaxf(m, fmaxf(fmaxf(f.x, f.y),
                                        fmaxf(f.z, f.w)));
        l = l * expf(m - m2) + expf(f.x - m2) + expf(f.y - m2)
          + expf(f.z - m2) + expf(f.w - m2);
        m = m2;
    }
    block_merge_hs<SB_BLOCK>(m, l, s_m, s_l);
    m = s_m[0];
    l = s_l[0];

    // ---- 跨块会合: 发布半行 (m,l), 等另一半, 合并 ----
    if (tid == 0) {
        const int p = row * 2 + half;
        m_part[p] = m;
        l_part[p] = l;
        __threadfence();                  // release: 分块值先于 cnt 可见
        atomicAdd(cnt + row, 1);
        // 自旋到本 launch 的配对完成。NCU kernel-replay 安全: 回放之间
        // 不会重跑驱动端的 memset（计数器跨回放累积）, 因此接受任意
        // >= 2 的偶数 —— 第 k 次回放观察到 2k, 两个分块值都已在本次
        // 回放中被重写（release 序在各自 increment 之前）。单次执行的
        // 语义与 != 2 完全相同（0 -> 1 自旋 -> 2 通过）。
        int c;
        do {
            c = volatile_read_i(cnt + row);
        } while (c < 2 || (c & 1) != 0);
        __threadfence();                  // acquire: 分块值现在可见
        const float m0 = m_part[row * 2 + 0];
        const float l0 = l_part[row * 2 + 0];
        const float m1 = m_part[row * 2 + 1];
        const float l1 = l_part[row * 2 + 1];
        const float M = fmaxf(m0, m1);
        // 两个 block 执行完全相同的运算序列 ⇒ 逐位一致
        const float L = l0 * expf(m0 - M) + l1 * expf(m1 - M);
        s_m[0] = M;
        s_l[0] = L;
    }
    __syncthreads();
    const float M = s_m[0];
    const float L = s_l[0];

    // ---- 遍 2: 重读本半行, 写 y = exp(x − M) / L ----
    const float inv_l = 1.0f / L;
    for (int v = tid; v < nvec_half; v += nthreads) {
        const float4 f = V4Hs<T>::load(xhalf, v);
        V4Hs<T>::store(yhalf, v, make_float4(
            expf(f.x - M) * inv_l, expf(f.y - M) * inv_l,
            expf(f.z - M) * inv_l, expf(f.w - M) * inv_l));
    }
}

// ---- 回退用: 与 softmax_vec4.cu 相同的 3 遍向量化内核（复制件,
//      保持本翻译单元自包含; 修改时与 softmax_vec4.cu 同步）----
template <typename T>
__global__ void hsplit2_fallback_vec4_kernel(const T* __restrict__ x,
                                             T* __restrict__ y,
                                             int H) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const T* __restrict__ xrow = x + (size_t)row * H;
    T* __restrict__ yrow = y + (size_t)row * H;
    __shared__ float s_red[SB_BLOCK / 32];
    const int nvec = H / 4;

    float m = -FLT_MAX;
    for (int v = tid; v < nvec; v += nthreads) {
        const float4 f = V4Hs<T>::load(xrow, v);
        m = fmaxf(m, fmaxf(fmaxf(f.x, f.y), fmaxf(f.z, f.w)));
    }
    m = block_max<SB_BLOCK>(m, s_red);

    float l = 0.f;
    for (int v = tid; v < nvec; v += nthreads) {
        const float4 f = V4Hs<T>::load(xrow, v);
        l += expf(f.x - m) + expf(f.y - m) + expf(f.z - m) + expf(f.w - m);
    }
    l = block_sum<SB_BLOCK>(l, s_red);

    const float inv_l = 1.0f / l;
    for (int v = tid; v < nvec; v += nthreads) {
        const float4 f = V4Hs<T>::load(xrow, v);
        V4Hs<T>::store(yrow, v, make_float4(
            expf(f.x - m) * inv_l, expf(f.y - m) * inv_l,
            expf(f.z - m) * inv_l, expf(f.w - m) * inv_l));
    }
}

// ---- scratch 与容量: 进程级一次性 ----
struct HsGlobal {
    float* scratch = nullptr;      // [m_part 2M | l_part 2M | cnt M (int32)]
    int wave_capacity = 0;         // 每波次可驻留块数（SM 数 × 每 SM 块上限）
};

HsGlobal& hs_global() {
    static HsGlobal g;
    return g;
}

bool hs_init(int dev) {
    HsGlobal& g = hs_global();
    if (g.scratch == nullptr) {
        const size_t bytes = (size_t)kMaxScratchM * 5 * sizeof(float);
        if (cudaMalloc(&g.scratch, bytes) != cudaSuccess) return false;
    }
    if (g.wave_capacity == 0) {
        int threads_per_sm = 0, sm_count = 0, blocks_per_sm = 0;
        cudaDeviceGetAttribute(&threads_per_sm,
                               cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        cudaDeviceGetAttribute(&sm_count,
                               cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&blocks_per_sm,
                               cudaDevAttrMaxBlocksPerMultiprocessor, dev);
        int per_sm = threads_per_sm / SB_BLOCK;
        if (per_sm > blocks_per_sm) per_sm = blocks_per_sm;
        g.wave_capacity = per_sm * sm_count;
    }
    return true;
}

template <typename T>
void softmax_hsplit2_fwd(const at::Tensor& x, at::Tensor& out) {
    c10::cuda::CUDAGuard guard(x.device());
    const int M = x.size(0);
    const int H = x.size(1);
    const char* xp = reinterpret_cast<const char*>(x.data_ptr());
    char* yp = reinterpret_cast<char*>(out.data_ptr());
    const size_t align = V4Hs<T>::vec_align_bytes;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int dev = 0;
    cudaGetDevice(&dev);
    HsGlobal& g = hs_global();
    const bool hsplit_ok =
        (H % 8 == 0)
        && ((reinterpret_cast<uintptr_t>(xp) & (align - 1)) == 0)
        && ((reinterpret_cast<uintptr_t>(yp) & (align - 1)) == 0)
        && (M <= kMaxScratchM)
        && hs_init(dev)
        && (g.wave_capacity % 2 == 0);   // 死锁防护: 半对不能跨波次

    if (hsplit_ok) {
        const size_t cnt_bytes = (size_t)M * sizeof(int);
        cudaMemsetAsync(g.scratch + (size_t)kMaxScratchM * 4, 0,
                        cnt_bytes, stream);
        softmax_hsplit2_vec4_kernel<T><<<dim3(2, M), SB_BLOCK, 0, stream>>>(
            reinterpret_cast<const T*>(xp), reinterpret_cast<T*>(yp), H,
            g.scratch, g.scratch + (size_t)kMaxScratchM * 2,
            reinterpret_cast<int*>(g.scratch + (size_t)kMaxScratchM * 4));
    } else if ((H % 4 == 0)
               && ((reinterpret_cast<uintptr_t>(xp) & (align - 1)) == 0)
               && ((reinterpret_cast<uintptr_t>(yp) & (align - 1)) == 0)) {
        // 回退 1: 与 vec4 相同的 3 遍向量化内核
        hsplit2_fallback_vec4_kernel<T><<<dim3(M), SB_BLOCK, 0, stream>>>(
            reinterpret_cast<const T*>(xp), reinterpret_cast<T*>(yp), H);
    } else {
        // 回退 2: 与 baseline 相同的标量 3 遍内核
        softmax_scalar_kernel<T><<<dim3(M), SB_BLOCK, 0, stream>>>(
            reinterpret_cast<const T*>(xp), reinterpret_cast<T*>(yp), H);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void softmax_hsplit2_fwd(const at::Tensor& x, at::Tensor& out) {
    switch (x.scalar_type()) {
        case at::kHalf:
            softmax_hsplit2_fwd<__half>(x, out);
            break;
        case at::kFloat:
            softmax_hsplit2_fwd<float>(x, out);
            break;
        default:
            TORCH_CHECK(false, "softmax hsplit2 不支持该 dtype");
    }
}

}  // namespace

static struct SoftmaxHsplit2Registrar {
    SoftmaxHsplit2Registrar() {
        register_softmax_variant("softmax_hsplit2", softmax_hsplit2_fwd);
    }
} softmax_hsplit2_registrar;
