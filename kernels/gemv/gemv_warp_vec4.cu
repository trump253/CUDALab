// CUDALab GEMV — GEMV-0002 / GEMV-0003: warp-per-row + 16B load + ILP=4。
//
// 实验假设（GEMV-0002）: 在 GEMV-0001 的向量化之上, 把结构从
// "每行一个 256 线程 block" 改为 "**每行一个 warp**":
//   (1) 归约从 "5 步 warp shuffle + __syncthreads + shared + 二次
//       shuffle" 降为单一 5 步 warp shuffle —— 无 shared 写、无
//       barrier（NCU: barrier 2.9% stall, shared 32B, 占比不大但
//       串行化点每行两次）;
//   (2) 每线程在途 load 数提高: K=4096 fp16 时 512 个 16B 向量 /
//       32 lane = 16 个/线程, 用 4 条独立累加链（ILP=4, 每条步长
//       128 向量）→ 最多 4 个 16B load 同时在途/线程（baseline:
//       2B load × 顺序发射, 在途字节少 16×+）;
//   (3) block = 8 或 16 个 warp（GEMV-0002: 256 线程 / 8 行,
//       GEMV-0003: 512 线程 / 16 行）—— 同一代码两个 block size,
//       测试 "每 block 行数" 杠杆（occupancy 已 94%, 预期影响有限,
//       验证即可）。
//
// 覆盖证明: lane t 的 4 条链处理向量索引
//   { t + 32i + 128r : i∈0..3, r≥0 }
// = 所有 idx 满足 idx mod 128 ∈ {t, 32+t, 64+t, 96+t} —— 32 个 lane
// 的并集 = 全部 128 个剩余类, 每个 16B 向量恰好覆盖一次, 无重不漏。
// nvec 非 128 倍数时（如 K=11008 → nvec=1376 = 128×10+76）, 部分
// 链短 1-3 个向量, 边界由 idx < nvec 守卫（无 OOB）。
//
// 对齐契约（同 GEMV-0001; gemv_common.h）: W/x 基指针 16B 对齐 ∧
// K % epv == 0, 不满足 → host 侧回退 gemv_scalar_kernel（逐位一致
// 于 baseline）; 合法输入不得被拒。
//
// 数值: 累加顺序 = 4 条独立步长链部分和 → 链内和 → 单 warp shuffle
// 树。合法 FP32 累加顺序, 固定 arith 界适用; 与 baseline 不要求
// 逐位一致（对齐输入下）。

#include "gemv_common.h"

namespace {

template <typename T, int kBlock>
__global__ void gemv_warp_vec4_kernel(const T* __restrict__ W,
                                      const T* __restrict__ x,
                                      T* __restrict__ out,
                                      int64_t N, int64_t K) {
    constexpr int64_t epv = 16 / static_cast<int64_t>(sizeof(T));
    constexpr int rows_per_block = kBlock / 32;
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * rows_per_block + (threadIdx.x >> 5);
    if (row >= N) return;

    const U16* __restrict__ wrow =
        reinterpret_cast<const U16*>(W + row * K);
    const U16* __restrict__ xv = reinterpret_cast<const U16*>(x);
    const int64_t nvec = K / epv;
    const int lane = threadIdx.x & 31;

    // ILP=4: 4 条独立链, 步长 128 向量（= 4×32 lane）
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int64_t start = static_cast<int64_t>(lane) + 32 * i;
#pragma unroll 4
        for (int64_t idx = start; idx < nvec; idx += 128) {
            vec_acc<T>(wrow[idx], xv[idx], acc[i]);
        }
    }
    float s = (acc[0] + acc[1]) + (acc[2] + acc[3]);

    // 单 warp 归约（无 shared / 无 barrier）
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, off);
    }
    if (lane == 0) out[row] = el_from_float<T>(s);
}

template <typename T, int kBlock>
void launch_gemv_warp_vec4(const at::Tensor& W, const at::Tensor& x,
                           at::Tensor& out) {
    const int64_t N = W.size(0);
    const int rows_per_block = kBlock / 32;
    const int grid = static_cast<int>((N + rows_per_block - 1) / rows_per_block);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gemv_warp_vec4_kernel<T, kBlock><<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<const T*>(W.const_data_ptr()),
        reinterpret_cast<const T*>(x.const_data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()),
        W.size(0), W.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kBlock>
void warp_vec4_fwd(const at::Tensor& W, const at::Tensor& x,
                   at::Tensor& out) {
    if (W.scalar_type() == at::kHalf) {
        if (gemv_vec_contract_ok<__half>(W, x)) {
            launch_gemv_warp_vec4<__half, kBlock>(W, x, out);
        } else {
            launch_gemv_scalar_dispatch(W, x, out);
        }
    } else {
        if (gemv_vec_contract_ok<float>(W, x)) {
            launch_gemv_warp_vec4<float, kBlock>(W, x, out);
        } else {
            launch_gemv_scalar_dispatch(W, x, out);
        }
    }
}

void gemv_warp_vec4_b256_fwd(const at::Tensor& W, const at::Tensor& x,
                             at::Tensor& out) {
    warp_vec4_fwd<256>(W, x, out);
}

void gemv_warp_vec4_b512_fwd(const at::Tensor& W, const at::Tensor& x,
                             at::Tensor& out) {
    warp_vec4_fwd<512>(W, x, out);
}

static struct Registrar {
    Registrar() {
        register_gemv_variant("gemv_warp_vec4_b256", gemv_warp_vec4_b256_fwd);
        register_gemv_variant("gemv_warp_vec4_b512", gemv_warp_vec4_b512_fwd);
    }
} registrar;

}  // namespace
