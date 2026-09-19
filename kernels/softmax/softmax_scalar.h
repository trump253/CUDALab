// CUDALab Softmax — 标量 3 遍内核（baseline 与 vec4 变体的 fallback 共用）。
//
// 从 softmax_baseline.cu 原样移出的设备代码（SFM-0001 引入）: 向量化
// 变体在 H % 4 != 0 或指针未按向量宽度对齐时回退到该标量内核，
// 而不是拒绝输入 —— 输入契约与 baseline 完全一致。
// 归约 / 遍历逻辑未做任何修改。
#pragma once

#include "softmax_common.h"
#include <cfloat>

namespace {

constexpr int SB_BLOCK = 256;

// block 内 max 归约（warp shuffle + shared memory）。
template <int NT>
__device__ __forceinline__ float block_max(float v, float* smem) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    const int nwarp = NT / 32;
    if ((tid & 31) == 0) smem[tid >> 5] = v;
    __syncthreads();
    if (tid == 0) {
        float r = smem[0];
        for (int i = 1; i < nwarp; i++) r = fmaxf(r, smem[i]);
        smem[0] = r;
    }
    __syncthreads();
    return smem[0];
}

// block 内 sum 归约（与 block_max 相同的骨架）。
template <int NT>
__device__ __forceinline__ float block_sum(float v, float* smem) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    const int nwarp = NT / 32;
    if ((tid & 31) == 0) smem[tid >> 5] = v;
    __syncthreads();
    if (tid == 0) {
        float r = smem[0];
        for (int i = 1; i < nwarp; i++) r += smem[i];
        smem[0] = r;
    }
    __syncthreads();
    return smem[0];
}

// 标量 3 遍内核: 与 baseline 完全相同（max -> sum(exp) -> normalize，
// 遍 3 重读 x 并重算 exp）。
template <typename T>
__global__ void softmax_scalar_kernel(const T* __restrict__ x,
                                      T* __restrict__ y,
                                      int H) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const T* __restrict__ xrow = x + (size_t)row * H;
    T* __restrict__ yrow = y + (size_t)row * H;
    __shared__ float s_red[SB_BLOCK / 32];

    // ---- 遍 1: 行内 max（FP32）----
    float m = -FLT_MAX;
    for (int i = tid; i < H; i += nthreads)
        m = fmaxf(m, el_to_float(xrow[i]));
    m = block_max<SB_BLOCK>(m, s_red);

    // ---- 遍 2: sum(exp(x - max))（FP32）----
    float l = 0.f;
    for (int i = tid; i < H; i += nthreads)
        l += expf(el_to_float(xrow[i]) - m);
    l = block_sum<SB_BLOCK>(l, s_red);

    // ---- 遍 3: y = exp(x - max) / sum（重读 x、重算 exp）----
    const float inv_l = 1.0f / l;
    for (int i = tid; i < H; i += nthreads)
        yrow[i] = el_from_float<T>(expf(el_to_float(xrow[i]) - m) * inv_l);
}

}  // namespace
