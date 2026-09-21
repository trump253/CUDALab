// CUDALab QGEMV — QGEMV-0001: 16B 向量化 load, 结构不变（每行一个 block）。
//
// 实验假设（QGEMV-0001）: baseline（标量 1B int8 + 2B fp16 load）的
// 瓶颈假设是 **指令/字节比** —— 与 v0.5 GEMV baseline 同型但更甚:
// 每 32B W_q 流量 = 32 条 LDG.1（1B/lane 合并为 32B）+ 32 条 LDG.2
// （x 每元素一条）+ ~96 条计算（CVT + MUL(×scale) + FFMA per 元素）,
// 而 FP16 GEMV baseline 每 32B W 只有 0.5 条 LDG.1（64B/指令）+ 32
// 条 LDG.2 + ~64 条计算。把 W 侧 1B 标量 load 换成 16B 向量 load
// （16×int8 / uint4）: 每线程每次迭代取 16 个 int8 —— W 侧 load
// 指令 ÷16、x 侧 ÷8（每 W 向量配两个 x 向量）, 无需改变 block/行
// 结构 —— 隔离向量化的单独收益（scale 提升 / warp-per-row / ILP 是
// QGEMV-0002+ 的独立杠杆）。
//
// 结构（与 baseline 相同）:
//   grid  = N 个 block; block = 256 线程
//   每线程: i = tid, tid+256, ... 的 16B W 向量（K=4096: 每线程 1 个）
//           每 W 向量取两个连续 16B x 向量（xv0 + xv1, 共 16 个 fp16）
//   归约: 与 baseline 完全相同（warp shuffle + shared + warp0）
//
// 对齐契约（v0.6 总则; 见 qgemv_common.h qgemv_vec_contract_ok）:
//   W_q 基指针 16B 对齐 ∧ x 基指针 16B 对齐 ∧ K % 16 == 0
//   （K%16==0 同时保证 x 侧 16B 单位数 K/8 为偶数, 每 W 向量配对的
//   两个 x 向量完整落在行内）。
//   不满足 → host 侧回退 qgemv_scalar_kernel（与 baseline 同一来源,
//   输出逐位一致）; 合法输入（1 字节 offset 视图 / K 非 16 倍数）
//   不得被拒。negative 套件 per-variant 回退用例钉死该行为。
//
// 数值: 累加顺序 = 每线程步长 16B 向量部分和（每向量内 16 元素按
// 低→高序, 逐元素 q→fp32 ×scale ×x FMA）→ 同一归约树。合法 FP32
// 累加顺序, 固定 arith 界适用（每 term 最多 3 次 FP32 舍入, 与
// baseline 相同量级）; 与 baseline 不要求逐位一致（对齐输入下）。

#include "qgemv_common.h"

namespace {

__global__ void qgemv_vec16_row_kernel(const int8_t* __restrict__ Wq,
                                       const float* __restrict__ scale,
                                       const __half* __restrict__ x,
                                       __half* __restrict__ out,
                                       int64_t N, int64_t K) {
    constexpr int64_t epv = 16;  // 16B / 1B = 16 个 int8 per W 向量
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const U16Q* __restrict__ wrow =
        reinterpret_cast<const U16Q*>(Wq + row * K);
    const U16Q* __restrict__ xv = reinterpret_cast<const U16Q*>(x);
    const int64_t nvec = K / epv;  // 每行 16B W 向量数
    const float s = scale[row];    // 每行一次 4B load, 寄存器内复用

    // 步长 16B W 向量部分和（FP32 累加, 逐元素 q→fp32 ×s ×x）
    float acc = 0.f;
#pragma unroll 4
    for (int64_t i = threadIdx.x; i < nvec; i += blockDim.x) {
        U16Q w = wrow[i];
        U16Q xv0 = xv[2 * i];
        U16Q xv1 = xv[2 * i + 1];
        qgemv_vec_acc_dequant(w, xv0, xv1, s, acc);
    }

    // 归约: 与 baseline 相同
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    __shared__ float warp_sums[kQgemvScalarWarps];
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = (lane < kQgemvScalarWarps) ? warp_sums[lane] : 0.f;
#pragma unroll
        for (int off = kQgemvScalarWarps / 2; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) out[row] = __float2half_rn(acc);
    }
}

void qgemv_vec16_row_fwd(const at::Tensor& Wq, const at::Tensor& scale,
                         const at::Tensor& x, at::Tensor& out) {
    if (qgemv_vec_contract_ok(Wq, x)) {
        const int64_t N = Wq.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        qgemv_vec16_row_kernel<<<static_cast<int>(N), kQgemvScalarBlock, 0,
                                 stream>>>(
            static_cast<const int8_t*>(Wq.const_data_ptr()),
            static_cast<const float*>(scale.const_data_ptr()),
            reinterpret_cast<const __half*>(x.const_data_ptr()),
            reinterpret_cast<__half*>(out.data_ptr()),
            Wq.size(0), Wq.size(1));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        launch_qgemv_scalar(Wq, scale, x, out);  // 回退: 不得拒绝
    }
}

static struct Registrar {
    Registrar() { register_qgemv_variant("qgemv_vec16_row",
                                         qgemv_vec16_row_fwd); }
} registrar;

}  // namespace
