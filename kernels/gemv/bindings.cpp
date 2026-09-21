// CUDALab GEMV — PyTorch 扩展绑定层。
//
// 纯 C++ 文件（用宿主编译器编译）: 保存变体注册表和 Python 入口点。
// 所有 CUDA 代码都在同目录的 .cu 文件中，每个文件自注册其变体。
//
// GEMV 入口: W (N,K) / x (K,) /（可选的预分配）out (N,)。
// 输出 dtype = W dtype; 累加恒为 FP32（见 gemv_common.h 头部）。
//
// 所有输入验证都在 kernel launch 之前以 TORCH_CHECK 完成（negative
// test suite 依赖这一点）; 内核自身不含设备端断言。
//
// 与 RoPE 不同: GEMV **没有数据依赖验证**（没有 positions 值域之类的
// D2H 同步检查）, validate 全部是 host 元数据检查（dim/shape/dtype/
// 连续/设备）, 逐 launch 开销可忽略。因此**不提供** RoPE 式的
// validate=false 基准池开关 —— forward_into 每次调用都执行完整
// 验证, 基准池无需任何豁免（契约见 cudalab/operators/gemv.py）。
//
// native_timing（v0.5 新增计时口径）: Python 只调用一次本扩展, C++
// 内部连续 launch kernel N 次, CUDA Events 包围整个循环, elapsed/N
// 即"原生 kernel-loop 单发时间"。它与（1）API 路径口径（bench 引擎
// 经 Python↔C++ 边界、每样本 32 连发）和（2）NCU kernel duration
// （profiler replay 下纯 kernel 时间）是**三个不同口径**, 三者不
// 混用（v0.5 要求分开报告, 冲突时记录并调查）。
//   - 每次调用先做一次性完整 host 验证, 之后是 N 次**裸 launch**
//     （无逐 launch 验证, 无 stream 同步）—— 这正是"去掉 Python
//     边界开销"的定义;
//   - warmup 次不计时 launch + 同步; 随后 n_windows 个窗口, 每窗口
//     launches_per_window 次连续 launch, 窗口间同步;
//   - 返回每窗口单发 us + 中位/最小/最大/均值（窗口为统计单位,
//     与 bench 引擎的"round/block 为统计单位"一致）。

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>   // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "gemv_common.h"

namespace {
std::unordered_map<std::string, gemv_fn_t>& registry() {
    static std::unordered_map<std::string, gemv_fn_t> r;
    return r;
}

// ---- 输入验证（全部在 launch 前, 全部 host 元数据, 无同步）------------

void validate_gemv_W(const at::Tensor& W) {
    TORCH_CHECK(W.dim() == 2, "W 必须是 2 维 (N, K)，实际 ", W.dim(), " 维");
    TORCH_CHECK(W.size(0) > 0, "N 必须 > 0，实际 ", W.size(0));
    TORCH_CHECK(W.size(1) > 0, "K 必须 > 0，实际 ", W.size(1));
    TORCH_CHECK(W.dtype() == at::kHalf || W.dtype() == at::kFloat,
                "仅支持 float16 / float32，实际 ", W.dtype());
    TORCH_CHECK(W.is_contiguous(), "W 必须是连续内存");
    TORCH_CHECK(W.is_cuda(), "W 必须是 CUDA 张量");
}

void validate_gemv_x(const at::Tensor& W, const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 1, "x 必须是 1 维 (K,)，实际 ", x.dim(), " 维");
    TORCH_CHECK(x.size(0) == W.size(1),
                "x 长度必须等于 K: x=", x.size(0),
                " K=", W.size(1));
    TORCH_CHECK(x.dtype() == W.dtype(),
                "x 的 dtype 必须与 W 一致: x=", x.dtype(),
                " W=", W.dtype());
    TORCH_CHECK(x.is_contiguous(), "x 必须是连续内存");
    TORCH_CHECK(x.is_cuda(), "x 必须是 CUDA 张量");
    TORCH_CHECK(x.device() == W.device(),
                "x 必须与 W 在同一 CUDA 设备上");
}

void validate_gemv_out(const at::Tensor& W, const at::Tensor& x,
                       const at::Tensor& out) {
    TORCH_CHECK(out.dim() == 1, "out 必须是 1 维 (N,)，实际 ", out.dim(),
                " 维");
    TORCH_CHECK(out.size(0) == W.size(0),
                "out 长度必须等于 N: out=", out.size(0),
                " N=", W.size(0));
    TORCH_CHECK(out.dtype() == W.dtype(),
                "out 的 dtype 必须与 W 一致: out=", out.dtype(),
                " W=", W.dtype());
    TORCH_CHECK(out.is_contiguous(), "out 必须是连续内存");
    TORCH_CHECK(out.is_cuda(), "out 必须是 CUDA 张量");
    TORCH_CHECK(out.device() == W.device(),
                "out 必须与 W 在同一 CUDA 设备上");
}
}  // namespace

void register_gemv_variant(const std::string& name, gemv_fn_t fn) {
    auto& r = registry();
    if (r.count(name)) {
        TORCH_CHECK(false, "重复的 gemv 变体: ", name);
    }
    r.emplace(name, fn);
}

// 分配输出并计算（常规 API 路径）。
at::Tensor gemv_forward(const std::string& name, const at::Tensor& W,
                        const at::Tensor& x) {
    auto& r = registry();
    auto it = r.find(name);
    TORCH_CHECK(it != r.end(), "未知 gemv 变体 '", name, "'。可用: ", [&] {
        std::string s;
        for (auto& n : gemv_variant_list()) s += n + " ";
        return s;
    }());
    validate_gemv_W(W);
    validate_gemv_x(W, x);
    at::Tensor out = at::empty({W.size(0)}, W.options());
    c10::cuda::CUDAGuard guard(W.device());
    it->second(W, x, out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// 写入预分配、布局兼容的 `out`（基准测试用它把内存分配排除在计时
// 区域之外）。与 forward 共享验证逻辑。launch 之后立即检查启动错误。
void gemv_forward_into(const std::string& name, const at::Tensor& W,
                       const at::Tensor& x, at::Tensor& out) {
    auto& r = registry();
    auto it = r.find(name);
    TORCH_CHECK(it != r.end(), "未知 gemv 变体 '", name, "'。可用: ", [&] {
        std::string s;
        for (auto& n : gemv_variant_list()) s += n + " ";
        return s;
    }());
    validate_gemv_W(W);
    validate_gemv_x(W, x);
    validate_gemv_out(W, x, out);
    c10::cuda::CUDAGuard guard(W.device());
    it->second(W, x, out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- native_timing: 原生 kernel-loop 计时口径 --------------------------

py::dict gemv_native_timing(const std::string& name, const at::Tensor& W,
                            const at::Tensor& x, at::Tensor& out,
                            int64_t warmup, int64_t n_windows,
                            int64_t launches_per_window) {
    auto& r = registry();
    auto it = r.find(name);
    TORCH_CHECK(it != r.end(), "未知 gemv 变体 '", name, "'。可用: ", [&] {
        std::string s;
        for (auto& n : gemv_variant_list()) s += n + " ";
        return s;
    }());
    TORCH_CHECK(warmup >= 0, "warmup 必须 >= 0，实际 ", warmup);
    TORCH_CHECK(n_windows >= 1, "n_windows 必须 >= 1，实际 ", n_windows);
    TORCH_CHECK(launches_per_window >= 1,
                "launches_per_window 必须 >= 1，实际 ", launches_per_window);
    validate_gemv_W(W);
    validate_gemv_x(W, x);
    validate_gemv_out(W, x, out);

    c10::cuda::CUDAGuard guard(W.device());
    const gemv_fn_t fn = it->second;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // warmup: 不计时（覆盖 first-touch / 分支预热态）
    for (int64_t i = 0; i < warmup; ++i) fn(W, x, out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cudaStreamSynchronize(stream);

    cudaEvent_t ev0, ev1;
    C10_CUDA_CHECK(cudaEventCreate(&ev0));
    C10_CUDA_CHECK(cudaEventCreate(&ev1));
    std::vector<double> per_window_us;
    per_window_us.reserve(static_cast<size_t>(n_windows));
    for (int64_t w = 0; w < n_windows; ++w) {
        C10_CUDA_CHECK(cudaEventRecord(ev0, stream));
        for (int64_t i = 0; i < launches_per_window; ++i) fn(W, x, out);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cudaEventRecord(ev1, stream));
        C10_CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.f;
        C10_CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        // 单发时间 us = 窗口总耗时 / 窗口内 launch 数
        per_window_us.push_back(static_cast<double>(ms) * 1e3 /
                                static_cast<double>(launches_per_window));
    }
    C10_CUDA_CHECK(cudaEventDestroy(ev0));
    C10_CUDA_CHECK(cudaEventDestroy(ev1));

    std::vector<double> sorted = per_window_us;
    std::sort(sorted.begin(), sorted.end());
    const size_t n = sorted.size();
    const double median = (n % 2 == 1)
        ? sorted[n / 2]
        : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    const double mean =
        std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(n);
    std::vector<py::float_> window_list;
    for (double v : per_window_us) window_list.push_back(v);

    return py::dict(
        py::arg("variant") = py::cast(name),
        py::arg("warmup") = warmup,
        py::arg("n_windows") = n_windows,
        py::arg("launches_per_window") = launches_per_window,
        py::arg("window_median_us") = py::cast(window_list),
        py::arg("median_us") = median,
        py::arg("min_us") = sorted.front(),
        py::arg("max_us") = sorted.back(),
        py::arg("mean_us") = mean,
        py::arg("note") = py::cast(
            "原生 kernel-loop 口径: C++ 内连续 launch, CUDA events / N; "
            "不含 Python 边界开销（区别于 bench 引擎 API 路径口径与 "
            "NCU kernel duration）"));
}

std::vector<std::string> gemv_variant_list() {
    std::vector<std::string> names;
    for (auto& kv : registry()) names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    return names;
}

std::vector<std::string> gemv_all_variant_list() {
    return gemv_variant_list();  // 当前无被隔离变体
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gemv_forward,
          "GEMV: y[n] = Σ_k W[n,k]*x[k]（FP32 累加, 输出 dtype = W "
          "dtype）。分配新输出张量。",
          py::arg("variant"), py::arg("W"), py::arg("x"));
    m.def("forward_into", &gemv_forward_into,
          "GEMV，写入预分配的输出张量（基准测试路径; 验证恒为完整 "
          "host 元数据检查, GEMV 无数据依赖验证, 见文件头）。",
          py::arg("variant"), py::arg("W"), py::arg("x"), py::arg("out"));
    m.def("native_timing", &gemv_native_timing,
          "原生 kernel-loop 计时: warmup 次不计时 launch 后, n_windows "
          "个窗口各 launches_per_window 次连续 launch, CUDA events 包围 "
          "整窗, elapsed/N 为单发时间（us）。返回每窗口值 + "
          "median/min/max/mean（窗口为统计单位）。",
          py::arg("variant"), py::arg("W"), py::arg("x"), py::arg("out"),
          py::arg("warmup") = 200, py::arg("n_windows") = 10,
          py::arg("launches_per_window") = 64);
    m.def("variants", &gemv_variant_list,
          "正常（可 dispatch）的 gemv 变体列表");
    m.def("all_variants", &gemv_all_variant_list,
          "全部已注册 gemv 变体");
}
