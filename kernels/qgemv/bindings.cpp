// CUDALab QGEMV — PyTorch 扩展绑定层。
//
// 纯 C++ 文件（用宿主编译器编译）: 保存变体注册表和 Python 入口点。
// 所有 CUDA 代码都在同目录的 .cu 文件中，每个文件自注册其变体。
//
// QGEMV 入口: W_q (N,K) int8 / scale (N,) fp32 / x (K,) fp16 /
// （可选的预分配）out (N,) fp16。
// y[n] = Σ_k (scale[n]·W_q[n,k])·x[k]（FP32 累加, FP16 输出）。
//
// 所有输入验证都在 kernel launch 之前以 TORCH_CHECK 完成（negative
// test suite 依赖这一点）; 内核自身不含设备端断言。
//
// 与 GEMV 相同: QGEMV **没有数据依赖验证**（无 D2H 同步检查）,
// validate 全部是 host 元数据检查（dim/shape/dtype/连续/设备）,
// 逐 launch 开销可忽略。因此**不提供** RoPE 式的 validate=false
// 基准池开关 —— forward_into 每次调用都执行完整验证, 基准池无需
// 任何豁免（契约见 cudalab/operators/qgemv.py）。
//
// native_timing（v0.5 引入的第三种计时口径, v0.6 原样沿用）: Python
// 只调用一次本扩展, C++ 内部连续 launch kernel N 次, CUDA Events
// 包围整个循环, elapsed/N 即"原生 kernel-loop 单发时间"。它与（1）
// API 路径口径（bench 引擎经 Python↔C++ 边界、每样本 32 连发）和
// （2）NCU kernel duration（profiler replay 下纯 kernel 时间）是
// **三个不同口径**, 三者不混用（v0.6 要求分开报告, 冲突时记录并
// 调查, 不能挑最好看的数字）。
//   - 每次调用先做一次性完整 host 验证, 之后是 N 次**裸 launch**
//     （无逐 launch 验证, 无 stream 同步）;
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
#include <unordered_set>
#include <vector>

#include "qgemv_common.h"

namespace {
std::unordered_map<std::string, qgemv_fn_t>& registry() {
    static std::unordered_map<std::string, qgemv_fn_t> r;
    return r;
}

// ---- 隔离（quarantine）策略 ---------------------------------------------
// v0.6 当前**无**隔离变体。保留该机制与 v0.5 GEMV 相同的注册表协议:
// 若未来某候选被 REJECTED / 带静态 workspace 等风险, 加入
// quarantined_set() 即自动从 variants() 正常列表移除（显式命名调用
// 仍是受控历史审计入口）, CLI / bench 的门禁（scripts/cudalab.py
// _variant_gate_error + evaluator/bench.py _require_normal_variant）
// 无需改动。
const std::unordered_set<std::string>& quarantined_set() {
    static const std::unordered_set<std::string> q = {};
    return q;
}

// ---- 输入验证（全部在 launch 前, 全部 host 元数据, 无同步）------------

void validate_qgemv_Wq(const at::Tensor& Wq) {
    TORCH_CHECK(Wq.dim() == 2, "W_q 必须是 2 维 (N, K)，实际 ", Wq.dim(),
                " 维");
    TORCH_CHECK(Wq.size(0) > 0, "N 必须 > 0，实际 ", Wq.size(0));
    TORCH_CHECK(Wq.size(1) > 0, "K 必须 > 0，实际 ", Wq.size(1));
    TORCH_CHECK(Wq.dtype() == at::kChar,
                "W_q 必须是 int8（torch int8 为 kChar），实际 ", Wq.dtype());
    TORCH_CHECK(Wq.is_contiguous(), "W_q 必须是连续内存");
    TORCH_CHECK(Wq.is_cuda(), "W_q 必须是 CUDA 张量");
}

void validate_qgemv_scale(const at::Tensor& Wq, const at::Tensor& scale) {
    TORCH_CHECK(scale.dim() == 1, "scale 必须是 1 维 (N,)，实际 ",
                scale.dim(), " 维");
    TORCH_CHECK(scale.size(0) == Wq.size(0),
                "scale 长度必须等于 N: scale=", scale.size(0),
                " N=", Wq.size(0));
    TORCH_CHECK(scale.dtype() == at::kFloat,
                "scale 必须是 float32，实际 ", scale.dtype());
    TORCH_CHECK(scale.is_contiguous(), "scale 必须是连续内存");
    TORCH_CHECK(scale.is_cuda(), "scale 必须是 CUDA 张量");
    TORCH_CHECK(scale.device() == Wq.device(),
                "scale 必须与 W_q 在同一 CUDA 设备上");
}

void validate_qgemv_x(const at::Tensor& Wq, const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 1, "x 必须是 1 维 (K,)，实际 ", x.dim(), " 维");
    TORCH_CHECK(x.size(0) == Wq.size(1),
                "x 长度必须等于 K: x=", x.size(0),
                " K=", Wq.size(1));
    TORCH_CHECK(x.dtype() == at::kHalf,
                "x 必须是 float16（v0.6 主路径 FP16 activation）, 实际 ",
                x.dtype());
    TORCH_CHECK(x.is_contiguous(), "x 必须是连续内存");
    TORCH_CHECK(x.is_cuda(), "x 必须是 CUDA 张量");
    TORCH_CHECK(x.device() == Wq.device(),
                "x 必须与 W_q 在同一 CUDA 设备上");
}

void validate_qgemv_out(const at::Tensor& Wq, const at::Tensor& out) {
    TORCH_CHECK(out.dim() == 1, "out 必须是 1 维 (N,)，实际 ", out.dim(),
                " 维");
    TORCH_CHECK(out.size(0) == Wq.size(0),
                "out 长度必须等于 N: out=", out.size(0),
                " N=", Wq.size(0));
    TORCH_CHECK(out.dtype() == at::kHalf,
                "out 必须是 float16，实际 ", out.dtype());
    TORCH_CHECK(out.is_contiguous(), "out 必须是连续内存");
    TORCH_CHECK(out.is_cuda(), "out 必须是 CUDA 张量");
    TORCH_CHECK(out.device() == Wq.device(),
                "out 必须与 W_q 在同一 CUDA 设备上");
}

std::string unknown_variant_msg(const std::string& name) {
    std::string s = "未知 qgemv 变体 '" + name + "'。正常可用: ";
    for (auto& n : qgemv_variant_list()) s += n + " ";
    auto q = qgemv_quarantined_variant_list();
    if (!q.empty()) {
        s += "；被隔离（NOT_FOR_NORMAL_DISPATCH）: ";
        for (auto& n : q) s += n + " ";
        s += "（显式命名仍可调用，属受控历史审计入口）";
    }
    return s;
}
}  // namespace

void register_qgemv_variant(const std::string& name, qgemv_fn_t fn) {
    auto& r = registry();
    if (r.count(name)) {
        TORCH_CHECK(false, "重复的 qgemv 变体: ", name);
    }
    r.emplace(name, fn);
}

// 分配输出并计算（常规 API 路径）。
at::Tensor qgemv_forward(const std::string& name, const at::Tensor& Wq,
                         const at::Tensor& scale, const at::Tensor& x) {
    auto& r = registry();
    auto it = r.find(name);
    TORCH_CHECK(it != r.end(), unknown_variant_msg(name));
    validate_qgemv_Wq(Wq);
    validate_qgemv_scale(Wq, scale);
    validate_qgemv_x(Wq, x);
    at::Tensor out = at::empty({Wq.size(0)},
                               Wq.options().dtype(at::kHalf));
    c10::cuda::CUDAGuard guard(Wq.device());
    it->second(Wq, scale, x, out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// 写入预分配、布局兼容的 `out`（基准测试用它把内存分配排除在计时
// 区域之外）。与 forward 共享验证逻辑。launch 之后立即检查启动错误。
void qgemv_forward_into(const std::string& name, const at::Tensor& Wq,
                        const at::Tensor& scale, const at::Tensor& x,
                        at::Tensor& out) {
    auto& r = registry();
    auto it = r.find(name);
    TORCH_CHECK(it != r.end(), unknown_variant_msg(name));
    validate_qgemv_Wq(Wq);
    validate_qgemv_scale(Wq, scale);
    validate_qgemv_x(Wq, x);
    validate_qgemv_out(Wq, out);
    c10::cuda::CUDAGuard guard(Wq.device());
    it->second(Wq, scale, x, out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- native_timing: 原生 kernel-loop 计时口径 --------------------------

py::dict qgemv_native_timing(const std::string& name, const at::Tensor& Wq,
                             const at::Tensor& scale, const at::Tensor& x,
                             at::Tensor& out, int64_t warmup,
                             int64_t n_windows, int64_t launches_per_window) {
    auto& r = registry();
    auto it = r.find(name);
    TORCH_CHECK(it != r.end(), unknown_variant_msg(name));
    TORCH_CHECK(warmup >= 0, "warmup 必须 >= 0，实际 ", warmup);
    TORCH_CHECK(n_windows >= 1, "n_windows 必须 >= 1，实际 ", n_windows);
    TORCH_CHECK(launches_per_window >= 1,
                "launches_per_window 必须 >= 1，实际 ", launches_per_window);
    validate_qgemv_Wq(Wq);
    validate_qgemv_scale(Wq, scale);
    validate_qgemv_x(Wq, x);
    validate_qgemv_out(Wq, out);

    c10::cuda::CUDAGuard guard(Wq.device());
    const qgemv_fn_t fn = it->second;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // warmup: 不计时（覆盖 first-touch / 分支预热态）
    for (int64_t i = 0; i < warmup; ++i) fn(Wq, scale, x, out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cudaStreamSynchronize(stream);

    cudaEvent_t ev0, ev1;
    C10_CUDA_CHECK(cudaEventCreate(&ev0));
    C10_CUDA_CHECK(cudaEventCreate(&ev1));
    std::vector<double> per_window_us;
    per_window_us.reserve(static_cast<size_t>(n_windows));
    for (int64_t w = 0; w < n_windows; ++w) {
        C10_CUDA_CHECK(cudaEventRecord(ev0, stream));
        for (int64_t i = 0; i < launches_per_window; ++i)
            fn(Wq, scale, x, out);
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

// 默认（正常）变体列表：不含被隔离的变体（当前为空集, 机制保留,
// 见文件头部 quarantine 策略注释）。所有正常 dispatch / 基准 /
// 测试 / 剖析路径都使用本列表。
std::vector<std::string> qgemv_variant_list() {
    std::vector<std::string> names;
    for (auto& kv : registry()) {
        if (!quarantined_set().count(kv.first)) names.push_back(kv.first);
    }
    std::sort(names.begin(), names.end());
    return names;
}

// 全部已注册变体（含被隔离者, 当前与 qgemv_variant_list 相同）:
// 显式历史审计入口使用。
std::vector<std::string> qgemv_all_variant_list() {
    std::vector<std::string> names;
    for (auto& kv : registry()) names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    return names;
}

// 被隔离的变体列表（当前为空）: UNSAFE_HISTORICAL_EXPERIMENT /
// REJECTED / NOT_FOR_NORMAL_DISPATCH。
std::vector<std::string> qgemv_quarantined_variant_list() {
    std::vector<std::string> names;
    for (auto& kv : registry()) {
        if (quarantined_set().count(kv.first)) names.push_back(kv.first);
    }
    std::sort(names.begin(), names.end());
    return names;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &qgemv_forward,
          "QGEMV: y[n] = Σ_k (scale[n]·W_q[n,k])·x[k]（INT8 权重 + "
          "FP16 activation, FP32 累加, FP16 输出）。分配新输出张量。",
          py::arg("variant"), py::arg("W_q"), py::arg("scale"),
          py::arg("x"));
    m.def("forward_into", &qgemv_forward_into,
          "QGEMV，写入预分配的输出张量（基准测试路径; 验证恒为完整 "
          "host 元数据检查, QGEMV 无数据依赖验证, 见文件头）。",
          py::arg("variant"), py::arg("W_q"), py::arg("scale"),
          py::arg("x"), py::arg("out"));
    m.def("native_timing", &qgemv_native_timing,
          "原生 kernel-loop 计时: warmup 次不计时 launch 后, n_windows "
          "个窗口各 launches_per_window 次连续 launch, CUDA events 包围 "
          "整窗, elapsed/N 为单发时间（us）。返回每窗口值 + "
          "median/min/max/mean（窗口为统计单位）。",
          py::arg("variant"), py::arg("W_q"), py::arg("scale"),
          py::arg("x"), py::arg("out"),
          py::arg("warmup") = 200, py::arg("n_windows") = 10,
          py::arg("launches_per_window") = 64);
    m.def("variants", &qgemv_variant_list,
          "正常（可 dispatch）的 qgemv 变体列表；不含被隔离变体"
          "（v0.6 当前无隔离变体, 机制保留）");
    m.def("all_variants", &qgemv_all_variant_list,
          "全部已注册变体（含被隔离者, 仅供显式历史审计）");
    m.def("quarantined_variants", &qgemv_quarantined_variant_list,
          "被隔离的变体: UNSAFE_HISTORICAL_EXPERIMENT / REJECTED / "
          "NOT_FOR_NORMAL_DISPATCH（v0.6 当前为空）");
}
