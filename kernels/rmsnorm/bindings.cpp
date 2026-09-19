// CUDALab RMSNorm — PyTorch 扩展绑定层。
//
// 纯 C++ 文件（用宿主编译器编译）: 保存变体注册表和 Python 入口点。
// 所有 CUDA 代码都在同目录的 .cu 文件中，每个文件自注册其变体。

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>   // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <cmath>
#include <unordered_map>
#include <mutex>

#include "rmsnorm_common.h"

namespace {
std::unordered_map<std::string, rmsnorm_fn_t>& registry() {
    static std::unordered_map<std::string, rmsnorm_fn_t> r;
    return r;
}

// v0.2 (Finding B): forward 与 forward_into 共享同一套输入验证，
// 避免维护两套可能漂移的逻辑。所有检查都在 kernel launch 之前完成，
// 非法输入以明确异常拒绝（negative test suite 依赖这一点）。
void validate_common(const at::Tensor& x, const at::Tensor& w, double eps) {
    TORCH_CHECK(x.dim() == 2, "x 必须是 2 维 (M, H)，实际 ", x.dim(), " 维");
    TORCH_CHECK(w.dim() == 1, "w 必须是 1 维 (H,)，实际 ", w.dim(), " 维");
    TORCH_CHECK(x.size(0) > 0, "M 必须 > 0，实际 ", x.size(0));
    TORCH_CHECK(x.size(1) > 0, "H 必须 > 0，实际 ", x.size(1));
    TORCH_CHECK(x.size(1) == w.size(0),
                "H 不匹配: x.size(1)=", x.size(1), " w.size(0)=", w.size(0));
    TORCH_CHECK(x.dtype() == w.dtype(), "x 与 w 必须同 dtype");
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat,
                "仅支持 float16 / float32，实际 ", x.dtype());
    TORCH_CHECK(x.is_contiguous(), "x 必须是连续内存");
    TORCH_CHECK(w.is_contiguous(), "w 必须是连续内存");
    TORCH_CHECK(x.is_cuda(), "x 必须是 CUDA 张量");
    TORCH_CHECK(w.is_cuda(), "w 必须是 CUDA 张量");
    TORCH_CHECK(x.device() == w.device(),
                "x 与 w 必须在同一 CUDA 设备上");
    TORCH_CHECK(std::isfinite(eps) && eps >= 0.0,
                "eps 必须有限且 >= 0，实际 ", eps);
}

void validate_out(const at::Tensor& x, const at::Tensor& out) {
    TORCH_CHECK(out.dim() == 2, "out 必须是 2 维 (M, H)，实际 ", out.dim(), " 维");
    TORCH_CHECK(out.sizes().equals(x.sizes()),
                "out 的形状必须与 x 一致: out=", out.sizes(), " x=", x.sizes());
    TORCH_CHECK(out.dtype() == x.dtype(),
                "out 的 dtype 必须与 x 一致: out=", out.dtype(),
                " x=", x.dtype());
    TORCH_CHECK(out.is_contiguous(), "out 必须是连续内存");
    TORCH_CHECK(out.is_cuda(), "out 必须是 CUDA 张量");
    TORCH_CHECK(out.device() == x.device(), "out 必须与 x 在同一 CUDA 设备上");
}
}  // namespace

void register_rmsnorm_variant(const std::string& name, rmsnorm_fn_t fn) {
    auto& r = registry();
    if (r.count(name)) {
        TORCH_CHECK(false, "重复的 rmsnorm 变体: ", name);
    }
    r.emplace(name, fn);
}

// 写入预分配、布局兼容的 `out`（基准测试用它把内存分配排除在
// 计时区域之外）。v0.2: 与 forward 共享完整验证（Finding B），
// launch 之后立即检查启动错误（Finding C）。
void rmsnorm_forward_into(const std::string& name, const at::Tensor& x,
                          const at::Tensor& w, at::Tensor& out, double eps) {
    auto& r = registry();
    auto it = r.find(name);
    TORCH_CHECK(it != r.end(), "未知 rmsnorm 变体 '", name, "'");
    validate_common(x, w, eps);
    validate_out(x, out);
    c10::cuda::CUDAGuard guard(x.device());
    it->second(x, w, out, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor rmsnorm_forward(const std::string& name, const at::Tensor& x,
                           const at::Tensor& w, double eps) {
    auto& r = registry();
    auto it = r.find(name);
    TORCH_CHECK(it != r.end(),
                "未知 rmsnorm 变体 '", name,
                "'。可用: ", [&] {
                    std::string s;
                    for (auto& kv : r) s += kv.first + " ";
                    return s;
                }());
    validate_common(x, w, eps);
    at::Tensor out = at::empty_like(x);
    c10::cuda::CUDAGuard guard(x.device());
    it->second(x, w, out, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<std::string> rmsnorm_variant_list() {
    std::vector<std::string> names;
    for (auto& kv : registry()) names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    return names;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rmsnorm_forward,
          "RMSNorm: y = x * rsqrt(mean(x^2, -1) + eps) * w",
          py::arg("variant"), py::arg("x"), py::arg("w"), py::arg("eps") = 1e-5);
    m.def("forward_into", &rmsnorm_forward_into,
          "RMSNorm，写入预分配的输出张量（基准测试路径）",
          py::arg("variant"), py::arg("x"), py::arg("w"), py::arg("out"),
          py::arg("eps") = 1e-5);
    m.def("variants", &rmsnorm_variant_list, "列出已注册的 rmsnorm 变体");
}
