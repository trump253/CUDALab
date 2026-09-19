// CUDALab RMSNorm — PyTorch 扩展绑定层。
//
// 纯 C++ 文件（用宿主编译器编译）: 保存变体注册表和 Python 入口点。
// 所有 CUDA 代码都在同目录的 .cu 文件中，每个文件自注册其变体。

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <unordered_map>
#include <mutex>

#include "rmsnorm_common.h"

namespace {
std::unordered_map<std::string, rmsnorm_fn_t>& registry() {
    static std::unordered_map<std::string, rmsnorm_fn_t> r;
    return r;
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
// 计时区域之外）
void rmsnorm_forward_into(const std::string& name, const at::Tensor& x,
                          const at::Tensor& w, at::Tensor& out, double eps) {
    auto& r = registry();
    auto it = r.find(name);
    TORCH_CHECK(it != r.end(), "未知 rmsnorm 变体 '", name, "'");
    TORCH_CHECK(out.sizes().equals(x.sizes()) && out.dtype() == x.dtype(),
                "out 的形状/dtype 必须与 x 一致");
    TORCH_CHECK(out.is_contiguous(), "out 必须是连续内存");
    c10::cuda::CUDAGuard guard(x.device());
    it->second(x, w, out, eps);
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
    TORCH_CHECK(x.dim() == 2, "x 必须是 2 维 (M, H)，实际 ", x.dim(), " 维");
    TORCH_CHECK(w.dim() == 1, "w 必须是 1 维 (H,)，实际 ", w.dim(), " 维");
    TORCH_CHECK(x.size(1) == w.size(0),
                "H 不匹配: x.size(1)=", x.size(1), " w.size(0)=", w.size(0));
    TORCH_CHECK(x.dtype() == w.dtype(), "x 与 w 必须同 dtype");
    TORCH_CHECK(x.is_contiguous(), "x 必须是连续内存");
    TORCH_CHECK(w.is_contiguous(), "w 必须是连续内存");
    TORCH_CHECK(x.is_cuda(), "x 必须是 CUDA 张量");

    at::Tensor xc = x.is_contiguous() ? x : x.contiguous();
    at::Tensor out = at::empty_like(xc);
    c10::cuda::CUDAGuard guard(xc.device());
    it->second(xc, w, out, eps);
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
