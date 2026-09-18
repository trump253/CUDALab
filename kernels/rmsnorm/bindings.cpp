// CUDALab RMSNorm — PyTorch extension bindings.
//
// Pure C++ file (compiled with the host compiler): keeps the variant
// registry and the Python entry points. All CUDA code lives in the
// sibling .cu files, each of which self-registers its variant.

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
        TORCH_CHECK(false, "duplicate rmsnorm variant: ", name);
    }
    r.emplace(name, fn);
}

// writes into a pre-allocated, layout-compatible `out` (used by the
// benchmark to exclude allocation from the timed region)
void rmsnorm_forward_into(const std::string& name, const at::Tensor& x,
                          const at::Tensor& w, at::Tensor& out, double eps) {
    auto& r = registry();
    auto it = r.find(name);
    TORCH_CHECK(it != r.end(), "unknown rmsnorm variant '", name, "'");
    TORCH_CHECK(out.sizes().equals(x.sizes()) && out.dtype() == x.dtype(),
                "out must match x shape/dtype");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    c10::cuda::CUDAGuard guard(x.device());
    it->second(x, w, out, eps);
}

at::Tensor rmsnorm_forward(const std::string& name, const at::Tensor& x,
                           const at::Tensor& w, double eps) {
    auto& r = registry();
    auto it = r.find(name);
    TORCH_CHECK(it != r.end(),
                "unknown rmsnorm variant '", name,
                "'. Available: ", [&] {
                    std::string s;
                    for (auto& kv : r) s += kv.first + " ";
                    return s;
                }());
    TORCH_CHECK(x.dim() == 2, "x must be 2-D (M, H), got ", x.dim(), "D");
    TORCH_CHECK(w.dim() == 1, "w must be 1-D (H,), got ", w.dim(), "D");
    TORCH_CHECK(x.size(1) == w.size(0),
                "H mismatch: x.size(1)=", x.size(1), " w.size(0)=", w.size(0));
    TORCH_CHECK(x.dtype() == w.dtype(), "x and w must share dtype");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");

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
          "RMSNorm into a pre-allocated out tensor (benchmark path)",
          py::arg("variant"), py::arg("x"), py::arg("w"), py::arg("out"),
          py::arg("eps") = 1e-5);
    m.def("variants", &rmsnorm_variant_list, "list registered rmsnorm variants");
}
