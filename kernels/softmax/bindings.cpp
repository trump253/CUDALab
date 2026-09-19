// CUDALab Softmax — PyTorch 扩展绑定层。
//
// 纯 C++ 文件（用宿主编译器编译）: 保存变体注册表和 Python 入口点。
// 所有 CUDA 代码都在同目录的 .cu 文件中，每个文件自注册其变体。
//
// Softmax 无辅助张量: 入口只有 x (M,H) 与（可选的预分配）out (M,H)。

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>   // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <unordered_map>
#include <mutex>

#include "softmax_common.h"

namespace {
std::unordered_map<std::string, softmax_fn_t>& registry() {
    static std::unordered_map<std::string, softmax_fn_t> r;
    return r;
}

// v0.3: forward 与 forward_into 共享同一套输入验证，
// 避免维护两套可能漂移的逻辑。所有检查都在 kernel launch 之前完成，
// 非法输入以明确异常拒绝（negative test suite 依赖这一点）。
void validate_common(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 2, "x 必须是 2 维 (M, H)，实际 ", x.dim(), " 维");
    TORCH_CHECK(x.size(0) > 0, "M 必须 > 0，实际 ", x.size(0));
    TORCH_CHECK(x.size(1) > 0, "H 必须 > 0，实际 ", x.size(1));
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat,
                "仅支持 float16 / float32，实际 ", x.dtype());
    TORCH_CHECK(x.is_contiguous(), "x 必须是连续内存");
    TORCH_CHECK(x.is_cuda(), "x 必须是 CUDA 张量");
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

void register_softmax_variant(const std::string& name, softmax_fn_t fn) {
    auto& r = registry();
    if (r.count(name)) {
        TORCH_CHECK(false, "重复的 softmax 变体: ", name);
    }
    r.emplace(name, fn);
}

// 写入预分配、布局兼容的 `out`（基准测试用它把内存分配排除在
// 计时区域之外）。与 forward 共享完整验证，launch 之后立即检查
// 启动错误。
void softmax_forward_into(const std::string& name, const at::Tensor& x,
                          at::Tensor& out) {
    auto& r = registry();
    auto it = r.find(name);
    TORCH_CHECK(it != r.end(), "未知 softmax 变体 '", name, "'");
    validate_common(x);
    validate_out(x, out);
    c10::cuda::CUDAGuard guard(x.device());
    it->second(x, out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor softmax_forward(const std::string& name, const at::Tensor& x) {
    auto& r = registry();
    auto it = r.find(name);
    TORCH_CHECK(it != r.end(),
                "未知 softmax 变体 '", name,
                "'。可用: ", [&] {
                    std::string s;
                    for (auto& kv : r) s += kv.first + " ";
                    return s;
                }());
    validate_common(x);
    at::Tensor out = at::empty_like(x);
    c10::cuda::CUDAGuard guard(x.device());
    it->second(x, out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<std::string> softmax_variant_list() {
    std::vector<std::string> names;
    for (auto& kv : registry()) names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    return names;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_forward,
          "Softmax: y = exp(x - rowmax) / sum(exp(x - rowmax)), dim=-1",
          py::arg("variant"), py::arg("x"));
    m.def("forward_into", &softmax_forward_into,
          "Softmax，写入预分配的输出张量（基准测试路径）",
          py::arg("variant"), py::arg("x"), py::arg("out"));
    m.def("variants", &softmax_variant_list, "列出已注册的 softmax 变体");
}
