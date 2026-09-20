// CUDALab RoPE — PyTorch 扩展绑定层。
//
// 纯 C++ 文件（用宿主编译器编译）: 保存变体注册表和 Python 入口点。
// 所有 CUDA 代码都在同目录的 .cu 文件中，每个文件自注册其变体。
//
// RoPE 入口: x (M,D) / positions (M,) int64 / cos_t (L,D/2) /
// sin_t (L,D/2) 与（可选的预分配）out (M,D)。约定为 interleaved
// pair（见 rope_common.h 头部）。
//
// 所有输入验证都在 kernel launch 之前以 TORCH_CHECK 完成（negative
// test suite 依赖这一点）; 内核自身不含设备端断言。
//
// validate 参数（默认 true）:
//   validate=true  —— 完整验证, 含 positions 值域检查。值域检查
//       （0 <= p < L）需要一次小的 D2H 同步拷贝（M 个 int64）——
//       **同步** D2H 会强制流同步, 使逐 launch 开销 ~25-30 us,
//       破坏基准流的 steady-state（v0.4 首跑 baseline 28.8 us 的
//       根源, 已修正）。正常 API 调用 / 测试 / negative 套件一律
//       用默认 true。
//   validate=false —— 仅做 host 侧元数据检查（dim/dtype/连续/设备,
//       不访问数据、不同步, 与 rmsnorm/softmax 的验证开销同级）,
//       供**预验证、预分配**的基准池使用: 池构造时所有张量已验证,
//       positions 为构造时生成的 0..M-1（必然 < L=4096）, 契约由
//       cudalab/operators/rope.py 的 make_bench_pool 文档化。

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>   // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <algorithm>
#include <string>
#include <unordered_map>

#include "rope_common.h"

namespace {
std::unordered_map<std::string, rope_fn_t>& registry() {
    static std::unordered_map<std::string, rope_fn_t> r;
    return r;
}

// ---- 输入验证（全部在 launch 前） ----------------------------------------

void validate_rope_x(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 2, "x 必须是 2 维 (M, D)，实际 ", x.dim(), " 维");
    TORCH_CHECK(x.size(0) > 0, "M 必须 > 0，实际 ", x.size(0));
    TORCH_CHECK(x.size(1) > 0, "D 必须 > 0，实际 ", x.size(1));
    TORCH_CHECK(x.size(1) % 2 == 0,
                "D 必须是偶数（interleaved pair 约定），实际 D=", x.size(1));
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat,
                "仅支持 float16 / float32，实际 ", x.dtype());
    TORCH_CHECK(x.is_contiguous(), "x 必须是连续内存");
    TORCH_CHECK(x.is_cuda(), "x 必须是 CUDA 张量");
}

void validate_rope_table(const at::Tensor& t, const char* name,
                         const at::Tensor& x) {
    TORCH_CHECK(t.dim() == 2,
                std::string(name) + " 必须是 2 维 (L, D/2)，实际 " +
                std::to_string(t.dim()) + " 维");
    TORCH_CHECK(t.size(0) > 0,
                std::string(name) + " 的行数 L（max_seq_len）必须 > 0，实际 " +
                std::to_string(t.size(0)));
    TORCH_CHECK(t.size(1) == x.size(1) / 2,
                std::string(name) + " 的列数必须等于 D/2 = " +
                std::to_string(x.size(1) / 2) + "，实际 " +
                std::to_string(t.size(1)));
    TORCH_CHECK(t.dtype() == x.dtype(),
                std::string(name) + " 的 dtype 必须与 x 一致: ",
                t.dtype(), " vs ", x.dtype());
    TORCH_CHECK(t.is_contiguous(), std::string(name) + " 必须是连续内存");
    TORCH_CHECK(t.is_cuda(), std::string(name) + " 必须是 CUDA 张量");
    TORCH_CHECK(t.device() == x.device(),
                std::string(name) + " 必须与 x 在同一 CUDA 设备上");
}

// positions 的 host 元数据检查（不访问数据、不同步, 廉价, 始终执行）
void validate_rope_positions_meta(const at::Tensor& positions,
                                  const at::Tensor& x) {
    TORCH_CHECK(positions.dim() == 1,
                "positions 必须是 1 维 (M,)，实际 ", positions.dim(), " 维");
    TORCH_CHECK(positions.size(0) == x.size(0),
                "positions 长度必须等于 M: positions=", positions.size(0),
                " M=", x.size(0));
    TORCH_CHECK(positions.dtype() == at::kLong,
                "positions 必须是 int64，实际 ", positions.dtype());
    TORCH_CHECK(positions.is_contiguous(), "positions 必须是连续内存");
    TORCH_CHECK(positions.is_cuda(), "positions 必须是 CUDA 张量");
    TORCH_CHECK(positions.device() == x.device(),
                "positions 必须与 x 在同一 CUDA 设备上");
}

// positions 值域检查 0 <= p < L（launch 前的小 D2H 同步拷贝, 不是
// 设备端断言）。**同步** D2H 强制流同步, 逐 launch ~25-30 us, 因此
// 只在 validate=true 时执行; 基准池路径（validate=false）由池构造
// 契约保证 positions = 0..M-1 < L。
void validate_rope_positions_range(const at::Tensor& positions,
                                   const at::Tensor& cos_t) {
    auto pos_cpu = positions.to(at::kCPU);
    const int64_t* p = pos_cpu.const_data_ptr<int64_t>();
    const int64_t M = positions.size(0);
    int64_t mn = 0, mx = -1;
    for (int64_t i = 0; i < M; ++i) {
        mn = std::min(mn, p[i]);
        mx = std::max(mx, p[i]);
    }
    TORCH_CHECK(mn >= 0, "position 必须 >= 0，实际最小值 ", mn);
    TORCH_CHECK(mx < cos_t.size(0),
                "position 必须 < max_seq_len（cos 表行数）= ", cos_t.size(0),
                "，实际最大值 ", mx);
}

void validate_rope_inputs(const at::Tensor& x, const at::Tensor& positions,
                          const at::Tensor& cos_t, const at::Tensor& sin_t,
                          bool check_range) {
    validate_rope_x(x);
    validate_rope_table(cos_t, "cos", x);
    validate_rope_table(sin_t, "sin", x);
    TORCH_CHECK(cos_t.sizes().equals(sin_t.sizes()),
                "cos 与 sin 的形状必须一致: cos=", cos_t.sizes(),
                " sin=", sin_t.sizes());
    validate_rope_positions_meta(positions, x);
    if (check_range) {
        validate_rope_positions_range(positions, cos_t);
    }
}

void validate_rope_out(const at::Tensor& x, const at::Tensor& out) {
    TORCH_CHECK(out.dim() == 2,
                "out 必须是 2 维 (M, D)，实际 ", out.dim(), " 维");
    TORCH_CHECK(out.sizes().equals(x.sizes()),
                "out 的形状必须与 x 一致: out=", out.sizes(),
                " x=", x.sizes());
    TORCH_CHECK(out.dtype() == x.dtype(),
                "out 的 dtype 必须与 x 一致: out=", out.dtype(),
                " x=", x.dtype());
    TORCH_CHECK(out.is_contiguous(), "out 必须是连续内存");
    TORCH_CHECK(out.is_cuda(), "out 必须是 CUDA 张量");
    TORCH_CHECK(out.device() == x.device(),
                "out 必须与 x 在同一 CUDA 设备上");
}
}  // namespace

void register_rope_variant(const std::string& name, rope_fn_t fn) {
    auto& r = registry();
    if (r.count(name)) {
        TORCH_CHECK(false, "重复的 rope 变体: ", name);
    }
    r.emplace(name, fn);
}

// 写入预分配、布局兼容的 `out`（基准测试用它把内存分配排除在
// 计时区域之外）。与 forward 共享验证逻辑；validate 语义见文件头。
// launch 之后立即检查启动错误。
void rope_forward_into(const std::string& name, const at::Tensor& x,
                       const at::Tensor& positions,
                       const at::Tensor& cos_t, const at::Tensor& sin_t,
                       at::Tensor& out, bool validate) {
    auto& r = registry();
    auto it = r.find(name);
    TORCH_CHECK(it != r.end(), "未知 rope 变体 '", name, "'");
    // 元数据检查始终执行（廉价, 无同步）; 仅 positions 值域 D2H
    // 受 validate 门控（见 validate_rope_inputs 与文件头）。
    validate_rope_inputs(x, positions, cos_t, sin_t, validate);
    validate_rope_out(x, out);
    c10::cuda::CUDAGuard guard(x.device());
    it->second(x, positions, cos_t, sin_t, out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor rope_forward(const std::string& name, const at::Tensor& x,
                        const at::Tensor& positions,
                        const at::Tensor& cos_t, const at::Tensor& sin_t,
                        bool validate) {
    auto& r = registry();
    auto it = r.find(name);
    TORCH_CHECK(it != r.end(),
                "未知 rope 变体 '", name,
                "'。正常可用: ", [&] {
                    std::string s;
                    for (auto& n : rope_variant_list()) s += n + " ";
                    return s;
                }());
    // 元数据检查始终执行; positions 值域 D2H 受 validate 门控。
    validate_rope_inputs(x, positions, cos_t, sin_t, validate);
    at::Tensor out = at::empty_like(x);
    c10::cuda::CUDAGuard guard(x.device());
    it->second(x, positions, cos_t, sin_t, out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<std::string> rope_variant_list() {
    std::vector<std::string> names;
    for (auto& kv : registry()) names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    return names;
}

std::vector<std::string> rope_all_variant_list() {
    return rope_variant_list();  // 当前无被隔离变体
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rope_forward,
          "RoPE (interleaved pair): y[2i]=a*c-b*s, y[2i+1]=a*s+b*c, "
          "FP32 中间, 输出 dtype = x dtype。validate=false 仅供预验证"
          "基准池使用（跳过 positions 值域 D2H 检查，见文件头契约）。",
          py::arg("variant"), py::arg("x"), py::arg("positions"),
          py::arg("cos"), py::arg("sin"), py::arg("validate") = true);
    m.def("forward_into", &rope_forward_into,
          "RoPE，写入预分配的输出张量（基准测试路径）。validate 语义"
          "同 forward。",
          py::arg("variant"), py::arg("x"), py::arg("positions"),
          py::arg("cos"), py::arg("sin"), py::arg("out"),
          py::arg("validate") = true);
    m.def("variants", &rope_variant_list,
          "正常（可 dispatch）的 rope 变体列表");
    m.def("all_variants", &rope_all_variant_list,
          "全部已注册 rope 变体");
}
