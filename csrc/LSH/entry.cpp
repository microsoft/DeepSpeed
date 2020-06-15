#include <torch/extension.h>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <unordered_map>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

int create_lsh_layer(const torch::Tensor& weights) { return 0; }

std::vector<torch::Tensor> ds_lsh_forward(const torch::Tensor& input)
{
    // CHECK_INPUT(input);

    int bsz = input.size(0);

    auto output_options =
        torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).requires_grad(false);

    auto output = torch::ones({bsz, 256}, output_options);

    return {output};
}

std::vector<torch::Tensor> ds_lsh_backward(const torch::Tensor& grad_output,
                                           const torch::Tensor& input)
{
    auto g_output = grad_output.contiguous();
    // CHECK_INPUT(g_output);
    // CHECK_INPUT(input);

    auto grad_input = torch::empty_like(input);

    return {grad_input};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("create_lsh_layer", &create_lsh_layer, "Create DeepSpeed LSH Layer (CUDA)");
    m.def("forward", &ds_lsh_forward, "DeepSpeed LSH forward with (CUDA)");
    m.def("backward", &ds_lsh_backward, "DeepSpeed LSH backward with (CUDA)");
}
