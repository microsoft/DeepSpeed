#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>
#include "custom_cuda_layers.h"

template <typename T>
at::Tensor ds_quantize(at::Tensor& vals, int groups, int bits)
{
    auto t_size = vals.sizes();
    int size = 1;
    for (auto dim : t_size) size *= dim;

    if ((((size / groups) - 1) / 4096 + 1) <= MAX_REG) {
        launch_quantize_kernel(
            (T*)vals.data_ptr(), size, groups, bits, at::cuda::getCurrentCUDAStream());
    }
    return vals;
}

template <typename T>
at::Tensor ds_sr_quantize(at::Tensor& vals, int groups, int bits)
{
    auto t_size = vals.sizes();
    int size = 1;
    for (auto dim : t_size) size *= dim;

    if (((size / groups) / 4 / 1024) <= 256) {
        launch_sr_quantize_kernel(
            (T*)vals.data_ptr(), size, groups, bits, at::cuda::getCurrentCUDAStream());
    }
    return vals;
}

template <typename T>
at::Tensor ds_quantize_asym(at::Tensor& vals, int groups, int bits)
{
    auto t_size = vals.sizes();
    int size = 1;
    for (auto dim : t_size) size *= dim;

    if ((((size / groups) - 1) / 4096 + 1) <= MAX_REG) {
        launch_quantize_kernel_asym(
            (T*)vals.data_ptr(), size, groups, bits, at::cuda::getCurrentCUDAStream());
    }
    return vals;
}

template <typename T>
at::Tensor ds_sr_quantize_asym(at::Tensor& vals, int groups, int bits)
{
    auto t_size = vals.sizes();
    int size = 1;
    for (auto dim : t_size) size *= dim;

    if (((size / groups) / 4 / 1024) <= 256) {
        launch_sr_quantize_kernel_asym(
            (T*)vals.data_ptr(), size, groups, bits, at::cuda::getCurrentCUDAStream());
    }
    return vals;
}


std::vector<torch::Tensor> ds_quantizer(torch::Tensor& A, int num_bits, int groups)
{
    int M = 1;
    for (auto& s : A.sizes()) M *= s;
    int K = A.size(A.sizes().size() - 1);
    M = M / K;
    unsigned input_size = (M * K);
    unsigned min_max_size = (input_size - 1) / 4096 + 1;

    auto int8_options = torch::TensorOptions()
                            .dtype(torch::kInt8)
                            .layout(torch::kStrided)
                            .device(torch::kCUDA)
                            .requires_grad(false);
    auto float_options = torch::TensorOptions()
                             .dtype(torch::kFloat)
                             .layout(torch::kStrided)
                             .device(torch::kCUDA)
                             .requires_grad(false);
    auto min_max = torch::empty({min_max_size}, float_options);
    auto Q_A = torch::empty({M, K}, int8_options);
    auto groups_tensor = torch::empty({groups}, float_options);

    if (A.scalar_type() == at::kFloat)
        quantize_kernel((int8_t*)Q_A.data_ptr(),
                        (float*)A.data_ptr(),
                        (float*)min_max.data_ptr(),
                        (float*)groups_tensor.data_ptr(),
                        groups,
                        input_size,
                        num_bits,
                        at::cuda::getCurrentCUDAStream());
    else
        if (groups < M)
        {
            quantize_kernel1((int8_t*)Q_A.data_ptr(),
                        (__half*)A.data_ptr(),
                        (__half*)min_max.data_ptr(),
                        (float*)groups_tensor.data_ptr(),
                        groups,
                        input_size,
                        num_bits,
                        at::cuda::getCurrentCUDAStream());
        }
        else{
            quantize_kernel((int8_t*)Q_A.data_ptr(),
                        (__half*)A.data_ptr(),
                        (__half*)min_max.data_ptr(),
                        (float*)groups_tensor.data_ptr(),
                        groups,
                        input_size,
                        num_bits,
                        at::cuda::getCurrentCUDAStream());
        }
    return {Q_A, groups_tensor};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ds_quantize_fp32", &ds_quantize<float>, "DeepSpeed Quantize with fp32 (CUDA)");
    m.def("ds_quantize_fp16", &ds_quantize<__half>, "DeepSpeed Quantize with fp16 (CUDA)");
    m.def("ds_sr_quantize_fp32", &ds_sr_quantize<float>, "DeepSpeed Quantize with fp32 (CUDA)");
    m.def("ds_sr_quantize_fp16", &ds_sr_quantize<__half>, "DeepSpeed Quantize with fp16 (CUDA)");
    m.def("ds_quantize_asym_fp32", &ds_quantize_asym<float>, "DeepSpeed Quantize with fp32 (CUDA)");
    m.def(
        "ds_quantize_asym_fp16", &ds_quantize_asym<__half>, "DeepSpeed Quantize with fp16 (CUDA)");
    m.def("ds_sr_quantize_asym_fp32",
          &ds_sr_quantize_asym<float>,
          "DeepSpeed Quantize with fp32 (CUDA)");
    m.def("ds_sr_quantize_asym_fp16",
          &ds_sr_quantize_asym<__half>,
          "DeepSpeed Quantize with fp16 (CUDA)");
    m.def("ds_quantizer",
          &ds_quantizer,
          "DeepSpeed Quantize with fp16 (CUDA)");
}
