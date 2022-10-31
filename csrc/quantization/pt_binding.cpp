#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>
#include "custom_cuda_layers.h"
#include "quantization.h"

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

std::vector<at::Tensor> quantize_kernel(at::Tensor& input_vals,
                                     int groups,
                                     int numBits,
                                     quantize::Type quantType)
{
    auto params_options = at::TensorOptions()
                              .dtype(at::kFloat)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);
    auto scales = torch::empty({groups}, params_options);
    auto offsets = torch::empty({groups}, params_options);

    auto output_options = at::TensorOptions()
                              .dtype(at::kChar)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);
    int K = input_vals.size(1) / (numBits == 8 ? 1 : 2);
    int M = input_vals.size(0);

    auto output = torch::empty({M, K}, output_options);

    const int elems_per_group = at::numel(input_vals) / groups;

    if (numBits == 4) {
        if (quantType == quantize::Type::Symmetric) {
            launch_act_quant<4, quantize::Type::Symmetric>((int8_t*)output.data_ptr(),
                                                            (float*)scales.data_ptr(),
                                                            (float*)offsets.data_ptr(),
                                                            (__half*)input_vals.data_ptr(),
                                                            groups,
                                                            elems_per_group,
                                                            at::cuda::getCurrentCUDAStream());
        } else {
            launch_act_quant<4, quantize::Type::Asymmetric>((int8_t*)output.data_ptr(),
                                                             (float*)scales.data_ptr(),
                                                             (float*)offsets.data_ptr(),
                                                             (__half*)input_vals.data_ptr(),
                                                             groups,
                                                             elems_per_group,
                                                             at::cuda::getCurrentCUDAStream());
        }
    } else {
        if (quantType == quantize::Type::Symmetric) {
            launch_act_quant<8, quantize::Type::Symmetric>((int8_t*)output.data_ptr(),
                                                            (float*)scales.data_ptr(),
                                                            (float*)offsets.data_ptr(),
                                                            (__half*)input_vals.data_ptr(),
                                                            groups,
                                                            elems_per_group,
                                                            at::cuda::getCurrentCUDAStream());
        } else {
            launch_act_quant<8, quantize::Type::Asymmetric>((int8_t*)output.data_ptr(),
                                                             (float*)scales.data_ptr(),
                                                             (float*)offsets.data_ptr(),
                                                             (__half*)input_vals.data_ptr(),
                                                             groups,
                                                             elems_per_group,
                                                             at::cuda::getCurrentCUDAStream());
        }
    }

    return {output, scales, offsets};
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
    pybind11::enum_<quantize::Type>(m, "QuantizationType")
    .value("Symmetric", quantize::Type::Symmetric)
    .value("Asymmetric", quantize::Type::Asymmetric)
    .export_values();
    m.def("quantize", &quantize_kernel);
}
