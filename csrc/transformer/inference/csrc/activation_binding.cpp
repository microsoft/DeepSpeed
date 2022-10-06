#include <torch/extension.h>
#include "inference_context.h"
#include "inference_cuda_layers.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;


template <typename T>
at::Tensor ds_bias_gelu(at::Tensor& input, at::Tensor& bias)
{
    auto input_cont = input.contiguous();

    int bsz = input_cont.size(0) * input_cont.size(1);
    int intermediate_size = input_cont.size(2);

    launch_bias_gelu((T*)input_cont.data_ptr(),
                     (T*)bias.data_ptr(),
                     intermediate_size,
                     bsz,
                     Context::Instance().GetCurrentStream());
    return input_cont;
}

void ds_bias_gelu_fp32_bind(py::module_ &m){
    m.def("bias_gelu_fp32", &ds_bias_gelu<float>, "DeepSpeed Gelu with fp32 (CUDA)");
}

void ds_bias_gelu_fp16_bind(py::module_ &m){
    m.def("bias_gelu_fp16", &ds_bias_gelu<__half>, "DeepSpeed Gelu with fp32 (CUDA)");
}

#ifdef ACTIVATION_UNITTEST
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    ds_bias_gelu_fp32_bind(m);
    ds_bias_gelu_fp16_bind(m);
}
#endif