
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <vector>
#include "deepspeed_apis.h"

void bias_gelu(torch::Tensor& x, torch::Tensor& bias)
{
    bool is_float = (x.scalar_type() == at::kFloat);
    auto sizes = x.sizes();
    unsigned bsz = 1;
    for (auto size : sizes) bsz *= size;
    DeepSpeedAPI::bias_gelu(x.data_ptr(),
                            bias.data_ptr(),
                            bsz / x.size(sizes.size() - 1),
                            x.size(sizes.size() - 1),
                            is_float,
                            at::cuda::getStreamFromPool());
    // at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bias_gelu", &bias_gelu, "DeepSpeed bias_gelu forward(CUDA)");
}
