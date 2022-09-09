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


std::vector<torch::Tensor> ds_quantizer(torch::Tensor& A, int num_bits)
{
    int groups = 1;
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
    auto min = torch::empty({min_max_size}, float_options);
    auto max = torch::empty({min_max_size}, float_options);
    auto Q_A = torch::empty_like(A, int8_options);
    // auto scale = torch::empty({groups}, float_options);
    // auto zero_point = torch::empty({groups}, float_options);
    auto scale = torch::tensor(0, float_options);
    auto zero_point = torch::tensor(0, float_options);


    quantize_kernel((int8_t*)Q_A.data_ptr(),
                (__half*)A.data_ptr(),
                (__half*)min.data_ptr(),
                (__half*)max.data_ptr(),
                (float*)scale.data_ptr(),
                (float*)zero_point.data_ptr(),
                // groups,
                input_size,
                num_bits,
                at::cuda::getCurrentCUDAStream());

    // if (groups < M)
    // {
    //     // printf("groups < M\n");
    //     quantize_kernel1((int8_t*)Q_A.data_ptr(),
    //                 (__half*)A.data_ptr(),
    //                 (__half*)min.data_ptr(),
    //                 (__half*)max.data_ptr(),
    //                 (float*)scale.data_ptr(),
    //                 (float*)zero_point.data_ptr(),
    //                 // groups,
    //                 input_size,
    //                 num_bits,
    //                 at::cuda::getCurrentCUDAStream());
    // }
    // else{
    //     // printf("groups >= M\n");
    //     quantize_kernel((int8_t*)Q_A.data_ptr(),
    //                 (__half*)A.data_ptr(),
    //                 (__half*)min.data_ptr(),
    //                 (__half*)max.data_ptr(),
    //                 (float*)scale.data_ptr(),
    //                 (float*)zero_point.data_ptr(),
    //                 // groups,
    //                 input_size,
    //                 num_bits,
    //                 at::cuda::getCurrentCUDAStream());
    // }
    return {Q_A, zero_point,scale};
}


torch::Tensor ds_dequantizer(torch::Tensor& A,torch::Tensor& zero_point,torch::Tensor& scale, std::string dtype)
{
    unsigned input_size = 1;
    for (auto& s : A.sizes()) input_size *= s;

    torch::Dtype dtype_;
    if (dtype == "torch.float32" or dtype == "torch.cuda.FloatTensor")
    {
        dtype_=torch::kFloat;
    }
    else if (dtype == "torch.float16")
    {
        dtype_=torch::kHalf;
    }
    else
    {
        throw std::runtime_error("Got unsupported dtype:"+dtype);
    }

    auto dequantize_options = torch::TensorOptions()
                            .dtype(dtype_)
                            .layout(torch::kStrided)
                            .device(torch::kCUDA)
                            .requires_grad(false);

    auto Q_A = torch::empty_like(A, dequantize_options);
    if (dtype_ == torch::kFloat)
    {
        throw std::runtime_error("Got unsupported dtype:"+dtype);
        // dequantize_kernel((int8_t*)Q_A.data_ptr(),
        //             (float*)A.data_ptr(),
        //             (float*)scale.data_ptr(),
        //             (float*)zero_point.data_ptr(),
        //             input_size,
        //             at::cuda::getCurrentCUDAStream());
    }
    else if (dtype_ == torch::kHalf)
    {
        dequantize_kernel((int8_t*)A.data_ptr(),
                    (__half*)Q_A.data_ptr(),
                    (float*)scale.data_ptr(),
                    (float*)zero_point.data_ptr(),
                    input_size,
                    at::cuda::getCurrentCUDAStream());
    }
    else
    {
        throw std::runtime_error("Got unsupported dtype:"+dtype);
    }
    // dequantize_kernel((int8_t*)A.data_ptr(),
    //             (__half*)Q_A.data_ptr(),
    //             (float*)scale.data_ptr(),
    //             (float*)zero_point.data_ptr(),
    //             input_size,
    //             at::cuda::getCurrentCUDAStream());

    return Q_A;
}



torch::Tensor ds_dequantizer_chunks(torch::Tensor& A,torch::Tensor& zero_point_stats,torch::Tensor& scale_stats, std::string dtype)
{
    unsigned input_size = 1;
    for (auto& s : A.sizes()) input_size *= s;

    unsigned quant_stat_size = 1;
    for (auto& s : zero_point_stats.sizes()) quant_stat_size *= s;
    unsigned num_chunks = quant_stat_size;
    unsigned numel_per_chunk = input_size / num_chunks;


    torch::Dtype dtype_;
    if (dtype == "torch.float32" or dtype == "torch.cuda.FloatTensor")
    {
        dtype_=torch::kFloat;
    }
    else if (dtype == "torch.float16")
    {
        dtype_=torch::kHalf;
    }
    else
    {
        throw std::runtime_error("Got unsupported dtype:"+dtype);
    }

    auto dequantize_options = torch::TensorOptions()
                            .dtype(dtype_)
                            .layout(torch::kStrided)
                            .device(torch::kCUDA)
                            .requires_grad(false);

    auto Q_A = torch::empty_like(A, dequantize_options);


    dequantize_chunks_kernel((int8_t*)A.data_ptr(),
                (__half*)Q_A.data_ptr(),
                (float*)zero_point_stats.data_ptr(),
                (float*)scale_stats.data_ptr(),
                input_size,
                numel_per_chunk,
                at::cuda::getCurrentCUDAStream());

    return Q_A;
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
    m.def("ds_dequantizer",
          &ds_dequantizer,
          "DeepSpeed Dequantize with fp16 (CUDA)");
    m.def("ds_dequantizer_chunks",
          &ds_dequantizer_chunks,
          "DeepSpeed Dequantize with fp16 (CUDA) with respect to chunks");
}
