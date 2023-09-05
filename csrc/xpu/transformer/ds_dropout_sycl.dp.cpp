#include "common.hpp"
#include "context.hpp"
#include "dropout.hpp"

template <typename T>
std::vector<torch::Tensor> dropout_forward(float ratio,
                                           uint32_t dim,
                                           int bsz,
                                           const torch::Tensor& vals)
{
    CHECK_INPUT(vals);
    auto output = torch::empty_like(vals);

    auto uint8_options = torch::TensorOptions()
                             .dtype(torch::kInt8)
                             .layout(torch::kStrided)
                             .device(torch::kXPU)
                             .requires_grad(false);

    auto mask = torch::empty({bsz, dim}, uint8_options);

    const T* input_ptr = (const T*)vals.data_ptr();
    T* output_ptr = (T*)output.data_ptr();
    uint8_t* mask_ptr = (uint8_t*)mask.data_ptr();

    sycl::queue* q = ::SyclContext::Instance().GetCurrentStream();
    Dropout<T> _dropout = Dropout<T>(typename Dropout<T>::Config(ratio, dim));
    _dropout.SetMask(mask_ptr);
    _dropout.Forward(bsz, output_ptr, input_ptr, q);
    return {output, mask};
}

template <typename T>
std::vector<torch::Tensor> dropout_forward_with_bias(float ratio,
                                                     uint32_t dim,
                                                     int bsz,
                                                     const torch::Tensor& vals,
                                                     const torch::Tensor& bias,
                                                     const torch::Tensor& residual)
{
    CHECK_INPUT(vals);
    CHECK_INPUT(bias);
    CHECK_INPUT(residual);
    auto output = torch::empty_like(vals);

    auto uint8_options = torch::TensorOptions()
                             .dtype(torch::kInt8)
                             .layout(torch::kStrided)
                             .device(torch::kXPU)
                             .requires_grad(false);

    auto mask = torch::empty({bsz, dim}, uint8_options);

    const T* input_ptr = (const T*)vals.data_ptr();
    const T* bias_ptr = (const T*)bias.data_ptr();
    const T* residual_ptr = (const T*)residual.data_ptr();
    T* output_ptr = (T*)output.data_ptr();
    uint8_t* mask_ptr = (uint8_t*)mask.data_ptr();

    sycl::queue* q = ::SyclContext::Instance().GetCurrentStream();
    Dropout<T> _dropout = Dropout<T>(typename Dropout<T>::Config(ratio, dim));
    _dropout.SetMask(mask_ptr);
    _dropout.ForwardWithBias(bsz, output_ptr, input_ptr, residual_ptr, bias_ptr, q);
    return {output, mask};
}

template <typename T>
std::vector<torch::Tensor> dropout_backward(float ratio,
                                            uint32_t dim,
                                            int bsz,
                                            torch::Tensor& vals,
                                            torch::Tensor& mask,
                                            bool in_place)
{
    CHECK_INPUT(vals);
    CHECK_INPUT(mask);
    sycl::queue* q = ::SyclContext::Instance().GetCurrentStream();
    Dropout<T> _dropout = Dropout<T>(typename Dropout<T>::Config(ratio, dim));
    uint8_t* mask_ptr = (uint8_t*)mask.data_ptr();
    _dropout.SetMask(mask_ptr);
    if (in_place) {
        T* d_input_ptr = (T*)vals.data_ptr();
        _dropout.Backward(bsz, d_input_ptr, q);
        return {vals};
    } else {
        auto output = torch::empty_like(vals);
        const T* d_input_ptr = (const T*)vals.data_ptr();
        T* d_output_ptr = (T*)output.data_ptr();
        _dropout.Backward(bsz, d_output_ptr, d_input_ptr, q);
        return {output};
    }
}
