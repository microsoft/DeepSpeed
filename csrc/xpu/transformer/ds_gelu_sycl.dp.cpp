#include "common.hpp"
#include "context.hpp"
#include "gelu.hpp"

template <typename T>
std::vector<torch::Tensor> gelu_forward(int intermediate_size,
                                        int bsz_seq,
                                        const torch::Tensor& input,
                                        const torch::Tensor& bias)
{
    CHECK_INPUT(input);
    CHECK_INPUT(bias);
    const T* input_ptr = (const T*)input.data_ptr();
    const T* bias_ptr = (const T*)bias.data_ptr();
    auto output = torch::empty_like(input);
    T* output_ptr = (T*)output.data_ptr();
    sycl::queue* q = ::SyclContext::Instance().GetCurrentStream();
    Gelu<T> _gelu = Gelu<T>(typename Gelu<T>::Config(intermediate_size));
    _gelu.ForwardWithBiasAdd(bsz_seq, input_ptr, bias_ptr, output_ptr, q);
    return {output};
}

template <typename T>
std::vector<torch::Tensor> gelu_backward(torch::Tensor& d_output,
                                         int intermediate_size,
                                         int bsz_seq,
                                         const torch::Tensor& input,
                                         const torch::Tensor& bias)
{
    CHECK_INPUT(input);
    CHECK_INPUT(bias);
    const T* input_ptr = (const T*)input.data_ptr();
    const T* bias_ptr = (const T*)bias.data_ptr();
    T* d_output_ptr = (T*)d_output.data_ptr();
    sycl::queue* q = ::SyclContext::Instance().GetCurrentStream();
    Gelu<T> _gelu = Gelu<T>(typename Gelu<T>::Config(intermediate_size));
    _gelu.Backward(bsz_seq, d_output_ptr, input_ptr, bias_ptr, q);
    return {d_output};
}
