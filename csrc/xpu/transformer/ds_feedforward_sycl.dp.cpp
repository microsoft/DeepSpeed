#include "common.hpp"
#include "context.hpp"
#include "feed_forward.hpp"

template <typename T>
std::vector<torch::Tensor> feedforward_forward(int bsz,
                                               int seq_len,
                                               int hidden_size,
                                               const torch::Tensor& input,
                                               const torch::Tensor& weights)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weights);

    int batchSize = bsz * seq_len;
    int inputSize = hidden_size;
    int outputSize = 3 * hidden_size;
    auto options = torch::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    const T* input_ptr = (const T*)input.data_ptr();
    const T* weights_ptr = (const T*)weights.data_ptr();

    auto output = torch::empty({bsz, seq_len, outputSize}, options);

    T* output_ptr = (T*)output.data_ptr();

    sycl::queue* q = ::SyclContext::Instance().GetCurrentStream();

    FeedForward<T> _ff =
        FeedForward<T>(typename FeedForward<T>::Config(batchSize, outputSize, inputSize));

    _ff.Forward(batchSize, input_ptr, weights_ptr, output_ptr, q);
    return {output};
}

template <typename T>
std::vector<torch::Tensor> feedforward_backward(int bsz,
                                                int seq_len,
                                                int hidden_size,
                                                const torch::Tensor& grad_out,
                                                const torch::Tensor& input,
                                                const torch::Tensor& weights)
{
    CHECK_INPUT(grad_out);
    CHECK_INPUT(input);
    CHECK_INPUT(weights);

    int batchSize = bsz * seq_len;
    int inputSize = hidden_size;
    int outputSize = 3 * hidden_size;

    auto options = torch::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    const T* grad_out_ptr = (const T*)grad_out.data_ptr();
    const T* input_ptr = (const T*)input.data_ptr();
    const T* weights_ptr = (const T*)weights.data_ptr();

    auto grad_weights = torch::empty(weights.sizes(), options);
    auto grad_bias = torch::empty({outputSize}, options);
    auto grad_input = torch::empty(input.sizes(), options);

    T* grad_w_ptr = (T*)grad_weights.data_ptr();
    T* grad_b_ptr = (T*)grad_bias.data_ptr();
    T* grad_i_ptr = (T*)grad_input.data_ptr();
    sycl::queue* q = ::SyclContext::Instance().GetCurrentStream();

    FeedForward<T> _ff =
        FeedForward<T>(typename FeedForward<T>::Config(batchSize, outputSize, inputSize));

    _ff.Backward(
        batchSize, grad_out_ptr, input_ptr, weights_ptr, grad_w_ptr, grad_b_ptr, q, q, grad_i_ptr);
    return {grad_input, grad_weights, grad_bias};
}
