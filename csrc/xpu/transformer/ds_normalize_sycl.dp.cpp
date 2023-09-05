#include "common.hpp"
#include "context.hpp"
#include "normalize_layer.hpp"

template <typename T>
std::vector<torch::Tensor> normalize_forward(const int batch,
                                             const int seq_len,
                                             const int hidden_size,
                                             const torch::Tensor& residual,
                                             const torch::Tensor& gamma,
                                             const torch::Tensor& betta,
                                             torch::Tensor& mean,
                                             torch::Tensor& var,
                                             const bool preln,
                                             const bool wmean,
                                             const float epsilon)
{
    CHECK_INPUT(residual);
    CHECK_INPUT(gamma);
    CHECK_INPUT(betta);

    int bsz_seq = batch * seq_len;

    auto options = torch::TensorOptions()
                       .dtype(residual.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    auto output = torch::empty({batch, seq_len, hidden_size}, options);

    T* output_ptr = (T*)output.data_ptr();
    T* mean_ptr = (T*)mean.data_ptr();
    T* var_ptr = (T*)var.data_ptr();
    const T* residual_ptr = (const T*)residual.data_ptr();
    const T* gamma_ptr = (const T*)gamma.data_ptr();
    const T* betta_ptr = (const T*)betta.data_ptr();

    sycl::queue* q = ::SyclContext::Instance().GetCurrentStream();
    Normalize_Layer<T> _norm(
        typename Normalize_Layer<T>::Config(batch, seq_len, hidden_size, epsilon, true, wmean));
    _norm.SetMean(mean_ptr);
    _norm.SetVar(var_ptr);

    if (wmean)
        _norm.ForwardCheckpoint(bsz_seq, output_ptr, residual_ptr, gamma_ptr, betta_ptr, q);
    else
        _norm.Forward(bsz_seq, output_ptr, residual_ptr, gamma_ptr, betta_ptr, q);
    return {output};
}

template <typename T>
std::vector<torch::Tensor> normalize_backward(const int batch,
                                              const int seq_len,
                                              const int hidden_size,
                                              const torch::Tensor& input,
                                              const torch::Tensor& gamma,
                                              const torch::Tensor& betta,
                                              const torch::Tensor& output,
                                              const torch::Tensor& out1_grad,
                                              const torch::Tensor& out2_grad,
                                              torch::Tensor& mean,
                                              torch::Tensor& var,
                                              const bool preln,
                                              const bool wmean,
                                              const float epsilon)
{
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(out1_grad);
    CHECK_INPUT(out2_grad);
    CHECK_INPUT(gamma);
    CHECK_INPUT(betta);
    int bsz_seq = batch * seq_len;

    auto options = torch::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    auto gamma_grad = torch::empty({hidden_size}, options);
    auto betta_grad = torch::empty({hidden_size}, options);
    auto input_grad = torch::empty({batch, seq_len, hidden_size}, options);

    const T* input_ptr = (const T*)input.data_ptr();
    const T* out1_grad_ptr = (const T*)out1_grad.data_ptr();
    const T* out2_grad_ptr = (const T*)out2_grad.data_ptr();
    const T* gamma_ptr = (const T*)gamma.data_ptr();
    const T* betta_ptr = (const T*)betta.data_ptr();
    const T* output_ptr = (const T*)output.data_ptr();
    T* gamma_grad_ptr = (T*)gamma_grad.data_ptr();
    T* betta_grad_ptr = (T*)betta_grad.data_ptr();
    T* inp_grad_ptr = (T*)input_grad.data_ptr();
    T* mean_ptr = (T*)mean.data_ptr();
    T* var_ptr = (T*)var.data_ptr();
    sycl::queue* q = ::SyclContext::Instance().GetCurrentStream();

    Normalize_Layer<T> _norm(
        typename Normalize_Layer<T>::Config(batch, seq_len, hidden_size, epsilon, true, wmean));
    sycl::queue* qs[2] = {q, q};

    _norm.SetMean(mean_ptr);
    _norm.SetVar(var_ptr);

    if (preln) {
        if (wmean)
            _norm.BackwardFusedAdd(bsz_seq,
                                   out1_grad_ptr,
                                   out2_grad_ptr,
                                   gamma_ptr,
                                   gamma_grad_ptr,
                                   betta_grad_ptr,
                                   qs,
                                   inp_grad_ptr,
                                   input_ptr);
        else
            _norm.BackwardFusedAdd(bsz_seq,
                                   out1_grad_ptr,
                                   out2_grad_ptr,
                                   gamma_ptr,
                                   betta_ptr,
                                   gamma_grad_ptr,
                                   betta_grad_ptr,
                                   qs,
                                   inp_grad_ptr,
                                   output_ptr);
    } else {
        if (wmean)
            _norm.Backward(bsz_seq,
                           out1_grad_ptr,
                           gamma_ptr,
                           gamma_grad_ptr,
                           betta_grad_ptr,
                           qs,
                           inp_grad_ptr,
                           input_ptr);
        else {
            _norm.Backward(bsz_seq,
                           out1_grad_ptr,
                           gamma_ptr,
                           betta_ptr,
                           gamma_grad_ptr,
                           betta_grad_ptr,
                           qs,
                           inp_grad_ptr,
                           output_ptr);
        }
    }
    return {input_grad, gamma_grad, betta_grad};
}
