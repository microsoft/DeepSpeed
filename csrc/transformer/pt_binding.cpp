#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>
#include "context.h"
#include "custom_cuda_layers.h"

at::Tensor ds_softmax_dropout(at::Tensor& attn_scores,
                              at::Tensor& attn_mask,
                              float ratio,
                              bool bidirectional,
                              c10::optional<at::Generator> gen_,
                              at::Tensor& rel_pos,
                              int heads)
{
    auto attn_scores_c = attn_scores.contiguous();
    auto out = at::empty_like(attn_scores_c);
    int bsz = attn_scores_c.size(0);

    int seq_len = attn_scores_c.size(1);
    int len = attn_scores_c.sizes().size();
    if (len > 3) seq_len = attn_scores_c.size(2);

    int soft_len = attn_scores_c.size(2);
    if (len > 3) soft_len = attn_scores_c.size(3);

    // int heads = 1;
    // if (len > 3) heads = attn_scores_c.size(1);
    // Context::Instance().extract_rng_seed_offset(gen_);
    if (attn_scores_c.scalar_type() == at::kFloat)
        launch_softmax_dropout((float*)out.data_ptr(),
                               (float*)attn_scores_c.data_ptr(),
                               (bidirectional ? nullptr : (bool*)attn_mask.data_ptr()),
                               (float*)rel_pos.data_ptr(),
                               bsz,
                               heads,
                               seq_len,
                               soft_len,
                               ratio,
                               at::cuda::getCurrentCUDAStream());
    else if (attn_scores_c.scalar_type() == at::kHalf)
        launch_softmax_dropout((__half*)out.data_ptr(),
                               (__half*)attn_scores_c.data_ptr(),
                               (bidirectional ? nullptr : (bool*)attn_mask.data_ptr()),
                               (half*)rel_pos.data_ptr(),
                               bsz,
                               heads,
                               seq_len,
                               soft_len,
                               ratio,
                               at::cuda::getCurrentCUDAStream());
    else
        launch_softmax_dropout((__nv_bfloat16*)out.data_ptr(),
                               (__nv_bfloat16*)attn_scores_c.data_ptr(),
                               (bidirectional ? nullptr : (bool*)attn_mask.data_ptr()),
                               (__nv_bfloat16*)rel_pos.data_ptr(),
                               bsz,
                               heads,
                               seq_len,
                               soft_len,
                               ratio,
                               at::cuda::getCurrentCUDAStream());

    return out;
}

at::Tensor ds_softmax_dropout_backward(at::Tensor& out_grad, at::Tensor& attn_scores, float ratio)
{
    auto out_grad_c = out_grad.contiguous();
    auto attn_scores_c = attn_scores.contiguous();
    int bsz = attn_scores_c.size(0);

    int seq_len = attn_scores_c.size(1);
    int len = attn_scores_c.sizes().size();
    if (len > 3) seq_len = attn_scores_c.size(2);

    int soft_len = attn_scores_c.size(2);
    if (len > 3) soft_len = attn_scores_c.size(3);

    int heads = 1;
    if (len > 3) heads = attn_scores_c.size(1);
    if (attn_scores_c.scalar_type() == at::kFloat)
        launch_softmax_dropout_grad((float*)out_grad_c.data_ptr(),
                                    (float*)attn_scores_c.data_ptr(),
                                    bsz,
                                    heads,
                                    seq_len,
                                    soft_len,
                                    ratio,
                                    at::cuda::getCurrentCUDAStream());
    else if (attn_scores_c.scalar_type() == at::kHalf)
        launch_softmax_dropout_grad((__half*)out_grad_c.data_ptr(),
                                    (__half*)attn_scores_c.data_ptr(),
                                    bsz,
                                    heads,
                                    seq_len,
                                    soft_len,
                                    ratio,
                                    at::cuda::getCurrentCUDAStream());
    else
        launch_softmax_dropout_grad((__nv_bfloat16*)out_grad_c.data_ptr(),
                                    (__nv_bfloat16*)attn_scores_c.data_ptr(),
                                    bsz,
                                    heads,
                                    seq_len,
                                    soft_len,
                                    ratio,
                                    at::cuda::getCurrentCUDAStream());

    return out_grad;
}

std::vector<at::Tensor> ds_partial_norm(at::Tensor& input)
{
    
    int bsz = 1;
    for (int s : input.sizes())
        bsz *= s;
    int hidden = input.size(input.sizes().size() - 1);
    bsz = bsz / hidden;
    auto mean = at::empty({bsz}, input.options());
    auto var = at::empty({bsz}, input.options());
    if (input.scalar_type() == at::kFloat)
        launch_bias_residual_layer_norm((float*)input.data_ptr(),
                                        bsz,
                                        hidden,
                                        at::cuda::getCurrentCUDAStream(),
                                        (float*)vars.data_ptr(),
                                        (float*)means.data_ptr());
    else
        launch_bias_residual_layer_norm((__half*)input.data_ptr(),
                                        bsz,
                                        hidden,
                                        at::cuda::getCurrentCUDAStream(),
                                        (__half*)vars.data_ptr(),
                                        (__half*)means.data_ptr());
    return {mean, var};
}

std::vector<at::Tensor> ds_partial_norm_bwd(at::Tensor& input, 
                            at::Tensor& out_grad, at::Tensor& vars,
                            at::Tensor& gamm, at::Tensor& betta)
{
    
    int bsz = 1;
    for (int s : input.sizes())
        bsz *= s;
    int hidden = input.size(input.sizes().size() - 1);
    bsz = bsz / hidden;
    auto gamm_grad = at::empty({hidden}, input.options(bsz));
    auto beta_grad = at::empty({hidden}, input.options(bsz));
    auto inp_grad = at::empty_like(out_grad);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (input.scalar_type() == at::kFloat)
        launch_layerNorm_backward((float*)out_grad.data_ptr(),
                                       (float*)input.data_ptr(),
                                       (float*)vars.data_ptr(),
                                       (float*)gamma.data_ptr(),
                                       (float*)gamm_grad.data_ptr(),
                                       (float*)beta_grad.data_ptr(),
                                       __half* inp_grad,
                                       bsz,
                                       hidden,
                                       {stream, stream},
                                       true,
                                       (float*)betta.data_ptr());
    else
        launch_layerNorm_backward((__half*)out_grad.data_ptr(),
                                       (__half*)input.data_ptr(),
                                       (__half*)vars.data_ptr(),
                                       (__half*)gamma.data_ptr(),
                                       (__half*)gamm_grad.data_ptr(),
                                       (__half*)beta_grad.data_ptr(),
                                       __half* inp_grad,
                                       bsz,
                                       hidden,
                                       {stream, stream},
                                       true,
                                       (__half*)betta.data_ptr());
    return {inp_grad, gamm_grad, beta_grad};
}

std::vector<at::Tensor> ds_partial_norm1(at::Tensor& input, 
                            at::Tensor& mean, 
                            at::Tensor& var, 
                            at::Tensor& gamma, 
                            at::Tensor& beta,
                            float epsilon)
{
    
    int bsz = 1;
    for (int s : input.sizes())
        bsz *= s;
    int hidden = input.size(input.sizes().size() - 1);
    bsz = bsz / hidden;
    auto norm_out = torch::empty_like(input);
    if (input.scalar_type() == at::kFloat)
        launch_bias_residual_layer_norm1((float*)norm_out.data_ptr(),
                                        (float*)input.data_ptr(),
                                        bsz,
                                        hidden,
                                        at::cuda::getCurrentCUDAStream(),
                                        (float*)vars.data_ptr(),
                                        (float*)means.data_ptr(),
                                        (float*)gamma.data_ptr(),
                                        (float*)beta.data_ptr());
    else
        launch_bias_residual_layer_norm1((__half*)norm_out.data_ptr(),
                                        (__half*)input.data_ptr(),
                                        bsz,
                                        hidden,
                                        at::cuda::getCurrentCUDAStream(),
                                        (__half*)vars.data_ptr(),
                                        (__half*)means.data_ptr(),
                                        (__half*)gamma.data_ptr(),
                                        (__half*)beta.data_ptr());
    return {norm_out, vars};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("softd_forward", &ds_softmax_dropout, "DeepSpeed softmax_dropout forward (CUDA)");
    m.def("softd_backward",
          &ds_softmax_dropout_backward,
          "DeepSpeed softmax_dropout backward (CUDA)");
    m.def("partialNorm", &ds_partial_norm, "DeepSpeed softmax_dropout forward (CUDA)");
    m.def("partialNorm_bwd",
          &ds_partial_norm_bwd,
          "DeepSpeed softmax_dropout backward (CUDA)");
    m.def("partialNorm1", &ds_partial_norm1, "DeepSpeed softmax_dropout forward (CUDA)");
}
