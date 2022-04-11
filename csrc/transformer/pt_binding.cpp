#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>
#include "context.h"
#include "custom_cuda_layers.h"

at::Tensor ds_softmax_dropout(at::Tensor& attn_scores,
                              at::Tensor& attn_mask,
                              float ratio,
                              bool bidirectional)
{
    auto attn_scores_c = attn_scores.contiguous();
    auto out = at::empty_like(attn_scores_c);
    int bsz = attn_scores_c.size(0);

    int seq_len = attn_scores_c.size(1);
    int len = attn_scores_c.sizes().size();
    if (len > 3) seq_len = attn_scores_c.size(2);

    int soft_len = attn_scores_c.size(2);
    if (len > 3) soft_len = attn_scores_c.size(3);

    int heads = 1;
    if (len > 3) heads = attn_scores_c.size(1);

    if (attn_scores_c.scalar_type() == at::kFloat)
        launch_softmax_dropout((float*)out.data_ptr(), 
                               (float*)attn_scores_c.data_ptr(), 
                               (bidirectional ? nullptr : (float*)attn_mask.data_ptr()), 
                               bsz,
                               heads,
                               seq_len,
                               soft_len,
                               ratio, 
                               at::cuda::getCurrentCUDAStream());
    else if (attn_scores_c.scalar_type() == at::kHalf)
        launch_softmax_dropout((__half*)out.data_ptr(), 
                               (__half*)attn_scores_c.data_ptr(), 
                               (bidirectional ? nullptr : (__half*)attn_mask.data_ptr()), 
                               bsz,
                               heads,
                               seq_len,
                               soft_len,
                               ratio, 
                               at::cuda::getCurrentCUDAStream());
    else
        launch_softmax_dropout((__nv_bfloat16*)out.data_ptr(), 
                               (__nv_bfloat16*)attn_scores_c.data_ptr(), 
                               (bidirectional ? nullptr : (__nv_bfloat16*)attn_mask.data_ptr()), 
                               bsz,
                               heads,
                               seq_len,
                               soft_len,
                               ratio, 
                               at::cuda::getCurrentCUDAStream());

    return out;
}


at::Tensor ds_softmax_dropout_backward(at::Tensor& out_grad,
                              at::Tensor& attn_scores,
                              at::Tensor& attn_mask,
                              float ratio,
                              bool bidirectional)
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

    //if (attn_scores_c.scalar_type() == at::kFloat)
    //    launch_softmax_dropout_grad((float*)out_grad_c.data_ptr(), 
    //                            (float*)attn_scores_c.data_ptr(), 
    //                            (bidirectional ? nullptr : (float*)attn_mask.data_ptr()), 
    //                            bsz,
    //                            heads,
    //                            seq_len,
    //                            soft_len,
    //                            ratio, 
    //                            at::cuda::getCurrentCUDAStream());
    //else if (attn_scores_c.scalar_type() == at::kHalf)
    //    launch_softmax_dropout_grad((__half*)out_grad_c.data_ptr(), 
    //                            (__half*)attn_scores_c.data_ptr(), 
    //                            (bidirectional ? nullptr : (__half*)attn_mask.data_ptr()), 
    //                            bsz,
    //                            heads,
    //                            seq_len,
    //                            soft_len,
    //                            ratio, 
    //                            at::cuda::getCurrentCUDAStream());
    //else
    //    launch_softmax_dropout_grad((__nv_bfloat16*)out_grad_c.data_ptr(), 
    //                            (__nv_bfloat16*)attn_scores_c.data_ptr(), 
    //                            (bidirectional ? nullptr : (__nv_bfloat16*)attn_mask.data_ptr()), 
    //                            bsz,
    //                            heads,
    //                            seq_len,
    //                            soft_len,
    //                            ratio, 
    //                            at::cuda::getCurrentCUDAStream());
    //
    return out_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("softd_forward",
          &ds_softmax_dropout,
          "DeepSpeed softmax_dropout forward (CUDA)");
    m.def("softd_backward",
          &ds_softmax_dropout_backward,
          "DeepSpeed softmax_dropout backward (CUDA)");
}