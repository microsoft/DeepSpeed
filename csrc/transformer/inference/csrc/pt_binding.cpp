
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>
#include "context.h"
#include "cublas_wrappers.h"
#include "custom_cuda_layers.h"

std::array<int, 3> gemm_algos = std::array<int, 3>({99, 99, 99});

#define MAX_OUT_TOKES 1024

template <typename T>
at::Tensor ds_softmax(at::Tensor& attn_scores,
                      at::Tensor& attn_mask,
                      bool triangular,
                      bool recompute,
                      bool local_attention,
                      int window_size,
                      bool async_op)
{
    //auto attn_scores_c = attn_scores.contiguous();
    int bsz = attn_scores.size(0);

    int seq_len = attn_scores.size(1);
    int len = attn_scores.sizes().size();
    if (len > 3) seq_len = attn_scores.size(2);

    int soft_len = attn_scores.size(2);
    if (len > 3) soft_len = attn_scores.size(3);

    int heads = 1;
    if (len > 3) heads = attn_scores.size(1);

    launch_attn_softmax_v2((T*)attn_scores.data_ptr(),
                           (attn_mask.sizes().size() > 1 ? (T*)attn_mask.data_ptr() : nullptr),
                           triangular,
                           recompute,
                           local_attention,
                           window_size,
                           bsz,
                           heads,
                           seq_len,
                           soft_len,
                           1.0,
                           Context::Instance().GetCurrentStream(async_op));

    return attn_scores;
}

template <typename T>
void allocate_workspace(size_t hidden_dim,
                        size_t max_seq_len,
                        size_t batch_size,
                        unsigned num_layers,
                        size_t head_size = 128)
{
    size_t _workSpaceSize = 16 * (hidden_dim * batch_size * max_seq_len) + 
            (num_layers * batch_size * max_seq_len * hidden_dim * 2); //KV-cache
    printf("workspace size: %d \n", _workSpaceSize);
    Context::Instance().GenWorkSpace(_workSpaceSize * sizeof(T));
}

template <typename T>
at::Tensor einsum_sec_sm_ecm(at::Tensor& Q, at::Tensor& W)
{
    auto options = at::TensorOptions()
                       .dtype(Q.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    float alpha = 1;
    float gemm_beta = 0.0;

    if (!workspace) {
        allocate_workspace<T>(W.size(1), MAX_OUT_TOKES, Q.size(0), 1);
        workspace = (T*)Context::Instance().GetWorkSpace();
    }

    auto O = torch::from_blob(workspace, {Q.size(1), Q.size(2), W.size(1)}, options);
    unsigned m = W.size(1);
    unsigned n = Q.size(1) * Q.size(2);
    unsigned k = Q.size(0);
    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   m,
                   n,
                   k,
                   &alpha,
                   &gemm_beta,
                   (T*)W.data_ptr(),
                   (T*)Q.data_ptr(),
                   (T*)O.data_ptr(),
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return O;
}

template <typename T>
void attention_unfused(T* prev_key_cont,
                       T* query_cont,
                       at::Tensor& attn_mask,
                       T* prev_value_cont,
                       at::Tensor& output,
                       unsigned& bsz,
                       int& seq_len,
                       unsigned& soft_len,
                       int& heads,
                       float& norm_factor,
                       bool triangular,
                       bool recompute,
                       bool local_attention,
                       int window_size)
{
    float alpha = norm_factor;
    float gemm_beta = 0.0;
    T* temp_buf = (T*)output.data_ptr() + at::numel(output);
    auto attn_score = temp_buf + at::numel(output);
    int k = output.size(2) / heads;
    cublas_strided_batched_gemm(Context::Instance().GetCublasHandle(),
                                soft_len,
                                seq_len,
                                k,
                                &alpha,
                                &gemm_beta,
                                (T*)prev_key_cont,
                                (T*)query_cont,
                                (T*)attn_score,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                soft_len * k,
                                seq_len * k,
                                seq_len * soft_len,
                                bsz * heads,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    auto tmp_score = at::from_blob(attn_score, {bsz, heads, seq_len, soft_len}, output.options());
    tmp_score = ds_softmax<T>(tmp_score, attn_mask, triangular, recompute, local_attention, window_size, false);
    alpha = 1.0;
    cublas_strided_batched_gemm(Context::Instance().GetCublasHandle(),
                                k,
                                seq_len,
                                soft_len,
                                &alpha,
                                &gemm_beta,
                                (T*)prev_value_cont,
                                (T*)attn_score,
                                temp_buf,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                soft_len * k,
                                seq_len * soft_len,
                                seq_len * k,
                                bsz * heads,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    launch_transform4d_0213<T>(
        (T*)output.data_ptr(), temp_buf, bsz, heads, seq_len, output.size(2), Context::Instance().GetCurrentStream(false), 1);
}

template<typename T>
void apply_rotary_pos_emb1(T* query_key_value,
                           unsigned rotary_dim,
                           unsigned offset,
                           unsigned num_heads,
                           unsigned bsz, 
                           unsigned head_size,
                           unsigned seq_len)
{
    launch_apply_rotary_pos_emb1<T>(query_key_value,
                                           head_size,
                                           seq_len,
                                           rotary_dim,
                                           offset,
                                           num_heads,
                                           bsz,
                                           Context::Instance().GetCurrentStream());
}

template <typename T>
at::Tensor ds_softmax_context(at::Tensor& query_key_value,
                              at::Tensor& attn_mask,
                              int heads,
                              float norm_factor,
                              bool merging,
                              bool triangular,
                              bool local_attention,
                              int window_size,
                              bool no_masking,
                              unsigned layer_id,
                              int rotary_dim)
{
    unsigned bsz = query_key_value.size(0);
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    int seq_len = query_key_value.size(1);
    unsigned hidden_dim = (query_key_value.size(2) / 3);
    unsigned current_tokens = Context::Instance().current_tokens();
    size_t offset = 16 * (hidden_dim * bsz * MAX_OUT_TOKES) + 
            layer_id * 2 * bsz * MAX_OUT_TOKES * hidden_dim;
    size_t value_offset = bsz * MAX_OUT_TOKES * hidden_dim;
    auto query_cont = workspace + at::numel(query_key_value);
    
    if (rotary_dim > 0)
        apply_rotary_pos_emb1<T>(
                    (T*)query_key_value.data_ptr(),
                    rotary_dim,
                    current_tokens,
                    heads,
                    bsz,
                    hidden_dim / heads,
                    seq_len);

    auto kv_cache = (workspace + offset + hidden_dim * current_tokens);
    launch_transform_scale<T>(
        (T*)query_key_value.data_ptr(), 
        query_cont,
        kv_cache, 
        bsz, 
        seq_len, 
        current_tokens,
        value_offset,
        hidden_dim, 
        heads, 
        Context::Instance().GetCurrentStream(), 
        3,
        norm_factor);

    // Attn_Score [ batch Head Sequence-length Softmax-length]
    auto options = at::TensorOptions()
                       .dtype(query_key_value.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output =
        at::from_blob(query_cont + at::numel(query_key_value) / 3, 
        {bsz, seq_len, query_key_value.size(2) / 3}, options);
    attention_unfused<T>((workspace + offset),
                         query_cont,
                         attn_mask,  //(no_masking ? nullptr : (T*)attn_mask.data_ptr()),
                         (workspace + offset + value_offset),
                         output,
                         bsz,
                         seq_len,
                         current_tokens,
                         heads,
                         norm_factor,
                         (triangular && (current_tokens == seq_len)),
                         (current_tokens == seq_len),
                         local_attention,
                         window_size);
    Context::Instance().advance_tokens();

    return output;
}

template <typename T>
at::Tensor ds_bias_gelu(at::Tensor& input, at::Tensor& bias)
{

    int bsz = input.size(0) * input.size(1);
    int intermediate_size = input.size(2);

    launch_bias_gelu((T*)input.data_ptr(),
                     (T*)bias.data_ptr(),
                     intermediate_size,
                     bsz,
                     Context::Instance().GetCurrentStream());
    return input;
}

template <typename T>
at::Tensor ds_bias_residual(at::Tensor& input, at::Tensor& residual, at::Tensor& bias)
{
    int bsz = input.size(0) * input.size(1);
    launch_bias_residual((T*)input.data_ptr(),
                         (T*)residual.data_ptr(),
                         (T*)bias.data_ptr(),
                         bsz,
                         input.size(2),
                         (bias.size(0) > 1),
                         Context::Instance().GetCurrentStream());
    return input;
}

template <typename T>
at::Tensor ds_layernorm(at::Tensor& input_cont, at::Tensor& gamma, at::Tensor& betta, float epsilon)
{
    int bsz = input_cont.size(0) * input_cont.size(1);
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    auto inp_norm = torch::from_blob(workspace + (3 * input_cont.size(0) * MAX_OUT_TOKES * input_cont.size(2)), 
                    input_cont.sizes(), input_cont.options());
    launch_layer_norm((T*)inp_norm.data_ptr(),
                      (T*)input_cont.data_ptr(),
                      (T*)gamma.data_ptr(),
                      (T*)betta.data_ptr(),
                      epsilon,
                      bsz,
                      input_cont.size(2),
                      Context::Instance().GetCurrentStream());
    return inp_norm;
}

template <typename T>
at::Tensor qkv_unfused_cublas(at::Tensor& output,
                              at::Tensor& input,
                              at::Tensor& weight,
                              at::Tensor& bias,
                              at::Tensor& gamma,
                              at::Tensor& beta,
                              const float epsilon,
                              bool add_bias)
{
    auto inp_norm = ds_layernorm<T>(input, gamma, beta, epsilon);

    // cudaEventRecord(Context::Instance().GetCompEvent(1), Context::Instance().GetCurrentStream());

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    int bsz = input.size(0) * input.size(1);
    cublasSetStream(Context::Instance().GetCublasHandle(), Context::Instance().GetCurrentStream());
    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   weight.size(1),
                   bsz,
                   input.size(2),
                   &alpha,
                   &gemm_beta,
                   (T*)weight.data_ptr(),
                   (T*)inp_norm.data_ptr(),
                   (T*)output.data_ptr(),
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (add_bias)
        launch_bias_add((T*)output.data_ptr(),
                        (T*)bias.data_ptr(),
                        weight.size(1),
                        bsz,
                        Context::Instance().GetCurrentStream());
    return inp_norm;
}

template <typename T>
std::vector<at::Tensor> ds_qkv_gemm(at::Tensor& input,
                                    at::Tensor& weight,
                                    at::Tensor& bias,
                                    at::Tensor& gamma,
                                    at::Tensor& beta,
                                    const float epsilon,
                                    bool add_bias,
                                    unsigned num_layers,
                                    bool is_prompt)
{
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    if (!workspace) {
        cublasSetStream(Context::Instance().GetCublasHandle(), Context::Instance().GetCurrentStream());
        allocate_workspace<T>(input.size(2), MAX_OUT_TOKES, input.size(0), num_layers);
        workspace = (T*)Context::Instance().GetWorkSpace();
    }
    if(is_prompt) Context::Instance().reset_tokens(input.size(1));
    printf("cur tok: %d \n", Context::Instance().current_tokens());

    auto output = torch::from_blob(workspace, 
                                    {input.size(0), input.size(1), weight.size(1)}, input.options());
    int bsz = input.size(0) * input.size(1);
    auto inp_norm =
        qkv_unfused_cublas<T>(output, input, weight, bias, gamma, beta, epsilon, add_bias);

    return {output, inp_norm};
}

template <typename T>
void quantized_gemm(at::Tensor& output,
                    at::Tensor& input,
                    at::Tensor& weight,
                    at::Tensor& qscale,
                    int groups,
                    int merge_count)
{
    int bsz = input.size(0) * input.size(1);
    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    auto weight16 = at::empty({weight.size(0), weight.size(1)}, options);

    launch_dequantize((T*)weight16.data_ptr(),
                      (int8_t*)weight.data_ptr(),
                      (float*)qscale.data_ptr(),
                      weight.size(1),
                      weight.size(0),
                      groups,
                      merge_count,
                      Context::Instance().GetCurrentStream());

    cublasSetStream(Context::Instance().GetCublasHandle(), Context::Instance().GetCurrentStream());

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   weight.size(1),
                   bsz,
                   input.size(2),
                   &alpha,
                   &gemm_beta,
                   (T*)weight16.data_ptr(),
                   (T*)input.data_ptr(),
                   (T*)output.data_ptr(),
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

template <typename T>
at::Tensor ds_qkv_gemm_int8(at::Tensor& input,
                            at::Tensor& weight,
                            at::Tensor& bias,
                            at::Tensor& gamma,
                            at::Tensor& beta,
                            const float epsilon,
                            at::Tensor& q_scale,
                            int groups,
                            bool add_bias)
{
    int bsz = input.size(0) * input.size(1);
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);

    auto inp_norm = ds_layernorm<T>(input_cont, gamma, beta, epsilon);

    quantized_gemm<T>(output, inp_norm, weight, q_scale, groups, 0);
    if (add_bias)
        launch_bias_add((T*)output.data_ptr(),
                        (T*)bias.data_ptr(),
                        weight.size(1),
                        bsz,
                        Context::Instance().GetCurrentStream());

    return output;
}

template <typename T>
at::Tensor ds_linear_layer(at::Tensor& input, at::Tensor& weight, at::Tensor& bias)
{

    T* workspace = (T*)Context::Instance().GetWorkSpace();
    auto output = torch::from_blob(workspace, {input.size(0), input.size(1), weight.size(1)}, input.options());
    int bsz = input.size(0) * input.size(1);

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;

    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   weight.size(1),
                   bsz,
                   input.size(2),
                   &alpha,
                   &gemm_beta,
                   (T*)weight.data_ptr(),
                   (T*)input.data_ptr(),
                   (T*)output.data_ptr(),
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    launch_bias_add((T*)output.data_ptr(),
                    (T*)bias.data_ptr(),
                    weight.size(1),
                    bsz,
                    Context::Instance().GetCurrentStream());

    return output;
}

template <typename T>
at::Tensor ds_linear_layer_int8(at::Tensor& input,
                                at::Tensor& weight,
                                at::Tensor& bias,
                                at::Tensor& q_scale,
                                int groups)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    int bsz = input_cont.size(0) * input_cont.size(1);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);

    quantized_gemm<T>(output, input_cont, weight, q_scale, groups, 0);
    launch_bias_add((T*)output.data_ptr(),
                    (T*)bias.data_ptr(),
                    weight.size(1),
                    bsz,
                    Context::Instance().GetCurrentStream());
    return output;
}

template <typename T>
at::Tensor ds_vector_matmul(at::Tensor& input, at::Tensor& weight, bool async_op)
{
    int bsz = input.size(0) * input.size(1);
    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    
    T* workspace = (T*)Context::Instance().GetWorkSpace();

    auto output = torch::from_blob(workspace, {input.size(0), input.size(1), weight.size(1)}, input.options());
    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   weight.size(1),
                   bsz,
                   input.size(2),
                   &alpha,
                   &gemm_beta,
                   (T*)weight.data_ptr(),
                   (T*)input.data_ptr(),
                   (T*)output.data_ptr(),
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return output;
}

template <typename T>
at::Tensor ds_vector_matmul_int8(at::Tensor& input,
                                 at::Tensor& weight,
                                 at::Tensor& q_scale,
                                 int groups,
                                 int merge_count)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);

    quantized_gemm<T>(output, input_cont, weight, q_scale, groups, merge_count);
    return output;
}

template <typename T>
void mlp_unfused_cublas(at::Tensor& output,
                        at::Tensor& residual_add,
                        at::Tensor& input,
                        at::Tensor& residual,
                        at::Tensor& input_bias,
                        at::Tensor& weight,
                        at::Tensor& bias,
                        at::Tensor& gamma,
                        at::Tensor& beta,
                        const float epsilon,
                        bool preLayerNorm)
{
    int bsz = input.size(0) * input.size(1);
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    auto inp_norm = preLayerNorm ? torch::from_blob(workspace + 2 * at::numel(input), input.sizes(), input.options()) : residual_add;

    launch_residual_layer_norm((T*)inp_norm.data_ptr(),
                               (T*)residual_add.data_ptr(),
                               (T*)input.data_ptr(),
                               (T*)residual.data_ptr(),
                               (T*)input_bias.data_ptr(),
                               (T*)gamma.data_ptr(),
                               (T*)beta.data_ptr(),
                               epsilon,
                               bsz,
                               input.size(2),
                               preLayerNorm,
                               Context::Instance().GetCurrentStream());

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    cublasSetStream(Context::Instance().GetCublasHandle(), Context::Instance().GetCurrentStream());
    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   weight.size(1),
                   bsz,
                   input.size(2),
                   &alpha,
                   &gemm_beta,
                   (T*)weight.data_ptr(),
                   (T*)inp_norm.data_ptr(),
                   (T*)output.data_ptr(),
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    launch_bias_gelu((T*)output.data_ptr(),
                     (T*)bias.data_ptr(),
                     weight.size(1),
                     bsz,
                     Context::Instance().GetCurrentStream());
}
template <typename T>
std::vector<at::Tensor> ds_mlp_gemm(at::Tensor& input,
                                    at::Tensor& residual,
                                    at::Tensor& input_bias,
                                    at::Tensor& weight,
                                    at::Tensor& bias,
                                    at::Tensor& gamma,
                                    at::Tensor& beta,
                                    const float epsilon,
                                    bool preLayerNorm)
{
    T* workspace = (T*)Context::Instance().GetWorkSpace();

    auto output = torch::from_blob(workspace + 4 * at::numel(input), {input.size(0), input.size(1), weight.size(1)}, input.options());
    auto residual_add = torch::from_blob(workspace + at::numel(input), input.sizes(), input.options());
    int bsz = input.size(0) * input.size(1);

    mlp_unfused_cublas<T>(output,
                          residual_add,
                          input,
                          residual,
                          input_bias,
                          weight,
                          bias,
                          gamma,
                          beta,
                          epsilon,
                          preLayerNorm);

    return {output, residual_add};
}

template <typename T>
std::vector<at::Tensor> ds_mlp_gemm_int8(at::Tensor& input,
                                         at::Tensor& residual,
                                         at::Tensor& input_bias,
                                         at::Tensor& weight,
                                         at::Tensor& bias,
                                         at::Tensor& gamma,
                                         at::Tensor& beta,
                                         const float epsilon,
                                         at::Tensor& q_scale,
                                         int groups,
                                         bool preLayerNorm)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);

    int bsz = input_cont.size(0) * input_cont.size(1);
    auto inp_norm = at::empty_like(input_cont);

    auto residual_add = (preLayerNorm ? at::empty_like(input_cont) : inp_norm);
    // computing the blocking across K dimension
    launch_residual_layer_norm((T*)inp_norm.data_ptr(),
                               (T*)residual_add.data_ptr(),
                               (T*)input_cont.data_ptr(),
                               (T*)residual.data_ptr(),
                               (T*)input_bias.data_ptr(),
                               (T*)gamma.data_ptr(),
                               (T*)beta.data_ptr(),
                               epsilon,
                               bsz,
                               input_cont.size(2),
                               preLayerNorm,
                               Context::Instance().GetCurrentStream());

    quantized_gemm<T>(output, inp_norm, weight, q_scale, groups, 0);
    launch_bias_gelu((T*)output.data_ptr(),
                     (T*)bias.data_ptr(),
                     weight.size(1),
                     bsz,
                     Context::Instance().GetCurrentStream());

    return {output, residual_add};
}

template <typename T>
at::Tensor fused_gemm_gelu(at::Tensor& input,
                           at::Tensor& weight,
                           at::Tensor& bias,
                           at::Tensor& weight_out,
                           const float epsilon,
                           bool preLayerNorm,
                           bool async_op)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto intermediate =
        at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);
    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight_out.size(1)}, options);
    int bsz = input_cont.size(0) * input_cont.size(1);
    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    cublasSetStream(Context::Instance().GetCublasHandle(), Context::Instance().GetCurrentStream());
    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   weight.size(1),
                   bsz,
                   input.size(2),
                   &alpha,
                   &gemm_beta,
                   (T*)weight.data_ptr(),
                   (T*)input_cont.data_ptr(),
                   (T*)intermediate.data_ptr(),
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    launch_bias_gelu((T*)intermediate.data_ptr(),
                     (T*)bias.data_ptr(),
                     weight.size(1),
                     bsz,
                     Context::Instance().GetCurrentStream());

    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   weight_out.size(1),
                   bsz,
                   intermediate.size(2),
                   &alpha,
                   &gemm_beta,
                   (T*)weight_out.data_ptr(),
                   (T*)intermediate.data_ptr(),
                   (T*)output.data_ptr(),
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    // cudaEventRecord(Context::Instance().GetCompEvent(2),
    //                Context::Instance().GetCurrentStream(true));
    return output;
}

void gptj_residual_add(at::Tensor& output,
                       at::Tensor& input,
                       at::Tensor& attention_output,
                       at::Tensor& output_b)
{
    int bsz = input.size(0) * input.size(1);
    int hidden_size = input.size(2);
    // cudaStreamWaitEvent(
    //    Context::Instance().GetCurrentStream(), Context::Instance().GetCompEvent(2), 0);
    if (input.scalar_type() == at::kFloat)
        launch_gptj_residual_add<float>((float*)input.data_ptr(),
                                        (float*)output.data_ptr(),
                                        (float*)attention_output.data_ptr(),
                                        (float*)output_b.data_ptr(),
                                        hidden_size,
                                        bsz,
                                        Context::Instance().GetCurrentStream());
    else
        launch_gptj_residual_add<__half>((__half*)input.data_ptr(),
                                         (__half*)output.data_ptr(),
                                         (__half*)attention_output.data_ptr(),
                                         (__half*)output_b.data_ptr(),
                                         hidden_size,
                                         bsz,
                                         Context::Instance().GetCurrentStream());
}


std::vector<at::Tensor> apply_rotary_pos_emb(at::Tensor& mixed_query,
                                             at::Tensor& key_layer,
                                             unsigned rotary_dim,
                                             unsigned offset,
                                             unsigned num_heads)
{
    auto query_cont = mixed_query.contiguous();
    auto key_cont = key_layer.contiguous();

    unsigned bsz = mixed_query.size(0);
    unsigned head_size = mixed_query.size(2) / num_heads;
    unsigned seq_len = mixed_query.size(1);

    if (mixed_query.scalar_type() == at::kFloat)
        launch_apply_rotary_pos_emb<float>((float*)query_cont.data_ptr(),
                                           (float*)key_cont.data_ptr(),
                                           head_size,
                                           seq_len,
                                           rotary_dim,
                                           offset,
                                           num_heads,
                                           bsz,
                                           Context::Instance().GetCurrentStream());
    else
        launch_apply_rotary_pos_emb<__half>((__half*)query_cont.data_ptr(),
                                            (__half*)key_cont.data_ptr(),
                                            head_size,
                                            seq_len,
                                            rotary_dim,
                                            offset,
                                            num_heads,
                                            bsz,
                                            Context::Instance().GetCurrentStream());
    return {query_cont, key_cont};
}

template <typename T>
at::Tensor fused_gemm_gelu_int8(at::Tensor& input,
                                at::Tensor& weight,
                                at::Tensor& bias,
                                const float epsilon,
                                at::Tensor& q_scale,
                                int groups,
                                bool preLayerNorm)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);

    int bsz = input_cont.size(0) * input_cont.size(1);

    quantized_gemm<T>(output, input_cont, weight, q_scale, groups, 0);
    launch_bias_gelu((T*)output.data_ptr(),
                     (T*)bias.data_ptr(),
                     weight.size(1),
                     bsz,
                     Context::Instance().GetCurrentStream());

    return output;
}

at::Tensor moe_res_matmul(at::Tensor& moe_res, at::Tensor& coef, at::Tensor& output)
{
    int M = moe_res.size(0) * moe_res.size(1);
    int N = moe_res.size(2);
    Context::Instance().SynchComm();
    if (moe_res.scalar_type() == at::kFloat) {
        launch_moe_res_matmul<float>((float*)moe_res.data_ptr(),
                                     (float*)coef.data_ptr(),
                                     (float*)output.data_ptr(),
                                     M,
                                     N,
                                     at::cuda::getCurrentCUDAStream());
    } else {
        launch_moe_res_matmul<__half>((__half*)moe_res.data_ptr(),
                                      (__half*)coef.data_ptr(),
                                      (__half*)output.data_ptr(),
                                      M,
                                      N,
                                      at::cuda::getCurrentCUDAStream());
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("softmax_fp32", &ds_softmax<float>, "DeepSpeed SoftMax with fp32 (CUDA)");
    m.def("softmax_fp16", &ds_softmax<__half>, "DeepSpeed SoftMax with fp32 (CUDA)");
    m.def(
        "softmax_context_fp32", &ds_softmax_context<float>, "DeepSpeed attention with fp32 (CUDA)");
    m.def("softmax_context_fp16",
          &ds_softmax_context<__half>,
          "DeepSpeed attention with fp32 (CUDA)");
    m.def("bias_gelu_fp32", &ds_bias_gelu<float>, "DeepSpeed Gelu with fp32 (CUDA)");
    m.def("bias_gelu_fp16", &ds_bias_gelu<__half>, "DeepSpeed Gelu with fp32 (CUDA)");
    m.def("bias_residual_fp32",
          &ds_bias_residual<float>,
          "DeepSpeed residual-bias add with fp32 (CUDA)");
    m.def("bias_residual_fp16",
          &ds_bias_residual<__half>,
          "DeepSpeed residual-bias add with fp32 (CUDA)");
    m.def("layer_norm_fp32", &ds_layernorm<float>, "DeepSpeed layer-norm with fp32 (CUDA)");
    m.def("layer_norm_fp16", &ds_layernorm<__half>, "DeepSpeed layer-norm with fp16 (CUDA)");
    m.def("qkv_gemm_fp32", &ds_qkv_gemm<float>, "DeepSpeed qkv gemm with fp32 (CUDA)");
    m.def("qkv_gemm_fp16", &ds_qkv_gemm<__half>, "DeepSpeed qkv gemm with fp16 (CUDA)");
    m.def("qkv_gemm_int8", &ds_qkv_gemm_int8<__half>, "DeepSpeed qkv gemm with int8 (CUDA)");
    m.def("mlp_gemm_fp32", &ds_mlp_gemm<float>, "DeepSpeed mlp with fp32 (CUDA)");
    m.def("mlp_gemm_fp16", &ds_mlp_gemm<__half>, "DeepSpeed mlp with fp16 (CUDA)");
    m.def("mlp_gemm_int8", &ds_mlp_gemm_int8<__half>, "DeepSpeed mlp with int8 (CUDA)");
    m.def("vector_matmul_fp32", &ds_vector_matmul<float>, "DeepSpeed vector-MM with fp32 (CUDA)");
    m.def("vector_matmul_fp16", &ds_vector_matmul<__half>, "DeepSpeed vector-MM with fp16 (CUDA)");
    m.def("vector_matmul_int8",
          &ds_vector_matmul_int8<__half>,
          "DeepSpeed vector-MM with int8 (CUDA)");
    m.def("linear_layer_fp32", &ds_linear_layer<float>, "DeepSpeed linear_layer with fp32 (CUDA)");
    m.def("linear_layer_fp16", &ds_linear_layer<__half>, "DeepSpeed linear_layer with fp16 (CUDA)");
    m.def("linear_layer_int8",
          &ds_linear_layer_int8<__half>,
          "DeepSpeed linear_layer with int8 (CUDA)");
    m.def("fused_gemm_gelu_fp32", &fused_gemm_gelu<float>, "DeepSpeed mlp with fp32 (CUDA)");
    m.def("fused_gemm_gelu_fp16", &fused_gemm_gelu<__half>, "DeepSpeed mlp with fp16 (CUDA)");
    m.def("gptj_residual_add", &gptj_residual_add, "DeepSpeed mlp with fp16 (CUDA)");
    m.def("apply_rotary_pos_emb", &apply_rotary_pos_emb, "DeepSpeed mlp with fp16 (CUDA)");
    m.def("einsum_sec_sm_ecm_fp32",
          &einsum_sec_sm_ecm<float>,
          "DeepSpeed vector-MM with fp32 (CUDA)");

    m.def("einsum_sec_sm_ecm_fp16",
          &einsum_sec_sm_ecm<__half>,
          "DeepSpeed vector-MM with fp16 (CUDA)");
    m.def("moe_res_matmul", &moe_res_matmul, "DeepSpeed moe residual matmul (CUDA)");
}
