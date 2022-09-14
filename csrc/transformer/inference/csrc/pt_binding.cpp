/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>
#include <stdexcept>
#include <vector>
#include "inference_context.h"
#include "inference_cublas_wrappers.h"
#include "inference_cuda_layers.h"

std::array<int, 3> gemm_algos = std::array<int, 3>({99, 99, 99});

// NOTE: This activation function type enum should be always in sync
// with the python counterpart, otherwise the casting from python binding
// will be incorrect.
enum class ActivationFuncType { UNKNOWN = 0, GELU = 1, ReLU = 2 };

enum class TransformerType : uint8_t { UNKNOWN = 0, GPTType = 1, BERTType = 2 };

// NOTE: this is a temporary and dodgy solution to distinguish GPT and BERT style models
// based on the dimensions of the corresponding attention mask.
inline auto infer_transformer_type(torch::Tensor& attn_mask) -> TransformerType
{
    auto attn_mask_num_dims = attn_mask.sizes().size();

    if (attn_mask_num_dims > 2) {
        return TransformerType::GPTType;
    } else if (attn_mask_num_dims == 2) {
        return TransformerType::BERTType;
    } else {
        return TransformerType::UNKNOWN;
    }
}

// infer stride of attention mask memory layout based on the model type.
inline auto get_attn_mask_stride(torch::Tensor& attn_mask) -> int
{
    auto trnsfrmr_type = infer_transformer_type(attn_mask);

    if (trnsfrmr_type == TransformerType::GPTType) {
        return attn_mask.size(2);
    } else if (trnsfrmr_type == TransformerType::BERTType) {
        // Bert style models have always a mask stride of 1.
        return 1;
    } else if (trnsfrmr_type == TransformerType::UNKNOWN) {
        throw std::runtime_error("Unknown transformer type.");
    }

    // this is just to make the compiler happy.
    return 0;
}

template <typename T>
torch::Tensor ds_softmax(torch::Tensor& attn_scores,
                         torch::Tensor& attn_mask,
                         torch::Tensor& alibi,
                         int64_t triangular,       // bool
                         int64_t recompute,        // bool
                         int64_t local_attention,  // bool
                         int64_t window_size,
                         int64_t async_op,  // bool
                         double layer_scale,
                         int64_t head_offset,
                         int64_t mp_size)
{
    auto attn_scores_c = attn_scores.contiguous();
    int bsz = attn_scores_c.size(0);

    int seq_len = attn_scores_c.size(1);
    int len = attn_scores_c.sizes().size();
    if (len > 2) seq_len = attn_scores_c.size(2);

    int soft_len = attn_scores_c.size(2);
    if (len > 3) soft_len = attn_scores_c.size(3);

    int heads = 1;
    if (len > 1) heads = attn_scores_c.size(1);

    auto mask_stride = get_attn_mask_stride(attn_mask);

    launch_attn_softmax_v2((T*)attn_scores_c.data_ptr(),
                           (attn_mask.sizes().size() > 1 ? (T*)attn_mask.data_ptr() : nullptr),
                           (alibi.sizes().size() > 1 ? (T*)alibi.data_ptr() : nullptr),
                           layer_scale,
                           triangular,
                           recompute,
                           local_attention,
                           window_size,
                           bsz,
                           heads,
                           seq_len,
                           soft_len,
                           head_offset,
                           mask_stride,
                           mp_size,
                           Context::Instance().GetCurrentStream(async_op));

    return attn_scores_c;
}

template <typename T>
void allocate_workspace(size_t hidden_dim,
                        size_t max_seq_len,
                        size_t batch_size,
                        unsigned num_layers,
                        size_t head_size = 128)
{
    size_t _workSpaceSize = 16 * (hidden_dim * batch_size * max_seq_len) +
                            (num_layers * batch_size * max_seq_len * hidden_dim * 2);  // KV-cache
    Context::Instance().GenWorkSpace(_workSpaceSize * sizeof(T));
}

template <typename T>
torch::Tensor einsum_sec_sm_ecm(torch::Tensor& Q, torch::Tensor& W)
{
    auto options = torch::TensorOptions()
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

    auto O = at::from_blob(workspace, {Q.size(1), Q.size(2), W.size(1)}, options);
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
#ifdef __HIP_PLATFORM_HCC__
                   rocblas_gemm_algo_standard);
#else
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    return O;
}

template <typename T>
void attention_unfused(torch::Tensor& prev_key_cont,
                       torch::Tensor& query_cont,
                       torch::Tensor& attn_mask,
                       torch::Tensor& prev_value_cont,
                       torch::Tensor& output,
                       int64_t& bsz,
                       int64_t& seq_len,
                       int64_t& soft_len,
                       int64_t& heads,
                       double& norm_factor,
                       int64_t triangular,       // bool
                       int64_t recompute,        // bool
                       int64_t local_attention,  // bool
                       int64_t window_size)
{
    auto options = torch::TensorOptions()
                       .dtype(query_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    float alpha = (float)norm_factor;
    float gemm_beta = 0.0;
    auto attn_score = at::empty({bsz, heads, seq_len, soft_len}, options);
    int k = prev_value_cont.size(2) / heads;

    auto mask_stride = get_attn_mask_stride(attn_mask);

    cublasSetStream(Context::Instance().GetCublasHandle(), Context::Instance().GetCurrentStream());
    cublas_strided_batched_gemm(Context::Instance().GetCublasHandle(),
                                soft_len,
                                seq_len,
                                k,
                                &alpha,
                                &gemm_beta,
                                (T*)prev_key_cont.data_ptr(),
                                (T*)query_cont.data_ptr(),
                                (T*)attn_score.data_ptr(),
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                soft_len * k,
                                seq_len * k,
                                seq_len * soft_len,
                                bsz * heads,
#ifdef __HIP_PLATFORM_HCC__
                                rocblas_gemm_algo_standard);
#else
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    launch_attn_softmax_v2((T*)attn_score.data_ptr(),
                           (T*)(attn_mask.sizes().size() > 1 ? attn_mask.data_ptr() : nullptr),
                           (T*)nullptr,
                           1.0,
                           triangular,
                           recompute,
                           local_attention,
                           window_size,
                           bsz,
                           heads,
                           seq_len,
                           soft_len,
                           0,
                           mask_stride,
                           1,
                           Context::Instance().GetCurrentStream(false));
    alpha = 1.0;
    cublas_strided_batched_gemm(Context::Instance().GetCublasHandle(),
                                k,
                                seq_len,
                                soft_len,
                                &alpha,
                                &gemm_beta,
                                (T*)prev_value_cont.data_ptr(),
                                (T*)attn_score.data_ptr(),
                                (T*)output.data_ptr(),
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                soft_len * k,
                                seq_len * soft_len,
                                seq_len * k,
                                bsz * heads,
#ifdef __HIP_PLATFORM_HCC__
                                rocblas_gemm_algo_standard);
#else
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
}

template <typename T>
std::vector<torch::Tensor> ds_softmax_context1(torch::Tensor& query,
                                               torch::Tensor& prev_key,
                                               torch::Tensor& new_key,
                                               torch::Tensor& attn_mask,
                                               torch::Tensor& prev_value,
                                               torch::Tensor& new_value,
                                               int64_t heads,
                                               double norm_factor,
                                               int64_t merging,          // bool
                                               int64_t triangular,       // bool
                                               int64_t local_attention,  // bool
                                               int64_t window_size,
                                               bool no_masking /* bool */)
{
    auto query_cont = query.contiguous();
    auto prev_key_cont = prev_key.contiguous();
    auto prev_value_cont = prev_value.contiguous();

    int64_t new_size = (new_value.sizes().size() > 1 ? new_value.size(1) : 0);

    // Attn_Score [ batch Head Sequence-length Softmax-length]

    int64_t bsz = query_cont.size(0);
    int64_t seq_len = query_cont.size(1);
    int64_t soft_len = prev_value.size(1);

    auto options = torch::TensorOptions()
                       .dtype(query_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output =
        at::empty({prev_value.size(0), heads, seq_len, prev_value.size(2) / heads}, options);
    attention_unfused<T>(prev_key_cont,
                         query_cont,
                         attn_mask,  //(no_masking ? nullptr : (T*)attn_mask.data_ptr()),
                         prev_value_cont,
                         output,
                         bsz,
                         seq_len,
                         soft_len,
                         heads,
                         norm_factor,
                         (triangular && (new_size == 0)),
                         (new_size == 0),
                         local_attention,
                         window_size);

    return {output, prev_key, prev_value};
}

template <typename T>
void ds_softmax_internal(T* attn_scores,
                         torch::Tensor& attn_mask,
                         torch::Tensor& alibi,
                         float& layer_scale,
                         bool triangular,
                         bool recompute,
                         bool local_attention,
                         int window_size,
                         int bsz,
                         int seq_len,
                         int soft_len,
                         int heads)
{
    auto mask_stride = get_attn_mask_stride(attn_mask);

    launch_attn_softmax_v2((T*)attn_scores,
                           (attn_mask.sizes().size() > 1 ? (T*)attn_mask.data_ptr() : nullptr),
                           (alibi.sizes().size() > 1 ? (T*)alibi.data_ptr() : nullptr),
                           layer_scale,
                           triangular,
                           recompute,
                           local_attention,
                           window_size,
                           bsz,
                           heads,
                           seq_len,
                           soft_len,
                           0,
                           mask_stride,
                           1,
                           at::cuda::getCurrentCUDAStream());
}

template <typename T>
void attention_unfused(T* prev_key_cont,
                       T* query_cont,
                       torch::Tensor& attn_mask,
                       T* prev_value_cont,
                       T* output,
                       int64_t& bsz,
                       int64_t& k,
                       int64_t& seq_len,
                       int64_t& soft_len,
                       int64_t& heads,
                       double& norm_factor,
                       int64_t triangular,
                       int64_t recompute,
                       int64_t local_attention,
                       int64_t window_size,
                       torch::Tensor& alibi,
                       int64_t layer_id)
{
    float layer_scale = alibi.sizes().size() > 1 ? std::max(1, (int)layer_id) : 1.0;
    float alpha = norm_factor * norm_factor / layer_scale;
    float gemm_beta = 0.0;
    T* workspace = (T*)output + bsz * seq_len * heads * k;

    cublasSetStream(Context::Instance().GetCublasHandle(), Context::Instance().GetCurrentStream());
    cublas_strided_batched_gemm(Context::Instance().GetCublasHandle(),
                                soft_len,
                                seq_len,
                                k,
                                &alpha,
                                &gemm_beta,
                                (T*)prev_key_cont,
                                (T*)query_cont,
                                workspace,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                MAX_OUT_TOKES * k,
                                seq_len * k,
                                seq_len * soft_len,
                                bsz * heads,
#ifdef __HIP_PLATFORM_HCC__
                                rocblas_gemm_algo_standard);
#else
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    ds_softmax_internal<T>(workspace,
                           attn_mask,
                           alibi,
                           layer_scale,
                           triangular,
                           recompute,
                           local_attention,
                           window_size,
                           bsz,
                           seq_len,
                           soft_len,
                           heads);
    alpha = 1.0;
    cublas_strided_batched_gemm(Context::Instance().GetCublasHandle(),
                                k,
                                seq_len,
                                soft_len,
                                &alpha,
                                &gemm_beta,
                                (T*)prev_value_cont,
                                workspace,
                                (T*)output,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                MAX_OUT_TOKES * k,
                                seq_len * soft_len,
                                seq_len * k,
                                bsz * heads,
#ifdef __HIP_PLATFORM_HCC__
                                rocblas_gemm_algo_standard);
#else
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
}

template <typename T>
std::vector<torch::Tensor> ds_softmax_context(torch::Tensor& query_key_value,
                                              torch::Tensor& attn_mask,
                                              int64_t rotary_dim,
                                              int64_t rotate_half,       // bool
                                              int64_t rotate_every_two,  // bool
                                              int64_t heads,
                                              double norm_factor,
                                              int64_t triangular,       // bool
                                              int64_t local_attention,  // bool
                                              int64_t window_size,
                                              int64_t no_masking,  // bool
                                              int64_t layer_id,
                                              int64_t num_layers,
                                              torch::Tensor& alibi)
{
    int64_t bsz = query_key_value.size(0);
    int64_t seq_len = query_key_value.size(1);
    int64_t hidden_dim = query_key_value.size(2) / 3;

    int64_t is_prompt = (seq_len > 1);

    if (is_prompt) Context::Instance().reset_tokens(seq_len);
    int64_t soft_len = Context::Instance().current_tokens();

    int64_t k = hidden_dim / heads;
    auto options = torch::TensorOptions()
                       .dtype(query_key_value.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    T* workspace = (T*)Context::Instance().GetWorkSpace();
    int64_t buf_size = bsz * seq_len * hidden_dim;
    auto output = torch::from_blob(workspace + 4 * buf_size, {bsz, seq_len, hidden_dim}, options);

    auto query_cont = workspace + 8 * buf_size;
    int64_t offset =
        16 * (hidden_dim * bsz * MAX_OUT_TOKES) + layer_id * 2 * bsz * MAX_OUT_TOKES * hidden_dim;

    int64_t all_tokens = soft_len;
    auto kv_cache = workspace + offset + (hidden_dim / heads) * (is_prompt ? 0 : soft_len - 1);
    int64_t value_offset = bsz * MAX_OUT_TOKES * hidden_dim;

    T* temp_buf = (T*)output.data_ptr() + at::numel(output);
    launch_bias_add_transform_0213<T>((T*)query_cont,
                                      kv_cache,
                                      kv_cache + value_offset,
                                      (T*)query_key_value.data_ptr(),
                                      nullptr,
                                      bsz,
                                      seq_len,
                                      (is_prompt ? 0 : soft_len - 1),
                                      soft_len,
                                      hidden_dim,
                                      heads,
                                      rotary_dim,
                                      rotate_half,
                                      rotate_every_two,
                                      Context::Instance().GetCurrentStream(),
                                      3);
    if (rotary_dim > 0 && rotate_half)
        launch_apply_rotary_pos_emb(query_cont,
                                    kv_cache,
                                    k,
                                    seq_len,
                                    rotary_dim,
                                    (is_prompt ? 0 : soft_len - 1),
                                    heads,
                                    bsz,
                                    rotate_half,
                                    rotate_every_two,
                                    Context::Instance().GetCurrentStream());

    attention_unfused<T>(workspace + offset,
                         (T*)query_cont,
                         attn_mask,
                         workspace + offset + value_offset,
                         temp_buf,
                         bsz,
                         k,
                         seq_len,
                         all_tokens,
                         heads,
                         norm_factor,
                         (triangular && is_prompt),
                         is_prompt,
                         local_attention,
                         window_size,
                         alibi,
                         layer_id);
    launch_transform4d_0213<T>((T*)output.data_ptr(),
                               temp_buf,
                               bsz,
                               heads,
                               seq_len,
                               output.size(2),
                               Context::Instance().GetCurrentStream(false),
                               1);

    if (layer_id == num_layers - 1) Context::Instance().advance_tokens();
    auto prev_key = torch::from_blob(workspace + offset, {bsz, heads, all_tokens, k}, options);
    auto prev_value =
        torch::from_blob(workspace + offset + value_offset, {bsz, heads, all_tokens, k}, options);
    return {output, prev_key, prev_value};
}

template <typename T>
torch::Tensor ds_bias_gelu(torch::Tensor& input, torch::Tensor& bias)
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

template <typename T>
torch::Tensor ds_bias_relu(torch::Tensor& input, torch::Tensor& bias)
{
    auto input_cont = input.contiguous();

    int bsz = input_cont.size(0) * input_cont.size(1);
    int intermediate_size = input_cont.size(2);

    launch_bias_relu((T*)input_cont.data_ptr(),
                     (T*)bias.data_ptr(),
                     intermediate_size,
                     bsz,
                     Context::Instance().GetCurrentStream());
    return input_cont;
}

template <typename T>
torch::Tensor ds_bias_add(torch::Tensor& input, torch::Tensor& bias)
{
    auto input_cont = input.contiguous();

    int bsz = input_cont.size(0) * input_cont.size(1);
    int hidden_size = input_cont.size(2);

    launch_bias_add((T*)input_cont.data_ptr(),
                    (T*)bias.data_ptr(),
                    hidden_size,
                    bsz,
                    Context::Instance().GetCurrentStream());
    return input_cont;
}

template <typename T>
torch::Tensor ds_bias_residual(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& bias)
{
    auto input_cont = input.contiguous();
    auto residual_cont = residual.contiguous();

    int bsz = input_cont.size(0) * input_cont.size(1);
    // launch_bias_residual((T*)input_cont.data_ptr(),
    //                      (T*)residual_cont.data_ptr(),
    //                      (T*)bias.data_ptr(),
    //                      bsz,
    //                      input_cont.size(2),
    //                      (bias.size(0) > 1),
    //                      Context::Instance().GetCurrentStream());
    return input_cont;
}

template <typename T>
torch::Tensor ds_layernorm(torch::Tensor& input_cont,
                           torch::Tensor& gamma,
                           torch::Tensor& betta,
                           double epsilon)
{
    int bsz = input_cont.size(0) * input_cont.size(1);
    auto inp_norm = at::empty_like(input_cont);
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
void ds_layernorm_internal(T* workspace,
                           torch::Tensor& input,
                           torch::Tensor& gamma,
                           torch::Tensor& betta,
                           float epsilon)
{
    int bsz = input.size(0) * input.size(1);
    launch_layer_norm(workspace,
                      (T*)input.data_ptr(),
                      (T*)gamma.data_ptr(),
                      (T*)betta.data_ptr(),
                      epsilon,
                      bsz,
                      input.size(2),
                      Context::Instance().GetCurrentStream());
}

template <typename T>
void quantized_gemm(torch::Tensor& output,
                    T* input,
                    torch::Tensor& weight,
                    torch::Tensor& qscale,
                    int groups,
                    int bsz)
{
    auto weight16 = at::empty({weight.size(0), weight.size(1)}, output.options());

    launch_dequantize((T*)weight16.data_ptr(),
                      (int8_t*)weight.data_ptr(),
                      (float*)qscale.data_ptr(),
                      weight.size(0),
                      weight.size(1),
                      groups,
                      Context::Instance().GetCurrentStream());

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                   CUBLAS_OP_T,
                   CUBLAS_OP_N,
                   weight.size(0),
                   bsz,
                   weight.size(1),
                   &alpha,
                   &gemm_beta,
                   (T*)weight16.data_ptr(),
                   (T*)input,
                   (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                   rocblas_gemm_algo_standard);
#else
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
}

template <typename T>
torch::Tensor qkv_unfused_cublas(torch::Tensor& output,
                                 torch::Tensor& input,
                                 torch::Tensor& weight,
                                 torch::Tensor& q_scale,
                                 torch::Tensor& bias,
                                 torch::Tensor& gamma,
                                 torch::Tensor& beta,
                                 const float epsilon,
                                 bool add_bias,
                                 bool q_int8)
{
    int bsz = input.size(0) * input.size(1);
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    workspace += (3 * bsz * input.size(2));
    ds_layernorm_internal<T>(workspace, input, gamma, beta, epsilon);
    // cudaEventRecord(Context::Instance().GetCompEvent(1), Context::Instance().GetCurrentStream());

    if (q_int8) {
        quantized_gemm<T>(output, workspace, weight, q_scale, q_scale.size(0), bsz);
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;

        cublasSetStream(Context::Instance().GetCublasHandle(),
                        Context::Instance().GetCurrentStream());
        cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       weight.size(1),
                       bsz,
                       input.size(2),
                       &alpha,
                       &gemm_beta,
                       (T*)weight.data_ptr(),
                       workspace,
                       (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                       rocblas_gemm_algo_standard);
#else
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    }
    if (add_bias)
        launch_bias_add((T*)output.data_ptr(),
                        (T*)bias.data_ptr(),
                        q_int8 ? weight.size(0) : weight.size(1),
                        bsz,
                        Context::Instance().GetCurrentStream());
    return torch::from_blob(workspace, input.sizes(), input.options());
}

template <typename T>
std::vector<torch::Tensor> ds_qkv_gemm(torch::Tensor& input,
                                       torch::Tensor& weight,
                                       torch::Tensor& q_scale,
                                       torch::Tensor& bias,
                                       torch::Tensor& gamma,
                                       torch::Tensor& beta,
                                       const double epsilon,
                                       int64_t add_bias,  // bool
                                       int64_t num_layers,
                                       int64_t q_int8 /* bool */)
{
    int bsz = input.size(0) * input.size(1);
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    int out_size = q_int8 ? weight.size(0) : weight.size(1);
    if (!workspace) {
        cublasSetStream(Context::Instance().GetCublasHandle(),
                        Context::Instance().GetCurrentStream());
        allocate_workspace<T>(input.size(2), MAX_OUT_TOKES, input.size(0), num_layers);
        workspace = (T*)Context::Instance().GetWorkSpace();
    }
    auto options = torch::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::from_blob(workspace, {input.size(0), input.size(1), out_size}, options);
    auto inp_norm = qkv_unfused_cublas<T>(
        output, input, weight, q_scale, bias, gamma, beta, epsilon, add_bias, q_int8);

    return {output, inp_norm};
}

template <typename T>
void quantized_gemm(torch::Tensor& output,
                    torch::Tensor& input,
                    torch::Tensor& weight,
                    torch::Tensor& qscale,
                    int groups,
                    int merge_count)
{
    int bsz = input.size(0) * input.size(1);
    auto options = torch::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    auto weight16 = at::empty({weight.size(0), weight.size(1)}, options);

    launch_dequantize((T*)weight16.data_ptr(),
                      (int8_t*)weight.data_ptr(),
                      (float*)qscale.data_ptr(),
                      weight.size(0),
                      weight.size(1),
                      groups,
                      merge_count,
                      Context::Instance().GetCurrentStream());

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                   CUBLAS_OP_T,
                   CUBLAS_OP_N,
                   weight.size(0),
                   bsz,
                   input.size(2),
                   &alpha,
                   &gemm_beta,
                   (T*)weight16.data_ptr(),
                   (T*)input.data_ptr(),
                   (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                   rocblas_gemm_algo_standard);
#else
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
}

template <typename T>
torch::Tensor ds_qkv_gemm_int8(torch::Tensor& input,
                               torch::Tensor& weight,
                               torch::Tensor& bias,
                               torch::Tensor& gamma,
                               torch::Tensor& beta,
                               const double epsilon,
                               torch::Tensor& q_scale,
                               int64_t groups,
                               int64_t add_bias /* bool */)
{
    int bsz = input.size(0) * input.size(1);
    auto input_cont = input.contiguous();
    auto options = torch::TensorOptions()
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
torch::Tensor ds_linear_layer(torch::Tensor& input,
                              torch::Tensor& weight,
                              torch::Tensor& bias,
                              int64_t num_layers /* bool */)
{
    auto input_cont = input.contiguous();
    auto options = torch::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    int bsz = input.size(0) * input.size(1);
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    if (!workspace) {
        cublasSetStream(Context::Instance().GetCublasHandle(),
                        Context::Instance().GetCurrentStream());
        allocate_workspace<T>(input.size(2), MAX_OUT_TOKES, input.size(0), num_layers);
        workspace = (T*)Context::Instance().GetWorkSpace();
    }
    auto output = at::from_blob(workspace, {input.size(0), input.size(1), weight.size(1)}, options);

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    cublasSetStream(Context::Instance().GetCublasHandle(), Context::Instance().GetCurrentStream());

    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   weight.size(1),
                   bsz,
                   input_cont.size(2),
                   &alpha,
                   &gemm_beta,
                   (T*)weight.data_ptr(),
                   (T*)input_cont.data_ptr(),
                   (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                   rocblas_gemm_algo_standard);
#else
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif

    launch_bias_add((T*)output.data_ptr(),
                    (T*)bias.data_ptr(),
                    weight.size(1),
                    bsz,
                    Context::Instance().GetCurrentStream());

    return output;
}

template <typename T>
torch::Tensor ds_linear_layer_int8(torch::Tensor& input,
                                   torch::Tensor& weight,
                                   torch::Tensor& bias,
                                   torch::Tensor& q_scale,
                                   int64_t groups /* bool */)
{
    auto input_cont = input.contiguous();
    auto options = torch::TensorOptions()
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
torch::Tensor ds_vector_matmul(torch::Tensor& input,
                               torch::Tensor& weight,
                               int64_t async_op,  // bool
                               torch::Tensor& q_scale,
                               int64_t q_int8 /* bool */)
{
    auto input_cont = input.contiguous();
    auto options = torch::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    int out_size = q_int8 ? weight.size(0) : weight.size(1);
    int bsz = input_cont.size(0) * input_cont.size(1);
    auto output = at::empty({input_cont.size(0), input_cont.size(1), out_size}, options);
    if (q_int8) {
        quantized_gemm<T>(output, (T*)input_cont.data_ptr(), weight, q_scale, q_scale.size(0), bsz);
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
        cublasSetStream(Context::Instance().GetCublasHandle(),
                        Context::Instance().GetCurrentStream(async_op));
        cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       weight.size(1),
                       bsz,
                       input_cont.size(2),
                       &alpha,
                       &gemm_beta,
                       (T*)weight.data_ptr(),
                       (T*)input_cont.data_ptr(),
                       (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                       rocblas_gemm_algo_standard);
#else
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    }
    return output;
}

template <typename T>
torch::Tensor ds_vector_matmul_int8(torch::Tensor& input,
                                    torch::Tensor& weight,
                                    torch::Tensor& q_scale,
                                    int64_t groups,
                                    int64_t merge_count)
{
    auto input_cont = input.contiguous();
    auto options = torch::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);

    quantized_gemm<T>(output, input_cont, weight, q_scale, groups, merge_count);
    return output;
}

template <typename T>
torch::Tensor mlp_unfused_cublas(torch::Tensor& output,
                                 torch::Tensor& input,
                                 torch::Tensor& residual,
                                 torch::Tensor& input_bias,
                                 torch::Tensor& weight,
                                 torch::Tensor& bias,
                                 torch::Tensor& gamma,
                                 torch::Tensor& beta,
                                 const float epsilon,
                                 bool preLayerNorm,
                                 bool mlp_after_attn,
                                 torch::Tensor& q_scale,
                                 bool q_int8,
                                 ActivationFuncType act_func_type)
{
    int bsz = input.size(0) * input.size(1);
    auto inp_norm = at::empty_like(input);

    launch_residual_layer_norm((T*)inp_norm.data_ptr(),
                               (T*)nullptr,
                               (T*)input.data_ptr(),
                               (T*)residual.data_ptr(),
                               (T*)input_bias.data_ptr(),
                               (T*)gamma.data_ptr(),
                               (T*)beta.data_ptr(),
                               epsilon,
                               bsz,
                               input.size(2),
                               preLayerNorm,
                               mlp_after_attn,
                               Context::Instance().GetCurrentStream());

    if (q_int8) {
        quantized_gemm<T>(output, (T*)inp_norm.data_ptr(), weight, q_scale, q_scale.size(0), bsz);
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
        cublasSetStream(Context::Instance().GetCublasHandle(),
                        Context::Instance().GetCurrentStream());
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
#ifdef __HIP_PLATFORM_HCC__
                       rocblas_gemm_algo_standard);
#else
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    }
    if (act_func_type == ActivationFuncType::GELU) {
        launch_bias_gelu((T*)output.data_ptr(),
                         (T*)bias.data_ptr(),
                         q_int8 ? weight.size(0) : weight.size(1),
                         bsz,
                         Context::Instance().GetCurrentStream());
    } else if (act_func_type == ActivationFuncType::ReLU) {
        launch_bias_relu((T*)output.data_ptr(),
                         (T*)bias.data_ptr(),
                         q_int8 ? weight.size(0) : weight.size(1),
                         bsz,
                         Context::Instance().GetCurrentStream());
    }

    return inp_norm;
}

template <typename T>
std::vector<torch::Tensor> ds_mlp_gemm(torch::Tensor& input,
                                       torch::Tensor& residual,
                                       torch::Tensor& input_bias,
                                       torch::Tensor& weight,
                                       torch::Tensor& bias,
                                       torch::Tensor& gamma,
                                       torch::Tensor& beta,
                                       const double epsilon,
                                       int64_t preLayerNorm,    // bool
                                       int64_t mlp_after_attn,  // bool
                                       torch::Tensor& q_scale,
                                       int64_t q_int8,  // bool
                                       int64_t activation_type)
{
    auto input_cont = input.contiguous();
    auto options = torch::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    int out_size = q_int8 ? weight.size(0) : weight.size(1);
    auto output = at::from_blob((T*)Context::Instance().GetWorkSpace(),
                                {input_cont.size(0), input_cont.size(1), out_size},
                                options);
    int bsz = input_cont.size(0) * input_cont.size(1);

    auto act_func_type = static_cast<ActivationFuncType>(activation_type);
    auto res_add = mlp_unfused_cublas<T>(output,
                                         mlp_after_attn ? input : residual,
                                         residual,
                                         input_bias,
                                         weight,
                                         bias,
                                         gamma,
                                         beta,
                                         epsilon,
                                         preLayerNorm,
                                         mlp_after_attn,
                                         q_scale,
                                         q_int8,
                                         act_func_type);

    return {output, res_add};
}

template <typename T>
std::vector<torch::Tensor> ds_mlp_gemm_int8(torch::Tensor& input,
                                            torch::Tensor& residual,
                                            torch::Tensor& input_bias,
                                            torch::Tensor& weight,
                                            torch::Tensor& bias,
                                            torch::Tensor& gamma,
                                            torch::Tensor& beta,
                                            const double epsilon,
                                            torch::Tensor& q_scale,
                                            int64_t groups,
                                            int64_t preLayerNorm /* bool */)
{
    auto input_cont = input.contiguous();
    auto options = torch::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);

    int bsz = input_cont.size(0) * input_cont.size(1);
    auto inp_norm = at::empty_like(input_cont);

    auto residual_add = (preLayerNorm ? at::empty_like(input_cont) : inp_norm);
    quantized_gemm<T>(output, inp_norm, weight, q_scale, groups, 0);
    launch_bias_gelu((T*)output.data_ptr(),
                     (T*)bias.data_ptr(),
                     weight.size(1),
                     bsz,
                     Context::Instance().GetCurrentStream());

    return {output, residual_add};
}

template <typename T>
torch::Tensor fused_gemm_gelu(torch::Tensor& input,
                              torch::Tensor& weight,
                              torch::Tensor& bias,
                              torch::Tensor& weight_out,
                              const double epsilon,
                              int64_t preLayerNorm,  // bool
                              int64_t async_op /* bool */)
{
    auto input_cont = input.contiguous();
    auto options = torch::TensorOptions()
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
#ifdef __HIP_PLATFORM_HCC__
                   rocblas_gemm_algo_standard);
#else
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
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
#ifdef __HIP_PLATFORM_HCC__
                   rocblas_gemm_algo_standard);
#else
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    // cudaEventRecord(Context::Instance().GetCompEvent(2),
    //                Context::Instance().GetCurrentStream(true));
    return output;
}

void residual_add_bias(torch::Tensor& output,
                       torch::Tensor& input,
                       torch::Tensor& attention_output,
                       torch::Tensor& output_b,
                       torch::Tensor& attention_b,
                       int64_t mp_size,
                       int64_t mlp_after_attn,  // bool
                       int64_t add_bias,        // bool
                       int64_t preln /* bool */)
{
    int bsz = input.size(0) * input.size(1);
    int hidden_size = input.size(2);
    // cudaStreamWaitEvent(
    //    Context::Instance().GetCurrentStream(), Context::Instance().GetCompEvent(2), 0);
    if (input.scalar_type() == at::kFloat)
        if (mlp_after_attn)
            launch_bias_residual((float*)input.data_ptr(),
                                 (float*)output.data_ptr(),
                                 (float*)attention_output.data_ptr(),
                                 (float*)output_b.data_ptr(),
                                 (float*)attention_b.data_ptr(),
                                 bsz,
                                 hidden_size,
                                 mp_size,
                                 preln,
                                 Context::Instance().GetCurrentStream());
        else
            launch_gptj_residual_add<float>((float*)input.data_ptr(),
                                            (float*)output.data_ptr(),
                                            (float*)attention_output.data_ptr(),
                                            (float*)output_b.data_ptr(),
                                            (float*)(add_bias ? attention_b.data_ptr() : nullptr),
                                            hidden_size,
                                            bsz,
                                            mp_size,
                                            Context::Instance().GetCurrentStream());
    else if (mlp_after_attn)
        launch_bias_residual((__half*)input.data_ptr(),
                             (__half*)output.data_ptr(),
                             (__half*)attention_output.data_ptr(),
                             (__half*)output_b.data_ptr(),
                             (__half*)attention_b.data_ptr(),
                             bsz,
                             hidden_size,
                             mp_size,
                             preln,
                             Context::Instance().GetCurrentStream());
    else
        launch_gptj_residual_add<__half>((__half*)input.data_ptr(),
                                         (__half*)output.data_ptr(),
                                         (__half*)attention_output.data_ptr(),
                                         (__half*)output_b.data_ptr(),
                                         (__half*)(add_bias ? attention_b.data_ptr() : nullptr),
                                         hidden_size,
                                         bsz,
                                         mp_size,
                                         Context::Instance().GetCurrentStream());
}

std::vector<torch::Tensor> apply_rotary_pos_emb(torch::Tensor& mixed_query,
                                                torch::Tensor& key_layer,
                                                int64_t rotary_dim,
                                                int64_t offset,
                                                int64_t num_heads,
                                                int64_t rotate_half,  // bool
                                                int64_t rotate_every_two /* bool */)
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
                                           rotate_half,
                                           rotate_every_two,
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
                                            rotate_half,
                                            rotate_every_two,
                                            Context::Instance().GetCurrentStream());
    return {query_cont, key_cont};
}

template <typename T>
torch::Tensor fused_gemm_gelu_int8(torch::Tensor& input,
                                   torch::Tensor& weight,
                                   torch::Tensor& bias,
                                   const double epsilon,
                                   torch::Tensor& q_scale,
                                   int64_t groups,
                                   int64_t preLayerNorm /* bool */)
{
    auto input_cont = input.contiguous();
    auto options = torch::TensorOptions()
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

torch::Tensor moe_res_matmul(torch::Tensor& moe_res, torch::Tensor& coef, torch::Tensor& output)
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

TORCH_LIBRARY(transformer_inference, m)
{
    m.def("softmax_fp32", &ds_softmax<float>);
    m.def("softmax_fp16", &ds_softmax<__half>);
    m.def("softmax_context_fp32", &ds_softmax_context<float>);
    m.def("softmax_context_fp16", &ds_softmax_context<__half>);
    m.def("softmax_context_int8", &ds_softmax_context1<__half>);
    m.def("bias_gelu_fp32", &ds_bias_gelu<float>);
    m.def("bias_gelu_fp16", &ds_bias_gelu<__half>);
    m.def("bias_add_fp32", &ds_bias_add<float>);
    m.def("bias_add_fp16", &ds_bias_add<__half>);
    m.def("bias_relu_fp32", &ds_bias_relu<float>);
    m.def("bias_relu_fp16", &ds_bias_relu<__half>);
    m.def("bias_residual_fp32", &ds_bias_residual<float>);
    m.def("bias_residual_fp16", &ds_bias_residual<__half>);
    m.def("layer_norm_fp32", &ds_layernorm<float>);
    m.def("layer_norm_fp16", &ds_layernorm<__half>);
    m.def("qkv_gemm_fp32", &ds_qkv_gemm<float>);
    m.def("qkv_gemm_fp16", &ds_qkv_gemm<__half>);
    m.def("qkv_gemm_int8", &ds_qkv_gemm_int8<__half>);
    m.def("mlp_gemm_fp32", &ds_mlp_gemm<float>);
    m.def("mlp_gemm_fp16", &ds_mlp_gemm<__half>);
    m.def("mlp_gemm_int8", &ds_mlp_gemm_int8<__half>);
    m.def("vector_matmul_fp32", &ds_vector_matmul<float>);
    m.def("vector_matmul_fp16", &ds_vector_matmul<__half>);
    m.def("vector_matmul_int8", &ds_vector_matmul_int8<__half>);
    m.def("linear_layer_fp32", &ds_linear_layer<float>);
    m.def("linear_layer_fp16", &ds_linear_layer<__half>);
    m.def("linear_layer_int8", &ds_linear_layer_int8<__half>);
    m.def("fused_gemm_gelu_fp32", &fused_gemm_gelu<float>);
    m.def("fused_gemm_gelu_fp16", &fused_gemm_gelu<__half>);
    m.def("residual_add", &residual_add_bias);
    m.def("apply_rotary_pos_emb", &apply_rotary_pos_emb);
    m.def("einsum_sec_sm_ecm_fp32", &einsum_sec_sm_ecm<float>);
    m.def("einsum_sec_sm_ecm_fp16", &einsum_sec_sm_ecm<__half>);
    m.def("moe_res_matmul", &moe_res_matmul);
}
