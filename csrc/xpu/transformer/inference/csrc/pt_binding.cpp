#include <torch/extension.h>
#include <stdexcept>
#include <vector>
#include "compatible.hpp"
#include "inference_context.hpp"
#include "inference_onednn_wrappers.hpp"
#include "inference_onemkl_wrappers.hpp"
#include "inference_sycl_layers.hpp"

// NOTE: This activation function type enum should be always in sync
// with the python counterpart, otherwise the casting from python binding
// will be incorrect.
enum class ActivationFuncType { UNKNOWN = 0, GELU = 1, ReLU = 2 };

enum class TransformerType : uint8_t { UNKNOWN = 0, GPTType = 1, BERTType = 2 };

// NOTE: this is a temporary and dodgy solution to distinguish GPT and BERT style models
// based on the dimensions of the corresponding attention mask.
inline auto infer_transformer_type(at::Tensor& attn_mask) -> TransformerType
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
inline auto get_attn_mask_stride(at::Tensor& attn_mask) -> int
{
    auto trnsfrmr_type = infer_transformer_type(attn_mask);

    if (trnsfrmr_type == TransformerType::GPTType) {
        return attn_mask.size(2);
    } else if (trnsfrmr_type == TransformerType::BERTType) {
        // Bert style models have always a mask stride of 1.
        return 1;
    } else if (trnsfrmr_type == TransformerType::UNKNOWN) {
        return 0;
    }

    // this is just to make the compiler happy.
    return 0;
}

template <typename T>
void allocate_workspace(unsigned hidden_dim,
                        unsigned num_heads,
                        unsigned prompt_length,
                        unsigned batch_size,
                        unsigned num_layers,
                        unsigned mp_size = 1,
                        bool external_cache = false,
                        unsigned rank = 0,
                        unsigned max_out_tokens = 1024,
                        unsigned min_out_tokens = 1)
{
    InferenceContext::Instance().GenWorkSpace(num_layers,
                                              num_heads,
                                              batch_size,
                                              prompt_length,
                                              hidden_dim,
                                              mp_size,
                                              external_cache,
                                              sizeof(T),
                                              rank,
                                              max_out_tokens,
                                              min_out_tokens);
}

template <typename T>
at::Tensor ds_softmax(at::Tensor& attn_scores,
                      at::Tensor& attn_mask,
                      at::Tensor& alibi,
                      bool triangular,
                      bool recompute,
                      bool local_attention,
                      int window_size,
                      bool async_op,
                      float layer_scale,
                      int head_offset,
                      int mp_size)
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
                           InferenceContext::Instance().GetCurrentStream());

    return attn_scores_c;
}

at::Tensor ds_layer_norm(at::Tensor& input, at::Tensor& gamma, at::Tensor& beta, float epsilon)
{
    const int rows = input.size(0) * input.size(1);
    const int elems_per_row = input.size(2);
    auto output = at::empty_like(input);

    if (input.options().dtype() == torch::kFloat16) {
        launch_fused_ln((fp16*)output.data_ptr(),
                        (const fp16*)input.data_ptr(),
                        (const fp16*)gamma.data_ptr(),
                        (const fp16*)beta.data_ptr(),
                        epsilon,
                        rows,
                        elems_per_row,
                        InferenceContext::Instance().GetCurrentStream());
    } else {
        launch_fused_ln((float*)output.data_ptr(),
                        (const float*)input.data_ptr(),
                        (const float*)gamma.data_ptr(),
                        (const float*)beta.data_ptr(),
                        epsilon,
                        rows,
                        elems_per_row,
                        InferenceContext::Instance().GetCurrentStream());
    }

    return output;
}

/* Currently only used in unit testing */
at::Tensor ds_layer_norm_residual(at::Tensor& input,
                                  at::Tensor& bias,
                                  at::Tensor& residual,
                                  at::Tensor& gamma,
                                  at::Tensor& beta,
                                  float epsilon)
{
    const int rows = input.size(0) * input.size(1);
    const int elems_per_row = input.size(2);
    auto output = at::empty_like(input);

    if (input.options().dtype() == torch::kFloat16) {
        launch_fused_residual_ln((fp16*)output.data_ptr(),
                                 (const fp16*)input.data_ptr(),
                                 (const fp16*)residual.data_ptr(),
                                 (const fp16*)bias.data_ptr(),
                                 (const fp16*)gamma.data_ptr(),
                                 (const fp16*)beta.data_ptr(),
                                 epsilon,
                                 rows,
                                 elems_per_row,
                                 InferenceContext::Instance().GetCurrentStream());
    } else {
        launch_fused_residual_ln((float*)output.data_ptr(),
                                 (const float*)input.data_ptr(),
                                 (const float*)residual.data_ptr(),
                                 (const float*)bias.data_ptr(),
                                 (const float*)gamma.data_ptr(),
                                 (const float*)beta.data_ptr(),
                                 epsilon,
                                 rows,
                                 elems_per_row,
                                 InferenceContext::Instance().GetCurrentStream());
    }

    return output;
}

template <typename T>
at::Tensor& residual_add_bias(at::Tensor& hidden_state,
                              at::Tensor& residual,
                              const at::Tensor& attention_output,
                              const at::Tensor& attention_bias,
                              const at::Tensor& final_bias,
                              const int mp_size,
                              const bool mlp_after_attn,
                              const bool add_bias,
                              const bool preln)
{
    int bsz = residual.size(0) * residual.size(1);
    int hidden_size = residual.size(2);
    if (mlp_after_attn)
        launch_bias_residual(static_cast<T*>(residual.data_ptr()),
                             static_cast<T*>(hidden_state.data_ptr()),
                             static_cast<T*>(attention_output.data_ptr()),
                             static_cast<T*>(final_bias.data_ptr()),
                             static_cast<T*>(attention_bias.data_ptr()),
                             bsz,
                             hidden_size,
                             mp_size,
                             preln,
                             InferenceContext::Instance().GetCurrentStream());
    else
        throw std::runtime_error("mlp_after_attn=true is not supported!");
    return residual;
}

template <typename T>
void ds_layer_norm_internal(T* workspace,
                            at::Tensor& input,
                            at::Tensor& gamma,
                            at::Tensor& beta,
                            float epsilon)
{
    int bsz = input.size(0) * input.size(1);
    launch_fused_ln(workspace,
                    (const T*)input.data_ptr(),
                    (const T*)gamma.data_ptr(),
                    (const T*)beta.data_ptr(),
                    epsilon,
                    bsz,
                    input.size(2),
                    InferenceContext::Instance().GetCurrentStream());
}

template <typename T>
at::Tensor ds_layer_norm_test(at::Tensor& input, at::Tensor& gamma, at::Tensor& beta, float epsilon)
{
    int bsz = input.size(0) * input.size(1);

    T* workspace = (T*)InferenceContext::Instance().GetWorkSpace();

    launch_fused_ln(workspace,
                    (const T*)input.data_ptr(),
                    (const T*)gamma.data_ptr(),
                    (const T*)beta.data_ptr(),
                    epsilon,
                    bsz,
                    input.size(2),
                    InferenceContext::Instance().GetCurrentStream());

    auto output_stride = c10::TensorType::contiguousStridesOf(input.sizes());

    return at::from_blob(
        workspace, input.sizes(), output_stride, nullptr, input.options(), input.device());
}

template <typename T>
at::Tensor qkv_unfused_sycl(at::Tensor& output,
                            at::Tensor& input,
                            at::Tensor& weight,
                            at::Tensor& q_scale,
                            at::Tensor& bias,
                            at::Tensor& gamma,
                            at::Tensor& beta,
                            const float epsilon,
                            bool add_bias,
                            bool q_int8,
                            bool transposed_mode)
{
    int bsz = input.size(0) * input.size(1);
    T* workspace = (T*)InferenceContext::Instance().GetWorkSpace();
    workspace += (3 * bsz * input.size(2));
    ds_layer_norm_internal<T>(workspace, input, gamma, beta, epsilon);

    if (q_int8) {
        throw std::runtime_error("q_int8=true is not supported!");
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
#ifdef USE_MKL_GEMM
        onemkl_matmul_ex<T>(InferenceContext::Instance().GetCurrentStream(),
                            transposed_mode ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans,
                            oneapi::mkl::transpose::nontrans,
                            bsz,
                            weight.size(1),
                            input.size(2),
                            alpha,
                            gemm_beta,
                            workspace,
                            (T*)weight.data_ptr(),
                            (T*)output.data_ptr());
#else
        onednn_matmul_ex<T>(InferenceContext::Instance().GetCurrentStream(),
                            transposed_mode,
                            false,
                            bsz,
                            weight.size(1),
                            input.size(2),
                            alpha,
                            gemm_beta,
                            workspace,
                            (T*)weight.data_ptr(),
                            (T*)output.data_ptr());
#endif
    }
    if (add_bias)
        launch_bias_add((T*)output.data_ptr(),
                        (T*)bias.data_ptr(),
                        q_int8 ? weight.size(0) : weight.size(1),
                        bsz,
                        InferenceContext::Instance().GetCurrentStream());

    auto output_stride = c10::TensorType::contiguousStridesOf(input.sizes());
    return at::from_blob(
        workspace, input.sizes(), output_stride, nullptr, input.options(), input.device());
}

template <typename T>
std::vector<at::Tensor> ds_qkv_gemm(at::Tensor& input,
                                    at::Tensor& weight,
                                    at::Tensor& q_scale,
                                    at::Tensor& bias,
                                    at::Tensor& gamma,
                                    at::Tensor& beta,
                                    const float epsilon,
                                    bool add_bias,
                                    bool q_int8,
                                    bool transposed_mode)
{
    int bsz = input.size(0) * input.size(1);
    T* workspace = (T*)InferenceContext::Instance().GetWorkSpace();
    int out_size = (transposed_mode || q_int8) ? weight.size(0) : weight.size(1);

    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(false);

    auto output_stride =
        c10::TensorType::contiguousStridesOf({input.size(0), input.size(1), out_size});
    auto output = at::from_blob(workspace,
                                {input.size(0), input.size(1), out_size},
                                output_stride,
                                nullptr,
                                options,
                                input.device());
    auto inp_norm = qkv_unfused_sycl<T>(output,
                                        input,
                                        weight,
                                        q_scale,
                                        bias,
                                        gamma,
                                        beta,
                                        epsilon,
                                        add_bias,
                                        q_int8,
                                        transposed_mode);

    return {output, inp_norm};
}

template <typename T>
at::Tensor mlp_unfused_sycl(at::Tensor& output,
                            at::Tensor& input,
                            at::Tensor& residual,
                            at::Tensor& input_bias,
                            at::Tensor& weight,
                            at::Tensor& weight1,
                            at::Tensor& bias,
                            at::Tensor& gamma,
                            at::Tensor& beta,
                            const float epsilon,
                            bool preLayerNorm,
                            bool mlp_after_attn,
                            at::Tensor& q_scale,
                            at::Tensor& q_scale1,
                            bool q_int8,
                            ActivationFuncType act_func_type,
                            bool transposed_mode)
{
    int bsz = input.size(0) * input.size(1);
    T* inp_norm = (T*)InferenceContext::Instance().GetWorkSpace() + torch::numel(input) +
                  torch::numel(output);
    T* intermediate = inp_norm + torch::numel(input);

    if (mlp_after_attn) {
        launch_fused_residual_ln((T*)inp_norm,
                                 (const T*)input.data_ptr(),
                                 (const T*)residual.data_ptr(),
                                 (const T*)input_bias.data_ptr(),
                                 (const T*)gamma.data_ptr(),
                                 (const T*)beta.data_ptr(),
                                 epsilon,
                                 bsz,
                                 input.size(2),
                                 InferenceContext::Instance().GetCurrentStream());
    } else {
        ds_layer_norm_internal(inp_norm, input, gamma, beta, epsilon);
    }
    if (q_int8) {
        throw std::runtime_error("q_int8=true is not supported!");
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
#ifdef USE_MKL_GEMM
        onemkl_matmul_ex<T>(InferenceContext::Instance().GetCurrentStream(),
                            transposed_mode ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans,
                            oneapi::mkl::transpose::nontrans,
                            bsz,
                            weight.size(1),
                            input.size(2),
                            alpha,
                            gemm_beta,
                            inp_norm,
                            (T*)weight.data_ptr(),
                            intermediate);
#else
        onednn_matmul_ex<T>(InferenceContext::Instance().GetCurrentStream(),
                            transposed_mode,
                            false,
                            bsz,
                            weight.size(1),
                            input.size(2),
                            alpha,
                            gemm_beta,
                            inp_norm,
                            (T*)weight.data_ptr(),
                            intermediate);
#endif
    }
    if (act_func_type == ActivationFuncType::GELU) {
        launch_bias_gelu(intermediate,
                         (T*)bias.data_ptr(),
                         (transposed_mode || q_int8) ? weight.size(0) : weight.size(1),
                         bsz,
                         InferenceContext::Instance().GetCurrentStream());
    } else if (act_func_type == ActivationFuncType::ReLU) {
        throw std::runtime_error("act_func_type=relu is not supported!");
    }

    if (q_int8) {
        throw std::runtime_error("q_int8=true is not supported!");
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
#ifdef USE_MKL_GEMM
        onemkl_matmul_ex<T>(InferenceContext::Instance().GetCurrentStream(),
                            transposed_mode ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans,
                            oneapi::mkl::transpose::nontrans,
                            bsz,
                            weight1.size(transposed_mode ? 0 : 1),
                            weight1.size(transposed_mode ? 1 : 0),
                            alpha,
                            gemm_beta,
                            intermediate,
                            (T*)weight1.data_ptr(),
                            (T*)output.data_ptr());
#else
        onednn_matmul_ex<T>(InferenceContext::Instance().GetCurrentStream(),
                            transposed_mode,
                            false,
                            bsz,
                            weight1.size(transposed_mode ? 0 : 1),
                            weight1.size(transposed_mode ? 1 : 0),
                            alpha,
                            gemm_beta,
                            intermediate,
                            (T*)weight1.data_ptr(),
                            (T*)output.data_ptr());
#endif
    }

    auto output_stride = c10::TensorType::contiguousStridesOf(input.sizes());
    return at::from_blob(
        inp_norm, input.sizes(), output_stride, nullptr, input.options(), input.device());
}

template <typename T>
std::vector<at::Tensor> ds_mlp_gemm(at::Tensor& input,
                                    at::Tensor& residual,
                                    at::Tensor& input_bias,
                                    at::Tensor& weight_interm,
                                    at::Tensor& weight_out,
                                    at::Tensor& bias,
                                    at::Tensor& gamma,
                                    at::Tensor& beta,
                                    const float epsilon,
                                    bool preLayerNorm,
                                    bool mlp_after_attn,
                                    at::Tensor& q_scale,
                                    at::Tensor& q_scale1,
                                    bool q_int8,
                                    int activation_type,
                                    bool transposed_mode)
{
    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kXPU)
                       .requires_grad(false);

    int out_size = (q_int8 || transposed_mode) ? weight_out.size(0) : weight_out.size(1);
    auto output_stride =
        c10::TensorType::contiguousStridesOf({input.size(0), input.size(1), out_size});
    auto output =
        at::from_blob((T*)InferenceContext::Instance().GetWorkSpace() + torch::numel(input),
                      {input.size(0), input.size(1), out_size},
                      output_stride,
                      nullptr,
                      options,
                      input.device());
    int bsz = input.size(0) * input.size(1);

    auto act_func_type = static_cast<ActivationFuncType>(activation_type);
    auto res_add = mlp_unfused_sycl<T>(output,
                                       mlp_after_attn ? input : residual,
                                       residual,
                                       input_bias,
                                       weight_interm,
                                       weight_out,
                                       bias,
                                       gamma,
                                       beta,
                                       epsilon,
                                       preLayerNorm,
                                       mlp_after_attn,
                                       q_scale,
                                       q_scale1,
                                       q_int8,
                                       act_func_type,
                                       transposed_mode);

    return {output, res_add};
}

template <typename T>
at::Tensor fused_gemm_gelu(at::Tensor& input,
                           at::Tensor& weight,
                           at::Tensor& weight_scale,
                           at::Tensor& bias,
                           at::Tensor& weight_out,
                           at::Tensor& weight_out_scale,
                           bool q_int8,
                           bool transposed_mode)
{
    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    int intm_dim = (transposed_mode || q_int8) ? weight.size(0) : weight.size(1);

    auto intermediate = at::empty({input.size(0), input.size(1), intm_dim}, options);

    int bsz = input.size(0) * input.size(1);

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    if (q_int8) {
        throw std::runtime_error("q_int8=true is not supported!");
    } else {
#ifdef USE_MKL_GEMM
        onemkl_matmul_ex<T>(InferenceContext::Instance().GetCurrentStream(),
                            transposed_mode ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans,
                            oneapi::mkl::transpose::nontrans,
                            bsz,
                            intm_dim,
                            input.size(2),
                            alpha,
                            gemm_beta,
                            (T*)input.data_ptr(),
                            (T*)weight.data_ptr(),
                            (T*)intermediate.data_ptr());
#else
        onednn_matmul_ex<T>(InferenceContext::Instance().GetCurrentStream(),
                            transposed_mode,
                            false,
                            bsz,
                            intm_dim,
                            input.size(2),
                            alpha,
                            gemm_beta,
                            (T*)input.data_ptr(),
                            (T*)weight.data_ptr(),
                            (T*)intermediate.data_ptr());
#endif
    }
    launch_bias_gelu((T*)intermediate.data_ptr(),
                     (T*)bias.data_ptr(),
                     intm_dim,
                     bsz,
                     InferenceContext::Instance().GetCurrentStream());

    int out_size = (transposed_mode || q_int8) ? weight_out.size(0) : weight_out.size(1);
    auto output = at::empty({input.size(0), input.size(1), out_size}, options);
    if (q_int8) {
        throw std::runtime_error("q_int8=true is not supported!");
    } else {
#ifdef USE_MKL_GEMM
        onemkl_matmul_ex<T>(InferenceContext::Instance().GetCurrentStream(),
                            transposed_mode ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans,
                            oneapi::mkl::transpose::nontrans,
                            bsz,
                            out_size,
                            intm_dim,
                            alpha,
                            gemm_beta,
                            (T*)intermediate.data_ptr(),
                            (T*)weight_out.data_ptr(),
                            (T*)output.data_ptr());
#else
        onednn_matmul_ex<T>(InferenceContext::Instance().GetCurrentStream(),
                            transposed_mode,
                            false,
                            bsz,
                            out_size,
                            intm_dim,
                            alpha,
                            gemm_beta,
                            (T*)intermediate.data_ptr(),
                            (T*)weight_out.data_ptr(),
                            (T*)output.data_ptr());
#endif
    }
    return output;
}

template <typename T>
at::Tensor ds_vector_matmul(at::Tensor& input,
                            at::Tensor& weight,
                            bool async_op,
                            at::Tensor& q_scale,
                            bool q_int8,
                            bool transposed_mode)
{
    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kXPU)
                       .requires_grad(false);
    int out_size = q_int8 ? weight.size(0) : weight.size(1);
    int bsz = input.size(0) * input.size(1);

    T* workspace = (T*)InferenceContext::Instance().GetWorkSpace();
    auto output_stride =
        c10::TensorType::contiguousStridesOf({input.size(0), input.size(1), out_size});
    auto output = at::from_blob(workspace,
                                {input.size(0), input.size(1), out_size},
                                output_stride,
                                nullptr,
                                options,
                                input.device());
    if (q_int8) {
        throw std::runtime_error("q_int8=true is not supported!");
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
#ifdef USE_MKL_GEMM
        onemkl_matmul_ex<T>(InferenceContext::Instance().GetCurrentStream(),
                            transposed_mode ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans,
                            oneapi::mkl::transpose::nontrans,
                            bsz,
                            weight.size(transposed_mode ? 0 : 1),
                            input.size(2),
                            alpha,
                            gemm_beta,
                            (T*)input.data_ptr(),
                            (T*)weight.data_ptr(),
                            (T*)output.data_ptr());
#else
        onednn_matmul_ex<T>(InferenceContext::Instance().GetCurrentStream(),
                            transposed_mode,
                            false,
                            bsz,
                            weight.size(transposed_mode ? 0 : 1),
                            input.size(2),
                            alpha,
                            gemm_beta,
                            (T*)input.data_ptr(),
                            (T*)weight.data_ptr(),
                            (T*)output.data_ptr());
#endif
    }
    return output;
}

#define DISPATCH_VECTOR_ADD(T_TYPE, C_TYPE)                                         \
    if (a.scalar_type() == at::k##T_TYPE) {                                         \
        launch_vector_add<C_TYPE>((C_TYPE*)(a.data_ptr()),                          \
                                  (const C_TYPE*)(a.data_ptr()),                    \
                                  (const C_TYPE*)(b.data_ptr()),                    \
                                  gamma,                                            \
                                  total_elems,                                      \
                                  InferenceContext::Instance().GetCurrentStream()); \
    }

at::Tensor& _vector_add(at::Tensor& a, at::Tensor& b, float gamma)
{
    const int total_elems = a.numel();

    DISPATCH_VECTOR_ADD(Float, float)
    DISPATCH_VECTOR_ADD(Half, sycl::half)
#ifdef BF16_AVAILABLE
    DISPATCH_VECTOR_ADD(BFloat16, bf16)
#endif

    return a;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("layer_norm", &ds_layer_norm, "DeepSpeed layer norm (SYCL)");
    m.def("_vector_add", &_vector_add, "DeepSpeed vector add (SYCL)");
    m.def(
        "_layer_norm_residual", &ds_layer_norm_residual, "DeepSpeed layer norm + residual (SYCL)");

#define DEF_OPS(_name, _dtype)                                                                    \
    m.def("softmax_" #_name, &ds_softmax<_dtype>, "DeepSpeed SoftMax with " #_name " (SYCL)");    \
    m.def("layer_norm_" #_name, &ds_layer_norm_test<_dtype>, "DeepSpeed layer norm (SYCL)");      \
    m.def("qkv_gemm_" #_name, &ds_qkv_gemm<_dtype>, "DeepSpeed qkv gemm with " #_name " (SYCL)"); \
    m.def("mlp_gemm_" #_name, &ds_mlp_gemm<_dtype>, "DeepSpeed mlp with " #_name " (SYCL)");      \
    m.def("vector_matmul_" #_name,                                                                \
          &ds_vector_matmul<_dtype>,                                                              \
          "DeepSpeed vector-MM with " #_name " (SYCL)");                                          \
    m.def("residual_add_bias_" #_name,                                                            \
          &residual_add_bias<_dtype>,                                                             \
          "DeepSpeed residual add with " #_name " (SYCL)");                                       \
    m.def("fused_gemm_gelu_" #_name,                                                              \
          &fused_gemm_gelu<_dtype>,                                                               \
          "DeepSpeed mlp with " #_name " (SYCL)");                                                \
    m.def("allocate_workspace_" #_name,                                                           \
          &allocate_workspace<_dtype>,                                                            \
          "DeepSpeed memory allocation for GPT inference with " #_name " (SYCL)")

    DEF_OPS(fp32, float);
    DEF_OPS(fp16, fp16);
#ifndef USE_MKL_GEMM
    DEF_OPS(bf16, bf16);
#endif
}
