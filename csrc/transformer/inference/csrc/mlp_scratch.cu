//----------------------------------------------------------------------------------------------------------------
//                          DS MLP GEMM
//----------------------------------------------------------------------------------------------------------------
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
                                    bool transposed_mode,
                                    int layer_id)
{
    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    int out_size = (q_int8 || transposed_mode) ? weight_out.size(0) : weight_out.size(1);
    auto output =
        at::from_blob((T*)InferenceContext::Instance().GetWorkSpace() + torch::numel(input),
                      {input.size(0), input.size(1), out_size},
                      options);
    int bsz = input.size(0) * input.size(1);

    auto act_func_type = static_cast<ActivationFuncType>(activation_type);
    auto res_add = mlp_unfused_cublas<T>(output,
                                         mlp_after_attn ? input : residual, // TODO (lekurile): comprehend this
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
                                         transposed_mode,
                                         layer_id);
    return {output, res_add};
}

//----------------------------------------------------------------------------------------------------------------
//                          MLP UNFUSED CUBLAS
//----------------------------------------------------------------------------------------------------------------

template <typename T>
at::Tensor mlp_unfused_cublas(at::Tensor& output,
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
                              bool transposed_mode,
                              int layer_id)
{
    int bsz = input.size(0) * input.size(1);
    T* inp_norm = (T*)InferenceContext::Instance().GetWorkSpace() + torch::numel(input) +
                  torch::numel(output);
    T* intermediate = inp_norm + torch::numel(input);

    if (mlp_after_attn) {
        // OPT models come here
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
        quantized_gemm<T>(
            intermediate, inp_norm, weight, q_scale, q_scale.size(0), bsz, input.size(2));
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
        cublasSetStream(InferenceContext::Instance().GetCublasHandle(),
                        InferenceContext::Instance().GetCurrentStream());
        cublas_gemm_ex(InferenceContext::Instance().GetCublasHandle(),
                       (transposed_mode ? CUBLAS_OP_T : CUBLAS_OP_N),
                       CUBLAS_OP_N,
                       weight.size(transposed_mode ? 0 : 1),
                       bsz,
                       input.size(2),
                       &alpha,
                       &gemm_beta,
                       (T*)weight.data_ptr(),
                       inp_norm,
                       intermediate,
#ifdef __HIP_PLATFORM_HCC__
                       rocblas_gemm_algo_standard);
#else
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    }
    if (act_func_type == ActivationFuncType::GELU) {
        launch_bias_gelu(intermediate,
                         (T*)bias.data_ptr(),
                         (transposed_mode || q_int8) ? weight.size(0) : weight.size(1),
                         bsz,
                         InferenceContext::Instance().GetCurrentStream());
    } else if (act_func_type == ActivationFuncType::ReLU) {
        launch_bias_relu(intermediate,
                         (T*)bias.data_ptr(),
                         (transposed_mode || q_int8) ? weight.size(0) : weight.size(1),
                         bsz,
                         InferenceContext::Instance().GetCurrentStream());
    }
    if (q_int8) {
        quantized_gemm<T>(output.data_ptr(),
                          intermediate,
                          weight1,
                          q_scale1,
                          q_scale1.size(0),
                          bsz,
                          input.size(2));
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
        cublasSetStream(InferenceContext::Instance().GetCublasHandle(),
                        InferenceContext::Instance().GetCurrentStream());
        cublas_gemm_ex(InferenceContext::Instance().GetCublasHandle(),
                       (transposed_mode ? CUBLAS_OP_T : CUBLAS_OP_N),
                       CUBLAS_OP_N,
                       weight1.size(transposed_mode ? 0 : 1),
                       bsz,
                       weight1.size(transposed_mode ? 1 : 0),
                       &alpha,
                       &gemm_beta,
                       (T*)weight1.data_ptr(),
                       intermediate,
                       (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                       rocblas_gemm_algo_standard);
#else
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    }

    return torch::from_blob(inp_norm, input.sizes(), input.options());
}


//----------------------------------------------------------------------------------------------------------------
// (DONE, NEEDS TESTING)            Fused Residual Layer Norm
//----------------------------------------------------------------------------------------------------------------

template <typename T>
at::Tensor mlp_layer_norm(//at::Tensor& output,
                          at::Tensor& input,
                          at::Tensor& residual,
                          at::Tensor& input_bias,
                          at::Tensor& gamma,
                          at::Tensor& beta,
                          const float epsilon,
                          bool mlp_after_attn,
                          int layer_id)
{
    int bsz = input.size(0) * input.size(1);
    T* inp_norm = (T*)InferenceContext::Instance().GetWorkSpace() + torch::numel(input);
    //T* inp_norm = (T*)InferenceContext::Instance().GetWorkSpace() + torch::numel(input) +
    //              torch::numel(output);
    if (mlp_after_attn) {
        // OPT models come here
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

    return torch::from_blob(inp_norm, input.sizes(), input.options());
}


//----------------------------------------------------------------------------------------------------------------
// (DONE, NEEDS TESTING)            Cublas GEMM FC1
//----------------------------------------------------------------------------------------------------------------

// TODO (lekurile): can ds_vector_matmul be used here?

template <typename T>
at::Tensor mlp_gemm_fc(at::Tensor& inp_norm,
                       at::Tensor& input, // TODO (lekurile): change this to size param?
                       at::Tensor& weight,
                       at::Tensor& q_scale,
                       bool q_int8,
                       bool transposed_mode,
                       int layer_id)
{
    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    int out_size = (q_int8 || transposed_mode) ? weight.size(0) : weight.size(1);
    int bsz = input.size(0) * input.size(1);
    auto output =
        at::from_blob((T*)InferenceContext::Instance().GetWorkSpace(),
                      {input.size(0), input.size(1), out_size},
                      options);

    if (q_int8) {
        quantized_gemm<T>(
            output.data_ptr(), inp_norm, weight, q_scale, q_scale.size(0), bsz, input.size(2));
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
        cublasSetStream(InferenceContext::Instance().GetCublasHandle(),
                        InferenceContext::Instance().GetCurrentStream());
        cublas_gemm_ex(InferenceContext::Instance().GetCublasHandle(),
                       (transposed_mode ? CUBLAS_OP_T : CUBLAS_OP_N),
                       CUBLAS_OP_N,
                       weight.size(transposed_mode ? 0 : 1),
                       bsz,
                       input.size(2),
                       //weight1.size(transposed_mode ? 1 : 0), // TODO (lekurile): which sizing to use?
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
    return output;
}


//----------------------------------------------------------------------------------------------------------------
// (DONE, TESTING)            Activation (ReLU or GELU)
//----------------------------------------------------------------------------------------------------------------

template <typename T>
at::Tensor mlp_activation(at::Tensor& input,
                          at::Tensor& input_mlp,
                          at::Tensor& weight,
                          at::Tensor& bias,
                          bool q_int8,
                          int activation_type,
                          bool transposed_mode,
                          int layer_id)
{
    auto act_func_type = static_cast<ActivationFuncType>(activation_type);
    int bsz = input_mlp.size(0) * input_mlp.size(1);

    if (act_func_type == ActivationFuncType::GELU) {
        launch_bias_gelu((T*)input.data_ptr(),
                         (T*)bias.data_ptr(),
                         (transposed_mode || q_int8) ? weight.size(0) : weight.size(1),
                         bsz,
                         InferenceContext::Instance().GetCurrentStream());
    } else if (act_func_type == ActivationFuncType::ReLU) {
        launch_bias_relu((T*)input.data_ptr(),
                         (T*)bias.data_ptr(),
                         (transposed_mode || q_int8) ? weight.size(0) : weight.size(1),
                         bsz,
                         InferenceContext::Instance().GetCurrentStream());
    }

    return input;
}


//----------------------------------------------------------------------------------------------------------------
// (IN PROGRESS)            Cublas GEMM FC2
//----------------------------------------------------------------------------------------------------------------

template <typename T>
at::Tensor mlp_unfused_cublas(at::Tensor& output,
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
                              bool transposed_mode,
                              int layer_id)
{
    if (q_int8) {
        quantized_gemm<T>(output.data_ptr(),
                          intermediate,
                          weight1,
                          q_scale1,
                          q_scale1.size(0),
                          bsz,
                          input.size(2));
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
        cublasSetStream(InferenceContext::Instance().GetCublasHandle(),
                        InferenceContext::Instance().GetCurrentStream());
        cublas_gemm_ex(InferenceContext::Instance().GetCublasHandle(),
                       (transposed_mode ? CUBLAS_OP_T : CUBLAS_OP_N),
                       CUBLAS_OP_N,
                       weight1.size(transposed_mode ? 0 : 1),
                       bsz,
                       weight1.size(transposed_mode ? 1 : 0),
                       &alpha,
                       &gemm_beta,
                       (T*)weight1.data_ptr(),
                       intermediate,
                       (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                       rocblas_gemm_algo_standard);
#else
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    }

    return X;
}


//----------------------------------------------------------------------------------------------------------------
// (IN PROGRESS)            Final Return
//----------------------------------------------------------------------------------------------------------------
    return torch::from_blob(inp_norm, input.sizes(), input.options());
