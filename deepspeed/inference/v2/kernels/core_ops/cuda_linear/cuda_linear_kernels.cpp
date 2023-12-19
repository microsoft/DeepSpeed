// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "cuda_linear_kernels.h"


torch::Tensor Launch_QuantGEMM( torch::Tensor output,
                                torch::Tensor Weight1, // 2bit
                                torch::Tensor Weight2, // 4bit
                                torch::Tensor B,
                                torch::Tensor Scales,
                                const int  M_Global,
                                const int  N_Global,
                                const int  K_Global,
                                const int Split_K,
                                torch::Tensor workspace);

void cuda_wf6af16_linear(torch::Tensor& output,
                         torch::Tensor& hidden_states,
                         torch::Tensor& weights_4bit,
                         torch::Tensor& weights_2bit,
                         torch::Tensor& scale,
                         torch::Tensor& workspace,
                         int M,
                         int N,
                         int K,
                         int split_k)
{
    TORCH_CHECK(weights_2bit.device().type() == torch::kCUDA, "weight_2bit must be on CUDA");
    TORCH_CHECK(weights_4bit.device().type() == torch::kCUDA, "weight_4bit must be on CUDA");
    TORCH_CHECK(hidden_states.device().type() == torch::kCUDA, "X must be on CUDA");
    TORCH_CHECK(scale.device().type() == torch::kCUDA, "scale must be on CUDA");
    Launch_QuantGEMM(output, weights_2bit, weights_4bit, hidden_states, scale, M, N, K, split_k, workspace);
}

/*
 * Inputs:
 * (1) torch::Tensor weight[M, K] in FP16
 * Outputs:
 * (1) torch::Tensor weight_2bit and weight_4bit
*/
std::vector<torch::Tensor> preprocess_weight(torch::Tensor& Weight) {
    TORCH_CHECK(Weight.dim() == 2, "weight must be 2-dimensional");
    TORCH_CHECK(Weight.scalar_type() == torch::kFloat16, "weight must be FP16");
    TORCH_CHECK(Weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(Weight.device().type() == torch::kCPU, "weight must be on CPU");
    auto M = Weight.size(0);
    auto K = Weight.size(1);
    TORCH_CHECK(K % 4 == 0, "K must be multiple of 4");

    // Pack Weight
    auto Weight_ptr = Weight.data_ptr<at::Half>();
    std::vector<uint8_t> Weight_6bit_Packed(M*K*6/8);
    PackMatrix_Weight_FP6((uint16_t *)Weight_ptr, Weight_6bit_Packed.data(), M, K);

    // Split Weight
    auto Weight_2bit = torch::empty({M*K*2/8}, torch::kUInt8);
    auto Weight_4bit = torch::empty({M*K*4/8}, torch::kUInt8);
    GenMatrix_Weight_FP6(Weight_6bit_Packed.data(), Weight_2bit.data_ptr<uint8_t>(), Weight_4bit.data_ptr<uint8_t>(), M, K);

    return {Weight_2bit, Weight_4bit};
}

/*
 * Inputs:
 * (1) torch::Tensor Scale_In[M, K/GroupSize] in FP16
 * Outputs:
 * (1) torch::Tensor Scale_Out[M, K/GroupSize] in FP16
*/

torch::Tensor preprocess_scales(torch::Tensor& Scale, int M, int K) {
    // Preprocess scales
    TORCH_CHECK(Scale.dim() == 2, "scale must be 2-dimensional");
    TORCH_CHECK(Scale.size(0) == M, "scale must have same M as weight");
    TORCH_CHECK(Scale.is_contiguous(), "scale must be contiguous");
    TORCH_CHECK(Scale.device().type() == torch::kCPU, "scale must be on CPU");
    TORCH_CHECK(Scale.scalar_type() == torch::kFloat16, "scale must be FP16");
    auto GroupSize = K / Scale.size(1);
    TORCH_CHECK(GroupSize % 64 == 0, "GroupSize must be multiple of 64");
    auto New_Scale = torch::empty_like(Scale);
    auto Scale_out = New_Scale.data_ptr<at::Half>();
    auto Scale_in = New_Scale.data_ptr<at::Half>();
    GenMatrix_Scale_FP16((uint8_t*)Scale_out, (uint8_t*)Scale_in, M, K, GroupSize);
    return New_Scale;
}
>>>>>>> update
