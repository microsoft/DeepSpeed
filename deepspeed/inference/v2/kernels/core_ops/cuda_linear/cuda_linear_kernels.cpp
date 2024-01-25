// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "cuda_linear_kernels.h"
#include "GenMatrix_QuantLLM.h"

void Launch_QuantGEMM(torch::Tensor output,
                      torch::Tensor Weight1,  // 2bit
                      torch::Tensor Weight2,  // 4bit
                      torch::Tensor B,
                      torch::Tensor Scales,
                      const int M_Global,
                      const int N_Global,
                      const int K_Global,
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
    Launch_QuantGEMM(
        output, weights_2bit, weights_4bit, hidden_states, scale, M, N, K, split_k, workspace);
}

/*
 * Inputs:
 * (1) torch::Tensor weight[M, K] in FP16
 * Outputs:
 * (1) torch::Tensor weight_2bit and weight_4bit
 */
std::vector<torch::Tensor> preprocess_weight(torch::Tensor& Weight)
{
    TORCH_CHECK(Weight.dim() == 2, "weight must be 2-dimensional");
    TORCH_CHECK(Weight.scalar_type() == torch::kFloat16, "weight must be FP16");
    TORCH_CHECK(Weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(Weight.device().type() == torch::kCPU, "weight must be on CPU");
    auto M = Weight.size(0);
    auto K = Weight.size(1);
    TORCH_CHECK(K % 4 == 0, "K must be multiple of 4");

    // Pack Weight
    auto Weight_ptr = Weight.data_ptr<at::Half>();
    std::vector<uint8_t> Weight_6bit_Packed(M * K * 6 / 8);
    PackMatrix_Weight_FP6((uint16_t*)Weight_ptr, Weight_6bit_Packed.data(), M, K);

    // Split Weight
    auto Weight_2bit = torch::empty({M * K * 2 / 8}, torch::kUInt8);
    auto Weight_4bit = torch::empty({M * K * 4 / 8}, torch::kUInt8);
    GenMatrix_Weight_FP6(Weight_6bit_Packed.data(),
                         Weight_2bit.data_ptr<uint8_t>(),
                         Weight_4bit.data_ptr<uint8_t>(),
                         M,
                         K);

    return {Weight_2bit, Weight_4bit};
}
