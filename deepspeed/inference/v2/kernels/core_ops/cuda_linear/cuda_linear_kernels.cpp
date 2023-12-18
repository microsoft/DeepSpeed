// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "cuda_linear_kernels.h"

torch::Tensor Launch_QuantGEMM(torch::Tensor Weight1,  // 2bit
                               torch::Tensor Weight2,  // 4bit
                               torch::Tensor B,
                               torch::Tensor Scales,
                               const int M_Global,
                               const int N_Global,
                               const int K_Global,
                               const int Split_K);

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
    Launch_QuantGEMM(weights_2bit, weights_4bit, hidden_states, scale, M, N, K, split_k);
}

void get_4and2bit_weights(torch::Tensor& weights_4bit,
                          torch::Tensor& weights_2bit,
                          torch::Tensor& weights)
{
    // TODO: split `weights` and fill `weights_4bit` and `weights_2bit`
}