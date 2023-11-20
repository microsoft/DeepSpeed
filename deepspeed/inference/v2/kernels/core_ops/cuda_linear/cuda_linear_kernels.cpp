// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "cuda_linear_kernels.h"

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
}