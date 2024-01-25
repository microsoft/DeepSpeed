// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <ATen/cuda/CUDAContext.h>
#include "cuda_linear_kernels.h"

cudaError_t QuantGEMM_API(
    cudaStream_t stream,
    const uint4* Weight1,
    const uint4* Weight2,
    const half* Scales,
    const half* B,
    half* C,
    const size_t M_Global,
    const size_t N_Global,
    const size_t K_Global,
    float* Reduction_Workspace,  // Identical workspace for all QuantGEMM kernel launches
    int Split_K);

void Launch_QuantGEMM(torch::Tensor C,
                      torch::Tensor Weight1,  // 2bit
                      torch::Tensor Weight2,  // 4bit
                      torch::Tensor B,
                      torch::Tensor Scales,
                      const int M_Global,
                      const int N_Global,
                      const int K_Global,
                      const int Split_K,
                      torch::Tensor workspace)
{
    auto C_ptr = C.data_ptr<at::Half>();
    auto B_ptr = B.data_ptr<at::Half>();
    auto W1_ptr = Weight1.data_ptr<uint8_t>();
    auto W2_ptr = Weight2.data_ptr<uint8_t>();

    // auto workspace_size = M_Global * N_Global * Split_K;
    // auto workspace = torch::empty({workspace_size}, torch::kFloat16);

    auto status = QuantGEMM_API(at::cuda::getCurrentCUDAStream(),
                                (uint4*)W1_ptr,
                                (uint4*)W2_ptr,
                                (half*)Scales.data_ptr<at::Half>(),
                                (half*)B_ptr,
                                (half*)C_ptr,
                                M_Global,
                                N_Global,
                                K_Global,
                                workspace.data_ptr<float>(),
                                Split_K);
    if (status != cudaSuccess) {
        AT_ERROR("QuantGEMM_API failed with error: ", cudaGetErrorString(status));
    }
}
