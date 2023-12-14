// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "cuda_linear_kernels.h"
#include <ATen/cuda/CUDAContext.h>

cudaError_t QuantGEMM_API(cudaStream_t    stream,
                          const uint4     *Weight1,
                          const uint4     *Weight2,
                          const half      *Scales,
                          const int       QUANT_GROUP_SIZE_DIVIDED_BY_64,
                          const half      *B,
                          half            *C,
                          const size_t    M_Global,
                          const size_t    N_Global,
                          const size_t    K_Global, 
                          float           *Reduction_Workspace,  // Identical workspace for all QuantGEMM kernel launches
                          int             Split_K);

torch::Tensor Launch_QuantGEMM(torch::Tensor Weight1, // 2bit
                                torch::Tensor Weight2, // 4bit
                                torch::Tensor B,
                                torch::Tensor Scales,
                                const int M_Global,
                                const int N_Global,
                                const int K_Global,
                                const int Split_K)
{
    auto C = torch::empty({M_Global, N_Global}, torch::kFloat16);
    auto C_ptr = C.data_ptr<at::Half>();
    auto B_ptr = B.data_ptr<at::Half>();
    auto W1_ptr = Weight1.data_ptr<uint8_t>();
    auto W2_ptr = Weight2.data_ptr<uint8_t>();
    auto Group_Size = K_Global / Scales.size(1);

    auto workspace_size = M_Global * N_Global * Split_K;
    TORCH_CHECK(Split_K == 1, "Does not support Split_K != 0");
    // auto workspace = torch::empty({workspace_size}, torch::kFloat16);

    auto status = QuantGEMM_API(at::cuda::getCurrentCUDAStream(),
                               (uint4*)W1_ptr,
                               (uint4*)W2_ptr,
                               (half*)Scales.data_ptr<at::Half>(),
                               Group_Size / 64,
                               (half*)B_ptr,
                               (half*)C_ptr,
                               M_Global,
                               N_Global, 
                               K_Global,
                               // workspace.data_ptr<float>),
                               nullptr,
                               Split_K);
    if (status != cudaSuccess) {
        AT_ERROR("QuantGEMM_API failed with error: ", cudaGetErrorString(status));
    }
    return C;
}
