// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/*
 *half* Reduction_Workspace:  1. Requiring an extra memory space in device memory for un-reducted intermediate output tensors
 *                            2. Reduction_Workspace_Size = Split_K * M_Global * N_Global * sizeof(fp16)
 */
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
                          float           *Reduction_Workspace,                 // Identical workspace for all QuantGEMM kernel launches
                          int             Split_K);
