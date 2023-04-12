// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <cuda_fp16.h>
#include "ds_kernel_utils.h"
#include "memory_access_utils.h"

namespace pwise {
constexpr int granularity = 16;
constexpr int unroll = 4;
constexpr int threads = 256;
}  // namespace pwise

template <typename T>
__global__ void vector_add_kernel(T* out, const T* a, const T* b, int num_elems)
{
    constexpr int T_per_access = pwise::granularity / sizeof(T);

    const int block_offset = blockIdx.x * pwise::threads * pwise::unroll * T_per_access;
    const int thread_offset = threadIdx.x * T_per_access;
    const int total_offset = block_offset + thread_offset;
    constexpr int stride = pwise::threads * T_per_access;

#pragma unroll
    for (int i = 0; i < pwise::unroll; i++) {
        T temp_buf_a[T_per_access], temp_buf_b[T_per_access];

        const int iter_idx = total_offset + i * stride;

        mem_access::load_global<pwise::granularity>(temp_buf_a, a + iter_idx, iter_idx < num_elems);
        mem_access::load_global<pwise::granularity>(temp_buf_b, b + iter_idx, iter_idx < num_elems);

#pragma unroll
        for (int j = 0; j < T_per_access; j++) { temp_buf_a[j] += temp_buf_b[j]; }

        if (iter_idx < num_elems) {
            mem_access::store_global<pwise::granularity>(out + iter_idx, temp_buf_a);
        }
    }
}

template <typename T>
void launch_vector_add(T* out, const T* a, const T* b, int num_elems, cudaStream_t stream)
{
    constexpr int T_per_access = pwise::granularity / sizeof(T);
    constexpr int T_per_block = pwise::threads * T_per_access * pwise::unroll;

    dim3 block(pwise::threads);
    dim3 grid((num_elems + T_per_block - 1) / T_per_block);

    vector_add_kernel<<<grid, block, 0, stream>>>(out, a, b, num_elems);
}

template void launch_vector_add<float>(float* out,
                                       const float* a,
                                       const float* b,
                                       int num_elems,
                                       cudaStream_t stream);

template void launch_vector_add<__half>(__half* out,
                                        const __half* a,
                                        const __half* b,
                                        int num_elems,
                                        cudaStream_t stream);
