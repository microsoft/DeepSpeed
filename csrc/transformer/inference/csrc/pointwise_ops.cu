/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include "activation_utils.h"
#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "inference_cuda_layers.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;

/* Residual add functions */
template <typename T>
__global__ void fused_bias_residual_postln(T* residual,
                                           const T* hidden_state,
                                           const T* bias,
                                           const int total_count,
                                           const int intermediate_size)
{
#ifndef DISABLE_KERNEL_BUILD
    constexpr int granularity = 16;
    constexpr int T_per_access = granularity / sizeof(T);
    const int offset = (blockIdx.x * blockDim.x + threadIdx.x) * T_per_access;

    if (offset < total_count) {
        T res_data[T_per_access];
        const T hs_data[T_per_access], b_data[T_per_access];

        mem_access::load_global<granularity>(res_data, residual + offset);
        mem_access::load_global<granularity>(hs_data, hidden_state + offset);
        mem_access::load_global<granularity>(b_data, bias + (offset % intermediate_size));

#pragma unroll
        for (int i = 0; i < T_per_access; i++) {
            float up_res = conversion::to<float>(res_data[i]);
            float up_hs = conversion::to<float>(hs_data[i]);
            float up_b = conversion::to<float>(b_data[i]);
            res_data[i] = conversion::to<T>(up_res + up_hs + up_b);
        }

        mem_access::store_global<granularity>(residual + offset, res_data);
    }
#endif
}

template <typename T>
__global__ void fused_bias_residual_preln(T* residual,
                                          const T* hidden_state,
                                          const T* attn,
                                          const T* bias,
                                          const T* attn_bias,
                                          const int total_count,
                                          const int intermediate_size,
                                          const float mp_scale)
{
#ifndef DISABLE_KERNEL_BUILD
    constexpr int granularity = 16;
    constexpr int T_per_access = granularity / sizeof(T);
    const int offset = (blockIdx.x * blockDim.x + threadIdx.x) * T_per_access;

    if (offset < total_count) {
        T res_data[T_per_access];
        const T hs_data[T_per_access], attn_data[T_per_access], b_data[T_per_access],
            ab_data[T_per_access];

        mem_access::load_global<granularity>(res_data, residual + offset);
        mem_access::load_global<granularity>(hs_data, hidden_state + offset);
        mem_access::load_global<granularity>(attn_data, attn + offset);
        mem_access::load_global<granularity>(b_data, bias + (offset % intermediate_size));
        mem_access::load_global<granularity>(ab_data, attn_bias + (offset % intermediate_size));

#pragma unroll
        for (int i = 0; i < T_per_access; i++) {
            // These are no-ops for T=float
            float up_res = conversion::to<float>(res_data[i]);
            const float up_hs = conversion::to<float>(hs_data[i]);
            const float up_attn = conversion::to<float>(attn_data[i]);
            const float up_b = conversion::to<float>(b_data[i]);
            const float up_ab = conversion::to<float>(ab_data[i]);

            res_data[i] = conversion::to<T>((up_res + up_attn + up_b + up_ab) * mp_scale + up_hs);
        }

        mem_access::store_global<granularity>(residual + offset, res_data);
    }
#endif
}

template <typename T>
void launch_bias_residual(T* residual,
                          T* hidden_state,
                          T* attn,
                          T* bias,
                          T* attn_bias,
                          int batch,
                          int hidden_dim,
                          int mp_size,
                          bool preln,
                          cudaStream_t stream)
{
    constexpr int threads = 1024;
    constexpr int granularity = 16;

    const int total_count = batch * hidden_dim;
    const int elems_per_block = threads * (granularity / sizeof(T));
    dim3 block(threads);
    dim3 grid((total_count + elems_per_block - 1) / elems_per_block);

    if (preln) {
        fused_bias_residual_preln<<<grid, block, 0, stream>>>(
            residual, hidden_state, attn, bias, attn_bias, total_count, hidden_dim, 1.0 / mp_size);
    } else {
        fused_bias_residual_postln<<<grid, block, 0, stream>>>(
            residual, hidden_state, bias, total_count, hidden_dim);
    }
}

template void launch_bias_residual<inference_data_t>(inference_data_t*,
                                                     inference_data_t*,
                                                     inference_data_t*,
                                                     inference_data_t*,
                                                     inference_data_t*,
                                                     int,
                                                     int,
                                                     int,
                                                     bool,
                                                     cudaStream_t);

template <typename T>
__global__ void moe_res_matmul(T* residual, T* coef, T* mlp_out, int seq_len, int hidden_dim)
{
#ifndef DISABLE_KERNEL_BUILD
    constexpr int granularity = 16;
    constexpr int vals_per_access = granularity / sizeof(T);

    T* residual_seq = residual + blockIdx.x * hidden_dim;
    T* mlp_out_seq = mlp_out + blockIdx.x * hidden_dim;

    for (unsigned tid = threadIdx.x * vals_per_access; tid < hidden_dim;
         tid += blockDim.x * vals_per_access) {
        T mlp[vals_per_access];
        T res[vals_per_access];
        T coef1[vals_per_access];
        T coef2[vals_per_access];

        mem_access::load_global<granularity>(mlp, mlp_out_seq + tid);
        mem_access::load_global<granularity>(res, residual_seq + tid);
        mem_access::load_global<granularity>(coef1, coef + tid);
        mem_access::load_global<granularity>(coef2, coef + tid + hidden_dim);

#pragma unroll
        for (int idx = 0; idx < vals_per_access; idx++) {
            mlp[idx] = mlp[idx] * coef2[idx] + res[idx] * coef1[idx];
        }

        mem_access::store_global<granularity>(mlp_out_seq + tid, mlp);
    }
#endif
}

/* MoE Point-wise Kernel*/
template <typename T>
void launch_moe_res_matmul(T* residual,
                           T* coef,
                           T* mlp_out,
                           int seq_len,
                           int hidden_dim,
                           cudaStream_t stream)
{
    dim3 grid_dim(seq_len);
    dim3 block_dim(1024);
    moe_res_matmul<<<grid_dim, block_dim, 0, stream>>>(
        residual, coef, mlp_out, seq_len, hidden_dim);
}

template void launch_moe_res_matmul(inference_data_t* residual,
                                    inference_data_t* coef,
                                    inference_data_t* mlp_out,
                                    int seq_len,
                                    int hidden_dim,
                                    cudaStream_t stream);
