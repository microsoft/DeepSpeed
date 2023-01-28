/*
Copyright 2023 The Microsoft DeepSpeed Team

This file contains implementations of activation functions. Specific
kernels implemented include:
    Generic fused bias + activation kernel (gelu, relu, identity)
    Fused bias + GeGLU kernel
*/

#include "activation_utils.h"
#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "inference_cuda_layers.h"
#include "memory_access_utils.h"

/*
In-place activation(biasAdd(x)) for channels last
*/
template <typename T, activation::Type ActFn>
__global__ void fused_bias_act(T* input, const T* bias, int total_count, int intermediate_size)
{
#ifndef DISABLE_KERNEL_BUILD
    // Input restriction: intermediate_size % vals_per_access == 0
    constexpr int granularity = 16;
    constexpr int values_per_access = granularity / sizeof(T);
    const int offset = (blockIdx.x * blockDim.x + threadIdx.x) * values_per_access;

    if (offset < total_count) {
        T data[values_per_access];
        T data_bias[values_per_access];
        mem_access::load_global<granularity>(data, input + offset);
        mem_access::load_global<granularity>(data_bias, bias + (offset % intermediate_size));

#pragma unroll
        for (int i = 0; i < values_per_access; i++) {
            data[i] = activation::func<ActFn>(data[i] + data_bias[i])
        }

        mem_access::store_global<granularity>(input + offset, data);
    }
#endif
}

template <typename T>
void launch_bias_gelu(T* input,
                      const T* bias,
                      int intermediate_size,
                      int batch_size,
                      cudaStream_t stream)
{
    constexpr int threads = 1024;
    constexpr int granularity = 16;

    const int total_count = batch_size * intermediate_size;
    const int elems_per_block = threads * (granularity / sizeof(T));
    dim3 block_dims(threads);
    dim3 grid_dims((total_count + elems_per_block - 1) / elems_per_block);

    fused_bias_act<T, activation::Type::GELU>
        <<<grid_dims, block_dims, 0, stream>>>(input, bias, total_count, intermediate_size);
}

template void launch_bias_gelu<inference_data_t>(inference_data_t*,
                                                 const inference_data_t*,
                                                 int,
                                                 int,
                                                 cudaStream_t);

template <typename T>
void launch_bias_relu(T* input,
                      const T* bias,
                      int intermediate_size,
                      int batch_size,
                      cudaStream_t stream)
{
    constexpr int threads = 1024;
    constexpr int granularity = 16;

    const int total_count = batch_size * intermediate_size;
    const int elems_per_block = threads * (granularity / sizeof(T));
    dim3 block_dims(threads);
    dim3 grid_dims((total_count + elems_per_block - 1) / elems_per_block);

    fused_bias_act<T, activation::Type::ReLU>
        <<<grid_dims, block_dims, 0, stream>>>(input, bias, total_count, intermediate_size);
}

template void launch_bias_relu<inference_data_t>(inference_data_t*,
                                                 const inference_data_t*,
                                                 int,
                                                 int,
                                                 cudaStream_t);

template <typename T>
void launch_bias_add(T* input,
                     const T* bias,
                     int intermediate_size,
                     int batch_size,
                     cudaStream_t stream)
{
    constexpr int threads = 1024;
    constexpr int granularity = 16;

    const int total_count = batch_size * intermediate_size;
    const int elems_per_block = threads * (granularity / sizeof(T));
    dim3 block_dims(threads);
    dim3 grid_dims((total_count + elems_per_block - 1) / elems_per_block);

    fused_bias_act<T, activation::Type::Identity>
        <<<grid_dims, block_dims, 0, stream>>>(input, bias, total_count, intermediate_size);
}

template void launch_bias_add<inference_data_t>(inference_data_t*,
                                                const inference_data_t*,
                                                int,
                                                int,
                                                cudaStream_t);

/*

GeGLU kernel

This kernel requires different memory scheduling than the above kernel; as such, it is implemented
separately.
*/

namespace fused_geglu {
constexpr int threads = 256;
constexpr int steps = 2;
constexpr int granularity = 16;
}  // namespace fused_geglu

template <typename T>
__global__ void fused_bias_geglu(T* output,
                                 const T* activation,
                                 const T* bias,
                                 int base_channels,
                                 int total_elems)
{
#ifndef DISABLE_KERNEL_BUILD
    constexpr int T_per_access = fused_geglu::granularity / sizeof(T);
    constexpr int T_per_step = T_per_access * fused_geglu::threads;
    constexpr int T_per_block = T_per_step * fused_geglu::steps;

    const int id = blockIdx.x * T_per_block + threadIdx.x * T_per_access;

#pragma unroll
    for (int i = 0; i < fused_geglu::steps; i++) {
        T activation_buffer_1[T_per_access];
        T activation_buffer_2[T_per_access];
        T bias_buffer_1[T_per_access];
        T bias_buffer_2[T_per_access];

        const int iter_id = id + T_per_step * i;
        if (iter_id < total_elems) {
            const int channel_id = iter_id % base_channels;
            const int seq_id = iter_id / base_channels;
            const int seq_offset = seq_id * base_channels * 2;

            mem_access::load_global<fused_geglu::granularity>(activation_buffer_1,
                                                              activation + seq_offset + channel_id);
            mem_access::load_global<fused_geglu::granularity>(
                activation_buffer_2, activation + seq_offset + channel_id + base_channels);
            mem_access::load_global<fused_geglu::granularity>(bias_buffer_1, bias + channel_id);
            mem_access::load_global<fused_geglu::granularity>(bias_buffer_2,
                                                              bias + channel_id + base_channels);

            // Since the GeLU is going to happen at float, might as well
            // convert (in theory if we could be SFU bottlenecked and this could hurt?)
#pragma unroll
            for (int v = 0; v < T_per_access; v++) {
                T hidden_state = activation_buffer_1[v] + bias_buffer_1[v];
                T pre_gate = activation_buffer_2[v] + bias_buffer_2[v];
                float gate_f =
                    activation::func<activation::Type::OldGELU>(conversion::to<float>(pre_gate));
                T gate = conversion::to<T>(gate_f);
                activation_buffer_1[v] = hidden_state * gate;
            }

            mem_access::store_global<fused_geglu::granularity>(output + iter_id,
                                                               activation_buffer_1);
        }
    }
#endif
}

template <typename T>
void launch_fused_bias_geglu(T* output,
                             const T* activation,
                             const T* bias,
                             int rows,
                             int elems_per_row,
                             cudaStream_t stream)
{
    /*
    Fused bias GEGLU is a variant of the gated activation functions.
    The input here is a matrix of [batch, seq_len, 2 * intermediate_dim]
    where the second half of the channels act as GeLU gates for the first
    half.
    */

    // Re-derive the above figures
    constexpr int T_per_access = fused_geglu::granularity / sizeof(T);
    constexpr int T_per_step = T_per_access * fused_geglu::threads;
    constexpr int T_per_block = T_per_step * fused_geglu::steps;

    const int base_channels = elems_per_row / 2;
    const int total_elems = base_channels * rows;

    dim3 block(fused_geglu::threads);
    dim3 grid((total_elems + T_per_block - 1) / T_per_block);

    fused_bias_geglu<<<grid, block, 0, stream>>>(
        output, activation, bias, base_channels, total_elems);
}

template void launch_fused_bias_geglu(inference_data_t*,
                                      const inference_data_t*,
                                      const inference_data_t*,
                                      int,
                                      int,
                                      cudaStream_t);
