/*
Copyright 2023 The Microsoft DeepSpeed Team

This file includes kernel implementations for padding data without arithmetic.

Kernels implemented:
    - pad_heads_kernel: Pads a fused QKV Tensor along the head dimension.
    - pad_head_and_sequence: Pads a head transposed Tensor along the head and sequence dimensions.

In general, kernels like this are pure overhead and should be considered temporary solutions.
*/

#include "ds_kernel_utils.h"
#include "memory_access_utils.h"

/*
Pads a dense matrix along the head dimension.

Input: [batch_size, seq_length, 3 * heads, head_size]
Output: [batch_size, seq_length, 3 * heads, padded_head_size]
*/
template <typename T>
__global__ void pad_heads_kernel(T* padded_output,
                                 const T* unpadded_input,
                                 int total_heads,
                                 int head_size,
                                 int padded_head_size)
{
    constexpr int granularity = 16;
    constexpr int T_per_access = granularity / sizeof(T);

    // We pack multiple heads inside the same block to achieve higher
    // occupancy. The y-dimension of the block corresponds with the head_id
    // within the block.
    const int head_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int thread_offset = threadIdx.x * T_per_access;
    const int block_read_offset = head_id * head_size + thread_offset;
    const int block_write_offset = head_id * padded_head_size + thread_offset;

    if (head_id < total_heads) {
        bool do_mem = thread_id < head_size;

        T buffer[T_per_access];
        mem_access::load_global<granularity>(buffer, unpadded_input + block_read_offset, do_mem);
        mem_access::store_global<granularity>(padded_output + block_write_offset, buffer);
    }
}

template <typename T>
void launch_pad_heads(T* padded_output,
                      const T* unpadded_input,
                      int total_heads,
                      int head_size,
                      int padded_head_size,
                      cudaStream_t stream)
{
    constexpr int granularity = 16;
    constexpr int T_per_access = granularity / sizeof(T);

    const int threads_per_block = 512;
    const int threads_per_head = (padded_head_size + T_per_access - 1) / T_per_access;
    const int heads_per_block = (threads_per_block + threads_per_head - 1) / threads_per_head;
    const int blocks = (total_heads + heads_per_block - 1) / heads_per_block;

    dim3 grid(blocks);
    dim3 block(threads_per_head, heads_per_block);

    pad_heads_kernel<<<grid, block, 0, stream>>>(
        padded_output, unpadded_input, total_heads, head_size, padded_head_size);
}

template void launch_pad_heads<inference_data_t>(inference_data_t* padded_output,
                                                 const inference_data_t* unpadded_input,
                                                 int total_heads,
                                                 int head_size,
                                                 int padded_head_size,
                                                 cudaStream_t stream);

/*
Pad along both head and sequence dimensions. This is specialized for the current public
stable diffusion implementation which is accounting for some limitations of the flash attention
implementation with pre-padding.

TODO(cmikeh2): Update flash attention to support more flexible problem sizes and remove this

Concrete behaviors:
-Pad the head dimension to either 64 or 128
-Pad the sequence dimension of the key and value Tensors to 128 elements.

This kernel is brittle and designed to work under the SD constraints and should be looked at
carefully if you are trying to use it for other purposes.

NOTE: This kernel may introduce accuracy issues do the sequence length padding if the padded
sequence length is not correctly masked in the attention kernel (i.e., a sequence of 0-values is not
the same as the -inf mask).
 */
template <typename T>
__global__ void pad_head_and_sequence(T* padded_output,
                                      const T* unpadded_input,
                                      const int seq_len,
                                      const int padded_seq_len,
                                      const int head_size,
                                      const int padded_head_size)
{
    constexpr int granularity = 16;
    constexpr int T_per_access = granularity / sizeof(T);

    // At this point we assume the Tensor is of the shape [batch_size, heads, seq_len, head_size].
    // With this layout we can collapse the Tensor into [batch * heads, seq_len, head_size] and
    // index more straightforwardly.
    const int batch_head_id = blockIdx.x;
    // Kernel partitions the sequence dimension into threads of blockDim.y size. Assumption is that
    // the `padded_seq_len` is a multiple of blockDim.y.
    const int seq_id = blockIdx.y * blockDim.x + threadIdx.y;
    const int elem_id = threadIdx.x * T_per_access;

    const T* unpadded_input_ptr =
        unpadded_input + batch_head_id * seq_len * head_size + seq_id * head_size + elem_id;
    T* padded_output_ptr = padded_output + batch_head_id * padded_seq_len * padded_head_size +
                           seq_id * padded_head_size + elem_id;

    // We need to check if we are in bounds for both the sequence and head dimensions.
    bool do_mem = seq_id < seq_len && elem_id < head_size;
    T[T_per_access] buffer;
    mem_access::load_global<granularity>(buffer, unpadded_input_ptr, do_mem);
}

template <typename T>
void launch_pad_head_and_sequence(T* padded_output,
                                  const T* unpadded_input,
                                  const int num_batch_and_heads,
                                  const int seq_len,
                                  const int padded_seq_len,
                                  const int head_size,
                                  const int padded_head_size,
                                  cudaStream_t stream)
{
    constexpr int granularity = 16;
    constexpr int T_per_access = granularity / sizeof(T);

    // The 16 here should be safe in the conditions we see it.
    const int threads_per_head = (padded_head_size + T_per_access - 1) / T_per_access;
    dim3 grid(num_batch_and_heads, padded_seq_len / 16);
    dim3 block(threads_per_head, 16);

    pad_head_and_sequence<<<grid, block, 0, stream>>>(
        padded_output, unpadded_input, seq_len, padded_seq_len, head_size, padded_head_size);
}

template void launch_pad_head_and_sequence<inference_data_t>(inference_data_t* padded_output,
                                                             const inference_data_t* unpadded_input,
                                                             const int num_batch_and_heads,
                                                             const int seq_len,
                                                             const int padded_seq_len,
                                                             const int head_size,
                                                             const int padded_head_size,
                                                             cudaStream_t stream);
