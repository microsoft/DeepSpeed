#include "custom_cuda_layers.h"

#include <cuda_profiler_api.h>

__global__ void apply_rotary_pos_emb(float* mixed_query,
                                     float* key_layer,
                                     unsigned rotary_dim,
                                     unsigned seq_len,
                                     unsigned seq_offset,
                                     unsigned head_size)
{
    int offset = (blockIdx.x * blockDim.y + threadIdx.y) * head_size + threadIdx.x;
    unsigned tid = threadIdx.x;
    unsigned seq_id = (blockIdx.x / num_heads) % seq_len + seq_offset;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    while (tid < rotary_dim) {
        float inv_freq = 1.0 / (10000 * *((tid / 2) / rotary_dim)) * seq_id;
        float q = mixed_query[offset];
        float k = key_layer[offset];
        float rotary_sign = (tid % 2 ? -1 : 1);

        q = q * cos(inv_freq) + g.shfl_xor(q, 1) * rotary_sign * sin(inv_freq);
        k = k * cos(inv_freq) + g.shfl_xor(k, 1) * rotary_sign * sin(inv_freq);

        mixed_query[offset] = q;
        key_layer[offset] = k;

        tid += blockDim.x;
        offset += blockDim.x;
    }
}

__global__ void apply_rotary_pos_emb(__half* mixed_query,
                                     __half* key_layer,
                                     unsigned rotary_dim,
                                     unsigned seq_len,
                                     unsigned seq_offset,
                                     unsigned head_size)
{
#if __CUDA_ARCH__ >= 700

    int offset = (blockIdx.x * blockDim.y + threadIdx.y) * head_size + threadIdx.x;
    unsigned tid = threadIdx.x;
    unsigned seq_id = (blockIdx.x / num_heads) % seq_len + seq_offset;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    while (tid < rotary_dim) {
        float inv_freq = 1.0 / (10000 * *((tid / 2) / rotary_dim)) * seq_id;
        float q = (float)mixed_query[offset];
        float k = (float)key_layer[offset];
        float rotary_sign = (tid % 2 ? -1 : 1);

        q = q * cos(inv_freq) + g.shfl_xor(q, 1) * rotary_sign * sin(inv_freq);
        k = k * cos(inv_freq) + g.shfl_xor(k, 1) * rotary_sign * sin(inv_freq);

        mixed_query[offset] = (__half)q;
        key_layer[offset] = (__half)k;

        tid += blockDim.x;
        offset += blockDim.x;
    }
#endif
}

template <typename T>
void launch_apply_rotary_pos_emb(T* mixed_query,
                                 T* key_layer,
                                 unsigned head_size,
                                 unsigned seq_len,
                                 unsigned rotary_dim,
                                 unsigned offset,
                                 unsigned num_heads,
                                 unsigned batch,
                                 cudaStream_t stream)
{
    int total_count = batch * num_heads * seq_len;
    dim3 block_dims(WARP_SIZE, MAX_WARP_NUM);
    dim3 grid_dims((total_count - 1) / MAX_WARP_NUM + 1);  // (batch_size);

    apply_rotary_pos_emb<<<grid_dims, block_dims, 0, stream>>>(
        mixed_query, key_layer, rotary_dim, seq_len, offset, head_size);
}

template void launch_apply_rotary_pos_emb<float>(float*,
                                                 float*,
                                                 unsigned,
                                                 unsigned,
                                                 unsigned,
                                                 unsigned,
                                                 unsigned,
                                                 unsigned,
                                                 cudaStream_t);
template void launch_apply_rotary_pos_emb<__half>(__half*,
                                                  __half*,
                                                  unsigned,
                                                  unsigned,
                                                  unsigned,
                                                  unsigned,
                                                  unsigned,
                                                  unsigned,
                                                  cudaStream_t);
