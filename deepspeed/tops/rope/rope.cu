#include "rope.cuh"
#include "utils.h"

#include "reduction_utils.h"
#include <stdexcept>

namespace rot_half {
constexpr int threads = 256;
}  // namespace rot_half

template <typename T, int threadsPerHead, int granularity>
__global__ void apply_rotary_pos_half(T* mixed_query,
                                      T* key_layer,
                                      unsigned rotary_dim,
                                      unsigned seq_len,
                                      unsigned seq_offset,
                                      unsigned num_heads,
                                      unsigned head_size,
                                      unsigned total_count,
                                      float rope_theta)
{
    constexpr int T_per_thread = granularity / sizeof(T);
    constexpr int heads_per_block = rot_half::threads / threadsPerHead;

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<threadsPerHead> head_group = cg::tiled_partition<threadsPerHead>(tb);

    const int head_idx = blockIdx.x * heads_per_block + threadIdx.x / threadsPerHead;
    const int cur_seq_idx = head_idx / (total_count / seq_len);
    const int offset = head_idx * head_size;

    const int seq_idx = cur_seq_idx + seq_offset;
    const int half_dim = rotary_dim >> 1;
    const int half_dim_threads = half_dim / T_per_thread;

    if (head_idx < total_count) {
        const int base_neuron_idx = head_group.thread_rank() * T_per_thread;

        T q[T_per_thread], k[T_per_thread];
        mem_access::load_global<granularity>(q, mixed_query + offset + base_neuron_idx);
        mem_access::load_global<granularity>(k, key_layer + offset + base_neuron_idx);

#pragma unroll
        for (int i = 0; i < T_per_thread; i++) {
            const int neuron_idx = base_neuron_idx + i;
            if (neuron_idx < rotary_dim) {
                float inv_freq = (float)((neuron_idx % half_dim) * 2) / (float)rotary_dim;
                inv_freq = 1.0 / powf(rope_theta, inv_freq) * (float)seq_idx;

                float rotary_sign = (neuron_idx > (half_dim - 1) ? -1.0 : 1.0);
                float q_rot = conversion::to<float>(q[i]) * rotary_sign;
                float k_rot = conversion::to<float>(k[i]) * rotary_sign;

                const int target_lane = (neuron_idx < half_dim)
                                            ? head_group.thread_rank() + half_dim_threads
                                            : head_group.thread_rank() - half_dim_threads;

                const float q_rot_temp = head_group.shfl(q_rot, target_lane);
                const float k_rot_temp = head_group.shfl(k_rot, target_lane);

                q[i] = conversion::to<T>(conversion::to<float>(q[i]) * cosf(inv_freq) +
                                         q_rot_temp * sinf(inv_freq));
                k[i] = conversion::to<T>(conversion::to<float>(k[i]) * cosf(inv_freq) +
                                         k_rot_temp * sinf(inv_freq));
            }
        }

        mem_access::store_global<granularity>(mixed_query + offset + base_neuron_idx, q);
        mem_access::store_global<granularity>(key_layer + offset + base_neuron_idx, k);
    }
}


template <typename T, int threadsPerHead, int granularity>
__global__ void apply_rotary_pos_bwd_half(T* mixed_query_grad,
                                          T* key_layer_grad,
                                          unsigned rotary_dim,
                                          unsigned seq_len,
                                          unsigned seq_offset,
                                          unsigned num_heads,
                                          unsigned head_size,
                                          unsigned total_count,
                                          float rope_theta
                                        )
{
    constexpr int T_per_thread = granularity / sizeof(T);
    constexpr int heads_per_block = rot_half::threads / threadsPerHead;

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<threadsPerHead> head_group = cg::tiled_partition<threadsPerHead>(tb);

    const int head_idx = blockIdx.x * heads_per_block + threadIdx.x / threadsPerHead;
    const int cur_seq_idx = head_idx / (total_count / seq_len);
    const int offset = head_idx * head_size;

    const int seq_idx = cur_seq_idx + seq_offset;
    const int half_dim = rotary_dim >> 1;
    const int half_dim_threads = half_dim / T_per_thread;

    if (head_idx < total_count) {
        const int base_neuron_idx = head_group.thread_rank() * T_per_thread;

        T q[T_per_thread], k[T_per_thread];
        mem_access::load_global<granularity>(q, mixed_query_grad + offset + base_neuron_idx);
        mem_access::load_global<granularity>(k, key_layer_grad + offset + base_neuron_idx);

#pragma unroll
        for (int i = 0; i < T_per_thread; i++) {
            const int neuron_idx = base_neuron_idx + i;
            if (neuron_idx < rotary_dim) {
                float inv_freq = (float)((neuron_idx % half_dim) * 2) / (float)rotary_dim;
                inv_freq = 1.0 / powf(rope_theta, inv_freq) * (float)seq_idx;

                float rotary_sign = (neuron_idx > (half_dim - 1) ? 1.0 : -1.0);
                float q_rot = conversion::to<float>(q[i]) * rotary_sign;
                float k_rot = conversion::to<float>(k[i]) * rotary_sign;

                const int target_lane = (neuron_idx < half_dim)
                                            ? head_group.thread_rank() + half_dim_threads
                                            : head_group.thread_rank() - half_dim_threads;

                const float q_rot_temp = head_group.shfl(q_rot, target_lane);
                const float k_rot_temp = head_group.shfl(k_rot, target_lane);

                q[i] = conversion::to<T>(conversion::to<float>(q[i]) * cosf(inv_freq) +
                                         q_rot_temp * sinf(inv_freq));
                k[i] = conversion::to<T>(conversion::to<float>(k[i]) * cosf(inv_freq) +
                                         k_rot_temp * sinf(inv_freq));
            }
        }

        mem_access::store_global<granularity>(mixed_query_grad + offset + base_neuron_idx, q);
        mem_access::store_global<granularity>(key_layer_grad + offset + base_neuron_idx, k);
    }
}

#define LAUNCH_ROT_POS_EMB_BWD_HALF(HEAD_THREADS, ALIGNMENT)                                       \
    apply_rotary_pos_bwd_half<T, HEAD_THREADS, ALIGNMENT><<<grid, block, 0, stream>>>(mixed_query_grad, \
                                                                                  key_layer_grad,   \
                                                                                  rotary_dim,  \
                                                                                  seq_len,     \
                                                                                  offset,      \
                                                                                  num_heads,   \
                                                                                  head_size,   \
                                                                                  total_count, \
                                                                                  rope_theta);


#ifdef __HIP_PLATFORM_AMD__
#define LAUNCH_BWD_FOR_ALIGNMENT(ALIGNMENT)         \
    if (threads_per_head == 4) {                \
        LAUNCH_ROT_POS_EMB_BWD_HALF(4, ALIGNMENT);  \
    } else if (threads_per_head == 8) {         \
        LAUNCH_ROT_POS_EMB_BWD_HALF(8, ALIGNMENT);  \
    } else if (threads_per_head == 16) {        \
        LAUNCH_ROT_POS_EMB_BWD_HALF(16, ALIGNMENT); \
    } else if (threads_per_head == 32) {        \
        LAUNCH_ROT_POS_EMB_BWD_HALF(32, ALIGNMENT); \
    } else if (threads_per_head == 64) {        \
        LAUNCH_ROT_POS_EMB_BWD_HALF(64, ALIGNMENT); \
    } else {                                    \
        assert(false);                          \
    }
#else
#define LAUNCH_BWD_FOR_ALIGNMENT(ALIGNMENT)         \
    if (threads_per_head == 4) {                \
        LAUNCH_ROT_POS_EMB_BWD_HALF(4, ALIGNMENT);  \
    } else if (threads_per_head == 8) {         \
        LAUNCH_ROT_POS_EMB_BWD_HALF(8, ALIGNMENT);  \
    } else if (threads_per_head == 16) {        \
        LAUNCH_ROT_POS_EMB_BWD_HALF(16, ALIGNMENT); \
    } else if (threads_per_head == 32) {        \
        LAUNCH_ROT_POS_EMB_BWD_HALF(32, ALIGNMENT); \
    } else {                                    \
        assert(false);                          \
    }
#endif

#define LAUNCH_ROT_POS_EMB_HALF(HEAD_THREADS, ALIGNMENT)                                       \
    apply_rotary_pos_half<T, HEAD_THREADS, ALIGNMENT><<<grid, block, 0, stream>>>(mixed_query, \
                                                                                  key_layer,   \
                                                                                  rotary_dim,  \
                                                                                  seq_len,     \
                                                                                  offset,      \
                                                                                  num_heads,   \
                                                                                  head_size,   \
                                                                                  total_count, \
                                                                                  rope_theta);

#ifdef __HIP_PLATFORM_AMD__
#define LAUNCH_FOR_ALIGNMENT(ALIGNMENT)         \
    if (threads_per_head == 4) {                \
        LAUNCH_ROT_POS_EMB_HALF(4, ALIGNMENT);  \
    } else if (threads_per_head == 8) {         \
        LAUNCH_ROT_POS_EMB_HALF(8, ALIGNMENT);  \
    } else if (threads_per_head == 16) {        \
        LAUNCH_ROT_POS_EMB_HALF(16, ALIGNMENT); \
    } else if (threads_per_head == 32) {        \
        LAUNCH_ROT_POS_EMB_HALF(32, ALIGNMENT); \
    } else if (threads_per_head == 64) {        \
        LAUNCH_ROT_POS_EMB_HALF(64, ALIGNMENT); \
    } else {                                    \
        assert(false);                          \
    }
#else
#define LAUNCH_FOR_ALIGNMENT(ALIGNMENT)         \
    if (threads_per_head == 4) {                \
        LAUNCH_ROT_POS_EMB_HALF(4, ALIGNMENT);  \
    } else if (threads_per_head == 8) {         \
        LAUNCH_ROT_POS_EMB_HALF(8, ALIGNMENT);  \
    } else if (threads_per_head == 16) {        \
        LAUNCH_ROT_POS_EMB_HALF(16, ALIGNMENT); \
    } else if (threads_per_head == 32) {        \
        LAUNCH_ROT_POS_EMB_HALF(32, ALIGNMENT); \
    } else {                                    \
        assert(false);                          \
    }
#endif

template <typename T>
void launch_apply_rotary_pos_emb(T* mixed_query,
                                 T* key_layer,
                                 unsigned head_size,
                                 unsigned seq_len,
                                 unsigned rotary_dim,
                                 unsigned offset,
                                 unsigned num_heads,
                                 unsigned batch,
                                 float rope_theta,
                                 cudaStream_t stream)
{
    const int half_dim = rotary_dim >> 1;

    int alignment = sizeof(T);
    if (half_dim % (16 / sizeof(T)) == 0) {
        alignment = 16;
    } else if (half_dim % (8 / sizeof(T)) == 0) {
        alignment = 8;
    } else if (half_dim % (4 / sizeof(T)) == 0) {
        alignment = 4;
    } else {
        assert(false);
    }
    const int T_per_elem = alignment / sizeof(T);

    int total_count = batch * num_heads * seq_len;

    const int padded_head_size = next_pow2(head_size);

    assert(padded_head_size <= hw_warp_size * T_per_elem);

    const int threads_per_head = padded_head_size / T_per_elem;
    const int heads_per_block = rot_half::threads / threads_per_head;

    dim3 block(rot_half::threads);
    dim3 grid((total_count + heads_per_block - 1) / heads_per_block);

    if (alignment == 4) {
        LAUNCH_FOR_ALIGNMENT(4);
    } else if (alignment == 8) {
        LAUNCH_FOR_ALIGNMENT(8);
    } else if (alignment == 16) {
        LAUNCH_FOR_ALIGNMENT(16);
    } else {
        assert(false);
    }
}


#define INSTANTIATE_LAUNCH_ROTARY_POS_EMB(T)                   \
    template void launch_apply_rotary_pos_emb<T>(T*,           \
                                                 T*,           \
                                                 unsigned,     \
                                                 unsigned,     \
                                                 unsigned,     \
                                                 unsigned,     \
                                                 unsigned,     \
                                                 unsigned,     \
                                                 float,        \
                                                 cudaStream_t);

INSTANTIATE_LAUNCH_ROTARY_POS_EMB(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_ROTARY_POS_EMB(__nv_bfloat16);
#endif
INSTANTIATE_LAUNCH_ROTARY_POS_EMB(__half);

template <typename T>
void launch_apply_rotary_pos_bwd_emb(T* mixed_query_grad,
                                 T* key_layer_grad,
                                 unsigned head_size,
                                 unsigned seq_len,
                                 unsigned rotary_dim,
                                 unsigned offset,
                                 unsigned num_heads,
                                 unsigned batch,
                                 float rope_theta,
                                 cudaStream_t stream)
{
    const int half_dim = rotary_dim >> 1;

    int alignment = sizeof(T);
    if (half_dim % (16 / sizeof(T)) == 0) {
        alignment = 16;
    } else if (half_dim % (8 / sizeof(T)) == 0) {
        alignment = 8;
    } else if (half_dim % (4 / sizeof(T)) == 0) {
        alignment = 4;
    } else {
        assert(false);
    }
    const int T_per_elem = alignment / sizeof(T);

    int total_count = batch * num_heads * seq_len;

    const int padded_head_size = next_pow2(head_size);

    assert(padded_head_size <= hw_warp_size * T_per_elem);

    const int threads_per_head = padded_head_size / T_per_elem;
    const int heads_per_block = rot_half::threads / threads_per_head;

    dim3 block(rot_half::threads);
    dim3 grid((total_count + heads_per_block - 1) / heads_per_block);

    if (alignment == 4) {
        LAUNCH_BWD_FOR_ALIGNMENT(4);
    } else if (alignment == 8) {
        LAUNCH_BWD_FOR_ALIGNMENT(8);
    } else if (alignment == 16) {
        LAUNCH_BWD_FOR_ALIGNMENT(16);
    } else {
        assert(false);
    }
}

#define INSTANTIATE_LAUNCH_ROTARY_POS_BWD_EMB(T)                   \
    template void launch_apply_rotary_pos_bwd_emb<T>(T*,           \
                                                 T*,           \
                                                 unsigned,     \
                                                 unsigned,     \
                                                 unsigned,     \
                                                 unsigned,     \
                                                 unsigned,     \
                                                 unsigned,     \
                                                 float,        \
                                                 cudaStream_t);

INSTANTIATE_LAUNCH_ROTARY_POS_BWD_EMB(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_ROTARY_POS_BWD_EMB(__nv_bfloat16);
#endif
INSTANTIATE_LAUNCH_ROTARY_POS_BWD_EMB(__half);