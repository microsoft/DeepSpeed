/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include "memory_access_utils.h"
#include "quantization.h"
#include "quantization_utils.h"
#include "reduction_utils.h"

namespace cg = cooperative_groups;

/*
Pure quantization kernel with no fusion.
*/
template <int numBits, quantize::Type qType, int numChunks, int threadsPerGroup, int maxThreads>
__global__ void cached_quantization(int8_t* __restrict__ output_data,
                                    float* __restrict__ params,
                                    const __half* __restrict__ input_data,
                                    int groups,
                                    int elems_per_group)
{
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Indexing offsets
    const int block_offset =
        (tb.group_index().x * (maxThreads / threadsPerGroup) * elems_per_group) +
        (tb.thread_index().y * elems_per_group);
    const int elem_offset = tb.thread_index().x * quantize::h_per_load;
    const int base_offset = block_offset + elem_offset;
    const int stride = threadsPerGroup * quantize::h_per_load;

    const __half* input_base = input_data + base_offset;
    __half2 local_buffer[numChunks * quantize::h2_per_load];

#pragma unroll
    for (int i = 0; i < numChunks; i++) {
        mem_access::load_global<quantize::granularity>(local_buffer + i * quantize::h2_per_load,
                                                       input_base + i * stride,
                                                       elem_offset + i * stride < elems_per_group);
    }

    quantize::local_array<qType, numBits, numChunks, threadsPerGroup, maxThreads>(
        local_buffer, params, output_data, elems_per_group, groups);
}

/*
Pure quantization kernel with no fusion.
*/
template <int numBits, quantize::Type qType, int unrollSteps, int numThreads>
__global__ void pipelined_quantization(int8_t* __restrict__ output_data,
                                       float* __restrict__ params,
                                       const __half* __restrict__ input_data,
                                       int groups,
                                       int elems_per_group,
                                       int total_steps)
{
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    const int block_offset = tb.group_index().x * elems_per_group;
    const int elem_offset = tb.thread_index().x * quantize::h_per_load;
    const int stride = numThreads * quantize::h_per_load;

    const __half* input_block = input_data + block_offset;

    quantize::GroupStats<qType> q_stats;
    for (int i = 0; i < total_steps; i += unrollSteps) {
        __half2 unroll_buffer[quantize::h2_per_load * unrollSteps];

#pragma unroll
        for (int j = 0; j < unrollSteps; j++) {
            const int iteration_offset = elem_offset + (i + j) * stride;
            mem_access::load_global<quantize::granularity>(
                unroll_buffer + j * quantize::h2_per_load,
                input_block + iteration_offset,
                iteration_offset < elems_per_group);
        }

#pragma unroll
        for (int j = 0; j < unrollSteps * quantize::h2_per_load; j++) {
            q_stats.update(unroll_buffer[j]);
        }
    }

    auto q_params = q_stats.template get_params<numBits, numThreads>(tb, warp);

    if (tb.thread_index().x == 0) { q_params.store(params, tb.group_index().x); }

    // TODO(cmikeh2): Refactor into helper functions
    constexpr int packed_vals = 8 / numBits;
    constexpr int out_chunk_buffer_size = quantize::h_per_load / packed_vals;
    constexpr int out_stride = stride / packed_vals;
    const int out_block_offset = tb.group_index().x * elems_per_group / packed_vals;
    const int out_elem_offset = tb.thread_index().x * quantize::h_per_load / packed_vals;
    int8_t* output_block = output_data + out_block_offset;

    for (int i = 0; i < total_steps; i += unrollSteps) {
        __half2 unroll_buffer[quantize::h2_per_load * unrollSteps];

#pragma unroll
        for (int j = 0; j < unrollSteps; j++) {
            const int iteration_offset = elem_offset + (i + j) * stride;
            if (iteration_offset < elems_per_group) {
                mem_access::load_global<quantize::granularity>(
                    unroll_buffer + j * quantize::h2_per_load,
                    input_block + iteration_offset,
                    iteration_offset < elems_per_group);

                int8_t quant_buffer[out_chunk_buffer_size];
                quantize::_chunk<numBits, qType>(
                    quant_buffer, unroll_buffer + j * quantize::h2_per_load, q_params);

                mem_access::store_global<out_chunk_buffer_size>(
                    output_block + out_elem_offset + (i + j) * out_stride, quant_buffer);
            }
        }
    }
}

/********* Launcher methods ***********/
int next_pow2(const int val)
{
    int rounded_val = val - 1;
    rounded_val |= rounded_val >> 1;
    rounded_val |= rounded_val >> 2;
    rounded_val |= rounded_val >> 4;
    rounded_val |= rounded_val >> 8;
    return rounded_val + 1;
}

inline int32_t round_to_2(int32_t raw_value) { return (((raw_value - 1) >> 1) + 1) << 1; }
inline int32_t round_to_4(int32_t raw_value) { return (((raw_value - 1) >> 2) + 1) << 2; }
inline int32_t round_to_8(int32_t raw_value) { return (((raw_value - 1) >> 3) + 1) << 3; }
inline int32_t round_to_32(int32_t raw_value) { return (((raw_value - 1) >> 5) + 1) << 5; }

#define LAUNCH_CACHED_QUANT(q_bits, quant_type, unroll_factor, threads_per_group, max_threads) \
    cached_quantization<q_bits, quant_type, unroll_factor, threads_per_group, max_threads>     \
        <<<grid, block, 0, stream>>>(output_data, params, input_data, groups, elems_per_group);

template <int numBits, quantize::Type qType>
void launch_quant_impl(int8_t* output_data,
                       float* params,
                       const __half* input_data,
                       const int groups,
                       const int elems_per_group,
                       cudaStream_t stream)
{
    constexpr int max_threads = 256;
    const bool is_subblock_schedule = (elems_per_group <= 128) ? true : false;

    const int one_step_threads =
        next_pow2((elems_per_group + quantize::h_per_load - 1) / quantize::h_per_load);
    const int threads_per_group = (one_step_threads < max_threads) ? one_step_threads : max_threads;

    const int groups_per_block =
        is_subblock_schedule ? (max_threads + threads_per_group - 1) / threads_per_group : 1;
    const int groups_launch = (groups_per_block + groups - 1) / groups_per_block;

    dim3 block(threads_per_group, groups_per_block);
    dim3 grid(groups_launch);

    const int elems_per_step = threads_per_group * quantize::h_per_load;
    int unroll = (elems_per_group + elems_per_step - 1) / elems_per_step;

    if (is_subblock_schedule) {
        // <=128
        if (threads_per_group == 1) {
            LAUNCH_CACHED_QUANT(numBits, qType, 1, 1, max_threads);
        } else if (threads_per_group == 2) {
            LAUNCH_CACHED_QUANT(numBits, qType, 1, 2, max_threads);
        } else if (threads_per_group == 4) {
            LAUNCH_CACHED_QUANT(numBits, qType, 1, 4, max_threads);
        } else if (threads_per_group == 8) {
            LAUNCH_CACHED_QUANT(numBits, qType, 1, 8, max_threads);
        } else if (threads_per_group == 16) {
            LAUNCH_CACHED_QUANT(numBits, qType, 1, 16, max_threads);
        }
    } else if (unroll == 1) {
        // 129 - 2048 elems
        // (this can launch with 1-7 warps as well)
        LAUNCH_CACHED_QUANT(numBits, qType, 1, max_threads, max_threads);
    } else {
        unroll = round_to_2(unroll);
        if (unroll == 2) {
            // up to 4k elems
            LAUNCH_CACHED_QUANT(numBits, qType, 2, max_threads, max_threads);
        } else if (unroll == 4) {
            // up to 8k elems
            LAUNCH_CACHED_QUANT(numBits, qType, 4, max_threads, max_threads);
        } else if (unroll == 6) {
            // up to 12k elems
            LAUNCH_CACHED_QUANT(numBits, qType, 6, max_threads, max_threads);
        } else if (unroll == 8) {
            // up to 16k elems
            LAUNCH_CACHED_QUANT(numBits, qType, 8, max_threads, max_threads);
        } else {
            // We can nominally support implementations with the cached version,
            // but the use cases I'm aware of for it are offline and less latency
            // sensitive. For now, let's save compile time and instantiate fewer
            // templates.
            pipelined_quantization<numBits, qType, 8, max_threads><<<grid, block, 0, stream>>>(
                output_data, params, input_data, groups, elems_per_group, unroll);
        }
    }
}

#define QUANTIZATION_CASE(TYPE, BITS)                                          \
    case TYPE:                                                                 \
        launch_quant_impl<BITS, TYPE>(                                         \
            output_data, params, input_data, groups, elems_per_group, stream); \
        break;

void launch_quant(int8_t* output_data,
                  float* params,
                  const __half* input_data,
                  const int groups,
                  const int elems_per_group,
                  int num_bits,
                  quantize::Type q_type,
                  cudaStream_t stream)
{
    if (num_bits == 4) {
        switch (q_type) {
            QUANTIZATION_CASE(quantize::Type::Symmetric, 4)
            QUANTIZATION_CASE(quantize::Type::Asymmetric, 4)
        }
    } else {
        switch (q_type) {
            QUANTIZATION_CASE(quantize::Type::Symmetric, 8)
            QUANTIZATION_CASE(quantize::Type::Asymmetric, 8)
        }
    }
}
