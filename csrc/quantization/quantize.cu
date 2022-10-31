#include <cstdio>
#include "custom_cuda_layers.h"
#include "memory_access_utils.h"
#include "quantization.h"
#include "quantization_utils.h"
#include "reduction_utils.h"
#include "reduction_utils.h"

namespace cg = cooperative_groups;

/*
Pure quantization kernel with no fusion.
*/
template <int q_bits, quantize::Type quant_type, int UNROLL, int internal_unroll>
__global__ void activation_quantization(int8_t* __restrict__ output_data,
                                        float* __restrict__ scales,
                                        float* __restrict__ offsets,
                                        const __half* __restrict__ input_data,
                                        int groups,
                                        int elems_per_group)
{
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Indexing offsets
    const int block_offset = tb.group_index().x * elems_per_group;
    const int elem_offset = tb.thread_index().x * quantize::h_per_load;
    const int base_offset = block_offset + elem_offset;
    const int stride = tb.size() * quantize::h_per_load;

    const __half* input_base = input_data + base_offset;//..

    __half2 local_buffer[UNROLL * internal_unroll * quantize::h2_per_load];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        // Convenience helper, should resolve to register indices and not realize.
        __half2* iteration_buffer = local_buffer + i * internal_unroll * quantize::h2_per_load;
#pragma unroll
        for (int j = 0; j < internal_unroll; j++) {
            const int iteration = i * internal_unroll + j;
            mem_access::load_global<quantize::granularity>(
                iteration_buffer + j * quantize::h2_per_load,
                input_base + iteration * stride,
                elem_offset + iteration * stride < elems_per_group);
        }
    }

    quantize::local_array<quant_type, q_bits, UNROLL * internal_unroll>(
        local_buffer, scales, offsets, output_data, elems_per_group);
}

/********* Launcher methods ***********/

int32_t round_to_32(int32_t raw_value) { return (((raw_value - 1) >> 5) + 1) << 5; }

#define LAUNCH_ACTIVATION_QUANT(q_bits, quant_type, unroll_factor, internal_unroll) \
    activation_quantization<q_bits, quant_type, unroll_factor, internal_unroll>     \
        <<<grid, block, 0, stream>>>(output_data, scales, offsets, input_data, groups, elems_per_group);

template <int numBits, quantize::Type qType>
void launch_act_quant(int8_t* output_data,
                      float* scales,
                      float* offsets,
                      const __half* input_data,
                      int groups,
                      int elems_per_group,
                      cudaStream_t stream)
{
    constexpr int max_threads = 256;

    constexpr int internal_unroll = 2;

    constexpr int h_per_step = quantize::h_per_load * internal_unroll;


    // Scheduling concern: may be slightly faster for some inputs to assign multiple stages of
    // warp-sized blocks rather than stepping up to 64/96 threads
    const int one_step_threads = round_to_32((elems_per_group + h_per_step - 1) / h_per_step);
    const int threads = (one_step_threads < max_threads) ? one_step_threads : max_threads;

    dim3 block(threads);
    dim3 grid(groups);

    const int elems_per_step = threads * h_per_step;
    const int external_unroll = (elems_per_group + elems_per_step - 1) / elems_per_step;

    if (external_unroll == 1) {
        // 0 - 4096 elems
        // (this can launch with 1-7 warps as well)
        LAUNCH_ACTIVATION_QUANT(numBits, qType, 1, internal_unroll);
    } else if (external_unroll == 2) {
        // 4097 - 8192 elems
        LAUNCH_ACTIVATION_QUANT(numBits, qType, 2, internal_unroll);
    } else if (external_unroll == 3) {
        // 8193 - 12288 elems
        LAUNCH_ACTIVATION_QUANT(numBits, qType, 3, internal_unroll);
    } else if (external_unroll == 4) {
        // 12289 - 16384 elems
        LAUNCH_ACTIVATION_QUANT(numBits, qType, 4, internal_unroll);
    }
}

template void launch_act_quant<8, quantize::Type::Symmetric>(int8_t* output_data,
                                                              float* scales,
                                                              float* offsets,
                                                              const __half* input_data,
                                                              int groups,
                                                              int elems_per_group,
                                                              cudaStream_t stream);

template void launch_act_quant<8, quantize::Type::Asymmetric>(int8_t* output_data,
                                                               float* scales,
                                                               float* offsets,
                                                               const __half* input_data,
                                                               int groups,
                                                               int elems_per_group,
                                                               cudaStream_t stream);

template void launch_act_quant<4, quantize::Type::Symmetric>(int8_t* output_data,
                                                              float* scales,
                                                              float* offsets,
                                                              const __half* input_data,
                                                              int groups,
                                                              int elems_per_group,
                                                              cudaStream_t stream);

template void launch_act_quant<4, quantize::Type::Asymmetric>(int8_t* output_data,
                                                               float* scales,
                                                               float* offsets,
                                                               const __half* input_data,
                                                               int groups,
                                                               int elems_per_group,
                                                               cudaStream_t stream);
