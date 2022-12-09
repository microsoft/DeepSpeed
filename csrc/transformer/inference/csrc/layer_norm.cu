/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include "conversion_utils.h"
#include "inference_cuda_layers.h"
#include "memory_access_utils.h"
#include "reduction_utils.h"

namespace cg = cooperative_groups;
using rop = reduce::ROpType;

namespace ln {
constexpr int granularity = 16;
}  // namespace ln

/*
Primary layer norm implementation. Assumes elems_per_row % 8
is equal to 0.

Args:
    output: buffer for output data
    vals: buffer for input data
    gamma: gain for normalization
    beta: bias for normalization
    epsilon: numeric stability
    elems_per_row: number of elements each block will normalize
*/
template <typename T, int UNROLL,
          int internal_unroll,
          int threads_per_group,
          int max_threads>
__global__ void fused_ln(T* output,
                         const T* vals,
                         const T* gamma,
                         const T* beta,
                         float epsilon,
                         int elems_per_row)
{
    constexpr int T_per_load = ln::granularity / sizeof(T);

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // X-dimension of the block
    const int block_offset =
        (tb.group_index().x * (max_threads / threads_per_group) * elems_per_row) +
        (tb.thread_index().y * elems_per_row);
    const int thread_offset = tb.thread_index().x * T_per_load;
    const int base_offset = block_offset + thread_offset;
    const int stride = tb.size() * T_per_load;

    // TODO(cmikeh2): refactor to reduction utility library
    float sum = reduce::init<rop::Add, float>();

    const T* input_base = vals + base_offset;
    T local_buffer[UNROLL * internal_unroll * T_per_load];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        T* iteration_buffer = local_buffer + i * internal_unroll * T_per_load;

#pragma unroll
        for (int j = 0; j < internal_unroll; j++) {
            const int iteration = i * internal_unroll + j;
            mem_access::load_global<ln::granularity>(
                iteration_buffer + j * T_per_load,
                input_base + iteration * stride,
                thread_offset + iteration * stride < elems_per_row);
        }

#pragma unroll
        for (int j = 0; j < internal_unroll * T_per_load; j++) {
            float up_cast = conversion::to<float>(iteration_buffer[j]);
            sum = reduce::element<rop::Add>(sum, up_cast);
        }
    }

    reduce::partitioned_block<rop::Add, threads_per_group>(tb, warp, sum);
    const float mean = sum / elems_per_row;

    float mean_diff = reduce::init<rop::Add, float>();

#pragma unroll
    for (int i = 0; i < UNROLL * internal_unroll; i++) {
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            // Using a 0 value here skews the variance, have to if-guard
            if (thread_offset + i * stride < elems_per_row) {
                float diff = (conversion::to<float>(local_buffer[i * T_per_load + j]) - mean);
                mean_diff = reduce::element<rop::Add>(mean_diff, diff * diff);
            }
        }
    }

    reduce::partitioned_block<rop::Add, threads_per_group>(tb, warp, mean_diff);
    const float variance = mean_diff / elems_per_row;
    const float denom = __frsqrt_rn(variance + epsilon);

    const T mean_compute = conversion::to<T>(mean);
    const T denom_compute = conversion::to<T>(denom);

    T* block_output = output + block_offset;

#pragma unroll
    for (int i = 0; i < UNROLL * internal_unroll; i++) {
        T* iteration_buffer = local_buffer + i * T_per_load;
        const int iter_idx = i * stride + thread_offset;
        const bool do_loads = iter_idx < elems_per_row;

        T gamma_local[T_per_load], beta_local[T_per_load];

        mem_access::load_global<ln::granularity>(gamma_local, gamma + iter_idx, do_loads);
        mem_access::load_global<ln::granularity>(beta_local, beta + iter_idx, do_loads);

#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            iteration_buffer[j] = (iteration_buffer[j] - mean_compute) * denom_compute;
            iteration_buffer[j] = iteration_buffer[j] * gamma_local[j] + beta_local[j];
        }

        if (do_loads) {
            mem_access::store_global<ln::granularity>(block_output + iter_idx, iteration_buffer);
        }
    }
}

#define LAUNCH_FUSED_LN(unroll_factor, internal_unroll, threads_per_group, max_threads) \
    fused_ln<T, unroll_factor, internal_unroll, threads_per_group, max_threads>         \
        <<<grid, block, 0, stream>>>(output, vals, gamma, beta, epsilon, elems_per_row);

template <typename T>
void launch_fused_ln(T* output,
                     const T* vals,
                     const T* gamma,
                     const T* beta,
                     float epsilon,
                     int rows,
                     int elems_per_row,
                     cudaStream_t stream)
{
    // 8 for __half, 4 for float
    constexpr int T_per_load = ln::granularity / sizeof(T);

    constexpr int max_threads = 256;

    // For Flaoat, unroll 4, for __half, unroll 2
    constexpr int internal_unroll = sizeof(T) == 4 ? 4 : 2;

    const bool is_subblock_schedule = (elems_per_row <= 128) ? true : false;
    const int h_per_step = is_subblock_schedule ? T_per_load : T_per_load * internal_unroll;

    // Scheduling concern: may be slightly faster for some inputs to assign multiple stages of
    // warp-sized blocks rather than stepping up to 64/96 threads
    const int one_step_threads = next_pow2((elems_per_row + h_per_step - 1) / h_per_step);
    const int threads_per_group = (one_step_threads < max_threads) ? one_step_threads : max_threads;
    const int warps_per_group = threads_per_group / hw_warp_size;

    const int groups_per_block_max =
        is_subblock_schedule ? (max_threads + threads_per_group - 1) / threads_per_group : 1;
    const int groups_per_block = (rows < groups_per_block_max) ? rows : groups_per_block_max;
    const int groups_launch = (groups_per_block + rows - 1) / groups_per_block;

    dim3 block(threads_per_group, groups_per_block);
    dim3 grid(groups_launch);

    const int elems_per_step = threads_per_group * h_per_step;
    const int external_unroll = (elems_per_row + elems_per_step - 1) / elems_per_step;

    if (is_subblock_schedule) {
        // <=128
        if (threads_per_group == 1) {
            LAUNCH_FUSED_LN(1, 1, 1, max_threads);
        } else if (threads_per_group == 2) {
            LAUNCH_FUSED_LN(1, 1, 2, max_threads);
        } else if (threads_per_group == 4) {
            LAUNCH_FUSED_LN(1, 1, 4, max_threads);
        } else if (threads_per_group == 8) {
            LAUNCH_FUSED_LN(1, 1, 8, max_threads);
        } else if (threads_per_group == 16) {
            LAUNCH_FUSED_LN(1, 1, 16, max_threads);
        }
    } else if (external_unroll == 1) {
        // 129 - 4096 elems
        // (this can launch with 1-7 warps as well)
        LAUNCH_FUSED_LN(1, internal_unroll, max_threads, max_threads);
    } else if (external_unroll == 2) {
        // 4097 - 8192 elems
        LAUNCH_FUSED_LN(2, internal_unroll, max_threads, max_threads);
    } else if (external_unroll == 3) {
        // 8193 - 12288 elems
        LAUNCH_FUSED_LN(3, internal_unroll, max_threads, max_threads);
    } else if (external_unroll == 4) {
        // 12289 - 16384 elems
        LAUNCH_FUSED_LN(4, internal_unroll, max_threads, max_threads);
    }
}

template void launch_fused_ln(__half*,
                              const __half*,
                              const __half*,
                              const __half*,
                              float,
                              int,
                              int,
                              cudaStream_t);
template void
launch_fused_ln(float*, const float*, const float*, const float*, float, int, int, cudaStream_t);

/*
Fused resiual + bias + layer norm implementation. Assumes elems_per_row % 8
is equal to 0.

TODO(cmikeh2): Goal is to deprecate this implementation. The bias + residual
need to be fused into compute-bound producer operations.

Args:
    output: buffer for output data
    res_output: output of residual addition
    vals: buffer for input data
    residual: residual data
    bias: bias of of input data
    gamma: gain for normalization
    beta: bias for normalization
    epsilon: numeric stability
    elems_per_row: number of elements each block will normalize
Template arg:
    StoreResidual: controls whether the residual calculation is stored
        or not. When set to false, the input `res_output` is unused.
*/
template <typename T,
          int UNROLL,
          int internal_unroll,
          int threads_per_group,
          int max_threads,
          bool PreLnResidual>
__global__ void fused_residual_ln(T* output,
                                  T* res_output,
                                  const T* vals,
                                  const T* residual,
                                  const T* bias,
                                  const T* gamma,
                                  const T* beta,
                                  float epsilon,
                                  int elems_per_row)
{
    constexpr int T_per_load = ln::granularity / sizeof(T);

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // X-dimension of the block
    const int block_offset =
        (tb.group_index().x * (max_threads / threads_per_group) * elems_per_row) +
        (tb.thread_index().y * elems_per_row);
    const int thread_offset = tb.thread_index().x * T_per_load;
    const int base_offset = block_offset + thread_offset;
    const int stride = tb.size() * T_per_load;

    float sum = reduce::init<rop::Add, float>();

    const T* input_base = vals + base_offset;
    const T* residual_base = residual + base_offset;
    const T* bias_base = bias + thread_offset;

    T local_buffer[UNROLL * internal_unroll * T_per_load];

    // Unlike a vanilla layernorm, since we're fusing the two adds as well
    // an inner unroll seems to be less valuable. If anything, a double unroll
    // makes the most sense if we find we are having performance issues.
#pragma unroll
    for (int i = 0; i < UNROLL * internal_unroll; i++) {
        T* iteration_buffer = local_buffer + i * T_per_load;
        T residual_buffer[T_per_load];
        T bias_buffer[T_per_load];

        mem_access::load_global<ln::granularity>(
            iteration_buffer, input_base + i * stride, thread_offset + i * stride < elems_per_row);
        mem_access::load_global<ln::granularity>(residual_buffer,
                                                 residual_base + i * stride,
                                                 thread_offset + i * stride < elems_per_row);
        mem_access::load_global<ln::granularity>(
            bias_buffer, bias_base + i * stride, thread_offset + i * stride < elems_per_row);

#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            float vals_up_cast = conversion::to<float>(iteration_buffer[j]);
            float res_up_cast = conversion::to<float>(residual_buffer[j]);
            float bias_up_cast = conversion::to<float>(bias_buffer[j]);
            vals_up_cast += res_up_cast + bias_up_cast;
            sum = reduce::element<rop::Add>(sum, vals_up_cast);
            iteration_buffer[j] = conversion::to<T>(vals_up_cast);
        }

        if (PreLnResidual && (thread_offset + i * stride < elems_per_row)) {
            mem_access::store_global<ln::granularity>(res_output + base_offset + i * stride,
                                                      iteration_buffer);
        }
    }

    reduce::partitioned_block<rop::Add, threads_per_group>(tb, warp, sum);
    const float mean = sum / elems_per_row;

    float mean_diff = reduce::init<rop::Add, float>();
#pragma unroll
    for (int i = 0; i < UNROLL * internal_unroll; i++) {
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            // Using a 0 value here skews the variance, have to if-guard
            if (thread_offset + i * stride < elems_per_row) {
                float diff = (conversion::to<float>(local_buffer[i * T_per_load + j]) - mean);
                mean_diff = reduce::element<rop::Add>(mean_diff, diff * diff);
            }
        }
    }

    reduce::partitioned_block<rop::Add, threads_per_group>(tb, warp, mean_diff);
    const float variance = mean_diff / elems_per_row;
    const float denom = __frsqrt_rn(variance + epsilon);

    const T mean_compute = conversion::to<T>(mean);
    const T denom_compute = conversion::to<T>(denom);

    T* block_output = output + block_offset;

#pragma unroll
    for (int i = 0; i < UNROLL * internal_unroll; i++) {
        T* iteration_buffer = local_buffer + i * T_per_load;
        const int iter_idx = i * stride + thread_offset;
        const bool do_loads = iter_idx < elems_per_row;

        T gamma_local[T_per_load], beta_local[T_per_load];

        mem_access::load_global<ln::granularity>(gamma_local, gamma + iter_idx, do_loads);
        mem_access::load_global<ln::granularity>(beta_local, beta + iter_idx, do_loads);

#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            iteration_buffer[j] = (iteration_buffer[j] - mean_compute) * denom_compute;
            iteration_buffer[j] = iteration_buffer[j] * gamma_local[j] + beta_local[j];
        }

        if (do_loads) {
            mem_access::store_global<ln::granularity>(block_output + iter_idx, iteration_buffer);
        }
    }
}

// TODO(cmikeh2): There's a bunch of redundancy here that needs to be removed/simplified.
#define LAUNCH_FUSED_RES_LN(unroll_factor, internal_unroll, threads_per_group, max_threads)     \
    fused_residual_ln<T, unroll_factor, internal_unroll, threads_per_group, max_threads, false> \
        <<<grid, block, 0, stream>>>(                                                           \
            output, nullptr, vals, residual, bias, gamma, beta, epsilon, elems_per_row);

template <typename T>
void launch_fused_residual_ln(T* output,
                              const T* vals,
                              const T* residual,
                              const T* bias,
                              const T* gamma,
                              const T* beta,
                              float epsilon,
                              int rows,
                              int elems_per_row,
                              cudaStream_t stream)
{
    // 8 for __half, 4 for float
    constexpr int T_per_load = ln::granularity / sizeof(T);

    constexpr int max_threads = 256;

    // For Flaoat, unroll 4, for __half, unroll 2
    constexpr int internal_unroll = sizeof(T) == 4 ? 4 : 2;

    const bool is_subblock_schedule = (elems_per_row <= 128) ? true : false;
    const int h_per_step = is_subblock_schedule ? T_per_load : T_per_load * internal_unroll;

    // Scheduling concern: may be slightly faster for some inputs to assign multiple stages of
    // warp-sized blocks rather than stepping up to 64/96 threads
    const int one_step_threads = next_pow2((elems_per_row + h_per_step - 1) / h_per_step);
    const int threads_per_group = (one_step_threads < max_threads) ? one_step_threads : max_threads;
    const int warps_per_group = threads_per_group / hw_warp_size;

    const int groups_per_block_max =
        is_subblock_schedule ? (max_threads + threads_per_group - 1) / threads_per_group : 1;
    const int groups_per_block = (rows < groups_per_block_max) ? rows : groups_per_block_max;
    const int groups_launch = (groups_per_block + rows - 1) / groups_per_block;

    dim3 block(threads_per_group, groups_per_block);
    dim3 grid(groups_launch);

    const int elems_per_step = threads_per_group * h_per_step;
    const int external_unroll = (elems_per_row + elems_per_step - 1) / elems_per_step;

    if (is_subblock_schedule) {
        // <=128
        if (threads_per_group == 1) {
            LAUNCH_FUSED_RES_LN(1, 1, 1, max_threads);
        } else if (threads_per_group == 2) {
            LAUNCH_FUSED_RES_LN(1, 1, 2, max_threads);
        } else if (threads_per_group == 4) {
            LAUNCH_FUSED_RES_LN(1, 1, 4, max_threads);
        } else if (threads_per_group == 8) {
            LAUNCH_FUSED_RES_LN(1, 1, 8, max_threads);
        } else if (threads_per_group == 16) {
            LAUNCH_FUSED_RES_LN(1, 1, 16, max_threads);
        }
    } else if (external_unroll == 1) {
        // 129 - 4096 elems
        // (this can launch with 1-7 warps as well)
        LAUNCH_FUSED_RES_LN(1, internal_unroll, max_threads, max_threads);
    } else if (external_unroll == 2) {
        // 4097 - 8192 elems
        LAUNCH_FUSED_RES_LN(2, internal_unroll, max_threads, max_threads);
    } else if (external_unroll == 3) {
        // 8193 - 12288 elems
        LAUNCH_FUSED_RES_LN(3, internal_unroll, max_threads, max_threads);
    } else if (external_unroll == 4) {
        // 12289 - 16384 elems
        LAUNCH_FUSED_RES_LN(4, internal_unroll, max_threads, max_threads);
    }
}

#define LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(unroll_factor, internal_unroll, threads_per_group, max_threads) \
    fused_residual_ln<T, unroll_factor, internal_unroll, threads_per_group, max_threads, true>    \
        <<<grid, block, 0, stream>>>(                                                             \
            norm_output, res_output, vals, residual, bias, gamma, beta, epsilon, elems_per_row);

template <typename T>
void launch_fused_residual_ln_store_pre_ln_res(T* norm_output,
                                    T* res_output,
                                    const T* vals,
                                    const T* residual,
                                    const T* bias,
                                    const T* gamma,
                                    const T* beta,
                                    float epsilon,
                                    int rows,
                                    int elems_per_row,
                                    cudaStream_t stream)
{
    // 8 for __half, 4 for float
    constexpr int T_per_load = ln::granularity / sizeof(T);

    constexpr int max_threads = 256;

    // For Flaoat, unroll 4, for __half, unroll 2
    constexpr int internal_unroll = sizeof(T) == 4 ? 4 : 2;

    const bool is_subblock_schedule = (elems_per_row <= 128) ? true : false;
    const int h_per_step = is_subblock_schedule ? T_per_load : T_per_load * internal_unroll;

    // Scheduling concern: may be slightly faster for some inputs to assign multiple stages of
    // warp-sized blocks rather than stepping up to 64/96 threads
    const int one_step_threads = next_pow2((elems_per_row + h_per_step - 1) / h_per_step);
    const int threads_per_group = (one_step_threads < max_threads) ? one_step_threads : max_threads;
    const int warps_per_group = threads_per_group / hw_warp_size;

    const int groups_per_block_max =
        is_subblock_schedule ? (max_threads + threads_per_group - 1) / threads_per_group : 1;
    const int groups_per_block = (rows < groups_per_block_max) ? rows : groups_per_block_max;
    const int groups_launch = (groups_per_block + rows - 1) / groups_per_block;

    dim3 block(threads_per_group, groups_per_block);
    dim3 grid(groups_launch);

    const int elems_per_step = threads_per_group * h_per_step;
    const int external_unroll = (elems_per_row + elems_per_step - 1) / elems_per_step;

    if (is_subblock_schedule) {
        // <=128
        if (threads_per_group == 1) {
            LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(1, 1, 1, max_threads);
        } else if (threads_per_group == 2) {
            LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(1, 1, 2, max_threads);
        } else if (threads_per_group == 4) {
            LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(1, 1, 4, max_threads);
        } else if (threads_per_group == 8) {
            LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(1, 1, 8, max_threads);
        } else if (threads_per_group == 16) {
            LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(1, 1, 16, max_threads);
        }
    } else if (external_unroll == 1) {
        // 129 - 4096 elems
        // (this can launch with 1-7 warps as well)
        LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(1, internal_unroll, max_threads, max_threads);
    } else if (external_unroll == 2) {
        // 4097 - 8192 elems
        LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(2, internal_unroll, max_threads, max_threads);
    } else if (external_unroll == 3) {
        // 8193 - 12288 elems
        LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(3, internal_unroll, max_threads, max_threads);
    } else if (external_unroll == 4) {
        // 12289 - 16384 elems
        LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(4, internal_unroll, max_threads, max_threads);
    }
}

// No-store specializations
template void launch_fused_residual_ln(__half*,
                                       const __half*,
                                       const __half*,
                                       const __half*,
                                       const __half*,
                                       const __half*,
                                       float,
                                       int,
                                       int,
                                       cudaStream_t);

template void launch_fused_residual_ln(float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       float,
                                       int,
                                       int,
                                       cudaStream_t);

// Store specializations
template void launch_fused_residual_ln_store_pre_ln_res(__half*,
                                             __half*,
                                             const __half*,
                                             const __half*,
                                             const __half*,
                                             const __half*,
                                             const __half*,
                                             float,
                                             int,
                                             int,
                                             cudaStream_t);

template void launch_fused_residual_ln_store_pre_ln_res(float*,
                                             float*,
                                             const float*,
                                             const float*,
                                             const float*,
                                             const float*,
                                             const float*,
                                             float,
                                             int,
                                             int,
                                             cudaStream_t);
