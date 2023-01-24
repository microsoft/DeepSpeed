/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include "activation_utils.h"
#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "inference_cuda_layers.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;

DS_D_INLINE float gelu(const float x)
{
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param = 0.044715;
    return x * 0.5f * (1.0f + tanhf(sqrt_param * (x + mul_param * x * x * x)));
}

/*
In-place activation(biasAdd(x)) for channels last
*/
template <typename T, activation::Type ActFn>
__global__ void fused_bias_act(T* input, const T* bias, int total_count, int intermediate_size)
{
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

template <typename T>
__global__ void fused_bias_residual_postln(T* residual,
                                           const T* hidden_state,
                                           const T* bias,
                                           const int total_count,
                                           const int intermediate_size)
{
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
}

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

__global__ void pad_data_kernel(__half* padded_output,
                                __half* output,
                                int head_size,
                                int padded_head_size)
{
    float4* padded_output_cast = reinterpret_cast<float4*>(padded_output);
    float4* output_cast = reinterpret_cast<float4*>(output);
    int bid = blockIdx.x * (blockDim.y) + threadIdx.y;
    int idx = threadIdx.x;
    padded_output_cast += (bid * padded_head_size);
    output_cast += (bid * head_size);
    float4 ZERO;
    const __half2 zero_h = __float2half2_rn(0.f);
    __half2* ZERO_h = reinterpret_cast<__half2*>(&ZERO);
#pragma unroll
    for (int i = 0; i < 4; i++) ZERO_h[i] = zero_h;
    if (idx < head_size)
        padded_output_cast[idx] = output_cast[idx];
    else
        padded_output_cast[idx] = ZERO;
}
__global__ void pad_data_kernel(float* padded_output,
                                float* output,
                                int head_size,
                                int padded_head_size)
{
}
template <typename T>
void pad_data(T* padded_output,
              T* output,
              int bsz,
              int head_size,
              int padded_head_size,
              cudaStream_t stream)
{
    dim3 grid_dim((bsz - 1) / 16 + 1);
    dim3 block_dim(padded_head_size / 8, 16);
    pad_data_kernel<<<grid_dim, block_dim, 0, stream>>>(
        padded_output, output, head_size / 8, padded_head_size / 8);
}
template void pad_data(__half* padded_output,
                       __half* output,
                       int bsz,
                       int head_size,
                       int padded_head_size,
                       cudaStream_t stream);
template void pad_data(float* padded_output,
                       float* output,
                       int bsz,
                       int head_size,
                       int padded_head_size,
                       cudaStream_t stream);

__global__ void pad_head_seq_kernel(__half* padded_output,
                                    __half* output,
                                    int seq_len,
                                    int padded_seq_len,
                                    int head_size,
                                    int padded_head_size)
{
    float4* padded_output_cast = reinterpret_cast<float4*>(padded_output);
    float4* output_cast = reinterpret_cast<float4*>(output);
    int bsz = blockIdx.x;
    int bid = blockIdx.y * (blockDim.y) + threadIdx.y;
    int idx = threadIdx.x;
    padded_output_cast += (bsz * padded_seq_len + bid) * padded_head_size;
    output_cast += (bsz * seq_len + bid) * head_size;
    float4 ZERO;
    const __half2 zero_h = __float2half2_rn(0.f);
    __half2* ZERO_h = reinterpret_cast<__half2*>(&ZERO);
#pragma unroll
    for (int i = 0; i < 4; i++) ZERO_h[i] = zero_h;

    if (idx < head_size && bid < seq_len)
        padded_output_cast[idx] = output_cast[idx];
    else
        padded_output_cast[idx] = ZERO;
}
__global__ void pad_head_seq_kernel(float* padded_output,
                                    float* output,
                                    int seq_len,
                                    int padded_seq_len,
                                    int head_size,
                                    int padded_head_size)
{
}
template <typename T>
void pad_head_seq(T* padded_output,
                  T* output,
                  int bsz,
                  int seq_len,
                  int padded_seq_len,
                  int head_size,
                  int padded_head_size,
                  cudaStream_t stream)
{
    dim3 grid_dim(bsz, padded_seq_len / 16);
    dim3 block_dim(padded_head_size / 8, 16);
    pad_head_seq_kernel<<<grid_dim, block_dim, 0, stream>>>(
        padded_output, output, seq_len, padded_seq_len, head_size / 8, padded_head_size / 8);
}
template void pad_head_seq(__half* padded_output,
                           __half* output,
                           int bsz,
                           int seq_len,
                           int padded_seq_len,
                           int head_size,
                           int padded_head_size,
                           cudaStream_t stream);
template void pad_head_seq(float* padded_output,
                           float* output,
                           int bsz,
                           int seq_len,
                           int padded_seq_len,
                           int head_size,
                           int padded_head_size,
                           cudaStream_t stream);

// TODO(cmikeh2): evaluate different GeLU performance
__device__ __forceinline__ float old_gelu(float val)
{
    // 1 / sqrt(2)
    constexpr float rsqrt_2 = 0.707106769084930419922;
    return val * 0.5f * (1.0f + erff(val * rsqrt_2));
}

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
            // convert
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
