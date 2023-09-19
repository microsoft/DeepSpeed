// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "conversion_utils.h"
#include "inference_cuda_layers.h"

#define MAX_QUANTIZE_GROUPING 1024

#define loop_unroll 1
#define loop_unroll_bits 1

template <typename T>
__global__ void dequantize_kernel(T* output,
                                  const int8_t* input,
                                  const float* qscale,
                                  int output_size,
                                  int hidden_dim,
                                  int groups,
                                  int merge_count)
{
    unsigned merge_hidden = hidden_dim >> merge_count;
    unsigned quantization_stride = (merge_hidden * output_size) / groups;

    unsigned bid = blockIdx.x;
    unsigned tid = threadIdx.x;

    while (tid < output_size) {
        unsigned w_index = bid / merge_hidden;
        unsigned q_index = tid + bid * output_size;

        auto q = input[q_index];

        unsigned merge_hidden_total = w_index * merge_hidden;
        unsigned scale_index =
            ((((bid - merge_hidden_total) + tid * merge_hidden) / quantization_stride)
             << merge_count) +
            w_index;

        float scale_data = qscale[scale_index];

        output[q_index] = conversion::to<T>(scale_data * (float)q);
        tid += blockDim.x;
    }
}

template <typename T>
void launch_dequantize(T* output,
                       const int8_t* input,
                       const float* qscale,
                       unsigned output_size,
                       unsigned hidden_dim,
                       unsigned groups,
                       unsigned merge_count,
                       cudaStream_t stream)
{
    unsigned threads = 1024;
    dim3 block_dims(threads);
    dim3 grid_dims(hidden_dim);

    dequantize_kernel<<<grid_dims, block_dims, 0, stream>>>(
        output, input, qscale, output_size, hidden_dim, groups, merge_count);
}

#define INSTANTIATE_DEQUANTIZE_MERGE(T) \
    template void launch_dequantize<T>( \
        T*, const int8_t*, const float*, unsigned, unsigned, unsigned, unsigned, cudaStream_t);

INSTANTIATE_DEQUANTIZE_MERGE(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_DEQUANTIZE_MERGE(__nv_bfloat16);
#endif
INSTANTIATE_DEQUANTIZE_MERGE(__half);

__global__ void dequantize_kernel(float* output,
                                  const int8_t* input,
                                  const float* qscale,
                                  int hidden_dim,
                                  unsigned merge_hidden,
                                  int cnt)
{
}
template <typename T>
__global__ void dequantize_kernel(T* output,
                                  const int8_t* input,
                                  const float* qscale,
                                  unsigned hidden_dim,
                                  unsigned merge_hidden,
                                  int cnt)
{
    unsigned bid = blockIdx.x * gridDim.y + blockIdx.y;
    unsigned tid = threadIdx.x;

    float local_scale = qscale[blockIdx.x];

    const float* input_cast = reinterpret_cast<const float*>(input);
    float2* output_cast = reinterpret_cast<float2*>(output);

    input_cast += bid * merge_hidden;
    output_cast += bid * merge_hidden;

    for (int c = 0; c < cnt; c++) {
        if (tid < merge_hidden) {
            float q = input_cast[tid];
            int8_t* q_int8 = (int8_t*)&q;

            float2 q_f;
            T* q_h = (T*)&q_f;

            q_h[0] = conversion::to<T>(local_scale * (float)q_int8[0]);
            q_h[1] = conversion::to<T>(local_scale * (float)q_int8[1]);
            q_h[2] = conversion::to<T>(local_scale * (float)q_int8[2]);
            q_h[3] = conversion::to<T>(local_scale * (float)q_int8[3]);
            output_cast[tid] = q_f;
            tid += blockDim.x;
        }
    }
}

template <typename T>
void launch_dequantize(T* output,
                       const int8_t* input,
                       const float* qscale,
                       unsigned output_size,
                       unsigned hidden_dim,
                       unsigned groups,
                       cudaStream_t stream)
{
    unsigned threads = 1024;
    hidden_dim /= 4;
    unsigned thd_cnt = (hidden_dim - 1) / threads + 1;

    assert(output_size % groups == 0);
    unsigned blocks = output_size / groups;

    dim3 block_dims(threads);
    dim3 grid_dims(groups, blocks);

    dequantize_kernel<<<grid_dims, block_dims, 0, stream>>>(
        output, input, qscale, hidden_dim, hidden_dim, thd_cnt);
}

#define INSTANTIATE_DEQUANTIZE_NO_MERGE(T) \
    template void launch_dequantize<T>(    \
        T*, const int8_t*, const float*, unsigned, unsigned, unsigned, cudaStream_t);

INSTANTIATE_DEQUANTIZE_NO_MERGE(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_DEQUANTIZE_NO_MERGE(__nv_bfloat16);
#endif
INSTANTIATE_DEQUANTIZE_NO_MERGE(__half);

struct q4_data{
    int8_t a: 4, b:4; // two 4-bit data
};

__global__ void dequantize_kernel_4bits(float* output,
                                  const int8_t* input,
                                  const float* qscale,
                                  int hidden_dim,
                                  unsigned merge_hidden,
                                  int cnt)
{
}
template <typename T>
__global__ void dequantize_kernel_4bits(T* output,
                                  const int8_t* input,
                                  const float* qscale,
                                  unsigned hidden_dim,
                                  unsigned merge_hidden,
                                  int cnt)
{
    unsigned bid = blockIdx.x * gridDim.y + blockIdx.y;
    unsigned tid = threadIdx.x;

    float local_scale = qscale[blockIdx.x];

    const float* input_cast = reinterpret_cast<const float*>(input);
    float4* output_cast = reinterpret_cast<float4*>(output);

    input_cast += bid * merge_hidden;
    output_cast += bid * merge_hidden;

    for (int c = 0; c < cnt; c++) 
    {
        if (tid < merge_hidden) {
            float q = input_cast[tid];
            int8_t* q_int4 = (int8_t*)&q;

            float4 q_f;
            T* q_h = (T*)&q_f;
            q4_data q4[4];

            q4[0].a = (q_int4[0] & 0xf);
            q4[0].b = ((q_int4[0] & 0xf0) >> 4);
            q_h[0] = conversion::to<T>(local_scale * (float)q4[0].a);
            q_h[1] = conversion::to<T>(local_scale * (float)q4[0].b);

            q4[1].a = (q_int4[1] & 0xf);
            q4[1].b = ((q_int4[1] & 0xf0) >> 4);
            q_h[2] = conversion::to<T>(local_scale * (float)q4[1].a);
            q_h[3] = conversion::to<T>(local_scale * (float)q4[1].b);

            q4[2].a = (q_int4[2] & 0xf);
            q4[2].b = ((q_int4[2] & 0xf0) >> 4);
            q_h[4] = conversion::to<T>(local_scale * (float)q4[2].a);
            q_h[5] = conversion::to<T>(local_scale * (float)q4[2].b);

            q4[3].a = (q_int4[3] & 0xf);
            q4[3].b = ((q_int4[3] & 0xf0) >> 4);
            q_h[6] = conversion::to<T>(local_scale * (float)q4[3].a);
            q_h[7] = conversion::to<T>(local_scale * (float)q4[3].b);
            
            output_cast[tid] = q_f;
            tid += blockDim.x;
        }
    }
}

template <typename T>
void launch_dequantize_4bits(T* output,
                       const int8_t* input,
                       const float* qscale,
                       unsigned output_size,
                       unsigned hidden_dim,
                       unsigned groups,
                       cudaStream_t stream)
{
    unsigned threads = 1024;
    hidden_dim /= 8;
    unsigned thd_cnt = (hidden_dim - 1) / threads + 1;

    assert(output_size % groups == 0);
    unsigned blocks = output_size / groups;

    dim3 block_dims(thd_cnt == 1 ? hidden_dim : threads);
    dim3 grid_dims(groups, blocks);

    dequantize_kernel_4bits<<<grid_dims, block_dims, 0, stream>>>(
        output, input, qscale, hidden_dim, hidden_dim, thd_cnt);
}

#define INSTANTIATE_DEQUANTIZE_4BIT(T) \
    template void launch_dequantize_4bits<T>(    \
        T*, const int8_t*, const float*, unsigned, unsigned, unsigned, cudaStream_t);

INSTANTIATE_DEQUANTIZE_4BIT(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_DEQUANTIZE_4BIT(__nv_bfloat16);
#endif
INSTANTIATE_DEQUANTIZE_4BIT(__half);



struct q2_data{
    int8_t a: 2, b: 2, c: 2, d: 2; // four 2-bit data
};

__global__ void dequantize_kernel_2bits(float* output,
                                  const int8_t* input,
                                  const float* qscale,
                                  int hidden_dim,
                                  unsigned merge_hidden,
                                  int cnt)
{
}
template <typename T>
__global__ void dequantize_kernel_2bits(T* output,
                                  const int8_t* input,
                                  const float* qscale,
                                  unsigned hidden_dim,
                                  unsigned merge_hidden,
                                  int cnt)
{
    unsigned bid = blockIdx.x * gridDim.y + blockIdx.y;
    unsigned tid = threadIdx.x;

    float local_scale = qscale[blockIdx.x];

    const int16_t* input_cast = reinterpret_cast<const int16_t*>(input);

    float2* output_cast = reinterpret_cast<float2*>(output);

    input += bid * merge_hidden;
    output_cast += bid * merge_hidden;

    for (int c = 0; c < cnt; c++) {
        if (tid < merge_hidden) {
            int16_t q = input[tid];
            int8_t* q_int2 = (int8_t*)&q;

            float2 q_f;
            T* q_h = (T*)&q_f;
            q2_data q2[2];

            q2[0].a = (q_int2[0] & 0x3);
            q2[0].b = ((q_int2[0] & 0xC) >> 2);
            q2[0].c = ((q_int2[0] & 0x30) >> 4);
            q2[0].d = ((q_int2[0] & 0xC0) >> 6);
            q_h[0] = conversion::to<T>(local_scale * (float)q2[0].a);
            q_h[1] = conversion::to<T>(local_scale * (float)q2[0].b);
            q_h[2] = conversion::to<T>(local_scale * (float)q2[0].c);
            q_h[3] = conversion::to<T>(local_scale * (float)q2[0].d);

            q2[1].a = (q_int2[1] & 0x3);
            q2[1].b = ((q_int2[1] & 0xC) >> 2);
            q2[1].c = ((q_int2[1] & 0x30) >> 4);
            q2[1].d = ((q_int2[1] & 0xC0) >> 6);
            q_h[4] = conversion::to<T>(local_scale * (float)q2[1].a);
            q_h[5] = conversion::to<T>(local_scale * (float)q2[1].b);
            q_h[6] = conversion::to<T>(local_scale * (float)q2[1].c);
            q_h[7] = conversion::to<T>(local_scale * (float)q2[1].d);

            output_cast[tid] = q_f;
            tid += blockDim.x;
        }
    }
}

template <typename T>
void launch_dequantize_2bits(T* output,
                       const int8_t* input,
                       const float* qscale,
                       unsigned output_size,
                       unsigned hidden_dim,
                       unsigned groups,
                       cudaStream_t stream)
{
    unsigned threads = 1024;
    hidden_dim /= 8;
    unsigned thd_cnt = (hidden_dim - 1) / threads + 1;

    assert(output_size % groups == 0);
    unsigned blocks = output_size / groups;

    dim3 block_dims(threads);
    dim3 grid_dims(groups, blocks);

    dequantize_kernel_2bits<<<grid_dims, block_dims, 0, stream>>>(
        output, input, qscale, hidden_dim, hidden_dim, thd_cnt);
}

#define INSTANTIATE_DEQUANTIZE_2BIT(T) \
    template void launch_dequantize_2bits<T>(    \
        T*, const int8_t*, const float*, unsigned, unsigned, unsigned, cudaStream_t);

INSTANTIATE_DEQUANTIZE_2BIT(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_DEQUANTIZE_2BIT(__nv_bfloat16);
#endif
INSTANTIATE_DEQUANTIZE_2BIT(__half);


__global__ void dequantize_kernel_10bits(float* output,
                                  const int16_t* input,
                                  const float* qscale,
                                  int hidden_dim,
                                  unsigned merge_hidden,
                                  int cnt)
{
}
template <typename T>
__global__ void dequantize_kernel_10bits(T* output,
                                  const int16_t* input,
                                  const float* qscale,
                                  unsigned hidden_dim,
                                  unsigned merge_hidden,
                                  int cnt)
{
    unsigned bid = blockIdx.x * gridDim.y + blockIdx.y;
    unsigned tid = threadIdx.x;

    float local_scale = qscale[blockIdx.x];

    const float2* input_cast = reinterpret_cast<const float2*>(input);
    float2* output_cast = reinterpret_cast<float2*>(output);

    input_cast += bid * merge_hidden;
    output_cast += bid * merge_hidden;

    for (int c = 0; c < cnt; c++) {
        if (tid < merge_hidden) {
            float2 q = input_cast[tid];
            int16_t* q_int10 = (int16_t*)&q;

            float2 q_f;
            T* q_h = (T*)&q_f;

            q_h[0] = conversion::to<T>(local_scale * (float)q_int10[0]);
            q_h[1] = conversion::to<T>(local_scale * (float)q_int10[1]);
            q_h[2] = conversion::to<T>(local_scale * (float)q_int10[2]);
            q_h[3] = conversion::to<T>(local_scale * (float)q_int10[3]);
            output_cast[tid] = q_f;
            tid += blockDim.x;
        }
    }
}

template <typename T>
void launch_dequantize_10bits(T* output,
                       const int16_t* input,
                       const float* qscale,
                       unsigned output_size,
                       unsigned hidden_dim,
                       unsigned groups,
                       cudaStream_t stream)
{
    unsigned threads = 1024;
    hidden_dim /= 4;
    unsigned thd_cnt = (hidden_dim - 1) / threads + 1;

    assert(output_size % groups == 0);
    unsigned blocks = output_size / groups;

    dim3 block_dims(threads);
    dim3 grid_dims(groups, blocks);

    dequantize_kernel_10bits<<<grid_dims, block_dims, 0, stream>>>(
        output, input, qscale, hidden_dim, hidden_dim, thd_cnt);
}

#define INSTANTIATE_DEQUANTIZE_10BIT(T) \
    template void launch_dequantize_10bits<T>(    \
        T*, const int16_t*, const float*, unsigned, unsigned, unsigned, cudaStream_t);

INSTANTIATE_DEQUANTIZE_10BIT(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_DEQUANTIZE_10BIT(__nv_bfloat16);
#endif
INSTANTIATE_DEQUANTIZE_10BIT(__half);
