
#include "custom_cuda_layers.h"

#include <cuda_fp16.h>
namespace cg = cooperative_groups;



__global__ void min_max_local(const __half* vals, int total_count, __half* min_max)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    const float4* vals_cast = reinterpret_cast<const float4*>(vals);
    unsigned total_count_8 = total_count << 3;
    float4 data;

    int group_id = (blockIdx.x * gridDim.y + blockIdx.y);

    int group_index = id + group_id * blockDim.x;
    int id1 = id << 3;
    __half max = -10000.0;
    if (id < total_count) 
    {
        data = vals_cast[group_index];
        __half* data_h = reinterpret_cast<__half*>(&data);
        data_h[1] = (id1 + 1) < total_count_8 ? __habs(data_h[1]) : max;
        data_h[2] = (id1 + 2) < total_count_8 ? __habs(data_h[2]) : max;
        data_h[3] = (id1 + 3) < total_count_8 ? __habs(data_h[3]) : max;
        data_h[4] = (id1 + 4) < total_count_8 ? __habs(data_h[4]) : max;
        data_h[5] = (id1 + 5) < total_count_8 ? __habs(data_h[5]) : max;
        data_h[6] = (id1 + 6) < total_count_8 ? __habs(data_h[6]) : max;
        data_h[7] = (id1 + 7) < total_count_8 ? __habs(data_h[7]) : max;
        if (__hgt(data_h[0], max)) max = data_h[0];
        if (__hgt(data_h[1], max)) max = data_h[1];
        if (__hgt(data_h[2], max)) max = data_h[2];
        if (__hgt(data_h[3], max)) max = data_h[3];
        if (__hgt(data_h[4], max)) max = data_h[4];
        if (__hgt(data_h[5], max)) max = data_h[5];
        if (__hgt(data_h[6], max)) max = data_h[6];
        if (__hgt(data_h[7], max)) max = data_h[7];
    }

#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1) {
        auto temp = g.shfl_xor(max, i);
        if (__hgt(temp, max)) max = temp;
    }
    __shared__ __half partialMax[WARP_SIZE];
    if (lane == 0) partialMax[gid] = max;
    b.sync();

    max = partialMax[lane];

    b.sync();

#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1) {
        auto temp = g.shfl_xor(max, i);
        if (__hgt(temp, max)) max = temp;
    }

    if (threadIdx.x == 0) min_max[group_id] = max;
}

__global__ void min_max_local(const float* vals, int total_count, float* min_max)
{

}

__global__ void quantize_data(const __half* vals,
                              int8_t* vals_int,
                              int total_count,
                              __half* min_max,
                              int groups,
                              float* q_scale_d,
                              int num_bits)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    const float4* vals_cast = reinterpret_cast<const float4*>(vals);
    float4* max_cast = reinterpret_cast<float4*>(min_max);
    float2* vals_int_cast = reinterpret_cast<float2*>(vals_int);
    float4 data;

    int bid = blockIdx.x * gridDim.y + blockIdx.y;
    int group_index = id + bid * blockDim.x;
    __half max = -10000.0;

    if (id < total_count) {
        data = vals_cast[group_index];
    }


    min_max += groups * blockIdx.x;
    while (id < groups) {
        __half m_data = min_max[id];
        m_data = __habs(m_data);
        if (__hgt(m_data, max)) max = m_data;
        id += blockDim.x;
    }

#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1) {
        auto temp = g.shfl_xor(max, i);
        if (__hgt(temp, max)) max = temp;
    }
    __shared__ __half partialMax[WARP_SIZE];
    if (lane == 0) partialMax[gid] = max;
    b.sync();

    max = partialMax[lane];
    b.sync();

#pragma unroll
    for (int i = 1; i < warp_num; i <<= 1) {
        auto temp = g.shfl_xor(max, i);
        if (__hgt(temp, max)) max = temp;
    }
    max = g.shfl(max, 0);

    float q_scale = (1 << num_bits) / (2 * (float)max);

    float2 q_data_int;
    int8_t* q_data_8 = reinterpret_cast<int8_t*>(&q_data_int);

    if (threadIdx.x < total_count) {
        __half* data_h = reinterpret_cast<__half*>(&data);
        int32_t data_f[8];
        data_f[0] = round((float)data_h[0] * q_scale);
        data_f[1] = round((float)data_h[1] * q_scale);
        data_f[2] = round((float)data_h[2] * q_scale);
        data_f[3] = round((float)data_h[3] * q_scale);
        data_f[4] = round((float)data_h[4] * q_scale);
        data_f[5] = round((float)data_h[5] * q_scale);
        data_f[6] = round((float)data_h[6] * q_scale);
        data_f[7] = round((float)data_h[7] * q_scale);
        q_data_8[0] = (data_f[0] > 127) ? 127 : (data_f[0] < -128 ? -128 : data_f[0]);
        q_data_8[1] = (data_f[1] > 127) ? 127 : (data_f[1] < -128 ? -128 : data_f[1]);
        q_data_8[2] = (data_f[2] > 127) ? 127 : (data_f[2] < -128 ? -128 : data_f[2]);
        q_data_8[3] = (data_f[3] > 127) ? 127 : (data_f[3] < -128 ? -128 : data_f[3]);
        q_data_8[4] = (data_f[4] > 127) ? 127 : (data_f[4] < -128 ? -128 : data_f[4]);
        q_data_8[5] = (data_f[5] > 127) ? 127 : (data_f[5] < -128 ? -128 : data_f[5]);
        q_data_8[6] = (data_f[6] > 127) ? 127 : (data_f[6] < -128 ? -128 : data_f[6]);
        q_data_8[7] = (data_f[7] > 127) ? 127 : (data_f[7] < -128 ? -128 : data_f[7]);
        vals_int_cast[group_index] = q_data_int;
    }

    if (threadIdx.x == 0 && blockIdx.y == 0) {
        q_scale_d[blockIdx.x] = 1 / q_scale;
    }
}

__global__ void quantize_data_1(const __half* vals,
                              int8_t* vals_int,
                              int total_count,
                              float* q_scale_d,
                              int num_bits,
                              int group_size)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    const float4* vals_cast = reinterpret_cast<const float4*>(vals);
    float2* vals_int_cast = reinterpret_cast<float2*>(vals_int);

    int bid = blockIdx.x;
    unsigned group_index = bid * group_size;
    __half max = -10000.0;

    unsigned cnt = 0;
    float4 data[8];
    while (id < group_size) {
        data[cnt] = vals_cast[group_index + id];
        id += (blockDim.x);
        cnt++;
    }
    for(int i = 0;i < cnt;i++){
        __half* data_h = reinterpret_cast<__half*>(&data[i]);
        if (__hgt(__habs(data_h[0]), max)) max = __habs(data_h[0]);
        if (__hgt(__habs(data_h[1]), max)) max = __habs(data_h[1]);
        if (__hgt(__habs(data_h[2]), max)) max = __habs(data_h[2]);
        if (__hgt(__habs(data_h[3]), max)) max = __habs(data_h[3]);
        if (__hgt(__habs(data_h[4]), max)) max = __habs(data_h[4]);
        if (__hgt(__habs(data_h[5]), max)) max = __habs(data_h[5]);
        if (__hgt(__habs(data_h[6]), max)) max = __habs(data_h[6]);
        if (__hgt(__habs(data_h[7]), max)) max = __habs(data_h[7]);
    }

    #pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1) {
        auto temp = g.shfl_xor(max, i);
        if (__hgt(temp, max)) max = temp;
    }
    __shared__ __half partialMax[WARP_SIZE];

    if (lane == 0) partialMax[gid] = max;

    b.sync();

    max = partialMax[lane];
    b.sync();

    #pragma unroll
    for (int i = 1; i < warp_num; i <<= 1) {
        auto temp = g.shfl_xor(max, i);
        if (__hgt(temp, max)) max = temp;
    }
    max = g.shfl(max, 0);

    float q_scale = (1 << num_bits) / (2 * (float)max + 1e-5);

    group_index = threadIdx.x + bid * group_size;
    for(int i = 0;i < cnt;i++){
        float2 q_data_int;
        int8_t* q_data_8 = reinterpret_cast<int8_t*>(&q_data_int);
        __half* data_h = reinterpret_cast<__half*>(&data[i]);
        int32_t data_f[8];
        data_f[0] = round((float)data_h[0] * q_scale);
        data_f[1] = round((float)data_h[1] * q_scale);
        data_f[2] = round((float)data_h[2] * q_scale);
        data_f[3] = round((float)data_h[3] * q_scale);
        data_f[4] = round((float)data_h[4] * q_scale);
        data_f[5] = round((float)data_h[5] * q_scale);
        data_f[6] = round((float)data_h[6] * q_scale);
        data_f[7] = round((float)data_h[7] * q_scale);
        q_data_8[0] = (data_f[0] > 127) ? 127 : (data_f[0] < -128 ? -128 : data_f[0]);
        q_data_8[1] = (data_f[1] > 127) ? 127 : (data_f[1] < -128 ? -128 : data_f[1]);
        q_data_8[2] = (data_f[2] > 127) ? 127 : (data_f[2] < -128 ? -128 : data_f[2]);
        q_data_8[3] = (data_f[3] > 127) ? 127 : (data_f[3] < -128 ? -128 : data_f[3]);
        q_data_8[4] = (data_f[4] > 127) ? 127 : (data_f[4] < -128 ? -128 : data_f[4]);
        q_data_8[5] = (data_f[5] > 127) ? 127 : (data_f[5] < -128 ? -128 : data_f[5]);
        q_data_8[6] = (data_f[6] > 127) ? 127 : (data_f[6] < -128 ? -128 : data_f[6]);
        q_data_8[7] = (data_f[7] > 127) ? 127 : (data_f[7] < -128 ? -128 : data_f[7]);
        vals_int_cast[group_index] = q_data_int;
        group_index += (blockDim.x);
    }
    if (threadIdx.x == 0) q_scale_d[blockIdx.x] = 1 / q_scale;
}
__global__ void quantize_data_1(const float* vals,
    int8_t* vals_int,
    int total_count,
    float* q_scale_d,
    int num_bits)
{

}
        
__global__ void quantize_data(const float* vals,
                              int8_t* vals_int,
                              int total_count,
                              volatile float* min_max,
                              int groups,
                              float* q_scale_d,
                              int num_bits)
{

}

template <typename T>
void quantize_kernel(int8_t* vals_int,
                     const T* vals,
                     T* min_max,
                     float* q_scale_d,
                     int groups,
                     int total_count,
                     int num_bits,
                     cudaStream_t stream);
template <>
void quantize_kernel<float>(int8_t* vals_int,
                            const float* vals,
                            float* min_max,
                            float* q_scale_d,
                            int groups,
                            int total_count,
                            int num_bits,
                            cudaStream_t stream)
{
    dim3 grid_dim((total_count - 1) / 8192 + 1);
    dim3 block_dim(1024);

    min_max_local<<<grid_dim, block_dim, 0, stream>>>(vals, total_count / 4, min_max);
    quantize_data<<<grid_dim, block_dim, 0, stream>>>(
        vals, vals_int, total_count / 4, min_max, grid_dim.y / 4, q_scale_d, num_bits);
}

template <>
void quantize_kernel<__half>(int8_t* vals_int,
                             const __half* vals,
                             __half* min_max,
                             float* q_scale_d,
                             int groups,
                             int total_count,
                             int num_bits,
                             cudaStream_t stream)
{
    int threads = (total_count - 1) / (8 * groups) + 1;
    int group_size = threads;
    threads = 1024;
    dim3 grid_dim(groups);
    dim3 block_dim(threads);
    quantize_data_1<<<grid_dim, block_dim, 0, stream>>>(
        vals, vals_int, total_count / 8, q_scale_d, num_bits, group_size);
}

template <typename T>
void quantize_kernel1(int8_t* vals_int,
                     const T* vals,
                     T* min_max,
                     float* q_scale_d,
                     int groups,
                     int total_count,
                     int num_bits,
                     cudaStream_t stream);
template <>
void quantize_kernel1<float>(int8_t* vals_int,
                            const float* vals,
                            float* min_max,
                            float* q_scale_d,
                            int groups,
                            int total_count,
                            int num_bits,
                            cudaStream_t stream)
{
    dim3 grid_dim((total_count - 1) / 8192 + 1);
    dim3 block_dim(1024);

    min_max_local<<<grid_dim, block_dim, 0, stream>>>(vals, total_count / 4, min_max);
    quantize_data<<<grid_dim, block_dim, 0, stream>>>(
        vals, vals_int, total_count / 4, min_max, grid_dim.y / 4, q_scale_d, num_bits);
}

template <>
void quantize_kernel1<__half>(int8_t* vals_int,
                             const __half* vals,
                             __half* min_max,
                             float* q_scale_d,
                             int groups,
                             int total_count,
                             int num_bits,
                             cudaStream_t stream)
{
    int threads = (total_count - 1) / (8 * groups) + 1;
    int group_size = threads;
    threads = 1024;
    dim3 grid_dim(groups, (total_count - 1) / ((group_size < threads ? group_size : threads) * 8 * groups) + 1);
    dim3 block_dim(threads);
    min_max_local<<<grid_dim, block_dim, 0, stream>>>(vals, group_size, min_max);
    quantize_data<<<grid_dim, block_dim, 0, stream>>>(
        vals, vals_int, group_size, min_max, grid_dim.y, q_scale_d, num_bits);
}


