
#include "custom_cuda_layers.h"

#include <cuda_fp16.h>
namespace cg = cooperative_groups;

__global__ void min_max_local(const __half *vals, int total_count, __half *min, __half *max)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    // int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    const float4 *vals_cast = reinterpret_cast<const float4 *>(vals);
    unsigned total_count_8 = total_count << 3;
    float4 data;

    int group_id = (blockIdx.x * gridDim.y + blockIdx.y);

    int group_index = id + group_id * blockDim.x;
    int id1 = id << 3;
    __half finalMax = -10000.0;
    __half finalMin = 10000.0;
    if (id < total_count)
    {
        data = vals_cast[group_index];
        __half *data_h = reinterpret_cast<__half *>(&data);
        data_h[1] = (id1 + 1) < total_count_8 ? (data_h[1]) : finalMax;
        data_h[2] = (id1 + 2) < total_count_8 ? (data_h[2]) : finalMax;
        data_h[3] = (id1 + 3) < total_count_8 ? (data_h[3]) : finalMax;
        data_h[4] = (id1 + 4) < total_count_8 ? (data_h[4]) : finalMax;
        data_h[5] = (id1 + 5) < total_count_8 ? (data_h[5]) : finalMax;
        data_h[6] = (id1 + 6) < total_count_8 ? (data_h[6]) : finalMax;
        data_h[7] = (id1 + 7) < total_count_8 ? (data_h[7]) : finalMax;
        if (__hgt(data_h[0], finalMax))
            finalMax = data_h[0];
        if (__hgt(data_h[1], finalMax))
            finalMax = data_h[1];
        if (__hgt(data_h[2], finalMax))
            finalMax = data_h[2];
        if (__hgt(data_h[3], finalMax))
            finalMax = data_h[3];
        if (__hgt(data_h[4], finalMax))
            finalMax = data_h[4];
        if (__hgt(data_h[5], finalMax))
            finalMax = data_h[5];
        if (__hgt(data_h[6], finalMax))
            finalMax = data_h[6];
        if (__hgt(data_h[7], finalMax))
            finalMax = data_h[7];
        data_h[1] = (id1 + 1) < total_count_8 ? (data_h[1]) : finalMin;
        data_h[2] = (id1 + 2) < total_count_8 ? (data_h[2]) : finalMin;
        data_h[3] = (id1 + 3) < total_count_8 ? (data_h[3]) : finalMin;
        data_h[4] = (id1 + 4) < total_count_8 ? (data_h[4]) : finalMin;
        data_h[5] = (id1 + 5) < total_count_8 ? (data_h[5]) : finalMin;
        data_h[6] = (id1 + 6) < total_count_8 ? (data_h[6]) : finalMin;
        data_h[7] = (id1 + 7) < total_count_8 ? (data_h[7]) : finalMin;
        if (__hlt(data_h[0], finalMin))
            finalMin = data_h[0];
        if (__hlt(data_h[1], finalMin))
            finalMin = data_h[1];
        if (__hlt(data_h[2], finalMin))
            finalMin = data_h[2];
        if (__hlt(data_h[3], finalMin))
            finalMin = data_h[3];
        if (__hlt(data_h[4], finalMin))
            finalMin = data_h[4];
        if (__hlt(data_h[5], finalMin))
            finalMin = data_h[5];
        if (__hlt(data_h[6], finalMin))
            finalMin = data_h[6];
        if (__hlt(data_h[7], finalMin))
            finalMin = data_h[7];
    }

#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1)
    {
        auto tempMax = g.shfl_xor(finalMax, i);
        if (__hgt(tempMax, finalMax))
            finalMax = tempMax;
        auto tempMin = g.shfl_xor(finalMin, i);
        if (__hlt(tempMin, finalMin))
            finalMin = tempMin;
    }
    __shared__ __half partialMax[WARP_SIZE];
    if (lane == 0)
        partialMax[gid] = finalMax;
    __shared__ __half partialMin[WARP_SIZE];
    if (lane == 0)
        partialMin[gid] = finalMin;
    b.sync();

    finalMax = partialMax[lane];
    finalMin = partialMin[lane];

    b.sync();

#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1)
    {
        auto tempMax = g.shfl_xor(finalMax, i);
        if (__hgt(tempMax, finalMax))
            finalMax = tempMax;
        auto tempMin = g.shfl_xor(finalMin, i);
        if (__hlt(tempMin, finalMin))
            finalMin = tempMin;
    }

    // if (__hgt(finalMax, 10))
    //     printf("%d %f\n", group_id, (float)finalMax);

    if (threadIdx.x == 0)
        max[group_id] = finalMax;
    if (threadIdx.x == 0)
        min[group_id] = finalMin;
}

__global__ void quantize_data(const __half *vals,
                              int8_t *vals_int,
                              int total_count,
                              __half *min,
                              __half *max,
                              int groups,
                              float *q_scale_d,
                              float *q_zero_point_d,
                              int num_bits)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    const float4 *vals_cast = reinterpret_cast<const float4 *>(vals);
    // float4* max_cast = reinterpret_cast<float4*>(min_max);
    float2 *vals_int_cast = reinterpret_cast<float2 *>(vals_int);
    float4 data;

    int bid = blockIdx.x * gridDim.y + blockIdx.y;
    int group_index = id + bid * blockDim.x;
    __half finalMax = -10000.0;
    __half finalMin = 10000.0;

    if (id < total_count)
    {
        data = vals_cast[group_index];
    }

    // min_max += groups * blockIdx.x;
    while (id < groups)
    {
        // __half m_data = min_max[id];
        // m_data = __habs(m_data);
        if (__hgt(max[id], finalMax))
            finalMax = max[id];
        if (__hlt(min[id], finalMin))
            finalMin = min[id];
        id += blockDim.x;
    }

#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1)
    {
        auto tempMax = g.shfl_xor(finalMax, i);
        if (__hgt(tempMax, finalMax))
            finalMax = tempMax;
        auto tempMin = g.shfl_xor(finalMin, i);
        if (__hlt(tempMin, finalMin))
            finalMin = tempMin;
    }
    __shared__ __half partialMax[WARP_SIZE];
    if (lane == 0)
        partialMax[gid] = finalMax;
    __shared__ __half partialMin[WARP_SIZE];
    if (lane == 0)
        partialMin[gid] = finalMin;
    b.sync();

    finalMax = partialMax[lane];
    finalMin = partialMin[lane];
    b.sync();

#pragma unroll
    for (int i = 1; i < warp_num; i <<= 1)
    {
        auto tempMax = g.shfl_xor(finalMax, i);
        if (__hgt(tempMax, finalMax))
            finalMax = tempMax;
        auto tempMin = g.shfl_xor(finalMin, i);
        if (__hlt(tempMin, finalMin))
            finalMin = tempMin;
    }
    finalMax = g.shfl(finalMax, 0);
    finalMin = g.shfl(finalMin, 0);
    // printf("finalMax:%f\n",(float)(finalMax));
    // printf("finalMin:%f\n",(float)(finalMin));

    float q_scale = ((1 << num_bits)) / ((float)finalMax - (float)finalMin);
    float q_zero_point = ((float)finalMax - (float)finalMin) / 2 + (float)finalMin; // data-zero_point=centerized data

    float2 q_data_int;
    int8_t *q_data_8 = reinterpret_cast<int8_t *>(&q_data_int);

    if (threadIdx.x < total_count)
    {
        __half *data_h = reinterpret_cast<__half *>(&data);
        q_data_8[0] = trunc(((float)data_h[0] - q_zero_point) * q_scale - 0.5);
        q_data_8[1] = trunc(((float)data_h[1] - q_zero_point) * q_scale - 0.5);
        q_data_8[2] = trunc(((float)data_h[2] - q_zero_point) * q_scale - 0.5);
        q_data_8[3] = trunc(((float)data_h[3] - q_zero_point) * q_scale - 0.5);
        q_data_8[4] = trunc(((float)data_h[4] - q_zero_point) * q_scale - 0.5);
        q_data_8[5] = trunc(((float)data_h[5] - q_zero_point) * q_scale - 0.5);
        q_data_8[6] = trunc(((float)data_h[6] - q_zero_point) * q_scale - 0.5);
        q_data_8[7] = trunc(((float)data_h[7] - q_zero_point) * q_scale - 0.5);
        vals_int_cast[group_index] = q_data_int;
    }

    if (threadIdx.x == 0 && blockIdx.y == 0)
    {
        q_scale_d[blockIdx.x] = 1 / q_scale;
        q_zero_point_d[blockIdx.x] = q_zero_point;
    }
}




__global__ void quantize_data_simple(const __half *vals,
                                int8_t *vals_int,
                                int total_count,
                                float *q_scale_d,
                                float *q_zero_point_d,
                                int num_bits,
                                int group_size)
{
    // printf("quantize_data_simple kernel starts\n");
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    __half max = -10000.0;
    __half min = 10000.0;
    __half data;
    while (id < total_count)
    {
        data = vals[id];
        if (__hgt(data, max))
            max = data;
        if (__hlt(data, min))
            min = data;
        id += (blockDim.x);
    }
#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1)
    {
        auto tempMax = g.shfl_xor(max, i);
        if (__hgt(tempMax, max))
            max = tempMax;
        auto tempMin = g.shfl_xor(min, i);
        if (__hlt(tempMin, min))
            min = tempMin;
    }
    __shared__ __half partialMax[WARP_SIZE];
    __shared__ __half partialMin[WARP_SIZE];

    // printf("for threadIdx.x/blockDim.x = %d/%d, blockIdx.x/gridDim.x = %d/%d, gid is %d \n",threadIdx.x,blockDim.x,blockIdx.x,gridDim.x,gid);

    if (lane == 0)
        partialMax[gid] = max;
    if (lane == 0)
        partialMin[gid] = min;

    b.sync();

    max = partialMax[lane];
    min = partialMin[lane];
    b.sync();

#pragma unroll
    for (int i = 1; i < warp_num; i <<= 1)
    {
        auto tempMax = g.shfl_xor(max, i);
        if (__hgt(tempMax, max))
            max = tempMax;
        auto tempMin = g.shfl_xor(min, i);
        if (__hlt(tempMin, min))
            min = tempMin;
    }
    max = g.shfl(max, 0);
    min = g.shfl(min, 0);
    // printf("total_count:%d, max: %f, min: %f\n", total_count,(float)max, (float)min);

    float q_scale = ((1 << num_bits)-2) / ((float)max - (float)min);
    float q_zero_point = ((float)max - (float)min) / 2 + (float)min; // data-zero_point=centerized data

    // group_index = threadIdx.x + bid * group_size;
    id = threadIdx.x;
    while (id < total_count)
    {
        data = vals[id];
        vals_int[id] = round(((float)data - q_zero_point) * q_scale-0.5);
        id += (blockDim.x);
    }
    if (threadIdx.x == 0)
    {
        *q_scale_d = 1 / q_scale;
        *q_zero_point_d = q_zero_point;
        // printf("CUDA: min=%f,max=%f\n", (float)min, (float)max);
    }
}



__global__ void quantize_data_1(const __half *vals,
                                int8_t *vals_int,
                                int total_count,
                                float *q_scale_d,
                                float *q_zero_point_d,
                                int num_bits,
                                int group_size)
{
    // printf("quantize_data_1 starts\n");
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    const float4 *vals_cast = reinterpret_cast<const float4 *>(vals);
    // unsigned total_count_8 = total_count << 3;
    float2 *vals_int_cast = reinterpret_cast<float2 *>(vals_int);
    // int bid = blockIdx.x;
    // unsigned group_index = bid * group_size;
    __half max = -10000.0;
    __half min = 10000.0;

    // unsigned cnt = 0;
    // printf("newton pointer left\n");
    float4 data;
    // float4 * data = new float4[group_size/blockDim.x+1];
    // extern __shared__ float4 data[];
    while (id < total_count/8.0)
    {
        // printf("newton pointer left\n");
        // printf("id:%d, group_size:%d, total_count:%d, blockDim.x:%d\n",id,group_size,total_count,blockDim.x);
        data = vals_cast[id];
        // printf("newton pointer middle\n");
        // printf("newton pointer right\n");
        // cnt++;
        // printf("first pass on data starts\n");
        // if (cnt%8 == 0 && cnt != 0)
        // {
        // for (int i = 0; i < cnt; i++)
        // {
        __half *data_h = reinterpret_cast<__half *>(&data);
        // #pragma unroll
        // printf("for threadIdx.x = %d, blockDim.x = %d, gridDim.x = %d, data[0] at %d is %f\n",threadIdx.x,blockDim.x,gridDim.x,id,(float)(data_h[0]));
        // printf("data at %d is %f\n",blockDim.x*8*i+0,(float)(data_h[0]));
        // printf("data at %d is %f\n",blockDim.x*8*i+1,(float)(data_h[1]));
        // printf("data at %d is %f\n",blockDim.x*8*i+2,(float)(data_h[2]));
        // printf("data at %d is %f\n",blockDim.x*8*i+3,(float)(data_h[3]));
        // // printf("data at %d is %f\n",blockDim.x*8*i+4,(float)(data_h[4]));
        // printf("data at %d is %f\n",blockDim.x*8*i+5,(float)(data_h[5]));
        // // printf("data at %d is %f\n",blockDim.x*8*i+6,(float)(data_h[6]));
        // printf("data at %d is %f\n",blockDim.x*8*i+7,(float)(data_h[7]));
        if (__hgt((data_h[0]), max) && (id * 8 + 0 < total_count))
            max = (data_h[0]);
        if (__hgt((data_h[1]), max) && (id * 8 + 1 < total_count))
            max = (data_h[1]);
        if (__hgt((data_h[2]), max) && (id * 8 + 2 < total_count))
            max = (data_h[2]);
        if (__hgt((data_h[3]), max) && (id * 8 + 3 < total_count))
            max = (data_h[3]);
        if (__hgt((data_h[4]), max) && (id * 8 + 4 < total_count))
            max = (data_h[4]);
        if (__hgt((data_h[5]), max) && (id * 8 + 5 < total_count))
            max = (data_h[5]);
        if (__hgt((data_h[6]), max) && (id * 8 + 6 < total_count))
            max = (data_h[6]);
        if (__hgt((data_h[7]), max) && (id * 8 + 7 < total_count))
            max = (data_h[7]);

        if (__hlt((data_h[0]), min) && (id * 8 + 0 < total_count))
            min = (data_h[0]);
        if (__hlt((data_h[1]), min) && (id * 8 + 1 < total_count))
            min = (data_h[1]);
        if (__hlt((data_h[2]), min) && (id * 8 + 2 < total_count))
            min = (data_h[2]);
        if (__hlt((data_h[3]), min) && (id * 8 + 3 < total_count))
            min = (data_h[3]);
        if (__hlt((data_h[4]), min) && (id * 8 + 4 < total_count))
            min = (data_h[4]);
        if (__hlt((data_h[5]), min) && (id * 8 + 5 < total_count))
            min = (data_h[5]);
        if (__hlt((data_h[6]), min) && (id * 8 + 6 < total_count))
            min = (data_h[6]);
        if (__hlt((data_h[7]), min) && (id * 8 + 7 < total_count))
            min = (data_h[7]);
        // }
        id += (blockDim.x);
    }
// printf("first reduce in warp starts\n");
// delete [] data;
// printf("newton pointer right\n");
#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1)
    {
        auto tempMax = g.shfl_xor(max, i);
        if (__hgt(tempMax, max))
            max = tempMax;
        auto tempMin = g.shfl_xor(min, i);
        if (__hlt(tempMin, min))
            min = tempMin;
    }
    __shared__ __half partialMax[WARP_SIZE];
    __shared__ __half partialMin[WARP_SIZE];

    // printf("for threadIdx.x/blockDim.x = %d/%d, blockIdx.x/gridDim.x = %d/%d, gid is %d \n",threadIdx.x,blockDim.x,blockIdx.x,gridDim.x,gid);

    if (lane == 0)
        partialMax[gid] = max;
    if (lane == 0)
        partialMin[gid] = min;

    b.sync();

    max = partialMax[lane];
    min = partialMin[lane];
    b.sync();

// printf("second reduce in warp starts\n");
#pragma unroll
    for (int i = 1; i < warp_num; i <<= 1)
    {
        auto tempMax = g.shfl_xor(max, i);
        if (__hgt(tempMax, max))
            max = tempMax;
        auto tempMin = g.shfl_xor(min, i);
        if (__hlt(tempMin, min))
            min = tempMin;
    }
    max = g.shfl(max, 0);
    min = g.shfl(min, 0);
    // printf("total_count:%d, max: %f, min: %f\n", total_count,(float)max, (float)min);

    float q_scale = ((1 << num_bits)) / ((float)max - (float)min);
    float q_zero_point = ((float)max - (float)min) / 2 + (float)min; // data-zero_point=centerized data

    // group_index = threadIdx.x + bid * group_size;
    id = threadIdx.x;
    while (id < total_count/8)
    {
        float2 q_data_int;
        int8_t *q_data_8 = reinterpret_cast<int8_t *>(&q_data_int);
        data = vals_cast[id];
        __half *data_h = reinterpret_cast<__half *>(&data);
        // int32_t data_f[8];
        q_data_8[0] = trunc(((float)data_h[0] - q_zero_point) * q_scale);
        q_data_8[1] = trunc(((float)data_h[1] - q_zero_point) * q_scale);
        q_data_8[2] = trunc(((float)data_h[2] - q_zero_point) * q_scale);
        q_data_8[3] = trunc(((float)data_h[3] - q_zero_point) * q_scale - 0.5);
        q_data_8[4] = trunc(((float)data_h[4] - q_zero_point) * q_scale - 0.5);
        q_data_8[5] = trunc(((float)data_h[5] - q_zero_point) * q_scale - 0.5);
        q_data_8[6] = trunc(((float)data_h[6] - q_zero_point) * q_scale - 0.5);
        q_data_8[7] = trunc(((float)data_h[7] - q_zero_point) * q_scale - 0.5);
        vals_int_cast[id] = q_data_int;
        id += (blockDim.x);
    }
    if (threadIdx.x == 0)
    {
        // q_scale_d[blockIdx.x] = 1 / q_scale;
        // q_zero_point_d[blockIdx.x] = q_zero_point;
        *q_scale_d = 1 / q_scale;
        *q_zero_point_d = q_zero_point;
    }
}




__global__ void dequantize_data_chunks(__half *vals,
                                int8_t *vals_int,
                                float *zero_point_stats,
                                float *scale_stats,
                                unsigned int total_count,
                                unsigned int numel_per_chunk)
{
    // printf("quantize_data_1 starts\n");
    // cg::thread_block b = cg::this_thread_block();
    // cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    // int gid = threadIdx.x >> 5;
    // int lane = threadIdx.x & 0x1f;
    // int warp_num = blockDim.x >> 5;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int quant_stats_idx;
    float data;
    while (id < total_count)
    {
        // printf("newton pointer left\n");
        quant_stats_idx = id / numel_per_chunk;
        data=vals_int[id];
        // printf("id:%d, quant_stats_idx:%d, total_count:%d, blockDim.x:%d\n",id,quant_stats_idx,total_count,blockDim.x);
        // data = vals_int[id];
        vals[id] = data*scale_stats[quant_stats_idx] + zero_point_stats[quant_stats_idx];
        id += stride;
    }
}

__global__ void dequantize_data(__half *vals,
                                int8_t *vals_int,
                                unsigned int total_count,
                                float *q_scale,
                                float *q_zero_point)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float data;
    // if (id == 0)
    // {
    // printf("id:%d, blockIdx.x:%d, blockDim.x:%d, threadIdx.x:%d, gridDim.x:%d\n",id,blockIdx.x,blockDim.x,threadIdx.x,gridDim.x);
    // printf("data:%f,q_scale:%f, q_zero_point:%f\n",(float)vals_int[id],*q_scale,*q_zero_point);
    // }

    while (id < total_count)
    {
        data=__half2float(vals_int[id]);
        // printf("id:%d, total_count:%d, data:%f (%d)\n",id,total_count,data,vals_int[id]);
        // data = vals_int[id];
        vals[id] = data*q_scale[0] + q_zero_point[0];
        id += stride;
    }
}



template <typename T>
void quantize_kernel(int8_t *vals_int,
                     const T *vals,
                     T *min,
                     T *max,
                     float *q_scale_d,
                     float *q_zero_point,
                     //  int groups,
                     int total_count,
                     int num_bits,
                     cudaStream_t stream);

template <>
void quantize_kernel<__half>(int8_t *vals_int,
                             const __half *vals,
                             __half *min,
                             __half *max,
                             float *q_scale_d,
                             float *q_zero_point,
                             //  int groups,
                             int total_count,
                             int num_bits,
                             cudaStream_t stream)
{
    // int groups = 1;
    // int threads = (total_count - 1) / (8 * groups) + 1;
    // int group_size = threads;
    // threads = 1024;
    dim3 grid_dim(1);
    dim3 block_dim(1024);
    // int data_size=(group_size/threads+1)*sizeof(float)*4;
    // printf("total_count is %d\n", total_count);
    quantize_data_simple<<<grid_dim, block_dim, 0, stream>>>(
        vals, vals_int, total_count, q_scale_d, q_zero_point, num_bits, 0);
}

template <typename T>
void quantize_kernel1(int8_t *vals_int,
                      const T *vals,
                      T *min,
                      T *max,
                      float *q_scale_d,
                      float *q_zero_point,
                      //  int groups,
                      int total_count,
                      int num_bits,
                      cudaStream_t stream);

template <>
void quantize_kernel1<__half>(int8_t *vals_int,
                              const __half *vals,
                              __half *min,
                              __half *max,
                              float *q_scale_d,
                              float *q_zero_point,
                              //  int groups,
                              int total_count,
                              int num_bits,
                              cudaStream_t stream)
{
    int groups = 1;
    int threads = (total_count - 1) / (8 * groups) + 1;
    int group_size = threads;
    threads = 1024;
    dim3 grid_dim(groups, (total_count - 1) / ((group_size < threads ? group_size : threads) * 8 * groups) + 1);
    dim3 block_dim(threads);
    min_max_local<<<grid_dim, block_dim, 0, stream>>>(vals, group_size, min, max);
    quantize_data<<<grid_dim, block_dim, 0, stream>>>(
        vals, vals_int, group_size, min, max, grid_dim.y, q_scale_d, q_zero_point, num_bits);
}



template <typename T>
void dequantize_kernel(int8_t *vals_int,
                      T *vals,
                      float *q_scale_d,
                      float *q_zero_point,
                      int total_count,
                      cudaStream_t stream);

template <>
void dequantize_kernel<__half>(int8_t *vals_int,
                              __half *vals,
                              float *q_scale_d,
                              float *q_zero_point,
                              int total_count,
                              cudaStream_t stream)
{
    // dim3 grid_dim((total_count - 1) / 1024  + 1);
    dim3 grid_dim(1);
    dim3 block_dim(1024);
    dequantize_data<<<grid_dim, block_dim, 0, stream>>>(vals, vals_int,total_count, q_scale_d, q_zero_point);
}



template <typename T>
void dequantize_chunks_kernel(int8_t *vals_int,
                      T *vals,
                     float* zero_point_stats,
                     float* scale_stats,
                      unsigned int total_count,
                      unsigned int numel_per_chunk,
                      cudaStream_t stream);

template <>
void dequantize_chunks_kernel<__half>(int8_t *vals_int,
                              __half *vals,
                     float* zero_point_stats,
                     float* scale_stats,
                              unsigned int total_count,
                              unsigned int numel_per_chunk,
                              cudaStream_t stream)
{
    // printf("dequantize_chunks_kernel start with total size %d\n", total_count);
    dim3 grid_dim((total_count - 1) / 1024/8  + 1);
    // dim3 grid_dim(1);
    // printf("dequantize_chunks_kernel grid_dim.x:%d, grid_dim.y:%d\n",grid_dim.x,grid_dim.y);
    dim3 block_dim(1024);
    dequantize_data_chunks<<<grid_dim, block_dim, 0, stream>>>(vals, vals_int, zero_point_stats,scale_stats, total_count, numel_per_chunk);
}
