#include <math.h>
#include "custom_cuda_layers.h"
#include "general_kernels.h"

namespace cg = cooperative_groups;

dim3 get_attn_softmax_grid(int batch_size, int heads, int sequence_length, int threads)
{
    int seq_length4 = sequence_length / 4;
    int block_compute_size =
        (seq_length4 < threads ? (int)pow(2.0, floor(log2((float)(threads / seq_length4)))) : 1);
    // Note that the Y and Z dimensions are limited to 65535, while X is basically unlimited:
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
    // The batch size is typically relatively small, while the sequence length could potentially be
    // arbitrarily large. We therefore place the batch size second to avoid hitting the Y limit.
    unsigned x = heads * sequence_length / block_compute_size;
    unsigned y = batch_size;
    return {x, y};
}

// Fused attention + softmax
template <int tbSize, int blockStride, int tbSeq>
__global__ void attn_softmax(float* vals,
                             const float* attn_mask,
                             int heads,
                             int seq_length,
                             int iterations)
{
    __shared__ float partialSum[MAX_WARP_NUM];

    int warp_num = blockDim.x >> WARP_SIZE_BITS;

    int iteration_stride = blockDim.x;
    int block_width = blockStride * seq_length;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    int batch = blockIdx.y;
    int row = blockIdx.x;
    int max_threads_in_sequence = std::max(seq_length, tbSeq);
    int seq_lane = threadIdx.x % max_threads_in_sequence;

    int data_offset = batch * (gridDim.x * block_width) + row * block_width +
                      (threadIdx.x / max_threads_in_sequence) * seq_length;
    int mask_offset = batch * seq_length;

    int wid = threadIdx.x >> WARP_SIZE_BITS;
    int lane = threadIdx.x & 0x1f;

    float4* val_cast = reinterpret_cast<float4*>(vals);
    const float4* attn_mask_cast = reinterpret_cast<const float4*>(attn_mask);

    float4 data[MAX_THREAD_ITERATIONS];

    float max_val = minus_infinity;

    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) {
            float4 mask = attn_mask_cast[mask_offset + data_id];
            data[i] = val_cast[data_offset + data_id];

            data[i].x += mask.x;
            data[i].y += mask.y;
            data[i].z += mask.z;
            data[i].w += mask.w;

            max_val = (data[i].x > max_val ? data[i].x : max_val);
            max_val = (data[i].y > max_val ? data[i].y : max_val);
            max_val = (data[i].z > max_val ? data[i].z : max_val);
            max_val = (data[i].w > max_val ? data[i].w : max_val);
        } else {
            data[i].x = minus_infinity;
            data[i].y = minus_infinity;
            data[i].z = minus_infinity;
            data[i].w = minus_infinity;
        }
    }

    for (int i = 1; i < tbSize; i *= 2) {
        auto temp = g.shfl_xor(max_val, i);
        max_val = (temp > max_val ? temp : max_val);
    }

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = max_val;
        b.sync();

        if (lane < warp_num) max_val = partialSum[lane];

#ifndef __STOCHASTIC_MODE__
        b.sync();
#endif

        int iters = warp_num;
        if (seq_length < iteration_stride)
            iters = warp_num / (iteration_stride / max_threads_in_sequence);

        for (int i = 1; i < iters; i *= 2) {
            auto temp = g.shfl_xor(max_val, i);
            max_val = (temp > max_val ? temp : max_val);
        }

        max_val = g.shfl(max_val, threadIdx.x / tbSize);
    }

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        data[i].x = __expf(data[i].x - max_val);
        data[i].y = __expf(data[i].y - max_val);
        data[i].z = __expf(data[i].z - max_val);
        data[i].w = __expf(data[i].w - max_val);

        sum += (data[i].x + data[i].y + data[i].z + data[i].w);
    }

    for (int i = 1; i < tbSize; i *= 2) { sum += g.shfl_xor(sum, i); }

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = sum;
        b.sync();

        if (lane < warp_num) sum = partialSum[lane];

#ifndef __STOCHASTIC_MODE__
        b.sync();
#endif

        int iters = warp_num;
        if (seq_length < iteration_stride)
            iters = warp_num / (iteration_stride / max_threads_in_sequence);

        for (int i = 1; i < iters; i *= 2) { sum += g.shfl_xor(sum, i); }

        sum = g.shfl(sum, threadIdx.x / tbSize);
    }

    sum += 1e-6;

    for (int i = 0; i < iterations; i++) {
        data[i].x /= sum;
        data[i].y /= sum;
        data[i].z /= sum;
        data[i].w /= sum;

        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) val_cast[data_offset + data_id] = data[i];
    }
}

template <int tbSize, int blockStride, int tbSeq>
__global__ void attn_softmax(__half* vals,
                             const __half* attn_mask,
                             int heads,
                             int seq_length,
                             int iterations)
{
#ifdef HALF_PRECISION_AVAILABLE
    __shared__ float partialSum[MAX_WARP_NUM];

    int warp_num = blockDim.x >> WARP_SIZE_BITS;

    int iteration_stride = blockDim.x;
    int block_width = blockStride * seq_length;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    int batch = blockIdx.y;
    int row = blockIdx.x;
    int max_threads_in_sequence = std::max(seq_length, tbSeq);
    int seq_lane = threadIdx.x % max_threads_in_sequence;

    int data_offset = batch * (gridDim.x * block_width) + row * block_width +
                      (threadIdx.x / max_threads_in_sequence) * seq_length;
    int mask_offset = batch * seq_length;

    int wid = threadIdx.x >> WARP_SIZE_BITS;
    int lane = threadIdx.x & 0x1f;

    float2* val_cast = reinterpret_cast<float2*>(vals);
    const float2* attn_mask_cast = reinterpret_cast<const float2*>(attn_mask);

    val_cast += data_offset;
    attn_mask_cast += mask_offset;

    float2 low_data[MAX_THREAD_ITERATIONS];
    float2 high_data[MAX_THREAD_ITERATIONS];

    float max_val = minus_infinity;

    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) {
            float2 data = val_cast[data_id];
            float2 mask = attn_mask_cast[data_id];

            __half2* data_arr = reinterpret_cast<__half2*>(&data);
            __half2* mask_arr = reinterpret_cast<__half2*>(&mask);

            low_data[i] = __half22float2(data_arr[0]);
            high_data[i] = __half22float2(data_arr[1]);
            float2 low_mask = __half22float2(mask_arr[0]);
            float2 high_mask = __half22float2(mask_arr[1]);

            low_data[i].x += low_mask.x;
            low_data[i].y += low_mask.y;
            high_data[i].x += high_mask.x;
            high_data[i].y += high_mask.y;

            max_val = (low_data[i].x > max_val ? low_data[i].x : max_val);
            max_val = (low_data[i].y > max_val ? low_data[i].y : max_val);
            max_val = (high_data[i].x > max_val ? high_data[i].x : max_val);
            max_val = (high_data[i].y > max_val ? high_data[i].y : max_val);
        }
    }

    for (int i = 1; i < tbSize; i *= 2) {
        auto temp = g.shfl_xor(max_val, i);
        max_val = (temp > max_val ? temp : max_val);
    }

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = max_val;
        b.sync();

        if (lane < warp_num) max_val = partialSum[lane];

#ifndef __STOCHASTIC_MODE__
        b.sync();
#endif

        int iters = warp_num;
        if (seq_length < iteration_stride)
            iters = warp_num / (iteration_stride / max_threads_in_sequence);

        for (int i = 1; i < iters; i *= 2) {
            auto temp = g.shfl_xor(max_val, i);
            max_val = (temp > max_val ? temp : max_val);
        }

        max_val = g.shfl(max_val, threadIdx.x / tbSize);
    }

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) {
            low_data[i].x = __expf(low_data[i].x - max_val);
            low_data[i].y = __expf(low_data[i].y - max_val);
            high_data[i].x = __expf(high_data[i].x - max_val);
            high_data[i].y = __expf(high_data[i].y - max_val);

            sum += (low_data[i].x + low_data[i].y + high_data[i].x + high_data[i].y);
        }
    }

    for (int i = 1; i < tbSize; i *= 2) { sum += g.shfl_xor(sum, i); }

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = sum;
        b.sync();

        if (lane < warp_num) sum = partialSum[lane];

#ifndef __STOCHASTIC_MODE__
        b.sync();
#endif

        int iters = warp_num;
        if (seq_length < iteration_stride)
            iters = warp_num / (iteration_stride / max_threads_in_sequence);

        for (int i = 1; i < iters; i *= 2) { sum += g.shfl_xor(sum, i); }

        sum = g.shfl(sum, threadIdx.x / tbSize);
    }

    sum += 1e-6;

    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) {
            float2 result_f;
            __half2* result_h = reinterpret_cast<__half2*>(&result_f);

            low_data[i].x /= sum;
            low_data[i].y /= sum;
            high_data[i].x /= sum;
            high_data[i].y /= sum;

            result_h[0] = __float22half2_rn(low_data[i]);
            result_h[1] = __float22half2_rn(high_data[i]);

            val_cast[data_id] = result_f;
        }
    }

#endif
}

template <typename T>
void launch_attn_softmax(T*, const T*, int, int, int, cudaStream_t);

template <>
void launch_attn_softmax<float>(float* vals,
                                const float* attn_mask,
                                int batch_size,
                                int heads,
                                int sequence_length,
                                cudaStream_t stream)
{
    const int threads = 128;
    int seq_length4 = sequence_length / 4;

    dim3 grid_dim = get_attn_softmax_grid(batch_size, heads, sequence_length, threads);

    int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;

    dim3 block_dim(seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) /
                                            subblock_max_workload * threads)
                                         : threads);
    int iterations =
        (sequence_length < subblock_max_workload ? (seq_length4 + threads - 1) / threads
                                                 : MAX_THREAD_ITERATIONS);

    if (sequence_length <= 8)
        attn_softmax<2, (threads / 2), 2>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 16)
        attn_softmax<4, (threads / 4), 4>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 32)
        attn_softmax<8, (threads / 8), 8>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 64)
        attn_softmax<16, (threads / 16), 16>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 128)
        attn_softmax<32, (threads / 32), 32>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 256)
        attn_softmax<32, (threads / 64), 64>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else {
        const int threads = 256;
        dim3 grid_dim = get_attn_softmax_grid(batch_size, heads, sequence_length, threads);

        int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;

        dim3 block_dim(seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) /
                                                subblock_max_workload * threads)
                                             : threads);
        iterations =
            (sequence_length < subblock_max_workload ? (seq_length4 + threads - 1) / threads
                                                     : MAX_THREAD_ITERATIONS);
        if (sequence_length <= 512)
            attn_softmax<32, (threads / 128), 128><<<grid_dim, block_dim, 0, stream>>>(
                vals, attn_mask, heads, seq_length4, iterations);
        else if (sequence_length < (MAX_THREADS * MAX_THREAD_ITERATIONS * 4))
            attn_softmax<32, 1, 128><<<grid_dim, block_dim, 0, stream>>>(
                vals, attn_mask, heads, seq_length4, iterations);
        else
            throw std::runtime_error(
                "Unsupport Seq_Length! Check the restriction of the max_threads and "
                "max_thread_iterations!");
    }
}

template <>
void launch_attn_softmax<__half>(__half* vals,
                                 const __half* attn_mask,
                                 int batch_size,
                                 int heads,
                                 int sequence_length,
                                 cudaStream_t stream)
{
    const int threads = 128;
    int seq_length4 = sequence_length / 4;

    dim3 grid_dim = get_attn_softmax_grid(batch_size, heads, sequence_length, threads);

    int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;

    dim3 block_dim(seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) /
                                            subblock_max_workload * threads)
                                         : threads);

    int iterations =
        (sequence_length < subblock_max_workload ? (seq_length4 + threads - 1) / threads
                                                 : MAX_THREAD_ITERATIONS);

    if (sequence_length <= 8)
        attn_softmax<2, (threads / 2), 2>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 16)
        attn_softmax<4, (threads / 4), 4>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 32)
        attn_softmax<8, (threads / 8), 8>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 64)
        attn_softmax<16, (threads / 16), 16>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 128)
        attn_softmax<32, (threads / 32), 32>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 256)
        attn_softmax<32, (threads / 64), 64>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else {
        const int threads = 256;
        dim3 grid_dim = get_attn_softmax_grid(batch_size, heads, sequence_length, threads);

        int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;

        dim3 block_dim(seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) /
                                                subblock_max_workload * threads)
                                             : threads);
        iterations =
            (sequence_length < subblock_max_workload ? (seq_length4 + threads - 1) / threads
                                                     : MAX_THREAD_ITERATIONS);
        if (sequence_length <= 512)
            attn_softmax<32, (threads / 128), 128><<<grid_dim, block_dim, 0, stream>>>(
                vals, attn_mask, heads, seq_length4, iterations);
        else if (sequence_length < (MAX_THREADS * MAX_THREAD_ITERATIONS * 4))
            attn_softmax<32, 1, 128><<<grid_dim, block_dim, 0, stream>>>(
                vals, attn_mask, heads, seq_length4, iterations);
        else
            throw std::runtime_error(
                "Unsupport Seq_Length! Check the restriction of the max_threads and "
                "max_thread_iterations!");
    }
}

template <typename T, int tbSize, int blockStride>
__global__ void softmax_backward_kernel(T* out_grad, const T* soft_inp, int seq_length)
{
    __shared__ float partialSum[MAX_WARP_NUM];

    int warp_num = blockDim.x >> WARP_SIZE_BITS;  // warp-count = num_threads / WARP_SIZE (32)

    int iteration_stride = blockDim.x;
    int block_width = blockStride * seq_length;

    int iterations = (seq_length < (MAX_THREAD_ITERATIONS * iteration_stride)
                          ? (seq_length + iteration_stride - 1) / iteration_stride
                          : MAX_THREAD_ITERATIONS);

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;

    int wid = id >> WARP_SIZE_BITS;
    int lane = id & 0x1f;

    T val_reg[MAX_THREAD_ITERATIONS];
    T soft_reg[MAX_THREAD_ITERATIONS];
    float grad_reg = 0.0f;

#pragma unroll
    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + id;
        if (data_id < block_width) {
            val_reg[i] = out_grad[row * block_width + data_id];
            soft_reg[i] = soft_inp[row * block_width + data_id];

            grad_reg += ((float)val_reg[i] *
                         (float)soft_reg[i]);  // if done in half, the multiplication, we may lose
                                               // 2% of accuracy in computation!!
        }
    }
    for (int i = 1; i < tbSize; i *= 2) grad_reg += g.shfl_xor(grad_reg, i);

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = grad_reg;
        b.sync();

        if (lane < warp_num) grad_reg = partialSum[lane];

        int iters = warp_num;
        if (seq_length < iteration_stride) iters = warp_num / (iteration_stride / seq_length);

        for (int i = 1; i < iters; i *= 2) grad_reg += g.shfl_xor(grad_reg, i);

        grad_reg = g.shfl(grad_reg, id / tbSize);
    }

    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + id;
        if (data_id < block_width) {
            float temp = (float)soft_reg[i] * ((float)val_reg[i] - grad_reg);
            out_grad[row * block_width + data_id] = (T)temp;
        }
    }
}

template <typename T, int ITERATIONS>
__global__ void softmax_backward_kernel_v2(T* grad /* input & output*/,
                                           const T* output,
                                           int softmax_length)
{
    int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int offset = batch_idx * softmax_length + threadIdx.x;

    grad += offset;
    output += offset;

    T grad_reg[ITERATIONS];
    T output_reg[ITERATIONS];
    float sum = 0.0;

#pragma unroll
    for (int i = 0; i < ITERATIONS; ++i) {
        int curr_idx = threadIdx.x + i * WARP_SIZE;
        if (curr_idx < softmax_length) {
            grad_reg[i] = grad[i * WARP_SIZE];
            output_reg[i] = output[i * WARP_SIZE];
            sum += (float)grad_reg[i] * (float)output_reg[i];
        }
    }

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    for (int i = 1; i < WARP_SIZE; i <<= 1) sum += g.shfl_xor(sum, i);

#pragma unroll
    for (int i = 0; i < ITERATIONS; ++i) {
        int curr_idx = threadIdx.x + i * WARP_SIZE;
        if (curr_idx < softmax_length)
            grad[i * WARP_SIZE] = (float)output_reg[i] * ((float)grad_reg[i] - sum);
    }
}

template <typename T>
void launch_attn_softmax_backward_v2(T* out_grad,
                                     const T* soft_inp,
                                     int batch_size,
                                     int heads,
                                     int seq_length,
                                     cudaStream_t stream)
{
    const int warps_per_block = 4;
    dim3 grid_dim(batch_size * heads * seq_length / warps_per_block);
    dim3 block_dim(WARP_SIZE, warps_per_block);

    if (seq_length <= 32)
        softmax_backward_kernel_v2<T, 1>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else if (seq_length <= 64)
        softmax_backward_kernel_v2<T, 2>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else if (seq_length <= 128)
        softmax_backward_kernel_v2<T, 4>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else if (seq_length <= 256)
        softmax_backward_kernel_v2<T, 8>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else if (seq_length <= 384)
        softmax_backward_kernel_v2<T, 12>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else if (seq_length <= 512)
        softmax_backward_kernel_v2<T, 16>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else if (seq_length <= 768)
        softmax_backward_kernel_v2<T, 24>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else if (seq_length <= 1024)
        softmax_backward_kernel_v2<T, 32>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else if (seq_length <= 2048)
        softmax_backward_kernel_v2<T, 64>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else
        throw std::runtime_error(
            std::string("Special sequence length found in softmax backward, seq_length: ") +
            std::to_string(seq_length));
}

template void launch_attn_softmax_backward_v2<__half>(__half* out_grad,
                                                      const __half* soft_inp,
                                                      int batch_size,
                                                      int heads,
                                                      int seq_length,
                                                      cudaStream_t stream);
template void launch_attn_softmax_backward_v2<float>(float* out_grad,
                                                     const float* soft_inp,
                                                     int batch_size,
                                                     int heads,
                                                     int seq_length,
                                                     cudaStream_t stream);

template <int tbSize>
__global__ void softmax_dropout_kernel(const int N,
                                       const float ratio,
                                       float* out,
                                       float* Xdata,
                                       const bool* attn_mask,
                                       const float* rel_pos,
                                       std::pair<uint64_t, uint64_t> seed,
                                       int tb_blocks)
{
    // TODO: add the implementation for float
}

template <int tbSize>
__global__ void softmax_dropout_kernel(const int seq_length,
                                       const float ratio,
                                       __nv_bfloat16* out,
                                       __nv_bfloat16* Xdata,
                                       const bool* attn_mask,
                                       const __nv_bfloat16* rel_pos,
                                       std::pair<uint64_t, uint64_t> seed,
                                       int tb_blocks)
{
    unsigned warp_num = blockDim.x >> 5;

    unsigned iteration_stride = WARP_SIZE;
    unsigned block_width = warp_num * seq_length;

    unsigned iterations = (seq_length - 1) / iteration_stride + 1;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    unsigned wid = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 0x1f;

    unsigned batch = blockIdx.x;
    unsigned row = blockIdx.y;

    unsigned data_offset = batch * (gridDim.y * block_width) + row * block_width +
                           (wid * tb_blocks + lane / tbSize) * seq_length;
    unsigned mask_offset = batch * seq_length;

    float4* out_cast = reinterpret_cast<float4*>(out);
    float4* val_cast = reinterpret_cast<float4*>(Xdata);
    const float4* attn_mask_cast;
    if (attn_mask) attn_mask_cast = reinterpret_cast<const float4*>(attn_mask);

    val_cast += data_offset;
    out_cast += data_offset;
    if (attn_mask) attn_mask_cast += mask_offset;

    float2 low_data[MAX_THREAD_ITERATIONS];
    float2 high_data[MAX_THREAD_ITERATIONS];
    float2 low_data1[MAX_THREAD_ITERATIONS];
    float2 high_data1[MAX_THREAD_ITERATIONS];

    const float scale = 1. / (1. - ratio);
    __nv_bfloat162 h_scale = __float2bfloat162_rn(scale);
    __nv_bfloat16 h_zero = __float2bfloat16(0.0);

    float max_val = minus_infinity;

    for (int i = 0; i < iterations; i++) {
        unsigned data_id = i * iteration_stride + lane;
        if (data_id < seq_length) {
            float4 data = val_cast[data_id];
            __nv_bfloat162* data_arr = reinterpret_cast<__nv_bfloat162*>(&data);
            low_data[i] = __bfloat1622float2(data_arr[0]);
            high_data[i] = __bfloat1622float2(data_arr[1]);
            low_data1[i] = __bfloat1622float2(data_arr[2]);
            high_data1[i] = __bfloat1622float2(data_arr[3]);
        } else {
            low_data[i] = {minus_infinity, minus_infinity};
            high_data[i] = {minus_infinity, minus_infinity};
            low_data1[i] = {minus_infinity, minus_infinity};
            high_data1[i] = {minus_infinity, minus_infinity};
        }
    }

    for (int i = 0; i < iterations; i++) {
        float maxes[4];
        maxes[0] = (low_data[i].x > low_data[i].y ? low_data[i].x : low_data[i].y);
        maxes[1] = (high_data[i].x > high_data[i].y ? high_data[i].x : high_data[i].y);
        maxes[2] = (low_data1[i].x > low_data1[i].y ? low_data1[i].x : low_data1[i].y);
        maxes[3] = (high_data1[i].x > high_data1[i].y ? high_data1[i].x : high_data1[i].y);
        maxes[0] = (maxes[0] > maxes[1] ? maxes[0] : maxes[1]);
        maxes[2] = (maxes[2] > maxes[3] ? maxes[2] : maxes[3]);
        max_val = (maxes[0] > max_val ? maxes[0] : max_val);
        max_val = (maxes[2] > max_val ? maxes[2] : max_val);
    }
    for (int i = 1; i < tbSize; i *= 2) {
        auto temp = g.shfl_xor(max_val, i);
        max_val = (temp > max_val ? temp : max_val);
    }

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        low_data[i].x = __expf(low_data[i].x - max_val);
        low_data[i].y = __expf(low_data[i].y - max_val);
        high_data[i].x = __expf(high_data[i].x - max_val);
        high_data[i].y = __expf(high_data[i].y - max_val);
        low_data1[i].x = __expf(low_data1[i].x - max_val);
        low_data1[i].y = __expf(low_data1[i].y - max_val);
        high_data1[i].x = __expf(high_data1[i].x - max_val);
        high_data1[i].y = __expf(high_data1[i].y - max_val);

        sum += (low_data[i].x + low_data[i].y + high_data[i].x + high_data[i].y);
        sum += (low_data1[i].x + low_data1[i].y + high_data1[i].x + high_data1[i].y);
    }

    for (int i = 1; i < tbSize; i *= 2) { sum += g.shfl_xor(sum, i); }
    sum += 1e-6;

    int idx = (((blockIdx.x * gridDim.y + blockIdx.y) * warp_num) + wid) * (seq_length << 3) +
              (lane << 3);

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

    for (int i = 0; i < iterations; i++) {
        unsigned data_id = i * iteration_stride + lane;
        if (data_id < seq_length) {
            low_data[i].x /= sum;
            low_data[i].y /= sum;
            high_data[i].x /= sum;
            high_data[i].y /= sum;
            low_data1[i].x /= sum;
            low_data1[i].y /= sum;
            high_data1[i].x /= sum;
            high_data1[i].y /= sum;

            float4 result_f;
            __nv_bfloat162* result_h = reinterpret_cast<__nv_bfloat162*>(&result_f);

            result_h[0] = __float22bfloat162_rn(low_data[i]);
            result_h[1] = __float22bfloat162_rn(high_data[i]);
            result_h[2] = __float22bfloat162_rn(low_data1[i]);
            result_h[3] = __float22bfloat162_rn(high_data1[i]);

            float4 rand = curand_uniform4(&state);
            float4 rand1 = curand_uniform4(&state);

            result_h[0].x = (rand.x > ratio) ? result_h[0].x : h_zero;
            result_h[0].y = (rand.y > ratio) ? result_h[0].y : h_zero;
            result_h[1].x = (rand.z > ratio) ? result_h[1].x : h_zero;
            result_h[1].y = (rand.w > ratio) ? result_h[1].y : h_zero;
            result_h[2].x = (rand1.x > ratio) ? result_h[2].x : h_zero;
            result_h[2].y = (rand1.y > ratio) ? result_h[2].y : h_zero;
            result_h[3].x = (rand1.z > ratio) ? result_h[3].x : h_zero;
            result_h[3].y = (rand1.w > ratio) ? result_h[3].y : h_zero;

            result_h[0] = result_h[0] * h_scale;
            result_h[1] = result_h[1] * h_scale;
            result_h[2] = result_h[2] * h_scale;
            result_h[3] = result_h[3] * h_scale;

            out_cast[data_id] = result_f;
        }
    }
}
template <int tbSize>
__global__ void softmax_dropout_kernel(const int seq_length,
                                       const float ratio,
                                       __half* out,
                                       __half* Xdata,
                                       const bool* attn_mask,
                                       const __half* rel_pos,
                                       std::pair<uint64_t, uint64_t> seed,
                                       int tb_blocks)
{
    unsigned warp_num = blockDim.x >> 5;

    unsigned iteration_stride = tbSize;  // WARP_SIZE;
    unsigned block_width = warp_num * tb_blocks * seq_length;

    unsigned iterations = (seq_length - 1) / iteration_stride + 1;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    unsigned wid = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 0x1f;

    unsigned data_offset = (blockIdx.x * gridDim.y + blockIdx.y) * block_width +
                           (wid * tb_blocks + lane / tbSize) * seq_length;
    unsigned mask_offset = blockIdx.x * seq_length;

    float2* out_cast = reinterpret_cast<float2*>(out);
    const float2* rel_pos_cast = reinterpret_cast<const float2*>(rel_pos);
    float2* val_cast = reinterpret_cast<float2*>(Xdata);
    const float* attn_mask_cast;
    if (attn_mask) attn_mask_cast = reinterpret_cast<const float*>(attn_mask);

    val_cast += data_offset;
    rel_pos_cast += data_offset;
    out_cast += data_offset;
    attn_mask_cast += mask_offset;

    float2 low_data[MAX_THREAD_ITERATIONS];
    float2 high_data[MAX_THREAD_ITERATIONS];

    const float scale = 1. / (1. - ratio);
    __half2 h_scale = __float2half2_rn(scale);
    __half h_zero = __float2half(0.0);
    __half h_inf = __float2half(-10000.0);

    float max_val = minus_infinity;

    for (int i = 0; i < iterations; i++) {
        unsigned data_id = i * iteration_stride + (lane % tbSize);
        if (data_id < seq_length) {
            float2 data = val_cast[data_id];
            float2 rel_pos_data = rel_pos_cast[data_id];
            __half2* data_arr = reinterpret_cast<__half2*>(&data);
            __half2* rel_pos_arr = reinterpret_cast<__half2*>(&rel_pos_data);
            data_arr[0] = data_arr[0] + rel_pos_arr[0];
            data_arr[1] = data_arr[1] + rel_pos_arr[1];
            if(attn_mask){
                float mask_data = attn_mask_cast[data_id];
                bool* mask_bool = reinterpret_cast<bool*>(&mask_data);
                data_arr[0].x = mask_bool[0] ? data_arr[0].x : h_inf;
                data_arr[0].y = mask_bool[1] ? data_arr[0].y : h_inf;
                data_arr[1].x = mask_bool[2] ? data_arr[1].x : h_inf;
                data_arr[1].y = mask_bool[3] ? data_arr[1].y : h_inf;
            }
            val_cast[data_id] = data;

            low_data[i] = __half22float2(data_arr[0]);
            high_data[i] = __half22float2(data_arr[1]);
            float maxes[2];
            maxes[0] = (low_data[i].x > low_data[i].y ? low_data[i].x : low_data[i].y);
            maxes[1] = (high_data[i].x > high_data[i].y ? high_data[i].x : high_data[i].y);
            maxes[0] = (maxes[0] > maxes[1] ? maxes[0] : maxes[1]);
            max_val = (maxes[0] > max_val ? maxes[0] : max_val);
        } else {
            low_data[i] = {minus_infinity, minus_infinity};
            high_data[i] = {minus_infinity, minus_infinity};
        }
    }

    for (int i = 1; i < tbSize; i *= 2) {
        auto temp = g.shfl_xor(max_val, i);
        max_val = (temp > max_val ? temp : max_val);
    }

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        low_data[i].x = __expf(low_data[i].x - max_val);
        low_data[i].y = __expf(low_data[i].y - max_val);
        high_data[i].x = __expf(high_data[i].x - max_val);
        high_data[i].y = __expf(high_data[i].y - max_val);
        sum += (low_data[i].x + low_data[i].y + high_data[i].x + high_data[i].y);
    }

    for (int i = 1; i < tbSize; i *= 2) { sum += g.shfl_xor(sum, i); }
    sum += 1e-6;

    int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

    for (int i = 0; i < iterations; i++) {
        unsigned data_id = i * iteration_stride + (lane % tbSize);
        if (data_id < seq_length) {
            low_data[i].x /= sum;
            low_data[i].y /= sum;
            high_data[i].x /= sum;
            high_data[i].y /= sum;

            float2 result_f;
            __half2* result_h = reinterpret_cast<__half2*>(&result_f);
            result_h[0] = __float22half2_rn(low_data[i]);
            result_h[1] = __float22half2_rn(high_data[i]);

            float4 rand = curand_uniform4(&state);

            result_h[0].x = (rand.x > ratio) ? result_h[0].x : h_zero;
            result_h[0].y = (rand.y > ratio) ? result_h[0].y : h_zero;
            result_h[1].x = (rand.z > ratio) ? result_h[1].x : h_zero;
            result_h[1].y = (rand.w > ratio) ? result_h[1].y : h_zero;

            result_h[0] = result_h[0] * h_scale;
            result_h[1] = result_h[1] * h_scale;

            out_cast[data_id] = result_f;
        }
    }
}

template <typename T>
void launch_softmax_dropout(T* out,
                            T* vals,
                            const bool* attn_mask,
                            const T* rel_pos,
                            int bsz,
                            int heads,
                            int seq_length,
                            int softmax_length,
                            float ratio,
                            cudaStream_t stream)
{
    int threads = 256;
    int stride = 4;
    int warp_num = threads / WARP_SIZE;
    int pow_soft = (int)pow(2.0, ceil(log2((float)(softmax_length / stride))));
    dim3 grid_dim((bsz / heads),
                  (heads * seq_length) / warp_num / ((WARP_SIZE - 1) / pow_soft + 1));
    dim3 block_dim(threads);
    int total_count = bsz * heads * seq_length * softmax_length;

    uint64_t inc = (total_count - 1) / ((grid_dim.x * grid_dim.y) * threads) + 1;
    std::pair<uint64_t, uint64_t> seed = Context::Instance().IncrementOffset(inc << 2, true);
    if (softmax_length <= (stride * 2))
        softmax_dropout_kernel<2><<<grid_dim, block_dim, 0, stream>>>(
            softmax_length / stride, ratio, out, vals, attn_mask, rel_pos, seed, WARP_SIZE / 2);
    else if (softmax_length <= (stride * 4))
        softmax_dropout_kernel<4><<<grid_dim, block_dim, 0, stream>>>(
            softmax_length / stride, ratio, out, vals, attn_mask, rel_pos, seed, WARP_SIZE / 4);
    else if (softmax_length <= (stride * 8))
        softmax_dropout_kernel<8><<<grid_dim, block_dim, 0, stream>>>(
            softmax_length / stride, ratio, out, vals, attn_mask, rel_pos, seed, WARP_SIZE / 8);
    else if (softmax_length <= (stride * 16))
        softmax_dropout_kernel<16><<<grid_dim, block_dim, 0, stream>>>(
            softmax_length / stride, ratio, out, vals, attn_mask, rel_pos, seed, WARP_SIZE / 16);
    else
        softmax_dropout_kernel<32><<<grid_dim, block_dim, 0, stream>>>(
            softmax_length / stride, ratio, out, vals, attn_mask, rel_pos, seed, 1);
}

template void launch_softmax_dropout(float* out,
                                     float* vals,
                                     const bool* attn_mask,
                                     const float* rel_pos,
                                     int bsz,
                                     int heads,
                                     int seq_length,
                                     int softmax_length,
                                     float ratio,
                                     cudaStream_t stream);
template void launch_softmax_dropout(__nv_bfloat16* out,
                                     __nv_bfloat16* vals,
                                     const bool* attn_mask,
                                     const __nv_bfloat16* rel_pos,
                                     int bsz,
                                     int heads,
                                     int seq_length,
                                     int softmax_length,
                                     float ratio,
                                     cudaStream_t stream);
template void launch_softmax_dropout(__half* out,
                                     __half* vals,
                                     const bool* attn_mask,
                                     const __half* rel_pos,
                                     int bsz,
                                     int heads,
                                     int seq_length,
                                     int softmax_length,
                                     float ratio,
                                     cudaStream_t stream);

__global__ void dropout_softmax_grad_kernel(const int N,
                                            const float scale,
                                            float* Xdata,
                                            const float* input,
                                            std::pair<uint64_t, uint64_t> seed)
{
    // TODO: Add float version
}

__global__ void dropout_softmax_grad_kernel(const int seq_length,
                                            const float ratio,
                                            __half* Xdata,
                                            const __half* input,
                                            std::pair<uint64_t, uint64_t> seed)
{
    unsigned warp_num = blockDim.x >> 5;

    unsigned iteration_stride = WARP_SIZE;
    unsigned block_width = warp_num * seq_length;

    unsigned iterations = (seq_length + iteration_stride - 1) / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    unsigned wid = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 0x1f;

    unsigned data_offset =
        blockIdx.x * (gridDim.y * block_width) + blockIdx.y * block_width + wid * seq_length;

    const float2* val_cast = reinterpret_cast<const float2*>(input);
    float2* grad_cast = reinterpret_cast<float2*>(Xdata);

    val_cast += data_offset;
    grad_cast += data_offset;

    float2 low_data[MAX_THREAD_ITERATIONS];
    float2 high_data[MAX_THREAD_ITERATIONS];
    float2 low_grad_data[MAX_THREAD_ITERATIONS];
    float2 high_grad_data[MAX_THREAD_ITERATIONS];

    const float scale = 1. / (1. - ratio);
    __half2 h_scale = __float2half2_rn(scale);
    __half h_zero = __float2half(0.0);

    int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

    // float minus_infinity = -1 * std::numeric_limits<float>::infinity();
    float max_val = minus_infinity;

    for (int i = 0; i < iterations; i++) {
        unsigned data_id = i * iteration_stride + lane;
        if (data_id < seq_length) {
            float2 data = val_cast[data_id];

            __half2* data_arr = reinterpret_cast<__half2*>(&data);

            low_data[i] = __half22float2(data_arr[0]);
            high_data[i] = __half22float2(data_arr[1]);

            max_val = (low_data[i].x > max_val ? low_data[i].x : max_val);
            max_val = (low_data[i].y > max_val ? low_data[i].y : max_val);
            max_val = (high_data[i].x > max_val ? high_data[i].x : max_val);
            max_val = (high_data[i].y > max_val ? high_data[i].y : max_val);
        } else {
            low_data[i].x = minus_infinity;
            low_data[i].y = minus_infinity;
            high_data[i].x = minus_infinity;
            high_data[i].y = minus_infinity;
        }
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) {
        auto temp = g.shfl_xor(max_val, i);
        max_val = (temp > max_val ? temp : max_val);
    }

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        low_data[i].x = __expf(low_data[i].x - max_val);
        low_data[i].y = __expf(low_data[i].y - max_val);
        high_data[i].x = __expf(high_data[i].x - max_val);
        high_data[i].y = __expf(high_data[i].y - max_val);

        sum += (low_data[i].x + low_data[i].y + high_data[i].x + high_data[i].y);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_xor(sum, i); }
    sum += 1e-6;

    float sum1 = 0;

#pragma unroll
    for (int i = 0; i < iterations; ++i) {
        unsigned data_id = i * iteration_stride + lane;
        if (data_id < seq_length) {
            float2 grad = grad_cast[data_id];

            low_data[i].x /= sum;
            low_data[i].y /= sum;
            high_data[i].x /= sum;
            high_data[i].y /= sum;

            float4 rand = curand_uniform4(&state);

            __half2* grad_arr = reinterpret_cast<__half2*>(&grad);
            float2 low_grad = __half22float2(grad_arr[0]);
            float2 high_grad = __half22float2(grad_arr[1]);

            low_grad_data[i].x = ((rand.x > ratio) ? low_grad.x * scale : 0);
            low_grad_data[i].y = ((rand.y > ratio) ? low_grad.y * scale : 0);
            high_grad_data[i].x = ((rand.z > ratio) ? high_grad.x * scale : 0);
            high_grad_data[i].y = ((rand.w > ratio) ? high_grad.y * scale : 0);

            low_grad.x = low_data[i].x * low_grad_data[i].x;
            low_grad.y = low_data[i].y * low_grad_data[i].y;
            high_grad.x = high_data[i].x * high_grad_data[i].x;
            high_grad.y = high_data[i].y * high_grad_data[i].y;

            sum1 += (low_grad.x + low_grad.y + high_grad.x + high_grad.y);
        }
    }

    for (int i = 1; i < WARP_SIZE; i <<= 1) sum1 += g.shfl_xor(sum1, i);

#pragma unroll
    for (int i = 0; i < iterations; ++i) {
        unsigned data_id = i * iteration_stride + lane;
        float2 result_f;
        __half2* result_h = reinterpret_cast<__half2*>(&result_f);
        result_h[0].x = low_data[i].x * (low_grad_data[i].x - sum1);
        result_h[0].y = low_data[i].y * (low_grad_data[i].y - sum1);
        result_h[1].x = high_data[i].x * (high_grad_data[i].x - sum1);
        result_h[1].y = high_data[i].y * (high_grad_data[i].y - sum1);

        if (data_id < seq_length) grad_cast[data_id] = result_f;
    }
}

__global__ void dropout_softmax_grad_kernel(const int seq_length,
                                            const float ratio,
                                            __nv_bfloat16* Xdata,
                                            const __nv_bfloat16* input,
                                            std::pair<uint64_t, uint64_t> seed)
{
    unsigned warp_num = blockDim.x >> 5;

    unsigned iteration_stride = WARP_SIZE;
    unsigned block_width = warp_num * seq_length;

    unsigned iterations = (seq_length + iteration_stride - 1) / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    unsigned wid = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 0x1f;

    unsigned batch = blockIdx.x;
    unsigned row = blockIdx.y;

    unsigned data_offset =
        blockIdx.x * (gridDim.y * block_width) + row * block_width + wid * seq_length;
    unsigned mask_offset = blockIdx.x * seq_length;

    const float2* val_cast = reinterpret_cast<const float2*>(input);
    float2* grad_cast = reinterpret_cast<float2*>(Xdata);

    val_cast += data_offset;
    grad_cast += data_offset;

    float2 low_data[MAX_THREAD_ITERATIONS];
    float2 high_data[MAX_THREAD_ITERATIONS];
    float2 low_grad_data[MAX_THREAD_ITERATIONS];
    float2 high_grad_data[MAX_THREAD_ITERATIONS];

    const float scale = 1. / (1. - ratio);
    __nv_bfloat162 h_scale = __float2bfloat162_rn(scale);
    __nv_bfloat16 h_zero = __float2bfloat16(0.0);

    int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

    // float minus_infinity = -1 * std::numeric_limits<float>::infinity();
    float max_val = minus_infinity;

    for (int i = 0; i < iterations; i++) {
        unsigned data_id = i * iteration_stride + lane;
        if (data_id < seq_length) {
            float2 data = val_cast[data_id];

            __nv_bfloat162* data_arr = reinterpret_cast<__nv_bfloat162*>(&data);

            low_data[i] = __bfloat1622float2(data_arr[0]);
            high_data[i] = __bfloat1622float2(data_arr[1]);

            max_val = (low_data[i].x > max_val ? low_data[i].x : max_val);
            max_val = (low_data[i].y > max_val ? low_data[i].y : max_val);
            max_val = (high_data[i].x > max_val ? high_data[i].x : max_val);
            max_val = (high_data[i].y > max_val ? high_data[i].y : max_val);
        } else {
            low_data[i].x = minus_infinity;
            low_data[i].y = minus_infinity;
            high_data[i].x = minus_infinity;
            high_data[i].y = minus_infinity;
        }
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) {
        auto temp = g.shfl_xor(max_val, i);
        max_val = (temp > max_val ? temp : max_val);
    }

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        low_data[i].x = __expf(low_data[i].x - max_val);
        low_data[i].y = __expf(low_data[i].y - max_val);
        high_data[i].x = __expf(high_data[i].x - max_val);
        high_data[i].y = __expf(high_data[i].y - max_val);

        sum += (low_data[i].x + low_data[i].y + high_data[i].x + high_data[i].y);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_xor(sum, i); }
    sum += 1e-6;

    float sum1 = 0;

#pragma unroll
    for (int i = 0; i < iterations; ++i) {
        unsigned data_id = i * iteration_stride + lane;
        if (data_id < seq_length) {
            float2 grad = grad_cast[data_id];

            low_data[i].x /= sum;
            low_data[i].y /= sum;
            high_data[i].x /= sum;
            high_data[i].y /= sum;

            float4 rand = curand_uniform4(&state);

            __nv_bfloat162* grad_arr = reinterpret_cast<__nv_bfloat162*>(&grad);
            float2 low_grad = __bfloat1622float2(grad_arr[0]);
            float2 high_grad = __bfloat1622float2(grad_arr[1]);

            low_grad_data[i].x = ((rand.x > ratio) ? low_grad.x * scale : 0);
            low_grad_data[i].y = ((rand.y > ratio) ? low_grad.y * scale : 0);
            high_grad_data[i].x = ((rand.z > ratio) ? high_grad.x * scale : 0);
            high_grad_data[i].y = ((rand.w > ratio) ? high_grad.y * scale : 0);

            low_grad.x = low_data[i].x * low_grad_data[i].x;
            low_grad.y = low_data[i].y * low_grad_data[i].y;
            high_grad.x = high_data[i].x * high_grad_data[i].x;
            high_grad.y = high_data[i].y * high_grad_data[i].y;

            sum1 += (low_grad.x + low_grad.y + high_grad.x + high_grad.y);
        }
    }

    for (int i = 1; i < WARP_SIZE; i <<= 1) sum1 += g.shfl_xor(sum1, i);

#pragma unroll
    for (int i = 0; i < iterations; ++i) {
        unsigned data_id = i * iteration_stride + lane;
        float2 result_f;
        __nv_bfloat162* result_h = reinterpret_cast<__nv_bfloat162*>(&result_f);
        result_h[0].x = __float2bfloat16(low_data[i].x * (low_grad_data[i].x - sum1));
        result_h[0].y = __float2bfloat16(low_data[i].y * (low_grad_data[i].y - sum1));
        result_h[1].x = __float2bfloat16(high_data[i].x * (high_grad_data[i].x - sum1));
        result_h[1].y = __float2bfloat16(high_data[i].y * (high_grad_data[i].y - sum1));
        if (data_id < seq_length) grad_cast[data_id] = result_f;
    }
}

template <typename T>
void launch_softmax_dropout_grad(T* vals,
                                 const T* input,
                                 int bsz,
                                 int heads,
                                 int seq_length,
                                 int softmax_length,
                                 float ratio,
                                 cudaStream_t stream)
{
    dim3 grid_dim(bsz, (heads * seq_length) / 8);
    dim3 block_dim(256);

    auto seed = Context::Instance().RestoreOffset();
    dropout_softmax_grad_kernel<<<grid_dim, block_dim, 0, stream>>>(
        softmax_length / 4, ratio, vals, input, seed);
}

template void launch_softmax_dropout_grad(float* vals,
                                          const float* input,
                                          int bsz,
                                          int heads,
                                          int seq_length,
                                          int softmax_length,
                                          float ratio,
                                          cudaStream_t stream);
template void launch_softmax_dropout_grad(__nv_bfloat16* vals,
                                          const __nv_bfloat16* input,
                                          int bsz,
                                          int heads,
                                          int seq_length,
                                          int softmax_length,
                                          float ratio,
                                          cudaStream_t stream);
template void launch_softmax_dropout_grad(__half* vals,
                                          const __half* input,
                                          int bsz,
                                          int heads,
                                          int seq_length,
                                          int softmax_length,
                                          float ratio,
                                          cudaStream_t stream);
