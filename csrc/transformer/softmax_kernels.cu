/*#include "custom_cuda_layers.h"
#include "general_kernels.h"

namespace cg = cooperative_groups;

// Fused attention + softmax
template <int tbSize, int blockStride, int tbSeq>
__global__ void attn_softmax(float* vals,
                             const float* attn_mask,
                             int heads,
                             int seq_length,
                             int iterations)
{
    __shared__ float partialSum[MAX_WARP_NUM];

    int warp_num = blockDim.x >> 5;

    int iteration_stride = blockDim.x;
    int block_width = blockStride * seq_length;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    int batch = blockIdx.x;
    int row = blockIdx.y;
    int max_threads_in_sequence = std::max(seq_length, tbSeq);
    int seq_lane = threadIdx.x % max_threads_in_sequence;

    int data_offset = batch * (gridDim.y * block_width) + row * block_width +
                      (threadIdx.x / max_threads_in_sequence) * seq_length;
    int mask_offset = batch * seq_length;

    int wid = threadIdx.x >> 5;
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
        if (seq_length < iteration_stride) iters = warp_num / (iteration_stride / seq_length);

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
        if (seq_length < iteration_stride) iters = warp_num / (iteration_stride / seq_length);

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
#if __CUDA_ARCH__ >= 700
    __shared__ float partialSum[MAX_WARP_NUM];

    int warp_num = blockDim.x >> 5;

    int iteration_stride = blockDim.x;
    int block_width = blockStride * seq_length;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    int batch = blockIdx.x;
    int row = blockIdx.y;
    int max_threads_in_sequence = std::max(seq_length, tbSeq);
    int seq_lane = threadIdx.x % max_threads_in_sequence;

    int data_offset = batch * (gridDim.y * block_width) + row * block_width +
                      (threadIdx.x / max_threads_in_sequence) * seq_length;
    int mask_offset = batch * seq_length;

    int wid = threadIdx.x >> 5;
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
        if (seq_length < iteration_stride) iters = warp_num / (iteration_stride / seq_length);

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
        if (seq_length < iteration_stride) iters = warp_num / (iteration_stride / seq_length);

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
void launch_attn_softmax(T*, const T*, int, int, int, cudaStream_t, bool);

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
    int seq2 = sequence_length * seq_length4;

    int block_compute_size =
        (seq_length4 < threads ? ((threads / seq_length4) * seq_length4) : seq_length4);
    dim3 grid_dim(batch_size, heads * seq2 / block_compute_size);

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
        block_compute_size =
            (seq_length4 < threads ? ((threads / seq_length4) * seq_length4) : seq_length4);
        dim3 grid_dim(batch_size, heads * seq2 / block_compute_size);

        int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;

        dim3 block_dim(seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) /
                                                subblock_max_workload * threads)
                                             : threads);

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
    int seq2 = sequence_length * seq_length4;

    int block_compute_size =
        (seq_length4 < threads ? ((threads / seq_length4) * seq_length4) : seq_length4);
    dim3 grid_dim(batch_size, heads * seq2 / block_compute_size);

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
        block_compute_size =
            (seq_length4 < threads ? ((threads / seq_length4) * seq_length4) : seq_length4);
        dim3 grid_dim(batch_size, heads * seq2 / block_compute_size);

        int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;

        dim3 block_dim(seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) /
                                                subblock_max_workload * threads)
                                             : threads);

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

    int warp_num = blockDim.x >> 5;  // warp-count = num_threads / WARP_SIZE (32)

    int iteration_stride = blockDim.x;
    int block_width = blockStride * seq_length;

    int iterations = (seq_length < (MAX_THREAD_ITERATIONS * iteration_stride)
                          ? (seq_length + iteration_stride - 1) / iteration_stride
                          : MAX_THREAD_ITERATIONS);

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;

    int wid = id >> 5;
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
__global__ void softmax_backward_kernel_v2(T* grad ,
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
    if ((seq_length % WARP_SIZE) != 0 || seq_length > 2048)
        throw std::runtime_error("Invalid sequence length found in softmax backward.");

    const int warps_per_block = 4;
    dim3 grid_dim(batch_size * heads * seq_length / warps_per_block);
    dim3 block_dim(WARP_SIZE, warps_per_block);

    switch (seq_length) {
        case 32:
            softmax_backward_kernel_v2<T, 1>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 64:
            softmax_backward_kernel_v2<T, 2>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 128:
            softmax_backward_kernel_v2<T, 4>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 256:
            softmax_backward_kernel_v2<T, 8>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 384:
            softmax_backward_kernel_v2<T, 12>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 512:
            softmax_backward_kernel_v2<T, 16>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 768:
            softmax_backward_kernel_v2<T, 24>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 1024:
            softmax_backward_kernel_v2<T, 32>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 2048:
            softmax_backward_kernel_v2<T, 64>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        default:
            throw std::runtime_error(
                std::string("Special sequence length found in softmax backward, seq_length: ") +
                std::to_string(seq_length));
    }
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
*/

#include "custom_cuda_layers.h"
#include "general_kernels.h"


void CheckCudaErrorAux (const char *file, unsigned line)
{
	cudaError_t err = cudaGetLastError();
	if (err == cudaSuccess)
		return;
	std::cerr << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	throw std::runtime_error("CUDA ERROR!!!\n");
}

#define CUDA_CHECK_ERROR() CheckCudaErrorAux(__FILE__,__LINE__)

namespace cg = cooperative_groups;


// Fused attention + softmax
template <int tbSize, int blockStride, int tbSeq>
__global__ void attn_softmax(float* vals,
                             const float* attn_mask,
                             int heads,
                             int seq_length,
                             int iterations)
{
    __shared__ float partialSum[MAX_WARP_NUM];

    int warp_num = blockDim.x >> 5;

    int iteration_stride = blockDim.x;
    int block_width = blockStride * seq_length;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    int batch = blockIdx.x;
    int row = blockIdx.y;
    int max_threads_in_sequence = std::max(seq_length, tbSeq);
    int seq_lane = threadIdx.x % max_threads_in_sequence;

    int seq_id = row % (seq_length << 2);//(row % ((seq_length << 2) / blockStride)) * blockStride + (threadIdx.x / max_threads_in_sequence);
    int seq_id_4 = seq_id % 4;

    int data_offset = batch * (gridDim.y * block_width) + row * block_width +
                      (threadIdx.x / max_threads_in_sequence) * seq_length;
    int mask_offset = batch * seq_length;

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;

    float4* val_cast = reinterpret_cast<float4*>(vals);
    const float4* attn_mask_cast;
    if(attn_mask)attn_mask_cast = reinterpret_cast<const float4*>(attn_mask);

    float4 data[MAX_THREAD_ITERATIONS];

    float max_val = minus_infinity;

    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + seq_lane;
        if ((data_id <= seq_id || attn_mask) && data_id < seq_length) {

            data[i] = val_cast[data_offset + data_id];
            if(attn_mask)
            {
                float4 mask = attn_mask_cast[mask_offset + data_id];
                data[i].x += mask.x;
                data[i].y += mask.y;
                data[i].z += mask.z;
                data[i].w += mask.w;
            }
            else{
                if(data_id == seq_id && seq_id_4 < 3)
                {
                    data[i].w  = minus_infinity;
                    data[i].z =(seq_id_4 < 2 ? minus_infinity : data[i].z);
                    data[i].y=(seq_id_4 < 1 ? minus_infinity : data[i].y);
                }
            }

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
        if (seq_length < iteration_stride) iters = warp_num / (iteration_stride / seq_length);

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
        if (seq_length < iteration_stride) iters = warp_num / (iteration_stride / seq_length);

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
#if __CUDA_ARCH__ >= 700
    __shared__ float partialSum[MAX_WARP_NUM];

    int warp_num = blockDim.x >> 5;

    int iteration_stride = blockDim.x;
    int block_width = blockStride * seq_length;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    int batch = blockIdx.x;
    int row = blockIdx.y;
    int max_threads_in_sequence = std::max(seq_length, tbSeq);
    int seq_lane = threadIdx.x % max_threads_in_sequence;

    int seq_id = (row * blockStride + (threadIdx.x / max_threads_in_sequence)) % (seq_length << 2);//(row % ((seq_length << 2) / blockStride)) * blockStride + (threadIdx.x / max_threads_in_sequence);
    int seq_id_4 = seq_id % 4;
    seq_id >>= 2;

    int data_offset = batch * (gridDim.y * block_width) + row * block_width +
                      (threadIdx.x / max_threads_in_sequence) * seq_length;
    int mask_offset = batch * seq_length;

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;

    float2* val_cast = reinterpret_cast<float2*>(vals);
    const float2* attn_mask_cast;

    val_cast += data_offset;
    if(attn_mask){
        attn_mask_cast = reinterpret_cast<const float2*>(attn_mask);
        attn_mask_cast += mask_offset;
    }

    float2 low_data[MAX_THREAD_ITERATIONS];
    float2 high_data[MAX_THREAD_ITERATIONS];

    float max_val = minus_infinity;

    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + seq_lane;
        if ((data_id <= seq_id || attn_mask) && data_id < seq_length) {
            float2 data = val_cast[data_id];

            __half2* data_arr = reinterpret_cast<__half2*>(&data);

            low_data[i] = __half22float2(data_arr[0]);
            high_data[i] = __half22float2(data_arr[1]);

            if(attn_mask)
            {
                float2 mask = attn_mask_cast[data_id];
                __half2* mask_arr = reinterpret_cast<__half2*>(&mask);
                float2 low_mask = __half22float2(mask_arr[0]);
                float2 high_mask = __half22float2(mask_arr[1]);

                low_data[i].x += low_mask.x;
                low_data[i].y += low_mask.y;
                high_data[i].x += high_mask.x;
                high_data[i].y += high_mask.y;
            }
            else{
                if(data_id == seq_id && seq_id_4 < 3){
                    high_data[i].y  = minus_infinity;
                    high_data[i].x =(seq_id_4 < 2 ? minus_infinity : high_data[i].x);
                    low_data[i].y =(seq_id_4 < 1 ? minus_infinity : low_data[i].y);
                }
            }

            max_val = (low_data[i].x > max_val ? low_data[i].x : max_val);
            max_val = (low_data[i].y > max_val ? low_data[i].y : max_val);
            max_val = (high_data[i].x > max_val ? high_data[i].x : max_val);
            max_val = (high_data[i].y > max_val ? high_data[i].y : max_val);
        }
        else{
            low_data[i].x  = minus_infinity;
            low_data[i].y  = minus_infinity;
            high_data[i].x = minus_infinity;
            high_data[i].y = minus_infinity;
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
        if (seq_length < iteration_stride) iters = warp_num / (iteration_stride / seq_length);

        for (int i = 1; i < iters; i *= 2) {
            auto temp = g.shfl_xor(max_val, i);
            max_val = (temp > max_val ? temp : max_val);
        }

        max_val = g.shfl(max_val, threadIdx.x / tbSize);
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

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = sum;
        b.sync();

        if (lane < warp_num) sum = partialSum[lane];

#ifndef __STOCHASTIC_MODE__
        b.sync();
#endif

        int iters = warp_num;
        if (seq_length < iteration_stride) iters = warp_num / (iteration_stride / seq_length);

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
    int seq2 = sequence_length * seq_length4;

    int block_compute_size =
        (seq_length4 < threads ? ((threads / seq_length4) * seq_length4) : seq_length4);
    dim3 grid_dim(batch_size, heads * seq2 / block_compute_size);

    int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;

    dim3 block_dim(seq_length4 > threads ? (((sequence_length - 1) / subblock_max_workload + 1) * threads)
                                         : threads);
    int iterations =
        (sequence_length < subblock_max_workload ? ((seq_length4 - 1) / threads + 1)
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
        const int threads = 512;
        block_compute_size =
            (seq_length4 < threads ? ((threads / seq_length4) * seq_length4) : seq_length4);
        dim3 grid_dim(batch_size, heads * seq2 / block_compute_size);

        int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;
        dim3 block_dim(/*seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) /
                                                subblock_max_workload * threads)
                                             : */threads);

        iterations = (sequence_length < subblock_max_workload ? (seq_length4 + threads - 1) / threads
                                                 : MAX_THREAD_ITERATIONS);

        if (sequence_length <= 512)
            attn_softmax<32, (threads / 128), 128><<<grid_dim, block_dim, 0, stream>>>(
                vals, attn_mask, heads, seq_length4, iterations);
         if (sequence_length <= 1024)
            attn_softmax<32, (threads / 256), 256><<<grid_dim, block_dim, 0, stream>>>(
                vals, attn_mask, heads, seq_length4, iterations);
        else if (sequence_length < (MAX_THREADS * MAX_THREAD_ITERATIONS * 4))
            attn_softmax<32, 1, 512><<<grid_dim, block_dim, 0, stream>>>(
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
    int seq2 = sequence_length * seq_length4;

    int block_compute_size =
        (seq_length4 < threads ? ((threads / seq_length4) * seq_length4) : seq_length4);
    dim3 grid_dim(batch_size, heads * seq2 / block_compute_size);

    int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;

    dim3 block_dim(seq_length4 > threads ? (((sequence_length - 1) / subblock_max_workload + 1) * threads)
                                         : threads);

    int iterations =
        (sequence_length < subblock_max_workload ? ((seq_length4 - 1) / threads + 1)
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
        const int threads = 512;
        block_compute_size =
            (seq_length4 < threads ? ((threads / seq_length4) * seq_length4) : seq_length4);
        dim3 grid_dim(batch_size, heads * seq2 / block_compute_size);

        int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;
        iterations =
        (sequence_length < subblock_max_workload ? (seq_length4 + threads - 1) / threads
                                                 : MAX_THREAD_ITERATIONS);

        dim3 block_dim(/*seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) /
                                                subblock_max_workload * threads)
                                             : */threads);

        if (sequence_length <= 512)
            attn_softmax<32, (threads / 128), 128><<<grid_dim, block_dim, 0, stream>>>(
                vals, attn_mask, heads, seq_length4, iterations);
        if (sequence_length <= 1024)
            attn_softmax<32, (threads / 256), 256><<<grid_dim, block_dim, 0, stream>>>(
                vals, attn_mask, heads, seq_length4, iterations);
        else if (sequence_length < (MAX_THREADS * MAX_THREAD_ITERATIONS * 4))
            attn_softmax<32, 1, 512><<<grid_dim, block_dim, 0, stream>>>(
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

    int warp_num = blockDim.x >> 5;  // warp-count = num_threads / WARP_SIZE (32)

    int iteration_stride = blockDim.x;
    int block_width = blockStride * seq_length;

    int iterations = (seq_length < (MAX_THREAD_ITERATIONS * iteration_stride)
                          ? (seq_length + iteration_stride - 1) / iteration_stride
                          : MAX_THREAD_ITERATIONS);

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;

    int wid = id >> 5;
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
    if ((seq_length % WARP_SIZE) != 0 || seq_length > 2048)
        throw std::runtime_error("Invalid sequence length found in softmax backward.");

    const int warps_per_block = 4;
    dim3 grid_dim(batch_size * heads * seq_length / warps_per_block); // 2 * 16 * 512 = 16K
    dim3 block_dim(WARP_SIZE, warps_per_block);

    switch (seq_length) {
        case 32:
            softmax_backward_kernel_v2<T, 1>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 64:
            softmax_backward_kernel_v2<T, 2>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 128:
            softmax_backward_kernel_v2<T, 4>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 256:
            softmax_backward_kernel_v2<T, 8>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 384:
            softmax_backward_kernel_v2<T, 12>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 512:
            softmax_backward_kernel_v2<T, 16>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 768:
            softmax_backward_kernel_v2<T, 24>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 1024:
            softmax_backward_kernel_v2<T, 32>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 2048:
            softmax_backward_kernel_v2<T, 64>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        default:
            throw std::runtime_error(
                std::string("Special sequence length found in softmax backward, seq_length: ") +
                std::to_string(seq_length));
    }
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


template <int tbSize, int tbSeq>
__global__
void attn_softmax_v2(__half* vals,
                          const __half* attn_mask,
                          int total_count,
                          int heads,
                          int seq_length, int reduce_threads)
{

#if __CUDA_ARCH__ >= 700
    __shared__ float partialSum[MAX_WARP_NUM];

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    int iters = (total_count-1) / (gridDim.x * (blockDim.x >> 5)) + 1;

    float2 * val_cast = reinterpret_cast<float2 *>(vals);

    const float2 * attn_mask_cast;
    if(attn_mask){
        attn_mask_cast = reinterpret_cast<const float2*>(attn_mask);
    }

    float2 low_data[tbSeq];
    float2 high_data[tbSeq];

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    int iter_offset = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    val_cast += (iter_offset * seq_length);

    int iteration_stride = (blockDim.x >> 5) * gridDim.x;

    for(int iter = 0;iter < iters;iter++)
    {
        if(iter_offset < total_count)
        {
            int seq_id = ((iter_offset + (threadIdx.x >> 5)) % (seq_length << 2));
            int seq_id_4 = seq_id % 4;
            seq_id >>= 2;

            int mask_offset = (iter_offset / (heads * (seq_length << 2))) * seq_length;
            float max_val = minus_infinity;

            for(int i = 0;i < tbSeq;i++)
            {
                int data_id = i * WARP_SIZE + lane;
                if((data_id <= seq_id || attn_mask) && data_id < seq_length)
                {
                    float2 data = val_cast[data_id];

                    __half2 * data_arr = reinterpret_cast<__half2 *>(&data);

                    low_data[i] = __half22float2(data_arr[0]);
                    high_data[i] = __half22float2(data_arr[1]);

                    if(attn_mask)
                    {
                        float2 mask = attn_mask_cast[data_id + mask_offset];
                        __half2* mask_arr = reinterpret_cast<__half2*>(&mask);
                        float2 low_mask = __half22float2(mask_arr[0]);
                        float2 high_mask = __half22float2(mask_arr[1]);

                        low_data[i].x += low_mask.x;
                        low_data[i].y += low_mask.y;
                        high_data[i].x += high_mask.x;
                        high_data[i].y += high_mask.y;
                    }
                    else{
                        if(data_id == seq_id && seq_id_4 < 3){
                            high_data[i].y  = minus_infinity;
                            high_data[i].x =(seq_id_4 < 2 ? minus_infinity : high_data[i].x);
                            low_data[i].y =(seq_id_4 < 1 ? minus_infinity : low_data[i].y);
                        }
                    }
                    max_val = (low_data[i].x > max_val ? low_data[i].x : max_val);
                    max_val = (low_data[i].y > max_val ? low_data[i].y : max_val);
                    max_val = (high_data[i].x > max_val ?high_data[i].x : max_val);
                    max_val = (high_data[i].y > max_val ?high_data[i].y : max_val);
                }
                else{
                    low_data[i].x = minus_infinity;
                    low_data[i].y = minus_infinity;
                    high_data[i].x = minus_infinity;
                    high_data[i].y = minus_infinity;
                }
            }

            for (int i = 1; i < tbSize; i *= 2) {
                auto temp = g.shfl_xor(max_val, i);
                max_val = (temp > max_val ? temp : max_val);
            }

            if(reduce_threads > tbSize)
            {
                if(lane == 0)partialSum[wid] = max_val;
                b.sync();

                if (lane < warp_num)max_val = partialSum[lane];

                int iters = warp_num;
                if(reduce_threads < iteration_stride)iters /= (iteration_stride / reduce_threads);

                for (int i = 1; i < iters; i *= 2){
                    auto temp = g.shfl_xor(max_val, i);
                    max_val = (temp > max_val ? temp : max_val);
                }

                max_val = g.shfl(max_val, threadIdx.x / tbSize);
            }

            float sum = 0;
            for(int i = 0;i < tbSeq;i++)
            {
                low_data[i].x = __expf(low_data[i].x - max_val);
                low_data[i].y = __expf(low_data[i].y - max_val);
                high_data[i].x = __expf(high_data[i].x - max_val);
                high_data[i].y = __expf(high_data[i].y - max_val);

                sum += (low_data[i].x + low_data[i].y +
                            high_data[i].x + high_data[i].y);
            }

            for (int i = 1; i < tbSize; i *= 2)
                sum += g.shfl_xor(sum, i);

            if(reduce_threads > tbSize)
            {
                if(lane == 0)partialSum[wid] = sum;
                b.sync();

                if (lane < warp_num)sum = partialSum[lane];

                int iters = warp_num;
                if(reduce_threads < iteration_stride)iters /= (iteration_stride / reduce_threads);

                for (int i = 1; i < iters; i *= 2){
                    sum += g.shfl_xor(sum, i);
                }

                sum = g.shfl(max_val, threadIdx.x / tbSize);
            }
            sum += 1e-6;
            __half2 sum_h = __float2half2_rn(sum);

            for(int i = 0;i < tbSeq;i++)
            {
                int data_id = i * WARP_SIZE + lane;
                if(data_id < seq_length)
                {
                    float2 result_f;
                    __half2* result_h = reinterpret_cast<__half2*>(&result_f);

                    result_h[0] = __float22half2_rn(low_data[i]);
                    result_h[1] = __float22half2_rn(high_data[i]);

                    result_h[0] /= sum_h;
                    result_h[1] /= sum_h;

                    val_cast[data_id] = result_f;
                }
            }
            val_cast += (iteration_stride * seq_length);
            iter_offset += iteration_stride;
        }
    }
#endif

}




template <int tbSize, int tbSeq>
__global__
void attn_softmax_v2(float* vals,
                     const float* attn_mask,
                     int total_count,
                     int heads,
                     int sequence_length, int reduce_threads)
{

}


template <typename T>
void launch_attn_softmax_v2(T * vals, const T * attn_mask, int batch_size, int heads,
                                 int sequence_length, cudaStream_t stream, int threads, int blocks, int reduce_threads) {

    /*int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);*/

    int total_count = batch_size * heads * sequence_length;

    int seq_length4 = sequence_length / 4;

    dim3 grid_dim(blocks);//(80);//(batch_size * heads * sequence_length / (threads >> 5));
    dim3 block_dim(threads);

    if(sequence_length <= 128)
        attn_softmax_v2<32, 1> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, total_count, heads, seq_length4, reduce_threads);
    else if(sequence_length <= 256)
        attn_softmax_v2<32, 2> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, total_count, heads, seq_length4, reduce_threads);
    else if(sequence_length <= 512)
        attn_softmax_v2<32, 4> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, total_count, heads, seq_length4, reduce_threads);
    else if(sequence_length <= 1024)
        attn_softmax_v2<32, 8> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, total_count, heads, seq_length4, reduce_threads);
    else if(sequence_length <= 2048)
        attn_softmax_v2<32, 16> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, total_count, heads, seq_length4, reduce_threads);
    else if(sequence_length <= 4096)
        attn_softmax_v2<32, 32> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, total_count, heads, seq_length4, reduce_threads);
    else
            throw std::runtime_error("Unsupport Seq_Length! Check the restriction of the max_threads and max_thread_iterations!");
    
    CUDA_CHECK_ERROR();

}

template void launch_attn_softmax_v2(float * vals, const float * attn_mask,
                         int batch_size, int heads, int sequence_length, cudaStream_t stream, int threads, int blocks, int reduce_threads);
template void launch_attn_softmax_v2(__half * vals, const __half * attn_mask,
                         int batch_size, int heads, int sequence_length, cudaStream_t stream, int threads, int blocks, int reduce_threads);



template <int tbSize, int blockStride, int tbSeq>
__global__
void attn_softmax_v3(float* vals,
                          const float* attn_mask,
                          int total_count,
                          int heads,
                          int sequence_length)
{

}

template <int tbSize, int blockStride, int tbSeq>
__global__
void attn_softmax_v3(__half* vals,
                          const __half* attn_mask,
                          int total_count,
                          int heads,
                          int seq_length)
{
#if __CUDA_ARCH__ >= 700
    __shared__ float partialSum[MAX_WARP_NUM];

    int iteration_stride = blockDim.x;
    int iterations = (seq_length < (MAX_THREAD_ITERATIONS * iteration_stride) ?
                        (seq_length + iteration_stride - 1) / iteration_stride :
                        MAX_THREAD_ITERATIONS);

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    int max_threads_in_sequence = std::max(seq_length, tbSeq);

    int iters = (total_count-1) / (gridDim.x * blockStride) + 1;

    float2 * val_cast = reinterpret_cast<float2 *>(vals);
    const float2 * attn_mask_cast = reinterpret_cast<const float2 *>(attn_mask);

    int seq_lane = threadIdx.x % max_threads_in_sequence;

    float2 low_data[MAX_THREAD_ITERATIONS];
    float2 high_data[MAX_THREAD_ITERATIONS];

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;


    int iter_offset = blockIdx.x * blockStride + (threadIdx.x / max_threads_in_sequence);
    val_cast += (iter_offset * seq_length);
    int iter_stride = blockStride * gridDim.x;

    for(int iter = 0;iter < iters;iter++)
    {
        if(iter_offset < total_count)
        {

            int seq_id = iter_offset % (seq_length << 2);
            int seq_id_4 = seq_id % 4;
            seq_id >>= 2;

            int mask_offset = (iter_offset / (heads * (seq_length << 2))) * seq_length;

            float max_val = minus_infinity;

            for(int i = 0;i < iterations;i++)
            {
                int data_id = i * iteration_stride + seq_lane;
                if((data_id <= seq_id || attn_mask) && data_id < seq_length)
                {
                    float2 data = val_cast[data_id];

                    __half2 * data_arr = reinterpret_cast<__half2 *>(&data);

                    low_data[i] = __half22float2(data_arr[0]);
                    high_data[i] = __half22float2(data_arr[1]);

                    if(attn_mask)
                    {
                        float2 mask = attn_mask_cast[data_id + mask_offset];
                        __half2* mask_arr = reinterpret_cast<__half2*>(&mask);
                        float2 low_mask = __half22float2(mask_arr[0]);
                        float2 high_mask = __half22float2(mask_arr[1]);

                        low_data[i].x += low_mask.x;
                        low_data[i].y += low_mask.y;
                        high_data[i].x += high_mask.x;
                        high_data[i].y += high_mask.y;
                    }
                    else{
                        if(data_id == seq_id && seq_id_4 < 3){
                            high_data[i].y  = minus_infinity;
                            high_data[i].x =(seq_id_4 < 2 ? minus_infinity : high_data[i].x);
                            low_data[i].y =(seq_id_4 < 1 ? minus_infinity : low_data[i].y);
                        }
                    }

                    max_val = (low_data[i].x > max_val ? low_data[i].x : max_val);
                    max_val = (low_data[i].y > max_val ? low_data[i].y : max_val);
                    max_val = (high_data[i].x > max_val ?high_data[i].x  : max_val);
                    max_val = (high_data[i].y > max_val ?high_data[i].y : max_val);
                }
                else{
                    low_data[i].x = minus_infinity;
                    low_data[i].y = minus_infinity;
                    high_data[i].x = minus_infinity;
                    high_data[i].y = minus_infinity;
                }
            }

            for (int i = 1; i < tbSize; i *= 2) {
                auto temp = g.shfl_xor(max_val, i);
                max_val = (temp > max_val ? temp : max_val);
            }

            if(seq_length > tbSize)
            {
                if(lane == 0)partialSum[wid] = max_val;
                b.sync();

                if (lane < warp_num)max_val = partialSum[lane];

                int iters = warp_num;
                if(seq_length < iteration_stride)iters /= (iteration_stride / seq_length);

                for (int i = 1; i < iters; i *= 2){
                    auto temp = g.shfl_xor(max_val, i);
                    max_val = (temp > max_val ? temp : max_val);
                }

                max_val = g.shfl(max_val, threadIdx.x / tbSize);
            }

            float sum = 0;
            for(int i = 0;i < iterations;i++)
            {
                low_data[i].x = __expf(low_data[i].x - max_val);
                low_data[i].y = __expf(low_data[i].y - max_val);
                high_data[i].x = __expf(high_data[i].x - max_val);
                high_data[i].y = __expf(high_data[i].y - max_val);

                sum += (low_data[i].x + low_data[i].y +
                    high_data[i].x + high_data[i].y);
            }

            for (int i = 1; i < tbSize; i *= 2) {
                sum += g.shfl_xor(sum, i);
            }

            if(seq_length > tbSize)
            {
                if(lane == 0)partialSum[wid] = sum;
                b.sync();

                if (lane < warp_num)sum = partialSum[lane];

                int iters = warp_num;
                if(seq_length < iteration_stride)iters /= (iteration_stride / seq_length);

                for (int i = 1; i < iters; i *= 2){
                    sum += g.shfl_xor(sum, i);
                }

                sum = g.shfl(sum, threadIdx.x / tbSize);
            }

            sum += 1e-6;
            __half2 sum_h = __float2half2_rn(sum);

            for(int i = 0;i < iterations;i++)
            {
                int data_id = i * iteration_stride + seq_lane;
                if(data_id < seq_length)
                {
                    float2 result_f;
                    __half2* result_h = reinterpret_cast<__half2*>(&result_f);

                    result_h[0] = __float22half2_rn(low_data[i]);
                    result_h[1] = __float22half2_rn(high_data[i]);

                    result_h[0] /= sum_h;
                    result_h[1] /= sum_h;

                    val_cast[data_id] = result_f;
                }
            }
            iter_offset += iter_stride;
            val_cast += iter_stride * seq_length;
        }
    }
#endif

}

template <typename T>
void launch_attn_softmax_v3(T * vals, const T * attn_mask, int batch_size, int heads,
                                 int sequence_length, cudaStream_t stream) {

    /*int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);*/

    int total_count = batch_size * heads * sequence_length * sequence_length;

    int seq_length4 = sequence_length / 4;

    dim3 grid_dim(80);

    const int threads = 1024;
    dim3 block_dim(threads);

    total_count /= sequence_length;
    if(sequence_length <= 8)
        attn_softmax_v3<2,  (threads / 2), 2> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, total_count, heads, seq_length4);
    else if(sequence_length <= 16)
        attn_softmax_v3<4, (threads / 4), 4> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, total_count, heads, seq_length4);
    else if(sequence_length <= 32)
        attn_softmax_v3<8, (threads / 8), 8> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, total_count, heads, seq_length4);
    else if(sequence_length <= 64)
        attn_softmax_v3<16, (threads / 16), 16> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, total_count, heads, seq_length4);
    else if(sequence_length <= 128)
        attn_softmax_v3<32, (threads / 32), 32> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, total_count, heads, seq_length4);
    else if(sequence_length <= 256)
        attn_softmax_v3<32, (threads / 64), 64> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, total_count, heads, seq_length4);
    else if(sequence_length <= 512)
        attn_softmax_v3<32, (threads / 128), 128> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, total_count, heads, seq_length4);
    else if(sequence_length < (MAX_THREADS * MAX_THREAD_ITERATIONS * 4))
        attn_softmax_v3<32, 1, 128> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, total_count, heads, seq_length4);
    else
        throw std::runtime_error("Unsupport Seq_Length! Check the restriction of the max_threads and max_thread_iterations!");

    CUDA_CHECK_ERROR();
}

template void launch_attn_softmax_v3(float * vals, const float * attn_mask,
                         int batch_size, int heads, int sequence_length, cudaStream_t stream);
template void launch_attn_softmax_v3(__half * vals, const __half * attn_mask,
                         int batch_size, int heads, int sequence_length, cudaStream_t stream);



template <int tbSize, int blockStride, int tbSeq>
__global__
void attn_softmax_dropout(__half* vals,
                          const __half* attn_mask,
                          uint8_t* mask,
                          __half *masked_softmax,
                          float ratio,
                          std::pair<uint64_t, uint64_t> seed,
                          int total_count,
                          int heads,
                          int seq_length)
{

#if __CUDA_ARCH__ >= 700
    __shared__ float partialSum[MAX_WARP_NUM];

    int iterations = (seq_length + WARP_SIZE - 1) / WARP_SIZE;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    int iters = (total_count-1) / (gridDim.x * (blockDim.x >> 5)) + 1;

    float2 * val_cast = reinterpret_cast<float2 *>(vals);
    const float2 * attn_mask_cast;
    if(attn_mask)attn_mask_cast = reinterpret_cast<const float2 *>(attn_mask);
    uint32_t *mask_32 = reinterpret_cast<uint32_t*>(mask);

    const __half2 scale_h = __float2half2_rn(1. / (1. - ratio));

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, (((blockIdx.x * iters) * (blockDim.x >> 5) + (threadIdx.x >> 5)) * seq_length + (threadIdx.x & 0x1f)), seed.second, &state);

    uint32_t sparse_offset = (((total_count * (seq_length << 2)) * (ratio * 1.1)) / gridDim.x) * blockIdx.x +
                            (threadIdx.x >> 5) * ((((total_count * (seq_length << 2)) * (ratio * 1.1)) / gridDim.x) / (blockDim.x >> 5));

    float2 low_data[MAX_THREAD_ITERATIONS];
    float2 high_data[MAX_THREAD_ITERATIONS];
    uint32_t m_32[MAX_THREAD_ITERATIONS];

    int iter_offset = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int iter_stride = (blockDim.x >> 5) * gridDim.x;
    val_cast += (iter_offset * seq_length);

    for(int iter = 0;iter < iters;iter++)
    {
        if(iter_offset < total_count)
        {
            int mask_offset = (iter_offset / (heads * (seq_length << 2))) * seq_length;
            float max_val = minus_infinity;
            uint16_t non_zeros = iterations << 2;

            int seq_id = iter_offset % (seq_length << 2);
            int seq_id_4 = seq_id % 4;

            for(int i = 0;i < iterations;i++)
            {
                int data_id = i * WARP_SIZE + (threadIdx.x & 0x1f);
                if((data_id <= seq_id || attn_mask) && data_id < seq_length)
                {
                    {
                        float4 rand4 = curand_uniform4(&state);
                        uint8_t *m = reinterpret_cast<uint8_t*>(&m_32[i]);

                        m[0] = (uint8_t)(rand4.x > ratio);
                        m[1] = (uint8_t)(rand4.y > ratio);
                        m[2] = (uint8_t)(rand4.z > ratio);
                        m[3] = (uint8_t)(rand4.w > ratio);

                        non_zeros -= (m[0] + m[1] + m[2] + m[3]);
                    }

                    {
                        float2 data = val_cast[data_id];
                        __half2 * data_arr = reinterpret_cast<__half2 *>(&data);

                        low_data[i] = __half22float2(data_arr[0]);
                        high_data[i] = __half22float2(data_arr[1]);

                        if(attn_mask)
                        {
                            float2 mask_attn = attn_mask_cast[data_id + mask_offset];
                            __half2* mask_arr = reinterpret_cast<__half2*>(&mask_attn);
                            float2 low_mask = __half22float2(mask_arr[0]);
                            float2 high_mask = __half22float2(mask_arr[1]);

                            low_data[i].x += low_mask.x;
                            low_data[i].y += low_mask.y;
                            high_data[i].x += high_mask.x;
                            high_data[i].y += high_mask.y;
                        }
                        else{
                            if(data_id == seq_id && seq_id_4 < 3){
                                high_data[i].y  = minus_infinity;
                                high_data[i].x =(seq_id_4 < 2 ? minus_infinity : high_data[i].x);
                                low_data[i].y =(seq_id_4 < 1 ? minus_infinity : low_data[i].y);
                            }
                        }
                    }

                    max_val = (low_data[i].x > max_val ? low_data[i].x : max_val);
                    max_val = (low_data[i].y > max_val ? low_data[i].y : max_val);
                    max_val = (high_data[i].x > max_val ?high_data[i].x  : max_val);
                    max_val = (high_data[i].y > max_val ?high_data[i].y : max_val);
                }
                else{
                    low_data[i].x = minus_infinity;
                    low_data[i].y = minus_infinity;
                    high_data[i].x = minus_infinity;
                    high_data[i].y = minus_infinity;
                }
            }

            uint32_t total_non_zeros;
            // Blelloch Scan: reduction & down-sweep
            {
                // Reduction
                for(int i = 1;i < (tbSize >> 1);i <<= 1)
                {
                    auto temp = g.shfl_xor(non_zeros, i);
                    non_zeros += ((threadIdx.x + 1) % (1 << i) == 0 ? temp : 0);
                }
                auto temp = g.shfl_xor(non_zeros, (tbSize >> 1));
                if((threadIdx.x & 0x1f) == (tbSize - 1))
                {
                    total_non_zeros = non_zeros + temp;
                    non_zeros = 0;
                }
                // Down Sweep
                for (int i = (tbSize >> 1); i > 0;i >>= 1)
                {
                    auto scan_add = ((threadIdx.x + 1) / i) % 2;
                    auto temp = g.shfl_xor(non_zeros, i);
                    non_zeros = (((threadIdx.x + 1) % i == 0) ? (scan_add ? temp : (temp + non_zeros)) : non_zeros);
                }
                sparse_offset += non_zeros;
            }

            for (int i = 1; i < tbSize; i *= 2) {
                auto temp = g.shfl_xor(max_val, i);
                max_val = (temp > max_val ? temp : max_val);
            }

            float sum = 0;
            for(int i = 0;i < iterations;i++)
            {
                low_data[i].x = __expf(low_data[i].x - max_val);
                low_data[i].y = __expf(low_data[i].y - max_val);
                high_data[i].x = __expf(high_data[i].x - max_val);
                high_data[i].y = __expf(high_data[i].y - max_val);

                sum += (low_data[i].x + low_data[i].y +
                            high_data[i].x + high_data[i].y);
            }

            for (int i = 1; i < tbSize; i *= 2) {
                sum += g.shfl_xor(sum, i);
            }

            sum += 1e-6;
            __half2 sum_h = __float2half2_rn(sum);

            for(int i = 0;i < iterations;i++)
            {
                int data_id = i * WARP_SIZE + (threadIdx.x & 0x1f);
                if(data_id < seq_length)
                {
                    float2 result_f;
                    __half2* result_h = reinterpret_cast<__half2*>(&result_f);

                    {
                        result_h[0] = __float22half2_rn(low_data[i]);
                        result_h[1] = __float22half2_rn(high_data[i]);

                        result_h[0] /= sum_h;
                        result_h[1] /= sum_h;
                    }

                    {
                        float2 mask_f[2];
                        __half2 mask_h[2];

                        uint8_t *m = reinterpret_cast<uint8_t*>(&m_32[i]);

                        mask_f[0].x = (float)m[0];
                        mask_f[0].y = (float)m[1];
                        mask_f[1].x = (float)m[2];
                        mask_f[1].y = (float)m[3];

                        //if(!m[0] || !m[1] || !m[2] || !m[3])
                        {
                            low_data[i] = __half22float2(result_h[0]);
                            high_data[i] = __half22float2(result_h[1]);

                            if(!m[0])masked_softmax[sparse_offset++] = __float2half(low_data[i].x);
                            if(!m[1])masked_softmax[sparse_offset++] = __float2half(low_data[i].y);
                            if(!m[2])masked_softmax[sparse_offset++] = __float2half(high_data[i].x);
                            if(!m[3])masked_softmax[sparse_offset++] = __float2half(high_data[i].y);
                        }

                        mask_h[0] = __float22half2_rn(mask_f[0]);
                        mask_h[1] = __float22half2_rn(mask_f[1]);

                        result_h[0] *= scale_h * mask_h[0];
                        result_h[1] *= scale_h * mask_h[1];
                    }

                    val_cast[data_id] = result_f;
                    mask_32[data_id + iter_offset] = m_32[i];
                }
            }
            total_non_zeros = g.shfl(total_non_zeros, (tbSize - 1));
            sparse_offset += (total_non_zeros - non_zeros);
            iter_offset += iter_stride;
            val_cast += (iter_stride * seq_length);
        }
    }
#endif

}

template <int tbSize, int blockStride, int tbSeq>
__global__
void attn_softmax_dropout(float* vals,
                          const float* attn_mask,
                          uint8_t* mask,
                          float *masked_softmax,
                          float ratio,
                          std::pair<uint64_t, uint64_t> seed,
                          int total_count,
                          int heads,
                          int sequence_length)
{

}

template <typename T>
void launch_attn_softmax_dropout(T * vals, const T * attn_mask, uint8_t* mask, T *masked_softmax,
                                 float ratio, int batch_size, int heads,
                                 int sequence_length, cudaStream_t stream, int threads) {

    /*int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);*/

    int total_count = batch_size * heads * sequence_length * sequence_length;

    int seq_length4 = sequence_length / 4;
    int seq2 = sequence_length * seq_length4;

    //int block_compute_size = (seq_length4 < threads ? ((threads / seq_length4) * seq_length4) : seq_length4);
    int block_compute_size = threads / WARP_SIZE;
    dim3 grid_dim (80); // (heads * sequence_length / block_compute_size, batch_size);

    int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;

    if(threads == 1024){
        const int threads = 1024;//128;//1024;
        dim3 block_dim(seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) / subblock_max_workload * threads) : threads);

        uint64_t inc = total_count / grid_dim.x / grid_dim.y / block_dim.x;
        std::pair<uint64_t, uint64_t> seed = Context::Instance().IncrementOffset(inc);

        total_count /= sequence_length;
        if(sequence_length <= 8)
            attn_softmax_dropout<2,  (threads / 2), 2> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 16)
            attn_softmax_dropout<4, (threads / 4), 4> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 32)
            attn_softmax_dropout<8, (threads / 8), 8> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 64)
            attn_softmax_dropout<16, (threads / 16), 16> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 128)
            attn_softmax_dropout<32, (threads / 32), 32> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 256)
            attn_softmax_dropout<32, (threads / 64), 64> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 512)
            attn_softmax_dropout<32, (threads / 128), 128> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length < (MAX_THREADS * MAX_THREAD_ITERATIONS * 4))
            attn_softmax_dropout<32, 1, 128> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else
            throw std::runtime_error("Unsupport Seq_Length! Check the restriction of the max_threads and max_thread_iterations!");
    }
    else if(threads == 512){
        const int threads = 512;//128;//1024;
        dim3 block_dim(/* seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) / subblock_max_workload * threads) : */threads);

        uint64_t inc = total_count / grid_dim.x / grid_dim.y / block_dim.x;
        std::pair<uint64_t, uint64_t> seed = Context::Instance().IncrementOffset(inc);

        total_count /= sequence_length;
        if(sequence_length <= 8)
            attn_softmax_dropout<2,  (threads / 2), 2> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 16)
            attn_softmax_dropout<4, (threads / 4), 4> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 32)
            attn_softmax_dropout<8, (threads / 8), 8> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 64)
            attn_softmax_dropout<16, (threads / 16), 16> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 128)
            attn_softmax_dropout<32, (threads / 32), 32> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 256)
            attn_softmax_dropout<32, (threads / 64), 64> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 512)
            attn_softmax_dropout<32, (threads / 128), 128> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length < (MAX_THREADS * MAX_THREAD_ITERATIONS * 4))
            attn_softmax_dropout<32, 1, 128> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else
                throw std::runtime_error("Unsupport Seq_Length! Check the restriction of the max_threads and max_thread_iterations!");
    }
    else if(threads == 256){
        const int threads = 256;//128;//1024;
        dim3 block_dim(/* seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) / subblock_max_workload * threads) : */threads);

        uint64_t inc = total_count / grid_dim.x / grid_dim.y / block_dim.x;
        std::pair<uint64_t, uint64_t> seed = Context::Instance().IncrementOffset(inc);

        total_count /= sequence_length;
        if(sequence_length <= 8)
            attn_softmax_dropout<2,  (threads / 2), 2> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 16)
            attn_softmax_dropout<4, (threads / 4), 4> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 32)
            attn_softmax_dropout<8, (threads / 8), 8> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 64)
            attn_softmax_dropout<16, (threads / 16), 16> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 128)
            attn_softmax_dropout<32, (threads / 32), 32> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 256)
            attn_softmax_dropout<32, (threads / 64), 64> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 512)
            attn_softmax_dropout<32, (threads / 128), 128> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length < (MAX_THREADS * MAX_THREAD_ITERATIONS * 4))
            attn_softmax_dropout<32, 1, 128> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else
                throw std::runtime_error("Unsupport Seq_Length! Check the restriction of the max_threads and max_thread_iterations!");
    }
    else{
        const int threads = 128;//128;//1024;
        dim3 block_dim(/* seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) / subblock_max_workload * threads) : */threads);

        uint64_t inc = total_count / grid_dim.x / grid_dim.y / block_dim.x;
        std::pair<uint64_t, uint64_t> seed = Context::Instance().IncrementOffset(inc);

        total_count /= sequence_length;
        if(sequence_length <= 8)
            attn_softmax_dropout<2,  (threads / 2), 2> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 16)
            attn_softmax_dropout<4, (threads / 4), 4> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 32)
            attn_softmax_dropout<8, (threads / 8), 8> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 64)
            attn_softmax_dropout<16, (threads / 16), 16> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 128)
            attn_softmax_dropout<32, (threads / 32), 32> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 256)
            attn_softmax_dropout<32, (threads / 64), 64> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length <= 512)
            attn_softmax_dropout<32, (threads / 128), 128> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else if(sequence_length < (MAX_THREADS * MAX_THREAD_ITERATIONS * 4))
            attn_softmax_dropout<32, 1, 128> <<<grid_dim, block_dim, 0, stream>>>( vals, attn_mask, mask, masked_softmax, ratio, seed, total_count, heads, seq_length4);
        else
                throw std::runtime_error("Unsupport Seq_Length! Check the restriction of the max_threads and max_thread_iterations!");
    }
    CUDA_CHECK_ERROR();

}



template void launch_attn_softmax_dropout(float * vals, const float * attn_mask, uint8_t* mask,
                         float* masked_softmax, float ratio,
                         int batch_size, int heads, int sequence_length, cudaStream_t stream, int threads);
template void launch_attn_softmax_dropout(__half * vals, const __half * attn_mask, uint8_t* mask,
                         __half* masked_softmax, float ratio,
                         int batch_size, int heads, int sequence_length, cudaStream_t stream, int threads);



template <typename T, int ITERATIONS>
__global__ void softmax_dropout_backward_kernel(float* grad  , const float* output,
                                uint8_t *mask, float *masked_softmax,
                                float ratio, int softmax_length, int total_count)
{

}

template <typename T, int ITERATIONS>
__global__ void softmax_dropout_backward_kernel(__half* grad  , const __half* output,
                                uint8_t *mask, __half *masked_softmax, float ratio,
                                int softmax_length, int total_count)
{
    //__shared__ uint32_t non_zeros_shared[MAX_WARP_NUM];
    int wid = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_num = blockDim.x / WARP_SIZE;

    int iters = (total_count-1) / (warp_num * gridDim.x) + 1;
    int iter_stride = warp_num * softmax_length;

    int add_offset = blockIdx.x * iters * warp_num + wid;
    int offset = add_offset * softmax_length;

    const __half2 ratio_h = __float2half2_rn(ratio);

    float2 *grad_cast = reinterpret_cast<float2*>(grad);
    const float2 *output_cast = reinterpret_cast<const float2*>(output);
    uint32_t *mask_32 = reinterpret_cast<uint32_t*>(mask);

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    grad_cast += offset;
    output_cast += offset;
    mask_32 += offset;

    float2 grad_reg[ITERATIONS];
    float2 output_reg[ITERATIONS];

    __half2 *grad_reg_h;
    __half2 *output_reg_h;

    float2 mask_f[2];
    __half2 mask_h[2];

    uint32_t m_32_reg[ITERATIONS];

    int block_offset = ((total_count * (softmax_length << 2)) * (ratio * 1.1)) / gridDim.x;
    uint32_t sparse_offset = block_offset * blockIdx.x + wid * (block_offset / warp_num);// 23100 * blockIdx.x + wid * 722;

    for(int iter = 0;iter < iters;iter++)
    {
        if(add_offset < total_count)
        {
            float sum = 0.0;
            uint16_t non_zeros = ITERATIONS * 4;
            #pragma unroll
            for (int i = 0; i < ITERATIONS; ++i)
            {
                int curr_idx = lane + i * WARP_SIZE;
                if (curr_idx < softmax_length)
                {
                    grad_reg[i] = grad_cast[curr_idx];
                    output_reg[i] = output_cast[curr_idx];
                    m_32_reg[i] = mask_32[curr_idx];
                    uint8_t *m = reinterpret_cast<uint8_t*>(&m_32_reg[i]);

                    non_zeros -= (m[0] + m[1] + m[2] + m[3]);

                    mask_f[0].x = (float)m[0];
                    mask_f[0].y = (float)m[1];
                    mask_f[1].x = (float)m[2];
                    mask_f[1].y = (float)m[3];
                    mask_h[0] = __float22half2_rn(mask_f[0]);
                    mask_h[1] = __float22half2_rn(mask_f[1]);

                    grad_reg_h = reinterpret_cast<__half2*>(&grad_reg[i]);
                    output_reg_h = reinterpret_cast<__half2*>(&output_reg[i]);
                    grad_reg_h[0] *= (mask_h[0] * ratio_h);
                    grad_reg_h[1] *= (mask_h[1] * ratio_h);
                    output_reg_h[0] /= ratio_h;
                    output_reg_h[1] /= ratio_h;

                    __half2 result_h_0 = output_reg_h[0] * grad_reg_h[0];
                    __half2 result_h_1 = output_reg_h[1] * grad_reg_h[1];
                    float2 result_f_0 = __half22float2(result_h_0);
                    float2 result_f_1 = __half22float2(result_h_1);
                    sum += (result_f_0.x + result_f_0.y + result_f_1.x + result_f_1.y);
                }
            }
            for (int i = 1; i < WARP_SIZE; i <<= 1)
                    sum += g.shfl_xor(sum, i);

            // Blelloch Scan: reduction & down-sweep
            int bloch_width = WARP_SIZE >> 1;
            int index = threadIdx.x + 1;
            int k = 2;

            // Reduction
            for(int i = 1;i < bloch_width;i <<= 1)
            {
                auto temp = g.shfl_xor(non_zeros, i);
                non_zeros += (index % k == 0 ? temp : 0);
                k <<= 1;
            }
            uint32_t total_non_zeros;
            auto temp = g.shfl_xor(non_zeros, bloch_width);
            if(lane == (WARP_SIZE - 1))
            {
                //non_zeros_shared[wid] = non_zeros + temp;
                total_non_zeros = non_zeros + temp;
                non_zeros = 0;
            }
            total_non_zeros = g.shfl(total_non_zeros, (WARP_SIZE - 1));

            // Down Sweep
            for (int i = bloch_width; i > 0;i >>= 1)
            {
                auto scan_add = (index / i) % 2;
                auto temp = g.shfl_xor(non_zeros, i);
                non_zeros = ((index % i == 0) ? (scan_add ? temp : (temp + non_zeros)) : non_zeros);
            }

            sparse_offset += non_zeros;

            float2 grad_reg_f[2];
            float2 softmax_out[2];
            #pragma unroll
            for (int i = 0; i < ITERATIONS; ++i)
            {
                int curr_idx = threadIdx.x + i * WARP_SIZE;
                if (curr_idx < softmax_length)
                {
                    grad_reg_h = reinterpret_cast<__half2*>(&grad_reg[i]);
                    output_reg_h = reinterpret_cast<__half2*>(&output_reg[i]);
                    grad_reg_f[0] = __half22float2(grad_reg_h[0]);
                    grad_reg_f[1] = __half22float2(grad_reg_h[1]);

                    softmax_out[0] = __half22float2(output_reg_h[0]);
                    softmax_out[1] = __half22float2(output_reg_h[1]);

                    uint8_t *m = reinterpret_cast<uint8_t*>(&m_32_reg[i]);
                    if(!m[0])softmax_out[0].x = __half2float(masked_softmax[sparse_offset++]);
                    if(!m[1])softmax_out[0].y = __half2float(masked_softmax[sparse_offset++]);
                    if(!m[2])softmax_out[1].x = __half2float(masked_softmax[sparse_offset++]);
                    if(!m[3])softmax_out[1].y = __half2float(masked_softmax[sparse_offset++]);

                    output_reg_h[0] = __float22half2_rn(softmax_out[0]);
                    output_reg_h[1] = __float22half2_rn(softmax_out[1]);

                    grad_reg_f[0].x -= sum;
                    grad_reg_f[0].y -= sum;
                    grad_reg_f[1].x -= sum;
                    grad_reg_f[1].y -= sum;

                    grad_reg_h[0] = __float22half2_rn(grad_reg_f[0]);
                    grad_reg_h[1] = __float22half2_rn(grad_reg_f[1]);

                    output_reg_h[0] *= grad_reg_h[0];
                    output_reg_h[1] *= grad_reg_h[1];

                    grad_cast[curr_idx] = grad_reg[i];
                }
            }
            grad_cast += iter_stride;
            grad_cast += iter_stride;
            mask_32 += iter_stride;
            add_offset += warp_num;
            sparse_offset += (total_non_zeros - non_zeros);
        }
    }
}

template <typename T>
void launch_attn_softmax_dropout_grad(T *out_grad, const T *soft_inp, uint8_t* mask, T *masked_softmax,
                        float ratio, int batch_size, int heads, int seq_length, cudaStream_t stream, int threads)
{
    if ((seq_length % WARP_SIZE) != 0 || seq_length > 2048)
        throw std::runtime_error("Invalid sequence length found in softmax backward.");

    ratio = 1. / (1. - ratio);

    const int warps_per_block = 4;
    dim3 grid_dim(80); // (batch_size * heads * seq_length / warps_per_block);

    int total_count = batch_size * heads * seq_length;
    //if(prob_length == 0)prob_length = seq_length;
    int seq_length4 = seq_length/ 4;
    if(threads == 1024)
    {
        dim3 block_dim(1024); //(WARP_SIZE * warps_per_block);
        switch (seq_length)
        {
            case 32:
                softmax_dropout_backward_kernel<T, 1> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 64:
                softmax_dropout_backward_kernel<T, 1> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 128:
                softmax_dropout_backward_kernel<T, 1> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 256:
                softmax_dropout_backward_kernel<T, 2> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 384:
                softmax_dropout_backward_kernel<T, 3> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 512:
                softmax_dropout_backward_kernel<T, 4> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 768:
                softmax_dropout_backward_kernel<T, 6> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 1024:
                softmax_dropout_backward_kernel<T, 8> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 2048:
                softmax_dropout_backward_kernel<T, 16> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            default:
                throw std::runtime_error(std::string("Special sequence length found in softmax backward, seq_length: ") + std::to_string(seq_length));
        }
    }else if(threads == 512)
    {
        dim3 block_dim(512); //(WARP_SIZE * warps_per_block);
        switch (seq_length)
        {
            case 32:
                softmax_dropout_backward_kernel<T, 1> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 64:
                softmax_dropout_backward_kernel<T, 1> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 128:
                softmax_dropout_backward_kernel<T, 1> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 256:
                softmax_dropout_backward_kernel<T, 2> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 384:
                softmax_dropout_backward_kernel<T, 3> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 512:
                softmax_dropout_backward_kernel<T, 4> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 768:
                softmax_dropout_backward_kernel<T, 6> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 1024:
                softmax_dropout_backward_kernel<T, 8> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 2048:
                softmax_dropout_backward_kernel<T, 16> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            default:
                throw std::runtime_error(std::string("Special sequence length found in softmax backward, seq_length: ") + std::to_string(seq_length));
        }
    }
    else if(threads == 256)
    {
        dim3 block_dim(256); //(WARP_SIZE * warps_per_block);
        switch (seq_length)
        {
            case 32:
                softmax_dropout_backward_kernel<T, 1> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 64:
                softmax_dropout_backward_kernel<T, 1> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 128:
                softmax_dropout_backward_kernel<T, 1> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 256:
                softmax_dropout_backward_kernel<T, 2> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 384:
                softmax_dropout_backward_kernel<T, 3> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 512:
                softmax_dropout_backward_kernel<T, 4> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 768:
                softmax_dropout_backward_kernel<T, 6> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 1024:
                softmax_dropout_backward_kernel<T, 8> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 2048:
                softmax_dropout_backward_kernel<T, 16> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            default:
                throw std::runtime_error(std::string("Special sequence length found in softmax backward, seq_length: ") + std::to_string(seq_length));
        }
    }
    else{
            dim3 block_dim(128); //(WARP_SIZE * warps_per_block);
            switch (seq_length)
        {
            case 32:
                softmax_dropout_backward_kernel<T, 1> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 64:
                softmax_dropout_backward_kernel<T, 1> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 128:
                softmax_dropout_backward_kernel<T, 1> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 256:
                softmax_dropout_backward_kernel<T, 2> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 384:
                softmax_dropout_backward_kernel<T, 3> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 512:
                softmax_dropout_backward_kernel<T, 4> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 768:
                softmax_dropout_backward_kernel<T, 6> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 1024:
                softmax_dropout_backward_kernel<T, 8> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            case 2048:
                softmax_dropout_backward_kernel<T, 16> <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, mask, masked_softmax, ratio, seq_length4, total_count);
                break;
            default:
                throw std::runtime_error(std::string("Special sequence length found in softmax backward, seq_length: ") + std::to_string(seq_length));
        }
    }
}

template void launch_attn_softmax_dropout_grad(float *out_grad, const float *soft_inp, uint8_t* mask,
                                                float *masked_softmax, float ratio,
                                                int batch_size, int heads, int sequence_length,
                                                cudaStream_t stream, int threads);

template void launch_attn_softmax_dropout_grad(__half *out_grad, const __half *soft_inp, uint8_t* mask,
                                                __half *masked_softmax, float ratio,
                                                int batch_size, int heads, int sequence_length,
                                                cudaStream_t stream, int threads);
