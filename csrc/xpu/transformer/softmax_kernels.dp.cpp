// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
using namespace sycl;
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
using namespace cl::sycl;
#else
#error "Unsupported compiler"
#endif
#include "custom_sycl_layers.hpp"
#include "general_kernels.hpp"

#define MAX_SG_NUM (32)
// Fused attention + softmax
template <int tbSize, int blockStride, int tbSeq>
void attn_softmax(float* vals,
                  const float* attn_mask,
                  int heads,
                  int seq_length,
                  int iterations,
                  nd_item<3> item_ct1,
                  float* partialSum)
{
    int sg_num = item_ct1.get_local_range().get(2) >> 5;

    int iteration_stride = item_ct1.get_local_range().get(2);
    int block_width = blockStride * seq_length;

    // auto b = item_ct1.get_group();
    // cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);
    sub_group sg = item_ct1.get_sub_group();

    int batch = item_ct1.get_group(2);
    int row = item_ct1.get_group(1);
    int max_threads_in_sequence = std::max(seq_length, tbSeq);
    int seq_lane = item_ct1.get_local_id(2) % max_threads_in_sequence;

    int data_offset = batch * (item_ct1.get_group_range(1) * block_width) + row * block_width +
                      (item_ct1.get_local_id(2) / max_threads_in_sequence) * seq_length;
    int mask_offset = batch * seq_length;

    int wid = item_ct1.get_local_id(2) >> 5;
    int lane = item_ct1.get_local_id(2) & 0x1f;

    float4* val_cast = reinterpret_cast<float4*>(vals);
    const float4* attn_mask_cast = reinterpret_cast<const float4*>(attn_mask);

    float4 data[MAX_THREAD_ITERATIONS];

    float max_val = minus_infinity;

    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) {
            float4 mask = attn_mask_cast[mask_offset + data_id];
            data[i] = val_cast[data_offset + data_id];
            data[i].x() += mask.x();
            data[i].y() += mask.y();
            data[i].z() += mask.z();
            data[i].w() += mask.w();

            max_val = (data[i].x() > max_val ? data[i].x() : max_val);
            max_val = (data[i].y() > max_val ? data[i].y() : max_val);
            max_val = (data[i].z() > max_val ? data[i].z() : max_val);
            max_val = (data[i].w() > max_val ? data[i].w() : max_val);
        } else {
            data[i].x() = minus_infinity;
            data[i].y() = minus_infinity;
            data[i].z() = minus_infinity;
            data[i].w() = minus_infinity;
        }
    }

    for (int i = 1; i < tbSize; i *= 2) {
        auto temp = sg.shuffle_xor(max_val, i);
        max_val = (temp > max_val ? temp : max_val);
    }

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = max_val;
        item_ct1.barrier();

        if (lane < sg_num) max_val = partialSum[lane];

#ifndef __STOCHASTIC_MODE__
        item_ct1.barrier();
#endif

        int iters = sg_num;
        if (seq_length < iteration_stride)
            iters = sg_num / (iteration_stride / max_threads_in_sequence);

        for (int i = 1; i < iters; i *= 2) {
            auto temp = sg.shuffle_xor(max_val, i);
            max_val = (temp > max_val ? temp : max_val);
        }

        max_val = sg.shuffle(max_val, item_ct1.get_local_id(2) / tbSize);
    }

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        data[i].x() = sycl::exp(data[i].x() - max_val);
        data[i].y() = sycl::exp(data[i].y() - max_val);
        data[i].z() = sycl::exp(data[i].z() - max_val);
        data[i].w() = sycl::exp(data[i].w() - max_val);

        sum += (data[i].x() + data[i].y() + data[i].z() + data[i].w());
    }

    for (int i = 1; i < tbSize; i *= 2) { sum += sg.shuffle_xor(sum, i); }

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = sum;
        item_ct1.barrier();

        if (lane < sg_num) sum = partialSum[lane];

#ifndef __STOCHASTIC_MODE__
        item_ct1.barrier();
#endif

        int iters = sg_num;
        if (seq_length < iteration_stride)
            iters = sg_num / (iteration_stride / max_threads_in_sequence);

        for (int i = 1; i < iters; i *= 2) { sum += sg.shuffle_xor(sum, i); }

        sum = sg.shuffle(sum, item_ct1.get_local_id(2) / tbSize);
    }

    sum += 1e-6;

    for (int i = 0; i < iterations; i++) {
        data[i].x() /= sum;
        data[i].y() /= sum;
        data[i].z() /= sum;
        data[i].w() /= sum;

        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) val_cast[data_offset + data_id] = data[i];
    }
}

template <int tbSize, int blockStride, int tbSeq>
void attn_softmax(bf16* vals,
                  const bf16* attn_mask,
                  int heads,
                  int seq_length,
                  int iterations,
                  nd_item<3> item_ct1,
                  float* partialSum)
{
    int sg_num = item_ct1.get_local_range().get(2) >> 5;

    int iteration_stride = item_ct1.get_local_range().get(2);
    int block_width = blockStride * seq_length;

    // auto b = item_ct1.get_group();
    // cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);
    sub_group sg = item_ct1.get_sub_group();

    int batch = item_ct1.get_group(2);
    int row = item_ct1.get_group(1);
    int max_threads_in_sequence = std::max(seq_length, tbSeq);
    int seq_lane = item_ct1.get_local_id(2) % max_threads_in_sequence;

    int data_offset = batch * (item_ct1.get_group_range(1) * block_width) + row * block_width +
                      (item_ct1.get_local_id(2) / max_threads_in_sequence) * seq_length;
    int mask_offset = batch * seq_length;

    int wid = item_ct1.get_local_id(2) >> 5;
    int lane = item_ct1.get_local_id(2) & 0x1f;

    ushort4* val_cast = reinterpret_cast<ushort4*>(vals);
    const ushort4* attn_mask_cast = reinterpret_cast<const ushort4*>(attn_mask);

    float4 data[MAX_THREAD_ITERATIONS];

    float max_val = minus_infinity;

    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) {
            ushort4 mask_ushort = attn_mask_cast[mask_offset + data_id];
            ushort4 val_ushort = val_cast[data_offset + data_id];
            float4 mask = {float(mask_ushort.x()),
                           float(mask_ushort.y()),
                           float(mask_ushort.z()),
                           float(mask_ushort.w())};
            data[i] = {float(val_ushort.x()),
                       float(val_ushort.y()),
                       float(val_ushort.z()),
                       float(val_ushort.w())};

            data[i].x() += mask.x();
            data[i].y() += mask.y();
            data[i].z() += mask.z();
            data[i].w() += mask.w();

            max_val = (data[i].x() > max_val ? data[i].x() : max_val);
            max_val = (data[i].y() > max_val ? data[i].y() : max_val);
            max_val = (data[i].z() > max_val ? data[i].z() : max_val);
            max_val = (data[i].w() > max_val ? data[i].w() : max_val);
        } else {
            data[i].x() = minus_infinity;
            data[i].y() = minus_infinity;
            data[i].z() = minus_infinity;
            data[i].w() = minus_infinity;
        }
    }

    for (int i = 1; i < tbSize; i *= 2) {
        auto temp = sg.shuffle_xor(max_val, i);
        max_val = (temp > max_val ? temp : max_val);
    }

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = max_val;
        item_ct1.barrier();

        if (lane < sg_num) max_val = partialSum[lane];

#ifndef __STOCHASTIC_MODE__
        item_ct1.barrier();
#endif

        int iters = sg_num;
        if (seq_length < iteration_stride)
            iters = sg_num / (iteration_stride / max_threads_in_sequence);

        for (int i = 1; i < iters; i *= 2) {
            auto temp = sg.shuffle_xor(max_val, i);
            max_val = (temp > max_val ? temp : max_val);
        }

        max_val = sg.shuffle(max_val, item_ct1.get_local_id(2) / tbSize);
    }

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        data[i].x() = sycl::exp(data[i].x() - max_val);
        data[i].y() = sycl::exp(data[i].y() - max_val);
        data[i].z() = sycl::exp(data[i].z() - max_val);
        data[i].w() = sycl::exp(data[i].w() - max_val);

        sum += (data[i].x() + data[i].y() + data[i].z() + data[i].w());
    }

    for (int i = 1; i < tbSize; i *= 2) { sum += sg.shuffle_xor(sum, i); }

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = sum;
        item_ct1.barrier();

        if (lane < sg_num) sum = partialSum[lane];

#ifndef __STOCHASTIC_MODE__
        item_ct1.barrier();
#endif

        int iters = sg_num;
        if (seq_length < iteration_stride)
            iters = sg_num / (iteration_stride / max_threads_in_sequence);

        for (int i = 1; i < iters; i *= 2) { sum += sg.shuffle_xor(sum, i); }

        sum = sg.shuffle(sum, item_ct1.get_local_id(2) / tbSize);
    }

    sum += 1e-6;

    for (int i = 0; i < iterations; i++) {
        data[i].x() /= sum;
        data[i].y() /= sum;
        data[i].z() /= sum;
        data[i].w() /= sum;

        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) {
            ushort4 data_ushort = {
                bf16(data[i].x()), bf16(data[i].y()), bf16(data[i].z()), bf16(data[i].w())};
            val_cast[data_offset + data_id] = data_ushort;
        }
    }
}

template <int tbSize, int blockStride, int tbSeq>
void attn_softmax(half* vals,
                  const half* attn_mask,
                  int heads,
                  int seq_length,
                  int iterations,
                  nd_item<3> item_ct1,
                  float* partialSum)
{
    int sg_num = item_ct1.get_local_range(2) >> 5;

    int iteration_stride = item_ct1.get_local_range(2);
    int block_width = blockStride * seq_length;

    // cg::thread_block b = cg::this_thread_block();
    // cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);
    sub_group sg = item_ct1.get_sub_group();

    int batch = item_ct1.get_group(2);
    int row = item_ct1.get_group(1);
    int max_threads_in_sequence = std::max(seq_length, tbSeq);
    int seq_lane = item_ct1.get_local_id(2) % max_threads_in_sequence;

    int data_offset = batch * (item_ct1.get_group_range(1) * block_width) + row * block_width +
                      (item_ct1.get_local_id(2) / max_threads_in_sequence) * seq_length;
    int mask_offset = batch * seq_length;

    int wid = item_ct1.get_local_id(2) >> 5;
    int lane = item_ct1.get_local_id(2) & 0x1f;

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

            half2* data_arr = reinterpret_cast<half2*>(&data);
            half2* mask_arr = reinterpret_cast<half2*>(&mask);

            low_data[i] = data_arr[0].convert<float>();
            high_data[i] = data_arr[1].convert<float>();
            float2 low_mask = mask_arr[0].convert<float>();
            float2 high_mask = mask_arr[1].convert<float>();

            low_data[i].x() += low_mask.x();
            low_data[i].y() += low_mask.y();
            high_data[i].x() += high_mask.x();
            high_data[i].y() += high_mask.y();

            max_val = (low_data[i].x() > max_val ? low_data[i].x() : max_val);
            max_val = (low_data[i].y() > max_val ? low_data[i].y() : max_val);
            max_val = (high_data[i].x() > max_val ? high_data[i].x() : max_val);
            max_val = (high_data[i].y() > max_val ? high_data[i].y() : max_val);
        }
    }

    for (int i = 1; i < tbSize; i *= 2) {
        auto temp = sg.shuffle_xor(max_val, i);
        max_val = (temp > max_val ? temp : max_val);
    }

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = max_val;
        item_ct1.barrier();

        if (lane < sg_num) max_val = partialSum[lane];

#ifndef __STOCHASTIC_MODE__
        item_ct1.barrier();
#endif

        int iters = sg_num;
        if (seq_length < iteration_stride)
            iters = sg_num / (iteration_stride / max_threads_in_sequence);

        for (int i = 1; i < iters; i *= 2) {
            auto temp = sg.shuffle_xor(max_val, i);
            max_val = (temp > max_val ? temp : max_val);
        }

        max_val = sg.shuffle(max_val, item_ct1.get_local_id(2) / tbSize);
    }

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) {
            low_data[i] = sycl::exp(low_data[i] - max_val);
            high_data[i] = sycl::exp(high_data[i] - max_val);
            // low_data[i].x() = sycl::exp(low_data[i].x() - max_val);
            // low_data[i].y() = sycl::exp(low_data[i].y() - max_val);
            // high_data[i].x() = sycl::exp(high_data[i].x() - max_val);
            // high_data[i].y() = sycl::exp(high_data[i].y() - max_val);

            sum += (low_data[i].x() + low_data[i].y() + high_data[i].x() + high_data[i].y());
        }
    }

    for (int i = 1; i < tbSize; i *= 2) { sum += sg.shuffle_xor(sum, i); }

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = sum;
        item_ct1.barrier();

        if (lane < sg_num) sum = partialSum[lane];

#ifndef __STOCHASTIC_MODE__
        item_ct1.barrier();
#endif

        int iters = sg_num;
        if (seq_length < iteration_stride)
            iters = sg_num / (iteration_stride / max_threads_in_sequence);

        for (int i = 1; i < iters; i *= 2) { sum += sg.shuffle_xor(sum, i); }

        sum = sg.shuffle(sum, item_ct1.get_local_id(2) / tbSize);
    }

    sum += 1e-6;

    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) {
            float2 result_f;
            half2* result_h = reinterpret_cast<half2*>(&result_f);

            low_data[i].x() /= sum;
            low_data[i].y() /= sum;
            high_data[i].x() /= sum;
            high_data[i].y() /= sum;

            result_h[0] = low_data[i].convert<half, rounding_mode::rtn>();
            result_h[1] = high_data[i].convert<half, rounding_mode::rtn>();

            val_cast[data_id] = result_f;
        }
    }
}

template <typename T>
void launch_attn_softmax(T* vals,
                         const T* attn_mask,
                         int batch_size,
                         int heads,
                         int sequence_length,
                         queue* stream)
{
    const int threads = 128;
    int seq_length4 = sequence_length / 4;

    int block_compute_size =
        (seq_length4 < threads ? (int)pow(2.0, floor(log2((float)(threads / seq_length4)))) : 1);
    range<3> grid_dim(1, heads * sequence_length / block_compute_size, batch_size);

    int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;

    range<3> block_dim(1,
                       1,
                       seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) /
                                                subblock_max_workload * threads)
                                             : threads);
    int iterations =
        (sequence_length < subblock_max_workload ? (seq_length4 + threads - 1) / threads
                                                 : MAX_THREAD_ITERATIONS);

    if (sequence_length <= 8)
        stream->submit([&](handler& cgh) {
            accessor<float, 1, access::mode::read_write, access::target::local> data_block_acc_ct1(
                range<1>(MAX_SG_NUM), cgh);
            cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                                 attn_softmax<2, (threads / 2), 2>(
                                     vals,
                                     attn_mask,
                                     heads,
                                     seq_length4,
                                     iterations,
                                     item_ct1,
                                     data_block_acc_ct1.get_pointer());
                             });
        });
    else if (sequence_length <= 16)
        stream->submit([&](handler& cgh) {
            accessor<float, 1, access::mode::read_write, access::target::local> data_block_acc_ct1(
                range<1>(MAX_SG_NUM), cgh);
            cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                                 attn_softmax<4, (threads / 4), 4>(
                                     vals,
                                     attn_mask,
                                     heads,
                                     seq_length4,
                                     iterations,
                                     item_ct1,
                                     data_block_acc_ct1.get_pointer());
                             });
        });
    else if (sequence_length <= 32)
        stream->submit([&](handler& cgh) {
            accessor<float, 1, access::mode::read_write, access::target::local> data_block_acc_ct1(
                range<1>(MAX_SG_NUM), cgh);
            cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                                 attn_softmax<8, (threads / 8), 8>(
                                     vals,
                                     attn_mask,
                                     heads,
                                     seq_length4,
                                     iterations,
                                     item_ct1,
                                     data_block_acc_ct1.get_pointer());
                             });
        });
    else if (sequence_length <= 64)
        stream->submit([&](handler& cgh) {
            accessor<float, 1, access::mode::read_write, access::target::local> data_block_acc_ct1(
                range<1>(MAX_SG_NUM), cgh);
            cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                                 attn_softmax<16, (threads / 16), 16>(
                                     vals,
                                     attn_mask,
                                     heads,
                                     seq_length4,
                                     iterations,
                                     item_ct1,
                                     data_block_acc_ct1.get_pointer());
                             });
        });
    else if (sequence_length <= 128)
        stream->submit([&](handler& cgh) {
            accessor<float, 1, access::mode::read_write, access::target::local> data_block_acc_ct1(
                range<1>(MAX_SG_NUM), cgh);
            cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                                 attn_softmax<32, (threads / 32), 32>(
                                     vals,
                                     attn_mask,
                                     heads,
                                     seq_length4,
                                     iterations,
                                     item_ct1,
                                     data_block_acc_ct1.get_pointer());
                             });
        });
    else if (sequence_length <= 256)
        stream->submit([&](handler& cgh) {
            accessor<float, 1, access::mode::read_write, access::target::local> data_block_acc_ct1(
                range<1>(MAX_SG_NUM), cgh);
            cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                                 attn_softmax<32, (threads / 64), 64>(
                                     vals,
                                     attn_mask,
                                     heads,
                                     seq_length4,
                                     iterations,
                                     item_ct1,
                                     data_block_acc_ct1.get_pointer());
                             });
        });
    else {
        const int threads = 256;
        block_compute_size =
            (seq_length4 < threads ? (int)pow(2.0, floor(log2((float)(threads / seq_length4))))
                                   : 1);
        range<3> grid_dim(1, heads * sequence_length / block_compute_size, batch_size);

        int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;

        range<3> block_dim(1,
                           1,
                           seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) /
                                                    subblock_max_workload * threads)
                                                 : threads);
        iterations =
            (sequence_length < subblock_max_workload ? (seq_length4 + threads - 1) / threads
                                                     : MAX_THREAD_ITERATIONS);
        if (sequence_length <= 512) {
            stream->submit([&](handler& cgh) {
                accessor<float, 1, access::mode::read_write, access::target::local>
                    data_block_acc_ct1(range<1>(MAX_SG_NUM), cgh);
                cgh.parallel_for(
                    nd_range<3>(grid_dim * block_dim, block_dim),
                    [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                        attn_softmax<32, (threads / 128), 128>(vals,
                                                               attn_mask,
                                                               heads,
                                                               seq_length4,
                                                               iterations,
                                                               item_ct1,
                                                               data_block_acc_ct1.get_pointer());
                    });
            });
        } else if (sequence_length < (MAX_THREADS * MAX_THREAD_ITERATIONS * 4))
            stream->submit([&](handler& cgh) {
                accessor<float, 1, access::mode::read_write, access::target::local>
                    data_block_acc_ct1(range<1>(MAX_SG_NUM), cgh);
                cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                                 [=](nd_item<3> item_ct1)
                                     [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                                         attn_softmax<32, 1, 128>(vals,
                                                                  attn_mask,
                                                                  heads,
                                                                  seq_length4,
                                                                  iterations,
                                                                  item_ct1,
                                                                  data_block_acc_ct1.get_pointer());
                                     });
            });
        else
            throw std::runtime_error(
                "Unsupport Seq_Length! Check the restriction of the max_threads and "
                "max_thread_iterations!");
    }
}

template void launch_attn_softmax<float>(float* vals,
                                         const float* attn_mask,
                                         int batch_size,
                                         int heads,
                                         int sequence_length,
                                         queue* stream);

template void launch_attn_softmax<bf16>(bf16* vals,
                                        const bf16* attn_mask,
                                        int batch_size,
                                        int heads,
                                        int sequence_length,
                                        queue* stream);

template void launch_attn_softmax<half>(half* vals,
                                        const half* attn_mask,
                                        int batch_size,
                                        int heads,
                                        int sequence_length,
                                        queue* stream);

template <typename T, int tbSize, int blockStride>
void softmax_backward_kernel(T* out_grad,
                             const T* soft_inp,
                             int seq_length,
                             nd_item<3> item_ct1,
                             float* partialSum)
{
    int sg_num = item_ct1.get_local_range().get(2) >> 5;  // sg-count = num_threads / SG_SIZE (32)

    int iteration_stride = item_ct1.get_local_range().get(2);
    int block_width = blockStride * seq_length;

    int iterations = (seq_length < (MAX_THREAD_ITERATIONS * iteration_stride)
                          ? (seq_length + iteration_stride - 1) / iteration_stride
                          : MAX_THREAD_ITERATIONS);

    // auto b = item_ct1.get_group();
    // cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);
    sub_group sg = item_ct1.get_sub_group();

    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);

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
                         (float)soft_reg[i]);  // if done in half, the multiplication, we may
                                               // lose 2% of accuracy in computation!!
        }
    }
    for (int i = 1; i < tbSize; i *= 2) grad_reg += sg.shuffle_xor(grad_reg, i);

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = grad_reg;
        item_ct1.barrier();

        if (lane < sg_num) grad_reg = partialSum[lane];

        int iters = sg_num;
        if (seq_length < iteration_stride) iters = sg_num / (iteration_stride / seq_length);

        for (int i = 1; i < iters; i *= 2) grad_reg += sg.shuffle_xor(grad_reg, i);

        grad_reg = sg.shuffle(grad_reg, id / tbSize);
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
void softmax_backward_kernel_v2(T* grad /* input & output*/,
                                const T* output,
                                int softmax_length,
                                nd_item<3> item_ct1)
{
    int batch_idx =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(1) + item_ct1.get_local_id(1);
    int offset = batch_idx * softmax_length + item_ct1.get_local_id(2);

    grad += offset;
    output += offset;

    float sum = 0.0;
    if constexpr (std::is_same_v<T, bf16>) {
        float grad_reg[ITERATIONS];
        float output_reg[ITERATIONS];
        ushort* grad_cast = (ushort*)grad;
        const ushort* output_cast = (const ushort*)output;
        for (int i = 0; i < ITERATIONS; ++i) {
            int curr_idx = item_ct1.get_local_id(2) + i * MAX_SG_NUM;
            if (curr_idx < softmax_length) {
                grad_reg[i] = float(grad_cast[i * MAX_SG_NUM]);
                output_reg[i] = float(output_cast[i * MAX_SG_NUM]);
                sum += grad_reg[i] * output_reg[i];
            }
        }

        sub_group sg = item_ct1.get_sub_group();

        for (int i = 1; i < MAX_SG_NUM; i <<= 1) sum += sg.shuffle_xor(sum, i);

#pragma unroll
        for (int i = 0; i < ITERATIONS; ++i) {
            int curr_idx = item_ct1.get_local_id(2) + i * MAX_SG_NUM;
            if (curr_idx < softmax_length) {
                grad_cast[i * MAX_SG_NUM] = bf16(output_reg[i] * (grad_reg[i] - sum));
            }
        }
    } else {
        T grad_reg[ITERATIONS];
        T output_reg[ITERATIONS];

#pragma unroll
        for (int i = 0; i < ITERATIONS; ++i) {
            int curr_idx = item_ct1.get_local_id(2) + i * MAX_SG_NUM;
            if (curr_idx < softmax_length) {
                grad_reg[i] = grad[i * MAX_SG_NUM];
                output_reg[i] = output[i * MAX_SG_NUM];
                sum += (float)grad_reg[i] * (float)output_reg[i];
            }
        }
        sub_group sg = item_ct1.get_sub_group();

        for (int i = 1; i < MAX_SG_NUM; i <<= 1) sum += sg.shuffle_xor(sum, i);

#pragma unroll
        for (int i = 0; i < ITERATIONS; ++i) {
            int curr_idx = item_ct1.get_local_id(2) + i * MAX_SG_NUM;
            if (curr_idx < softmax_length)
                grad[i * MAX_SG_NUM] = (float)output_reg[i] * ((float)grad_reg[i] - sum);
        }
    }
}

template <typename T>
void launch_attn_softmax_backward_v2(T* out_grad,
                                     const T* soft_inp,
                                     int batch_size,
                                     int heads,
                                     int seq_length,
                                     queue* stream)
{
    const int sgs_per_block = 4;
    range<3> grid_dim(1, 1, batch_size * heads * seq_length / sgs_per_block);
    range<3> block_dim(1, sgs_per_block, MAX_SG_NUM);

    if (seq_length <= 32)
        stream->parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                                 softmax_backward_kernel_v2<T, 1>(
                                     out_grad, soft_inp, seq_length, item_ct1);
                             });
    else if (seq_length <= 64)
        stream->parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                                 softmax_backward_kernel_v2<T, 2>(
                                     out_grad, soft_inp, seq_length, item_ct1);
                             });
    else if (seq_length <= 128)
        stream->parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                                 softmax_backward_kernel_v2<T, 4>(
                                     out_grad, soft_inp, seq_length, item_ct1);
                             });
    else if (seq_length <= 256)
        stream->parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                                 softmax_backward_kernel_v2<T, 8>(
                                     out_grad, soft_inp, seq_length, item_ct1);
                             });
    else if (seq_length <= 384)
        stream->parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                                 softmax_backward_kernel_v2<T, 12>(
                                     out_grad, soft_inp, seq_length, item_ct1);
                             });
    else if (seq_length <= 512)
        stream->parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                                 softmax_backward_kernel_v2<T, 16>(
                                     out_grad, soft_inp, seq_length, item_ct1);
                             });
    else if (seq_length <= 768)
        stream->parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                                 softmax_backward_kernel_v2<T, 24>(
                                     out_grad, soft_inp, seq_length, item_ct1);
                             });
    else if (seq_length <= 1024)
        stream->parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                                 softmax_backward_kernel_v2<T, 32>(
                                     out_grad, soft_inp, seq_length, item_ct1);
                             });
    else if (seq_length <= 2048)
        stream->parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                                 softmax_backward_kernel_v2<T, 64>(
                                     out_grad, soft_inp, seq_length, item_ct1);
                             });
    else
        throw std::runtime_error(
            std::string("Special sequence length found in softmax backward, seq_length: ") +
            std::to_string(seq_length));
}

template void launch_attn_softmax_backward_v2<float>(float* out_grad,
                                                     const float* soft_inp,
                                                     int batch_size,
                                                     int heads,
                                                     int seq_length,
                                                     queue* stream);
template void launch_attn_softmax_backward_v2<bf16>(bf16* out_grad,
                                                    const bf16* soft_inp,
                                                    int batch_size,
                                                    int heads,
                                                    int seq_length,
                                                    queue* stream);
template void launch_attn_softmax_backward_v2<half>(half* out_grad,
                                                    const half* soft_inp,
                                                    int batch_size,
                                                    int heads,
                                                    int seq_length,
                                                    queue* stream);
