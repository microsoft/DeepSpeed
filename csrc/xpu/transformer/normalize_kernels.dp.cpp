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
/*
  Fused bias add, residual (elementwise) add, and normalization layer.

  For FP16, this kernel does not promote to FP32 in order to utilize the 2x
  throughput for
  __half2 instructions, and avoid the conversion overhead (1/8 of __hal2
  arithmetic).

  For specific launch constraints, see the launch functions.
*/

#define NORM_REG (128)
#define MAX_SG_NUM (32)
#define MAX_SG_NUM1 (MAX_SG_NUM + 1)
#define TILE_DIM (32)
template <bool is_mean>
void fused_bias_residual_layer_norm(float* vals,
                                    const float* residual,
                                    const float* gamma,
                                    const float* beta,
                                    float epsilon,
                                    bool preLayerNorm,
                                    bool training,
                                    float* vars,
                                    float* means,
                                    int row_stride,
                                    nd_item<3> item_ct1,
                                    float* shr)
{
    int iteration_stride = item_ct1.get_local_range(2);
    int iterations = row_stride / iteration_stride;

    // sycl::group<3> b = item_ct1.get_group();
    // cg::thread_block_tile<MAX_SG_NUM> g = cg::tiled_partition<MAX_SG_NUM>(b);
    sub_group sg = item_ct1.get_sub_group();

    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    // int gid = id / MAX_SG_NUM;
    int gid = id / MAX_SG_NUM;

    float vals_arr[NORM_REG];

    residual += (row * row_stride);
    vals += (row * row_stride);

    float sum = 0.f;
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = residual[i * iteration_stride + id];
        sum += vals_arr[i];
    }
    if (high_index < row_stride) {
        vals_arr[iterations] = residual[high_index];
        sum += vals_arr[iterations];
        iterations++;
    }

    // for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }
    for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += sg.shuffle_down(sum, i); }

    if (sg.get_local_id() == 0) shr[gid] = sum;

    item_ct1.barrier();

    if (sg.get_local_id() < (iteration_stride >> 5)) sum = shr[sg.get_local_id()];

#if !defined(__STOCHASTIC_MODE__)
    item_ct1.barrier();
#endif

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { sum += sg.shuffle_down(sum, i); }

    // sum = g.shfl(sum, 0);
    sum = sg.shuffle(sum, 0);
    float mean = sum / row_stride;
    // if (training)
    //     if (g.thread_rank() == 0) means[row] = mean;
    if constexpr (is_mean) {
        if (training)
            if (sg.get_local_id() == 0) means[row] = mean;
    }
    float variance = 0.f;
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] -= mean;
        variance += vals_arr[i] * vals_arr[i];
    }

    // for (int i = 1; i < 32; i *= 2) { variance += g.shfl_down(variance, i); }
    for (int i = 1; i < MAX_SG_NUM; i *= 2) { variance += sg.shuffle_down(variance, i); }

    // if (g.thread_rank() == 0) shr[gid] = variance;
    if (sg.get_local_id() == 0) shr[gid] = variance;

    item_ct1.barrier();

    if (sg.get_local_id() < (iteration_stride >> 5)) variance = shr[sg.get_local_id()];

#ifndef __STOCHASTIC_MODE__

    item_ct1.barrier();
#endif

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) {
        variance += sg.shuffle_down(variance, i);
    }
    variance = sg.shuffle(variance, 0);
    variance /= row_stride;
    variance += epsilon;

    if (training)
        if (sg.get_local_id() == 0) vars[row] = variance;

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = vals_arr[i] * rsqrt(variance);
        vals_arr[i] =
            vals_arr[i] * gamma[i * iteration_stride + id] + beta[i * iteration_stride + id];
        vals[i * iteration_stride + id] = vals_arr[i];
    }
    if ((high_index) < row_stride) {
        vals_arr[iterations] = vals_arr[iterations] * rsqrt(variance);
        vals_arr[iterations] = vals_arr[iterations] * gamma[high_index] + beta[high_index];
        vals[high_index] = vals_arr[iterations];
    }
}

template <bool is_mean>
void fused_bias_residual_layer_norm(bf16* vals,
                                    const bf16* residual,
                                    const bf16* gamma,
                                    const bf16* beta,
                                    float epsilon,
                                    bool preLayerNorm,
                                    bool training,
                                    bf16* vars,
                                    bf16* means,
                                    int row_stride,
                                    nd_item<3> item_ct1,
                                    float* shr)
{
    int iteration_stride = item_ct1.get_local_range(2);
    int iterations = row_stride / iteration_stride;

    // sycl::group<3> b = item_ct1.get_group();
    // cg::thread_block_tile<MAX_SG_NUM> g = cg::tiled_partition<MAX_SG_NUM>(b);
    sub_group sg = item_ct1.get_sub_group();

    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    // int gid = id / MAX_SG_NUM;
    int gid = id / MAX_SG_NUM;

    float vals_arr[NORM_REG];

    ushort* vals_cast = reinterpret_cast<ushort*>(vals);
    const ushort* residual_cast = reinterpret_cast<const ushort*>(residual);
    const ushort* gamma_cast = reinterpret_cast<const ushort*>(gamma);
    const ushort* beta_cast = reinterpret_cast<const ushort*>(beta);
    ushort* vars_cast = reinterpret_cast<ushort*>(vars);
    ushort* means_cast = reinterpret_cast<ushort*>(means);

    residual_cast += (row * row_stride);
    vals_cast += (row * row_stride);

    float sum = 0.f;
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = float(residual_cast[i * iteration_stride + id]);
        sum += vals_arr[i];
    }
    if (high_index < row_stride) {
        vals_arr[iterations] = float(residual_cast[high_index]);
        sum += vals_arr[iterations];
        iterations++;
    }

    for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += sg.shuffle_down(sum, i); }

    if (sg.get_local_id() == 0) shr[gid] = sum;

    item_ct1.barrier();

    // if (g.thread_rank() < (iteration_stride >> 5)) sum = shr[g.thread_rank()];
    if (sg.get_local_id() < (iteration_stride >> 5)) sum = shr[sg.get_local_id()];

#if !defined(__STOCHASTIC_MODE__)

    item_ct1.barrier();
#endif

    // for (int i = 1; i < (iteration_stride >> 5); i *= 2) { sum +=
    // g.shfl_down(sum, i); }
    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { sum += sg.shuffle_down(sum, i); }

    // sum = g.shfl(sum, 0);
    sum = sg.shuffle(sum, 0);
    float mean = sum / row_stride;
    // if (training)
    //     if (g.thread_rank() == 0) means[row] = mean;
    if constexpr (is_mean) {
        if (training)
            if (sg.get_local_id() == 0) means_cast[row] = bf16(mean);
    }
    float variance = 0.f;
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] -= mean;
        variance += vals_arr[i] * vals_arr[i];
    }

    // for (int i = 1; i < 32; i *= 2) { variance += g.shfl_down(variance, i); }
    for (int i = 1; i < MAX_SG_NUM; i *= 2) { variance += sg.shuffle_down(variance, i); }

    // if (g.thread_rank() == 0) shr[gid] = variance;
    if (sg.get_local_id() == 0) shr[gid] = variance;

    item_ct1.barrier();

    // if (g.thread_rank() < (iteration_stride >> 5)) variance =
    // shr[g.thread_rank()];
    if (sg.get_local_id() < (iteration_stride >> 5)) variance = shr[sg.get_local_id()];

#ifndef __STOCHASTIC_MODE__

    item_ct1.barrier();
#endif

    // for (int i = 1; i < (iteration_stride >> 5); i *= 2) { variance +=
    // g.shfl_down(variance, i); }
    for (int i = 1; i < (iteration_stride >> 5); i *= 2) {
        variance += sg.shuffle_down(variance, i);
    }
    // variance = g.shfl(variance, 0);
    variance = sg.shuffle(variance, 0);
    variance /= row_stride;
    variance += epsilon;
    // if (training)
    //     if (g.thread_rank() == 0) vars[row] = variance;
    if (training)
        if (sg.get_local_id() == 0) vars_cast[row] = bf16(variance);

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = vals_arr[i] * rsqrt(variance);
        vals_arr[i] = vals_arr[i] * float(gamma_cast[i * iteration_stride + id]) +
                      float(beta_cast[i * iteration_stride + id]);
        vals_cast[i * iteration_stride + id] = bf16(vals_arr[i]);
    }
    if ((high_index) < row_stride) {
        vals_arr[iterations] = vals_arr[iterations] * rsqrt(variance);
        vals_arr[iterations] =
            vals_arr[iterations] * float(gamma[high_index]) + float(beta[high_index]);
        vals_cast[high_index] = bf16(vals_arr[iterations]);
    }
}

template <bool is_mean>
void fused_bias_residual_layer_norm(half* vals,
                                    const half* residual,
                                    const half* gamma,
                                    const half* beta,
                                    float epsilon,
                                    bool preLayerNorm,
                                    bool training,
                                    half* vars,
                                    half* means,
                                    int row_stride,
                                    nd_item<3> item_ct1,
                                    float* shr)
{
    int iteration_stride = item_ct1.get_local_range(2);
    int iterations = row_stride / iteration_stride;

    // cg::thread_block b = cg::this_thread_block();
    // cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);
    sub_group sg = item_ct1.get_sub_group();

    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    // int gid = id >> 5;
    int gid = id / MAX_SG_NUM;

    float2 vals_f[NORM_REG];

    half2* vals_cast = reinterpret_cast<half2*>(vals);
    const half2* residual_cast = reinterpret_cast<const half2*>(residual);

    residual_cast += (row * row_stride);
    vals_cast += (row * row_stride);

    float sum = 0.f;
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    // for (int i = 0; i < iterations; i++) {
    //     vals_f[i] = __half22float2(residual_cast[i * iteration_stride + id]);
    //     sum += vals_f[i].x;
    //     sum += vals_f[i].y;
    // }
    for (int i = 0; i < iterations; i++) {
        vals_f[i] = residual_cast[i * iteration_stride + id]
                        .convert<float>();  // __half22float2(residual_cast[i *
                                            // iteration_stride + id]);
        sum += vals_f[i].x();
        sum += vals_f[i].y();
    }
    // if ((high_index) < row_stride) {
    //     vals_f[iterations] = __half22float2(residual_cast[high_index]);
    //     sum += vals_f[iterations].x;
    //     sum += vals_f[iterations].y;
    //     iterations++;
    // }
    if ((high_index) < row_stride) {
        vals_f[iterations] = residual_cast[high_index]
                                 .convert<float>();  // __half22float2(residual_cast[high_index]);
        sum += vals_f[iterations].x();
        sum += vals_f[iterations].y();
        iterations++;
    }

    // for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }
    for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += sg.shuffle_down(sum, i); }

    // if (g.thread_rank() == 0) shr[gid] = sum;
    if (sg.get_local_id() == 0) shr[gid] = sum;

    item_ct1.barrier();

    // if (g.thread_rank() < (iteration_stride >> 5)) sum = shr[g.thread_rank()];
    if (sg.get_local_id() < (iteration_stride >> 5)) sum = shr[sg.get_local_id()];

#ifndef __STOCHASTIC_MODE__
    // b.sync();
    item_ct1.barrier();
#endif

    // for (int i = 1; i < (iteration_stride >> 5); i *= 2) { sum +=
    // g.shfl_down(sum, i); }
    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { sum += sg.shuffle_down(sum, i); }
    // sum = g.shfl(sum, 0);
    sum = sg.shuffle(sum, 0);
    float mean = sum / (row_stride * 2);

    float variance = 0.f;
    for (int i = 0; i < iterations; i++) {
        vals_f[i].x() -= mean;
        vals_f[i].y() -= mean;
        variance += vals_f[i].x() * vals_f[i].x();
        variance += vals_f[i].y() * vals_f[i].y();
    }

    // for (int i = 1; i < 32; i *= 2) { variance += g.shfl_down(variance, i); }
    for (int i = 1; i < MAX_SG_NUM; i *= 2) { variance += sg.shuffle_down(variance, i); }

    // if (g.thread_rank() == 0) shr[gid] = variance;
    if (sg.get_local_id() == 0) shr[gid] = variance;

    // b.sync();
    item_ct1.barrier();

    // if (g.thread_rank() < (iteration_stride >> 5)) variance =
    // shr[g.thread_rank()];
    if (sg.get_local_id() < (iteration_stride >> 5)) variance = shr[sg.get_local_id()];

#ifndef __STOCHASTIC_MODE__
    // b.sync();
    item_ct1.barrier();
#endif

    // for (int i = 1; i < (iteration_stride >> 5); i *= 2) { variance +=
    // g.shfl_down(variance, i); }
    for (int i = 1; i < (iteration_stride >> 5); i *= 2) {
        variance += sg.shuffle_down(variance, i);
    }
    // variance = g.shfl(variance, 0);
    variance = sg.shuffle(variance, 0);
    variance /= (row_stride * 2);
    variance += epsilon;

    half2 variance_h =
        vec<float, 2>({variance, variance}).convert<half>();  // __float2half2_rn(variance);
    const half2* gamma_cast = reinterpret_cast<const half2*>(gamma);
    const half2* beta_cast = reinterpret_cast<const half2*>(beta);

    // if (training && g.thread_rank() == 0) {
    //     vars[row] = __float2half(variance);
    //     means[row] = __float2half(mean);
    // }
    if (training && sg.get_local_id() == 0) {
        vars[row] = vec<float, 1>(variance).convert<half>();  // __float2half(variance);
        if constexpr (is_mean) {
            means[row] = vec<float, 1>(mean).convert<half>();  // __float2half(mean);
        }
    }
    iterations = row_stride / iteration_stride;
    // for (int i = 0; i < iterations; i++) {
    //     half2 vals_arr = __float22half2_rn(vals_f[i]);
    //     vals_arr = vals_arr * h2rsqrt(variance_h);
    //     vals_arr =
    //         vals_arr * gamma_cast[i * iteration_stride + id] + beta_cast[i *
    //         iteration_stride + id];
    //     vals_cast[i * iteration_stride + id] = vals_arr;
    // }
    for (int i = 0; i < iterations; i++) {
        half2 vals_arr = vals_f[i].convert<half>();  // __float22half2_rn(vals_f[i]);
        vals_arr = vals_arr * rsqrt(variance_h);
        vals_arr =
            vals_arr * gamma_cast[i * iteration_stride + id] + beta_cast[i * iteration_stride + id];
        vals_cast[i * iteration_stride + id] = vals_arr;
    }
    // if ((high_index) < row_stride) {
    //     half2 vals_arr = __float22half2_rn(vals_f[iterations]);
    //     vals_arr = vals_arr * h2rsqrt(variance_h);
    //     vals_arr = vals_arr * gamma_cast[high_index] + beta_cast[high_index];
    //     vals_cast[high_index] = vals_arr;
    // }
    if ((high_index) < row_stride) {
        half2 vals_arr =
            vals_f[iterations].convert<half>();  // __float22half2_rn(vals_f[iterations]);
        vals_arr = vals_arr * rsqrt(variance_h);
        vals_arr = vals_arr * gamma_cast[high_index] + beta_cast[high_index];
        vals_cast[high_index] = vals_arr;
    }
}

template <typename T>
void launch_bias_residual_layer_norm(T* vals,
                                     const T* residual,
                                     const T* gamma,
                                     const T* beta,
                                     float epsilon,
                                     int batch_size,
                                     int hidden_dim,
                                     sycl::queue* stream,
                                     bool preLayerNorm,
                                     bool training,
                                     T* vars,
                                     T* means)
{
    int threads = THREADS;

    sycl::range<3> grid_dim(1, 1, batch_size);

    if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 1;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 2;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    sycl::range<3> block_dim(1, 1, threads);

    stream->submit([&](sycl::handler& cgh) {
        sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::access::target::local>
            shr_acc_ct1(sycl::range<1>(MAX_SG_NUM /*MAX_WARP_NUM*/), cgh);
        cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             fused_bias_residual_layer_norm<true>(vals,
                                                                  residual,
                                                                  gamma,
                                                                  beta,
                                                                  epsilon,
                                                                  preLayerNorm,
                                                                  training,
                                                                  vars,
                                                                  means,
                                                                  hidden_dim,
                                                                  item_ct1,
                                                                  shr_acc_ct1.get_pointer());
                         });
    });
}

template void launch_bias_residual_layer_norm<float>(float* vals,
                                                     const float* residual,
                                                     const float* gamma,
                                                     const float* beta,
                                                     float epsilon,
                                                     int batch_size,
                                                     int hidden_dim,
                                                     sycl::queue* stream,
                                                     bool preLayerNorm,
                                                     bool training,
                                                     float* vars,
                                                     float* means);
template void launch_bias_residual_layer_norm<bf16>(bf16* vals,
                                                    const bf16* residual,
                                                    const bf16* gamma,
                                                    const bf16* beta,
                                                    float epsilon,
                                                    int batch_size,
                                                    int hidden_dim,
                                                    sycl::queue* stream,
                                                    bool preLayerNorm,
                                                    bool training,
                                                    bf16* vars,
                                                    bf16* means);
template <>
void launch_bias_residual_layer_norm<half>(half* vals,
                                           const half* residual,
                                           const half* gamma,
                                           const half* beta,
                                           float epsilon,
                                           int batch_size,
                                           int hidden_dim,
                                           queue* stream,
                                           bool preLayerNorm,
                                           bool training,
                                           half* vars,
                                           half* means)
{
    int threads = 128;

    range<3> grid_dim(1, 1, batch_size);

    if (hidden_dim > 8192 && hidden_dim <= 16384)
        threads <<= 1;
    else if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 2;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 3;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    range<3> block_dim(1, 1, threads);

    stream->submit([&](handler& cgh) {
        sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::access::target::local>
            shr_acc_ct1(sycl::range<1>(MAX_SG_NUM /*MAX_WARP_NUM*/), cgh);
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             fused_bias_residual_layer_norm<true>(vals,
                                                                  residual,
                                                                  gamma,
                                                                  beta,
                                                                  epsilon,
                                                                  preLayerNorm,
                                                                  training,
                                                                  vars,
                                                                  means,
                                                                  hidden_dim / 2,
                                                                  item_ct1,
                                                                  shr_acc_ct1.get_pointer());
                         });
    });
}

/*
  To tune this launch the following restrictions must be met:

  For float:
  row_stride == hidden_size
  threads * iterations == row_stride
  threads is in [32, 64, 128, 256, 512, 1024]

  For half:
  row_stride == hidden_size / 2
  threads * iterations == row_stride
  threads is in [32, 64, 128, 256, 512, 1024]

*/

template <typename T>
void launch_bias_residual_layer_norm(T* vals,
                                     const T* residual,
                                     const T* gamma,
                                     const T* beta,
                                     float epsilon,
                                     int batch_size,
                                     int hidden_dim,
                                     queue* stream,
                                     bool preLayerNorm,
                                     bool training,
                                     T* vars)
{
    int threads = THREADS;

    range<3> grid_dim(1, 1, batch_size);

    // There are some limitations to call below functions, now just enumerate the
    // situations.

    if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 1;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 2;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    range<3> block_dim(1, 1, threads);

    stream->submit([&](handler& cgh) {
        accessor<float, 1, access_mode::read_write, access::target::local> shr_acc_ct1(
            range<1>(MAX_SG_NUM /*MAX_WARP_NUM*/), cgh);

        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             fused_bias_residual_layer_norm<false>(vals,
                                                                   residual,
                                                                   gamma,
                                                                   beta,
                                                                   epsilon,
                                                                   preLayerNorm,
                                                                   training,
                                                                   vars,
                                                                   nullptr,
                                                                   hidden_dim,
                                                                   item_ct1,
                                                                   shr_acc_ct1.get_pointer());
                         });
    });
}

template void launch_bias_residual_layer_norm<float>(float* vals,
                                                     const float* residual,
                                                     const float* gamma,
                                                     const float* beta,
                                                     float epsilon,
                                                     int batch_size,
                                                     int hidden_dim,
                                                     queue* stream,
                                                     bool preLayerNorm,
                                                     bool training,
                                                     float* vars);
template void launch_bias_residual_layer_norm<bf16>(bf16* vals,
                                                    const bf16* residual,
                                                    const bf16* gamma,
                                                    const bf16* beta,
                                                    float epsilon,
                                                    int batch_size,
                                                    int hidden_dim,
                                                    queue* stream,
                                                    bool preLayerNorm,
                                                    bool training,
                                                    bf16* vars);
template <>
void launch_bias_residual_layer_norm<half>(half* vals,
                                           const half* residual,
                                           const half* gamma,
                                           const half* beta,
                                           float epsilon,
                                           int batch_size,
                                           int hidden_dim,
                                           queue* stream,
                                           bool preLayerNorm,
                                           bool training,
                                           half* vars)
{
    int threads = 128;

    range<3> grid_dim(1, 1, batch_size);

    // There are some limitations to call below functions, now just enumerate the
    // situations.

    if (hidden_dim > 8192 && hidden_dim <= 16384)
        threads <<= 1;
    else if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 2;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 3;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    range<3> block_dim(1, 1, threads);

    stream->submit([&](handler& cgh) {
        sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::access::target::local>
            shr_acc_ct1(sycl::range<1>(MAX_SG_NUM /*MAX_WARP_NUM*/), cgh);
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             fused_bias_residual_layer_norm<false>(vals,
                                                                   residual,
                                                                   gamma,
                                                                   beta,
                                                                   epsilon,
                                                                   preLayerNorm,
                                                                   training,
                                                                   vars,
                                                                   nullptr,
                                                                   hidden_dim / 2,
                                                                   item_ct1,
                                                                   shr_acc_ct1.get_pointer());
                         });
    });
}

/* Normalize Gamma & Betta gradients
 * Compute gradients using either X_hat or
 * normalize input (invertible).
 * Combine transpose with gradients computation.
 */

template <typename T>
void LayerNormBackward1(const T* out_grad,
                        const T* vals_hat,
                        const T* gamma,
                        const T* betta,
                        T* gamma_grad,
                        T* betta_grad,
                        int rows,
                        int width,
                        bool invertible,
                        nd_item<3> item_ct1,
                        float* betta_buffer,
                        float* gamma_buffer)
{
    // group<3> b = item_ct1.get_group();
    // cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);
    sub_group sg = item_ct1.get_sub_group();

    int idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
    int offset = item_ct1.get_local_id(1) * width + idx;
    int y_stride = width * TILE_DIM;

    float betta_reg = (invertible ? (float)betta[idx] : 0.0f);
    float gamma_reg = (float)gamma[idx];

    // Loop across matrix height
    float betta_tmp = 0;
    float gamma_tmp = 0;
    for (int r = item_ct1.get_local_id(1); r < rows; r += TILE_DIM) {
        float grad = (float)out_grad[offset];
        float val = (invertible ? ((float)vals_hat[offset] - betta_reg) / gamma_reg
                                : (float)vals_hat[offset]);
        betta_tmp += grad;
        gamma_tmp += (val * grad);

        offset += y_stride;
    }

    // betta_buffer[item_ct1.get_local_id(2)][item_ct1.get_local_id(1)] =
    // betta_tmp; gamma_buffer[item_ct1.get_local_id(2)][item_ct1.get_local_id(1)]
    // = gamma_tmp;
    betta_buffer[item_ct1.get_local_id(2) * MAX_SG_NUM1 + item_ct1.get_local_id(1)] = betta_tmp;
    gamma_buffer[item_ct1.get_local_id(2) * MAX_SG_NUM1 + item_ct1.get_local_id(1)] = gamma_tmp;

    item_ct1.barrier();

    // Sum the shared buffer.
    // float s1 =
    // betta_buffer[item_ct1.get_local_id(1)][item_ct1.get_local_id(2)]; float s2
    // = gamma_buffer[item_ct1.get_local_id(1)][item_ct1.get_local_id(2)];
    float s1 = betta_buffer[item_ct1.get_local_id(1) * MAX_SG_NUM1 + item_ct1.get_local_id(2)];
    float s2 = gamma_buffer[item_ct1.get_local_id(1) * MAX_SG_NUM1 + item_ct1.get_local_id(2)];

#ifndef __STOCHASTIC_MODE__

    item_ct1.barrier();
#endif

    // for (int i = 1; i < TILE_DIM; i <<= 1) {
    //     s1 += g.shfl_down(s1, i);
    //     s2 += g.shfl_down(s2, i);
    // }
    for (int i = 1; i < TILE_DIM; i <<= 1) {
        s1 += sg.shuffle_down(s1, i);
        s2 += sg.shuffle_down(s2, i);
    }

    if (item_ct1.get_local_id(2) == 0) {
        int pos = item_ct1.get_group(2) * TILE_DIM + item_ct1.get_local_id(1);
        betta_grad[pos] = s1;
        gamma_grad[pos] = s2;
    }
}

/* Normalize Gamma & Betta gradients
 * Compute gradients using either X_hat or
 * normalize input (invertible).
 * Combine transpose with gradients computation.
 */

template <>
void LayerNormBackward1<bf16>(const bf16* out_grad,
                              const bf16* vals_hat,
                              const bf16* gamma,
                              const bf16* betta,
                              bf16* gamma_grad,
                              bf16* betta_grad,
                              int rows,
                              int width,
                              bool invertible,
                              nd_item<3> item_ct1,
                              float* betta_buffer,
                              float* gamma_buffer)
{
    // group<3> b = item_ct1.get_group();
    // cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);
    sub_group sg = item_ct1.get_sub_group();

    int idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
    int offset = item_ct1.get_local_id(1) * width + idx;
    int y_stride = width * TILE_DIM;

    const ushort* out_grad_cast = reinterpret_cast<const ushort*>(out_grad);
    const ushort* vals_hat_cast = reinterpret_cast<const ushort*>(vals_hat);
    const ushort* gamma_cast = reinterpret_cast<const ushort*>(gamma);
    const ushort* betta_cast = reinterpret_cast<const ushort*>(betta);
    ushort* gamma_grad_cast = reinterpret_cast<ushort*>(gamma_grad);
    ushort* betta_grad_cast = reinterpret_cast<ushort*>(betta_grad);

    float betta_reg = (invertible ? float(betta_cast[idx]) : 0.0f);
    float gamma_reg = float(gamma_cast[idx]);

    // Loop across matrix height
    float betta_tmp = 0;
    float gamma_tmp = 0;
    for (int r = item_ct1.get_local_id(1); r < rows; r += TILE_DIM) {
        float grad = float(out_grad_cast[offset]);
        float val = (invertible ? (float(vals_hat_cast[offset]) - betta_reg) / gamma_reg
                                : float(vals_hat_cast[offset]));
        betta_tmp += grad;
        gamma_tmp += (val * grad);

        offset += y_stride;
    }

    // betta_buffer[item_ct1.get_local_id(2)][item_ct1.get_local_id(1)] =
    // betta_tmp; gamma_buffer[item_ct1.get_local_id(2)][item_ct1.get_local_id(1)]
    // = gamma_tmp;
    betta_buffer[item_ct1.get_local_id(2) * MAX_SG_NUM1 + item_ct1.get_local_id(1)] = betta_tmp;
    gamma_buffer[item_ct1.get_local_id(2) * MAX_SG_NUM1 + item_ct1.get_local_id(1)] = gamma_tmp;

    item_ct1.barrier();

    // Sum the shared buffer.
    // float s1 =
    // betta_buffer[item_ct1.get_local_id(1)][item_ct1.get_local_id(2)]; float s2
    // = gamma_buffer[item_ct1.get_local_id(1)][item_ct1.get_local_id(2)];
    float s1 = betta_buffer[item_ct1.get_local_id(1) * MAX_SG_NUM1 + item_ct1.get_local_id(2)];
    float s2 = gamma_buffer[item_ct1.get_local_id(1) * MAX_SG_NUM1 + item_ct1.get_local_id(2)];

#ifndef __STOCHASTIC_MODE__

    item_ct1.barrier();
#endif

    // for (int i = 1; i < TILE_DIM; i <<= 1) {
    //     s1 += g.shfl_down(s1, i);
    //     s2 += g.shfl_down(s2, i);
    // }
    for (int i = 1; i < TILE_DIM; i <<= 1) {
        s1 += sg.shuffle_down(s1, i);
        s2 += sg.shuffle_down(s2, i);
    }

    if (item_ct1.get_local_id(2) == 0) {
        int pos = item_ct1.get_group(2) * TILE_DIM + item_ct1.get_local_id(1);
        betta_grad_cast[pos] = bf16(s1);
        gamma_grad_cast[pos] = bf16(s2);
    }
}

/* Normalize Gamma & Betta gradients
 * Compute gradients using the input to
 * the normalize.
 * Combine transpose with gradients computation.
 */

template <typename T>
void LayerNormBackward1(const T* out_grad,
                        const T* X_data,
                        const T* vars,
                        const T* means,
                        T* gamma_grad,
                        T* betta_grad,
                        int rows,
                        int width,
                        nd_item<3> item_ct1,
                        float* betta_buffer,
                        float* gamma_buffer)
{
    // group<3> b = item_ct1.get_group();
    // cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);
    sub_group sg = item_ct1.get_sub_group();

    int idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
    int offset = item_ct1.get_local_id(1) * width + idx;
    int y_stride = width * TILE_DIM;

    int pos = item_ct1.get_group(2) * TILE_DIM + item_ct1.get_local_id(1);
    // Loop across matrix height

    float betta_tmp = 0;
    float gamma_tmp = 0;
    for (int r = item_ct1.get_local_id(1); r < rows; r += TILE_DIM) {
        float grad = (float)out_grad[offset];
        float val = (float)X_data[offset];
        val = (val - (float)means[r]) * rsqrt((float)vars[r]);
        betta_tmp += grad;
        gamma_tmp += (val * grad);

        offset += y_stride;
    }

    betta_buffer[item_ct1.get_local_id(2) * MAX_SG_NUM1 + item_ct1.get_local_id(1)] = betta_tmp;
    gamma_buffer[item_ct1.get_local_id(2) * MAX_SG_NUM1 + item_ct1.get_local_id(1)] = gamma_tmp;

    item_ct1.barrier();

    // Sum the shared buffer.
    float s1 = betta_buffer[item_ct1.get_local_id(1) * MAX_SG_NUM1 + item_ct1.get_local_id(2)];
    float s2 = gamma_buffer[item_ct1.get_local_id(1) * MAX_SG_NUM1 + item_ct1.get_local_id(2)];

#ifndef __STOCHASTIC_MODE__

    item_ct1.barrier();
#endif

    // for (int i = 1; i < TILE_DIM; i <<= 1) {
    //     s1 += g.shfl_down(s1, i);
    //     s2 += g.shfl_down(s2, i);
    // }
    for (int i = 1; i < TILE_DIM; i <<= 1) {
        s1 += sg.shuffle_down(s1, i);
        s2 += sg.shuffle_down(s2, i);
    }

    if (item_ct1.get_local_id(2) == 0) {
        betta_grad[pos] = s1;
        gamma_grad[pos] = s2;
    }
}

template <>
void LayerNormBackward1<bf16>(const bf16* out_grad,
                              const bf16* X_data,
                              const bf16* vars,
                              const bf16* means,
                              bf16* gamma_grad,
                              bf16* betta_grad,
                              int rows,
                              int width,
                              nd_item<3> item_ct1,
                              float* betta_buffer,
                              float* gamma_buffer)
{
    // group<3> b = item_ct1.get_group();
    // cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);
    sub_group sg = item_ct1.get_sub_group();

    int idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
    int offset = item_ct1.get_local_id(1) * width + idx;
    int y_stride = width * TILE_DIM;

    int pos = item_ct1.get_group(2) * TILE_DIM + item_ct1.get_local_id(1);
    // Loop across matrix height

    const ushort* out_grad_cast = reinterpret_cast<const ushort*>(out_grad);
    const ushort* X_data_cast = reinterpret_cast<const ushort*>(X_data);
    const ushort* vars_cast = reinterpret_cast<const ushort*>(vars);
    const ushort* means_cast = reinterpret_cast<const ushort*>(means);
    ushort* gamma_grad_cast = reinterpret_cast<ushort*>(gamma_grad);
    ushort* betta_grad_cast = reinterpret_cast<ushort*>(betta_grad);

    float betta_tmp = 0;
    float gamma_tmp = 0;
    for (int r = item_ct1.get_local_id(1); r < rows; r += TILE_DIM) {
        float grad = float(out_grad_cast[offset]);
        float val = float(X_data_cast[offset]);
        val = (val - float(means_cast[r])) * rsqrt(float(vars_cast[r]));
        betta_tmp += grad;
        gamma_tmp += (val * grad);

        offset += y_stride;
    }

    betta_buffer[item_ct1.get_local_id(2) * MAX_SG_NUM1 + item_ct1.get_local_id(1)] = betta_tmp;
    gamma_buffer[item_ct1.get_local_id(2) * MAX_SG_NUM1 + item_ct1.get_local_id(1)] = gamma_tmp;

    item_ct1.barrier();

    // Sum the shared buffer.
    float s1 = betta_buffer[item_ct1.get_local_id(1) * MAX_SG_NUM1 + item_ct1.get_local_id(2)];
    float s2 = gamma_buffer[item_ct1.get_local_id(1) * MAX_SG_NUM1 + item_ct1.get_local_id(2)];

#ifndef __STOCHASTIC_MODE__

    item_ct1.barrier();
#endif

    // for (int i = 1; i < TILE_DIM; i <<= 1) {
    //     s1 += g.shfl_down(s1, i);
    //     s2 += g.shfl_down(s2, i);
    // }
    for (int i = 1; i < TILE_DIM; i <<= 1) {
        s1 += sg.shuffle_down(s1, i);
        s2 += sg.shuffle_down(s2, i);
    }

    if (item_ct1.get_local_id(2) == 0) {
        betta_grad_cast[pos] = bf16(s1);
        gamma_grad_cast[pos] = bf16(s2);
    }
}
/*

/* Backward Normalize (Input-Gradient)
* Using the means and variances from the input
* This type of backward is invertible!
* We do the backward using the X_hat (X - u) / sqrt(variance) or the output of
Normalization.
*/
template <bool is_fuseadd>
void LayerNormBackward2(const float* out_grad,
                        const float* out_grad_add,
                        const float* vals_hat,
                        const float* gamma,
                        const float* betta,
                        const float* vars,
                        float* inp_grad,
                        bool invertible,
                        int row_stride,
                        nd_item<3> item_ct1,
                        float* partialSum)
{
    int iteration_stride = item_ct1.get_local_range(2);
    int iterations = row_stride / iteration_stride;

    // group<3> b = item_ct1.get_group();
    // cg::thread_block_tile<MAX_SG_NUM> g = cg::tiled_partition<MAX_SG_NUM>(b);
    sub_group sg = item_ct1.get_sub_group();

    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    int wid = id / MAX_SG_NUM;
    int warp_num = (THREADS < row_stride ? THREADS : row_stride) / MAX_SG_NUM;

    out_grad += (row * row_stride);
    if constexpr (is_fuseadd) { out_grad_add += (row * row_stride); }
    vals_hat += (row * row_stride);
    inp_grad += (row * row_stride);

    float vals_arr[NORM_REG];
    float vals_hat_arr[NORM_REG];
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        float gamma_reg = gamma[i * iteration_stride + id];
        vals_arr[i] = out_grad[i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;
        vals_hat_arr[i] =
            (invertible ? (vals_hat[i * iteration_stride + id] - betta[i * iteration_stride + id]) /
                              gamma_reg
                        : vals_hat[i * iteration_stride + id]);
    }
    if ((high_index) < row_stride) {
        float gamma_reg = gamma[high_index];
        vals_arr[iterations] = out_grad[high_index];
        vals_arr[iterations] *= gamma_reg;
        vals_hat_arr[iterations] =
            (invertible ? (vals_hat[high_index] - betta[high_index]) / gamma_reg
                        : vals_hat[high_index]);
        iterations++;
    }

    float var_reg = vars[row];

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        sum +=
            vals_hat_arr[i] * vals_arr[i] * sqrt(var_reg);  // dval_hat = gamma * (x - u) * out_grad
        vals_arr[i] *= rsqrt(var_reg);  // dvar_inv = gamma * out_grad / sqrt(var)
    }

    // for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += g.shfl_down(sum, i); }
    for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += sg.shuffle_down(sum, i); }

    // if (g.thread_rank() == 0) partialSum[wid] = sum;
    if (sg.get_local_id() == 0) partialSum[wid] = sum;

    item_ct1.barrier();

    // if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];
    if (sg.get_local_id() < warp_num) sum = partialSum[sg.get_local_id()];

#ifndef __STOCHASTIC_MODE__

    item_ct1.barrier();
#endif

    // for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    for (int i = 1; i < warp_num; i *= 2) sum += sg.shuffle_down(sum, i);

    // sum = g.shfl(sum, 0);
    sum = sg.shuffle(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++) { vals_arr[i] += ((-sum * vals_hat_arr[i]) / var_reg); }

    sum = 0;
    for (int i = 0; i < iterations; i++) { sum += vals_arr[i]; }

    // for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += g.shfl_down(sum, i); }
    for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += sg.shuffle_down(sum, i); }

    // if (g.thread_rank() == 0) partialSum[wid] = sum;
    if (sg.get_local_id() == 0) partialSum[wid] = sum;

    item_ct1.barrier();

    // if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];
    if (sg.get_local_id() < warp_num) sum = partialSum[sg.get_local_id()];

#ifndef __STOCHASTIC_MODE__
    item_ct1.barrier();
#endif

    // for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    for (int i = 1; i < warp_num; i *= 2) sum += sg.shuffle_down(sum, i);
    // sum = g.shfl(sum, 0);
    sum = sg.shuffle(sum, 0);
    sum /= row_stride;

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++)
        if constexpr (is_fuseadd) {
            inp_grad[i * iteration_stride + id] =
                (vals_arr[i] - sum) + out_grad_add[i * iteration_stride + id];
        } else {
            inp_grad[i * iteration_stride + id] = (vals_arr[i] - sum);
        }
    if ((high_index) < row_stride)
        if constexpr (is_fuseadd) {
            inp_grad[high_index] = (vals_arr[iterations] - sum) + out_grad_add[high_index];
        } else {
            inp_grad[high_index] = (vals_arr[iterations] - sum);
        }
}

template <bool is_fuseadd>
void LayerNormBackward2(const bf16* out_grad,
                        const bf16* out_grad_add,
                        const bf16* vals_hat,
                        const bf16* gamma,
                        const bf16* betta,
                        const bf16* vars,
                        bf16* inp_grad,
                        bool invertible,
                        int row_stride,
                        nd_item<3> item_ct1,
                        float* partialSum)
{
    int iteration_stride = item_ct1.get_local_range(2);
    int iterations = row_stride / iteration_stride;

    // group<3> b = item_ct1.get_group();
    // cg::thread_block_tile<MAX_SG_NUM> g = cg::tiled_partition<MAX_SG_NUM>(b);
    sub_group sg = item_ct1.get_sub_group();

    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    int wid = id / MAX_SG_NUM;
    int warp_num = (THREADS < row_stride ? THREADS : row_stride) / MAX_SG_NUM;

    const ushort* out_grad_cast = reinterpret_cast<const ushort*>(out_grad);
    const ushort* out_grad_add_cast = reinterpret_cast<const ushort*>(out_grad_add);
    const ushort* vals_hat_cast = reinterpret_cast<const ushort*>(vals_hat);
    const ushort* gamma_cast = reinterpret_cast<const ushort*>(gamma);
    const ushort* betta_cast = reinterpret_cast<const ushort*>(betta);
    const ushort* vars_cast = reinterpret_cast<const ushort*>(vars);
    ushort* inp_grad_cast = reinterpret_cast<ushort*>(inp_grad);

    out_grad_cast += (row * row_stride);
    if constexpr (is_fuseadd) { out_grad_add_cast += (row * row_stride); }
    vals_hat_cast += (row * row_stride);
    inp_grad_cast += (row * row_stride);

    float vals_arr[NORM_REG];
    float vals_hat_arr[NORM_REG];
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        float gamma_reg = float(gamma_cast[i * iteration_stride + id]);
        vals_arr[i] = float(out_grad_cast[i * iteration_stride + id]);
        vals_arr[i] *= gamma_reg;
        vals_hat_arr[i] = (invertible ? (float(vals_hat_cast[i * iteration_stride + id]) -
                                         float(betta_cast[i * iteration_stride + id])) /
                                            gamma_reg
                                      : float(vals_hat_cast[i * iteration_stride + id]));
    }
    if ((high_index) < row_stride) {
        float gamma_reg = float(gamma_cast[high_index]);
        vals_arr[iterations] = float(out_grad_cast[high_index]);
        vals_arr[iterations] *= gamma_reg;
        vals_hat_arr[iterations] =
            (invertible
                 ? (float(vals_hat_cast[high_index]) - float(betta_cast[high_index])) / gamma_reg
                 : float(vals_hat_cast[high_index]));
        iterations++;
    }

    float var_reg = float(vars_cast[row]);

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        sum +=
            vals_hat_arr[i] * vals_arr[i] * sqrt(var_reg);  // dval_hat = gamma * (x - u) * out_grad
        vals_arr[i] *= rsqrt(var_reg);  // dvar_inv = gamma * out_grad / sqrt(var)
    }

    // for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += g.shfl_down(sum, i); }
    for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += sg.shuffle_down(sum, i); }

    // if (g.thread_rank() == 0) partialSum[wid] = sum;
    if (sg.get_local_id() == 0) partialSum[wid] = sum;

    item_ct1.barrier();

    // if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];
    if (sg.get_local_id() < warp_num) sum = partialSum[sg.get_local_id()];

#ifndef __STOCHASTIC_MODE__
    item_ct1.barrier();
#endif

    // for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    for (int i = 1; i < warp_num; i *= 2) sum += sg.shuffle_down(sum, i);

    // sum = g.shfl(sum, 0);
    sum = sg.shuffle(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++) { vals_arr[i] += ((-sum * vals_hat_arr[i]) / var_reg); }

    sum = 0;
    for (int i = 0; i < iterations; i++) { sum += vals_arr[i]; }

    // for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += g.shfl_down(sum, i); }
    for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += sg.shuffle_down(sum, i); }

    // if (g.thread_rank() == 0) partialSum[wid] = sum;
    if (sg.get_local_id() == 0) partialSum[wid] = sum;

    item_ct1.barrier();

    // if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];
    if (sg.get_local_id() < warp_num) sum = partialSum[sg.get_local_id()];

#ifndef __STOCHASTIC_MODE__
    item_ct1.barrier();
#endif

    // for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    for (int i = 1; i < warp_num; i *= 2) sum += sg.shuffle_down(sum, i);
    // sum = g.shfl(sum, 0);
    sum = sg.shuffle(sum, 0);
    sum /= row_stride;

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++)
        if constexpr (is_fuseadd) {
            inp_grad_cast[i * iteration_stride + id] =
                bf16((vals_arr[i] - sum) + float(out_grad_add_cast[i * iteration_stride + id]));
        } else {
            inp_grad_cast[i * iteration_stride + id] = bf16((vals_arr[i] - sum));
        }
    if ((high_index) < row_stride)
        if constexpr (is_fuseadd) {
            inp_grad_cast[high_index] =
                bf16((vals_arr[iterations] - sum) + float(out_grad_add_cast[high_index]));
        } else {
            inp_grad_cast[high_index] = bf16((vals_arr[iterations] - sum));
        }
}

template <bool is_fuseadd>
void LayerNormBackward2(const half* out_grad,
                        const half* out_grad_add,
                        const half* vals_hat,
                        const half* gamma,
                        const half* betta,
                        const half* vars,
                        half* inp_grad,
                        bool invertible,
                        int row_stride,
                        nd_item<3> item_ct1,
                        float* partialSum)
{
    int iteration_stride = item_ct1.get_local_range(2);
    int iterations = row_stride / iteration_stride;

    // group<3> b = item_ct1.get_group();
    // cg::thread_block_tile<MAX_SG_NUM> g = cg::tiled_partition<MAX_SG_NUM>(b);
    sub_group sg = item_ct1.get_sub_group();

    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    int wid = id / MAX_SG_NUM;
    int warp_num = (iteration_stride < row_stride ? iteration_stride : row_stride) / MAX_SG_NUM;

    half2 vals_arr[NORM_REG];
    float2 vals_arr_f[NORM_REG];
    half2 vals_hat_arr[NORM_REG];

    half2* inp_grad_h = reinterpret_cast<half2*>(inp_grad);
    const half2* out_grad_h = reinterpret_cast<const half2*>(out_grad);
    const half2* out_grad_add_h = reinterpret_cast<const half2*>(out_grad_add);
    const half2* vals_hat_h = reinterpret_cast<const half2*>(vals_hat);

    inp_grad_h += (row * row_stride);
    out_grad_h += (row * row_stride);
    if constexpr (is_fuseadd) { out_grad_add_h += (row * row_stride); }
    vals_hat_h += (row * row_stride);

    const half2* gamma_h = reinterpret_cast<const half2*>(gamma);
    const half2* betta_h = (invertible ? reinterpret_cast<const half2*>(betta) : nullptr);
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        half2 gamma_reg = gamma_h[i * iteration_stride + id];
        vals_arr[i] = out_grad_h[i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;
        vals_hat_arr[i] =
            (invertible
                 ? (vals_hat_h[i * iteration_stride + id] - betta_h[i * iteration_stride + id]) /
                       gamma_reg
                 : vals_hat_h[i * iteration_stride + id]);
    }
    if ((high_index) < row_stride) {
        half2 gamma_reg = gamma_h[high_index];
        vals_arr[iterations] = out_grad_h[high_index];
        vals_arr[iterations] *= gamma_reg;
        vals_hat_arr[iterations] =
            (invertible ? (vals_hat_h[high_index] - betta_h[high_index]) / gamma_reg
                        : vals_hat_h[high_index]);
        iterations++;
    }
    half var_h = vars[row];
    half2 var_reg = half2{var_h, var_h};

    float sum = 0.f;
    for (int i = 0; i < iterations; i++) {
        half2 result_h = (vals_hat_arr[i] * vals_arr[i] * sqrt(var_reg));
        float2 result_f = result_h.convert<float, rounding_mode::automatic>();
        sum += result_f.x();
        sum += result_f.y();
        vals_arr[i] *= rsqrt(var_reg);
    }

    // for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += g.shfl_down(sum, i); }
    for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += sg.shuffle_down(sum, i); }

    // if (g.thread_rank() == 0) partialSum[wid] = sum;
    if (sg.get_local_id() == 0) partialSum[wid] = sum;

    item_ct1.barrier();

    // if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];
    if (sg.get_local_id() < warp_num) sum = partialSum[sg.get_local_id()];

#ifndef __STOCHASTIC_MODE__
    item_ct1.barrier();
#endif

    // for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    for (int i = 1; i < warp_num; i *= 2) sum += sg.shuffle_down(sum, i);

    // sum = g.shfl(sum, 0);
    sum = sg.shuffle(sum, 0);
    sum /= (2 * row_stride);
    half2 sum_h = float2{sum, sum}.convert<half>();

    for (int i = 0; i < iterations; i++) {
        half2 temp = ((-sum_h * vals_hat_arr[i]) / (var_reg));
        vals_arr_f[i] = vals_arr[i].convert<float, rounding_mode::automatic>();
        float2 temp_f = temp.convert<float, rounding_mode::automatic>();
        vals_arr_f[i].x() += temp_f.x();
        vals_arr_f[i].y() += temp_f.y();
    }
    sum = 0.f;

    for (int i = 0; i < iterations; i++) {
        sum += (vals_arr_f[i].x());
        sum += (vals_arr_f[i].y());
    }

    // for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += g.shfl_down(sum, i); }
    for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += sg.shuffle_down(sum, i); }

    // if (g.thread_rank() == 0) partialSum[wid] = sum;
    if (sg.get_local_id() == 0) partialSum[wid] = sum;

    item_ct1.barrier();

    // if (sg.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];
    if (sg.get_local_id() < warp_num) sum = partialSum[sg.get_local_id()];

#ifndef __STOCHASTIC_MODE__
    item_ct1.barrier();
#endif

    // for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    for (int i = 1; i < warp_num; i *= 2) sum += sg.shuffle_down(sum, i);

    // sum = sg.shfl(sum, 0);
    sum = sg.shuffle(sum, 0);
    sum /= (2 * row_stride);

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) {
        vals_arr_f[i].x() -= sum;
        vals_arr_f[i].y() -= sum;
        half2 temp = vals_arr_f[i].convert<half>();
        if constexpr (is_fuseadd) {
            inp_grad_h[i * iteration_stride + id] =
                temp + out_grad_add_h[i * iteration_stride + id];
        } else {
            inp_grad_h[i * iteration_stride + id] = temp;
        }
    }
    if ((high_index) < row_stride) {
        vals_arr_f[iterations].x() -= sum;
        vals_arr_f[iterations].y() -= sum;
        half2 temp = vals_arr_f[iterations].convert<half>();
        if constexpr (is_fuseadd) {
            inp_grad_h[high_index] = temp + out_grad_add_h[high_index];
        } else {
            inp_grad_h[high_index] = temp;
        }
    }
}

template <typename T>
void launch_layerNorm_backward(const T* out_grad,
                               const T* vals_hat,
                               const T* vars,
                               const T* gamma,
                               T* gamma_grad,
                               T* betta_grad,
                               T* inp_grad,
                               int batch,
                               int hidden_dim,
                               queue* stream[2],
                               bool invertible,
                               const T* betta)
{
    int threads = THREADS;

    range<3> grid_dim(1, 1, hidden_dim / TILE_DIM);
    range<3> block_dim(1, TILE_DIM, TILE_DIM);

    stream[0]->submit([&](handler& cgh) {
        accessor<float, 2, access_mode::read_write, access::target::local> betta_buffer(
            range<2>(MAX_SG_NUM /*MAX_WARP_NUM*/, MAX_SG_NUM1), cgh);
        accessor<float, 2, access_mode::read_write, access::target::local> gamma_buffer(
            range<2>(MAX_SG_NUM /*MAX_WARP_NUM*/, MAX_SG_NUM1), cgh);

        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             LayerNormBackward1<T>(out_grad,
                                                   vals_hat,
                                                   gamma,
                                                   betta,
                                                   gamma_grad,
                                                   betta_grad,
                                                   batch,
                                                   hidden_dim,
                                                   invertible,
                                                   item_ct1,
                                                   betta_buffer.get_pointer(),
                                                   gamma_buffer.get_pointer());
                         });
    });
    // LayerNormBackward1<float><<<grid_dim, block_dim, 0, stream[0]>>>(
    //     out_grad, vals_hat, gamma, betta, gamma_grad, betta_grad, batch,
    //     hidden_dim, invertible);
    range<3> grid_dim2(1, 1, batch);

    if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 1;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 2;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    range<3> block_dim2(1, 1, threads);

    stream[1]->submit([&](handler& cgh) {
        accessor<float, 1, access_mode::read_write, access::target::local> partialSum_acc_ct1(
            range<1>(MAX_SG_NUM /*MAX_WARP_NUM*/), cgh);

        cgh.parallel_for(nd_range<3>(grid_dim2 * block_dim2, block_dim2),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             LayerNormBackward2<false>(out_grad,
                                                       nullptr,
                                                       vals_hat,
                                                       gamma,
                                                       betta,
                                                       vars,
                                                       inp_grad,
                                                       invertible,
                                                       hidden_dim,
                                                       item_ct1,
                                                       partialSum_acc_ct1.get_pointer());
                         });
    });
}

template void launch_layerNorm_backward<float>(const float* out_grad,
                                               const float* vals_hat,
                                               const float* vars,
                                               const float* gamma,
                                               float* gamma_grad,
                                               float* betta_grad,
                                               float* inp_grad,
                                               int batch,
                                               int hidden_dim,
                                               queue* stream[2],
                                               bool invertible,
                                               const float* betta);

template void launch_layerNorm_backward<bf16>(const bf16* out_grad,
                                              const bf16* vals_hat,
                                              const bf16* vars,
                                              const bf16* gamma,
                                              bf16* gamma_grad,
                                              bf16* betta_grad,
                                              bf16* inp_grad,
                                              int batch,
                                              int hidden_dim,
                                              queue* stream[2],
                                              bool invertible,
                                              const bf16* betta);

template <>
void launch_layerNorm_backward<half>(const half* out_grad,
                                     const half* vals_hat,
                                     const half* vars,
                                     const half* gamma,
                                     half* gamma_grad,
                                     half* betta_grad,
                                     half* inp_grad,
                                     int batch,
                                     int hidden_dim,
                                     queue* stream[2],
                                     bool invertible,
                                     const half* betta)
{
    int threads = THREADS;

    range<3> grid_dim(1, 1, hidden_dim / TILE_DIM);
    range<3> block_dim(1, TILE_DIM, TILE_DIM);

    // LayerNormBackward1<half><<<grid_dim, block_dim, 0, stream[0]>>>(
    //     out_grad, vals_hat, gamma, betta, gamma_grad, betta_grad, batch,
    //     hidden_dim, invertible);
    stream[0]->submit([&](handler& cgh) {
        accessor<float, 2, access_mode::read_write, access::target::local> betta_buffer(
            range<2>(MAX_SG_NUM /*MAX_WARP_NUM*/, MAX_SG_NUM1), cgh);
        accessor<float, 2, access_mode::read_write, access::target::local> gamma_buffer(
            range<2>(MAX_SG_NUM /*MAX_WARP_NUM*/, MAX_SG_NUM1), cgh);
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             LayerNormBackward1<half>(out_grad,
                                                      vals_hat,
                                                      gamma,
                                                      betta,
                                                      gamma_grad,
                                                      betta_grad,
                                                      batch,
                                                      hidden_dim,
                                                      invertible,
                                                      item_ct1,
                                                      betta_buffer.get_pointer(),
                                                      gamma_buffer.get_pointer());
                         });
    });

    range<3> grid_dim2(1, 1, batch);

    if (hidden_dim > 8192 && hidden_dim <= 16384)
        threads <<= 1;
    else if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 2;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 3;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    range<3> block_dim2(1, 1, threads / 2);

    stream[1]->submit([&](handler& cgh) {
        accessor<float, 1, access_mode::read_write, access::target::local> partialSum_acc_ct1(
            range<1>(MAX_SG_NUM /*MAX_WARP_NUM*/), cgh);

        cgh.parallel_for(nd_range<3>(grid_dim2 * block_dim2, block_dim2),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             LayerNormBackward2<false>(out_grad,
                                                       nullptr,
                                                       vals_hat,
                                                       gamma,
                                                       betta,
                                                       vars,
                                                       inp_grad,
                                                       invertible,
                                                       hidden_dim / 2,
                                                       item_ct1,
                                                       partialSum_acc_ct1.get_pointer());
                         });
    });
}

/* Backward Normalize (Input-Gradient)
 * Using the means and variances from the input
 * This type of backward is not invertible!
 * We do the backward using the input (X)
 */
template <bool is_fuseadd>
void LayerNormBackward2(const float* out_grad,
                        const float* out_grad_add,
                        const float* X_vals,
                        const float* gamma,
                        const float* vars,
                        const float* means,
                        float* inp_grad,
                        int row_stride,
                        nd_item<3> item_ct1,
                        float* partialSum)
{
    int iteration_stride = item_ct1.get_local_range(2);
    int iterations = row_stride / iteration_stride;

    // group<3> b = item_ct1.get_group();
    // cg::thread_block_tile<MAX_SG_NUM> g = cg::tiled_partition<MAX_SG_NUM>(b);
    sub_group sg = item_ct1.get_sub_group();

    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    int wid = id / MAX_SG_NUM;
    int warp_num = (THREADS < row_stride ? THREADS : row_stride) / MAX_SG_NUM;

    out_grad += (row * row_stride);
    if constexpr (is_fuseadd) { out_grad_add += (row * row_stride); }
    X_vals += (row * row_stride);
    inp_grad += (row * row_stride);

    float vals_arr[NORM_REG];
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        float gamma_reg = gamma[i * iteration_stride + id];
        vals_arr[i] = out_grad[i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;
    }
    if ((high_index) < row_stride) {
        float gamma_reg = gamma[high_index];
        vals_arr[iterations] = out_grad[high_index];
        vals_arr[iterations] *= gamma_reg;
        iterations++;
    }

    float var_reg = vars[row];
    float mean_reg = means[row];

    float sum = 0;
    float xu[NORM_REG];
    for (int i = 0; i < iterations; i++) {
        xu[i] = (X_vals[i * iteration_stride + id] - mean_reg);
        sum += vals_arr[i] * xu[i];
        vals_arr[i] *= rsqrt(var_reg);
    }

    // for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += g.shfl_down(sum, i); }
    for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += sg.shuffle_down(sum, i); }

    // if (g.thread_rank() == 0) partialSum[wid] = sum;
    if (sg.get_local_id() == 0) partialSum[wid] = sum;

    item_ct1.barrier();

    // if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];
    if (sg.get_local_id() < warp_num) sum = partialSum[sg.get_local_id()];

#ifndef __STOCHASTIC_MODE__

    item_ct1.barrier();
#endif

    // for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    for (int i = 1; i < warp_num; i *= 2) sum += sg.shuffle_down(sum, i);

    // sum = g.shfl(sum, 0);
    sum = sg.shuffle(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++) {
        vals_arr[i] += (-sum * xu[i] * rsqrt(var_reg) / (var_reg));
    }

    sum = 0;
    for (int i = 0; i < iterations; i++) { sum += vals_arr[i]; }

    // for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += g.shfl_down(sum, i); }
    for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += sg.shuffle_down(sum, i); }

    // if (g.thread_rank() == 0) partialSum[wid] = sum;
    if (sg.get_local_id() == 0) partialSum[wid] = sum;

    item_ct1.barrier();

    // if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];
    if (sg.get_local_id() < warp_num) sum = partialSum[sg.get_local_id()];

#ifndef __STOCHASTIC_MODE__
    item_ct1.barrier();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += sg.shuffle_down(sum, i);
    sum = sg.shuffle(sum, 0);
    sum /= row_stride;

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++)
        if constexpr (is_fuseadd) {
            inp_grad[i * iteration_stride + id] =
                (vals_arr[i] - sum) + out_grad_add[i * iteration_stride + id];
        } else {
            inp_grad[i * iteration_stride + id] = (vals_arr[i] - sum);
        }
    if ((high_index) < row_stride)
        if constexpr (is_fuseadd) {
            inp_grad[high_index] = (vals_arr[iterations] - sum) + out_grad_add[high_index];
        } else {
            inp_grad[high_index] = (vals_arr[iterations] - sum);
        }
}

/* Backward Normalize (Input-Gradient)
 * Using the means and variances from the input
 * This type of backward is not invertible!
 * We do the backward using the input (X)
 */
template <bool is_fuseadd>
void LayerNormBackward2(const bf16* out_grad,
                        const bf16* out_grad_add,
                        const bf16* X_vals,
                        const bf16* gamma,
                        const bf16* vars,
                        const bf16* means,
                        bf16* inp_grad,
                        int row_stride,
                        nd_item<3> item_ct1,
                        float* partialSum)
{
    int iteration_stride = item_ct1.get_local_range(2);
    int iterations = row_stride / iteration_stride;

    // group<3> b = item_ct1.get_group();
    // cg::thread_block_tile<MAX_SG_NUM> g = cg::tiled_partition<MAX_SG_NUM>(b);
    sub_group sg = item_ct1.get_sub_group();

    const ushort* out_grad_cast = reinterpret_cast<const ushort*>(out_grad);
    const ushort* out_grad_add_cast = reinterpret_cast<const ushort*>(out_grad_add);
    const ushort* X_vals_cast = reinterpret_cast<const ushort*>(X_vals);
    const ushort* gamma_cast = reinterpret_cast<const ushort*>(gamma);
    const ushort* vars_cast = reinterpret_cast<const ushort*>(vars);
    const ushort* means_cast = reinterpret_cast<const ushort*>(means);
    ushort* inp_grad_cast = reinterpret_cast<ushort*>(inp_grad);

    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    int wid = id / MAX_SG_NUM;
    int warp_num = (THREADS < row_stride ? THREADS : row_stride) / MAX_SG_NUM;

    out_grad_cast += (row * row_stride);
    if constexpr (is_fuseadd) { out_grad_add_cast += (row * row_stride); }
    X_vals_cast += (row * row_stride);
    inp_grad_cast += (row * row_stride);

    float vals_arr[NORM_REG];
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        float gamma_reg = float(gamma_cast[i * iteration_stride + id]);
        vals_arr[i] = float(out_grad_cast[i * iteration_stride + id]);
        vals_arr[i] *= gamma_reg;
    }
    if ((high_index) < row_stride) {
        float gamma_reg = float(gamma_cast[high_index]);
        vals_arr[iterations] = float(out_grad_cast[high_index]);
        vals_arr[iterations] *= gamma_reg;
        iterations++;
    }

    float var_reg = float(vars_cast[row]);
    float mean_reg = float(means_cast[row]);

    float sum = 0;
    float xu[NORM_REG];
    for (int i = 0; i < iterations; i++) {
        xu[i] = (float(X_vals_cast[i * iteration_stride + id]) - mean_reg);
        sum += vals_arr[i] * xu[i];
        vals_arr[i] *= rsqrt(var_reg);
    }

    // for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += g.shfl_down(sum, i); }
    for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += sg.shuffle_down(sum, i); }

    // if (g.thread_rank() == 0) partialSum[wid] = sum;
    if (sg.get_local_id() == 0) partialSum[wid] = sum;

    item_ct1.barrier();

    // if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];
    if (sg.get_local_id() < warp_num) sum = partialSum[sg.get_local_id()];

#ifndef __STOCHASTIC_MODE__
    item_ct1.barrier();
#endif

    // for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    for (int i = 1; i < warp_num; i *= 2) sum += sg.shuffle_down(sum, i);

    // sum = g.shfl(sum, 0);
    sum = sg.shuffle(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++) {
        vals_arr[i] += (-sum * xu[i] * rsqrt(var_reg) / (var_reg));
    }

    sum = 0;
    for (int i = 0; i < iterations; i++) { sum += vals_arr[i]; }

    // for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += g.shfl_down(sum, i); }
    for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += sg.shuffle_down(sum, i); }

    // if (g.thread_rank() == 0) partialSum[wid] = sum;
    if (sg.get_local_id() == 0) partialSum[wid] = sum;
    item_ct1.barrier();

    // if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];
    if (sg.get_local_id() < warp_num) sum = partialSum[sg.get_local_id()];

#ifndef __STOCHASTIC_MODE__
    item_ct1.barrier();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += sg.shuffle_down(sum, i);
    sum = sg.shuffle(sum, 0);
    sum /= row_stride;

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++)
        if constexpr (is_fuseadd) {
            inp_grad_cast[i * iteration_stride + id] =
                bf16((vals_arr[i] - sum) + float(out_grad_add_cast[i * iteration_stride + id]));
        } else {
            inp_grad_cast[i * iteration_stride + id] = bf16(vals_arr[i] - sum);
        }
    if ((high_index) < row_stride)
        if constexpr (is_fuseadd) {
            inp_grad_cast[high_index] =
                bf16((vals_arr[iterations] - sum) + float(out_grad_add_cast[high_index]));
        } else {
            inp_grad_cast[high_index] = bf16(vals_arr[iterations] - sum);
        }
}

template <bool is_fuseadd>
void LayerNormBackward2(const half* out_grad,
                        const half* out_grad_add,
                        const half* X_vals,
                        const half* gamma,
                        const half* vars,
                        const half* means,
                        half* inp_grad,
                        int row_stride,
                        nd_item<3> item_ct1,
                        float* partialSum)
{
    int iteration_stride = item_ct1.get_local_range(2);
    int iterations = row_stride / iteration_stride;

    // group<3> b = item_ct1.get_group();
    // cg::thread_block_tile<MAX_SG_NUM> g = cg::tiled_partition<MAX_SG_NUM>(b);
    sub_group sg = item_ct1.get_sub_group();

    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    int wid = id / MAX_SG_NUM;
    int warp_num = (iteration_stride < row_stride ? iteration_stride : row_stride) / MAX_SG_NUM;

    half2 vals_arr[NORM_REG];
    float2 vals_arr_f[NORM_REG];

    half2* inp_grad_h = reinterpret_cast<half2*>(inp_grad);
    const half2* out_grad_h = reinterpret_cast<const half2*>(out_grad);
    const half2* out_grad_add_h = reinterpret_cast<const half2*>(out_grad_add);
    const half2* vals_hat_h = reinterpret_cast<const half2*>(X_vals);

    inp_grad_h += (row * row_stride);
    out_grad_h += (row * row_stride);
    if constexpr (is_fuseadd) { out_grad_add_h += (row * row_stride); }
    vals_hat_h += (row * row_stride);

    const half2* gamma_h = reinterpret_cast<const half2*>(gamma);
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        half2 gamma_reg = gamma_h[i * iteration_stride + id];
        vals_arr[i] = out_grad_h[i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;  // out_grad * gamma
    }
    if ((high_index) < row_stride) {
        half2 gamma_reg = gamma_h[high_index];
        vals_arr[iterations] = out_grad_h[high_index];
        vals_arr[iterations] *= gamma_reg;  // out_grad * gamma
        iterations++;
    }
    half mean_h = means[row];
    half var_h = vars[row];
    half2 var_reg = half2{var_h, var_h};
    half2 mean_reg = half2{mean_h, mean_h};
    half2 xu[NORM_REG];

    float sum = 0.f;
    for (int i = 0; i < iterations; i++) {
        xu[i] = (vals_hat_h[i * iteration_stride + id] - mean_reg);
        half2 result_h = (xu[i] * vals_arr[i]);
        float2 result_f = result_h.convert<float, rounding_mode::automatic>();
        sum += result_f.x();
        sum += result_f.y();
        vals_arr[i] *= rsqrt(var_reg);
    }

    for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += sg.shuffle_down(sum, i); }

    if (sg.get_local_id() == 0) partialSum[wid] = sum;

    item_ct1.barrier();

    if (sg.get_local_id() < warp_num) sum = partialSum[sg.get_local_id()];

#ifndef __STOCHASTIC_MODE__
    item_ct1.barrier();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += sg.shuffle_down(sum, i);

    sum = sg.shuffle(sum, 0);
    sum /= (2 * row_stride);
    half2 sum_h = float2{sum, sum}.convert<half>();

    for (int i = 0; i < iterations; i++) {
        half2 xu_grad = ((-sum_h * xu[i] * rsqrt(var_reg)) / (var_reg));
        vals_arr_f[i] = vals_arr[i].convert<float, rounding_mode::automatic>();
        float2 xu_grad_f = xu_grad.convert<float, rounding_mode::automatic>();
        vals_arr_f[i].x() += xu_grad_f.x();
        vals_arr_f[i].y() += xu_grad_f.y();
    }

    sum = 0.f;
    for (int i = 0; i < iterations; i++) {
        sum += (vals_arr_f[i].x());
        sum += (vals_arr_f[i].y());
    }

    for (int i = 1; i < MAX_SG_NUM; i *= 2) { sum += sg.shuffle_down(sum, i); }

    if (sg.get_local_id() == 0) partialSum[wid] = sum;

    item_ct1.barrier();

    if (sg.get_local_id() < warp_num) sum = partialSum[sg.get_local_id()];

#ifndef __STOCHASTIC_MODE__
    item_ct1.barrier();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += sg.shuffle_down(sum, i);

    sum = sg.shuffle(sum, 0);
    sum /= (2 * row_stride);

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) {
        vals_arr_f[i].x() -= sum;
        vals_arr_f[i].y() -= sum;
        half2 temp = vals_arr_f[i].convert<half>();
        if constexpr (is_fuseadd) {
            inp_grad_h[i * iteration_stride + id] =
                temp + out_grad_add_h[i * iteration_stride + id];
        } else {
            inp_grad_h[i * iteration_stride + id] = temp;
        }
    }
    if ((high_index) < row_stride) {
        vals_arr_f[iterations].x() -= sum;
        vals_arr_f[iterations].y() -= sum;
        half2 temp = vals_arr_f[iterations].convert<half>();
        if constexpr (is_fuseadd) {
            inp_grad_h[high_index] = temp + out_grad_add_h[high_index];
        } else {
            inp_grad_h[high_index] = temp;
        }
    }
}

template <typename T>
void launch_layerNorm_backward(const T* out_grad,
                               const T* X_data,
                               const T* vars,
                               const T* means,
                               const T* gamma,
                               T* gamma_grad,
                               T* betta_grad,
                               T* inp_grad,
                               int batch,
                               int hidden_dim,
                               queue* stream[2])
{
    int threads = THREADS;

    range<3> grid_dim(1, 1, hidden_dim / TILE_DIM);
    range<3> block_dim(1, TILE_DIM, TILE_DIM);

    // LayerNormBackward1<float><<<grid_dim, block_dim, 0, stream[0]>>>(
    //     out_grad, X_data, vars, means, gamma_grad, betta_grad, batch,
    //     hidden_dim);
    stream[0]->submit([&](handler& cgh) {
        accessor<float, 2, access_mode::read_write, access::target::local> betta_buffer(
            range<2>(MAX_SG_NUM /*MAX_WARP_NUM*/, MAX_SG_NUM1), cgh);
        accessor<float, 2, access_mode::read_write, access::target::local> gamma_buffer(
            range<2>(MAX_SG_NUM /*MAX_WARP_NUM*/, MAX_SG_NUM1), cgh);

        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             LayerNormBackward1(out_grad,
                                                X_data,
                                                vars,
                                                means,
                                                gamma_grad,
                                                betta_grad,
                                                batch,
                                                hidden_dim,
                                                item_ct1,
                                                betta_buffer.get_pointer(),
                                                gamma_buffer.get_pointer());
                         });
    });

    range<3> grid_dim2(1, 1, batch);

    if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 1;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 2;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    range<3> block_dim2(1, 1, threads);
    stream[1]->submit([&](handler& cgh) {
        accessor<float, 1, access_mode::read_write, access::target::local> partialSum_acc_ct1(
            range<1>(MAX_SG_NUM /*MAX_WARP_NUM*/), cgh);

        cgh.parallel_for(nd_range<3>(grid_dim2 * block_dim2, block_dim2),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             LayerNormBackward2<false>(out_grad,
                                                       nullptr,
                                                       X_data,
                                                       gamma,
                                                       vars,
                                                       means,
                                                       inp_grad,
                                                       hidden_dim,
                                                       item_ct1,
                                                       partialSum_acc_ct1.get_pointer());
                         });
    });
}

template void launch_layerNorm_backward<float>(const float* out_grad,
                                               const float* X_data,
                                               const float* vars,
                                               const float* means,
                                               const float* gamma,
                                               float* gamma_grad,
                                               float* betta_grad,
                                               float* inp_grad,
                                               int batch,
                                               int hidden_dim,
                                               queue* stream[2]);
template void launch_layerNorm_backward<bf16>(const bf16* out_grad,
                                              const bf16* X_data,
                                              const bf16* vars,
                                              const bf16* means,
                                              const bf16* gamma,
                                              bf16* gamma_grad,
                                              bf16* betta_grad,
                                              bf16* inp_grad,
                                              int batch,
                                              int hidden_dim,
                                              queue* stream[2]);
template <>
void launch_layerNorm_backward<half>(const half* out_grad,
                                     const half* X_data,
                                     const half* vars,
                                     const half* means,
                                     const half* gamma,
                                     half* gamma_grad,
                                     half* betta_grad,
                                     half* inp_grad,
                                     int batch,
                                     int hidden_dim,
                                     queue* stream[2])
{
    int threads = THREADS;

    range<3> grid_dim(1, 1, hidden_dim / TILE_DIM);
    range<3> block_dim(1, TILE_DIM, TILE_DIM);

    // LayerNormBackward1<half><<<grid_dim, block_dim, 0, stream[0]>>>(
    //     out_grad, X_data, vars, means, gamma_grad, betta_grad, batch,
    //     hidden_dim);
    stream[0]->submit([&](handler& cgh) {
        accessor<float, 2, access_mode::read_write, access::target::local> betta_buffer(
            range<2>(MAX_SG_NUM /*MAX_WARP_NUM*/, MAX_SG_NUM1), cgh);
        accessor<float, 2, access_mode::read_write, access::target::local> gamma_buffer(
            range<2>(MAX_SG_NUM /*MAX_WARP_NUM*/, MAX_SG_NUM1), cgh);

        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             LayerNormBackward1<half>(out_grad,
                                                      X_data,
                                                      vars,
                                                      means,
                                                      gamma_grad,
                                                      betta_grad,
                                                      batch,
                                                      hidden_dim,
                                                      item_ct1,
                                                      betta_buffer.get_pointer(),
                                                      gamma_buffer.get_pointer());
                         });
    });

    range<3> grid_dim2(1, 1, batch);

    if (hidden_dim > 8192 && hidden_dim <= 16384)
        threads <<= 1;
    else if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 2;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 3;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    range<3> block_dim2(1, 1, threads / 2);
    stream[1]->submit([&](handler& cgh) {
        accessor<float, 1, access_mode::read_write, access::target::local> partialSum_acc_ct1(
            range<1>(MAX_SG_NUM /*MAX_WARP_NUM*/), cgh);

        cgh.parallel_for(nd_range<3>(grid_dim2 * block_dim2, block_dim2),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             LayerNormBackward2<false>(out_grad,
                                                       nullptr,
                                                       X_data,
                                                       gamma,
                                                       vars,
                                                       means,
                                                       inp_grad,
                                                       hidden_dim / 2,
                                                       item_ct1,
                                                       partialSum_acc_ct1.get_pointer());
                         });
    });
}

template <typename T>
void launch_layerNorm_backward_fused_add(const T* out_grad1,
                                         const T* out_grad2,
                                         const T* vals_hat,
                                         const T* vars,
                                         const T* gamma,
                                         T* gamma_grad,
                                         T* betta_grad,
                                         T* inp_grad,
                                         int batch,
                                         int hidden_dim,
                                         queue* stream[2],
                                         bool invertible,
                                         const T* betta)
{
    int threads = THREADS;

    range<3> grid_dim(1, 1, hidden_dim / TILE_DIM);
    range<3> block_dim(1, TILE_DIM, TILE_DIM);
    // LayerNormBackward1<float><<<grid_dim, block_dim, 0, stream[0]>>>(
    //     out_grad1, vals_hat, gamma, betta, gamma_grad, betta_grad, batch,
    //     hidden_dim, invertible);
    stream[0]->submit([&](handler& cgh) {
        accessor<float, 2, access_mode::read_write, access::target::local> betta_buffer(
            range<2>(MAX_SG_NUM /*MAX_WARP_NUM*/, MAX_SG_NUM1), cgh);
        accessor<float, 2, access_mode::read_write, access::target::local> gamma_buffer(
            range<2>(MAX_SG_NUM /*MAX_WARP_NUM*/, MAX_SG_NUM1), cgh);
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             LayerNormBackward1(out_grad1,
                                                vals_hat,
                                                gamma,
                                                betta,
                                                gamma_grad,
                                                betta_grad,
                                                batch,
                                                hidden_dim,
                                                invertible,
                                                item_ct1,
                                                betta_buffer.get_pointer(),
                                                gamma_buffer.get_pointer());
                         });
    });

    range<3> grid_dim2(1, 1, batch);

    if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 1;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 2;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    range<3> block_dim2(1, 1, threads);

    stream[1]->submit([&](handler& cgh) {
        accessor<float, 1, access_mode::read_write, access::target::local> partialSum_acc_ct1(
            range<1>(MAX_SG_NUM /*MAX_WARP_NUM*/), cgh);

        cgh.parallel_for(nd_range<3>(grid_dim2 * block_dim2, block_dim2),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             LayerNormBackward2<true>(out_grad1,
                                                      out_grad2,
                                                      vals_hat,
                                                      gamma,
                                                      betta,
                                                      vars,
                                                      inp_grad,
                                                      invertible,
                                                      hidden_dim,
                                                      item_ct1,
                                                      partialSum_acc_ct1.get_pointer());
                         });
    });
}

template void launch_layerNorm_backward_fused_add<float>(const float* out_grad1,
                                                         const float* out_grad2,
                                                         const float* vals_hat,
                                                         const float* vars,
                                                         const float* gamma,
                                                         float* gamma_grad,
                                                         float* betta_grad,
                                                         float* inp_grad,
                                                         int batch,
                                                         int hidden_dim,
                                                         queue* stream[2],
                                                         bool invertible,
                                                         const float* betta);

template void launch_layerNorm_backward_fused_add<bf16>(const bf16* out_grad1,
                                                        const bf16* out_grad2,
                                                        const bf16* vals_hat,
                                                        const bf16* vars,
                                                        const bf16* gamma,
                                                        bf16* gamma_grad,
                                                        bf16* betta_grad,
                                                        bf16* inp_grad,
                                                        int batch,
                                                        int hidden_dim,
                                                        queue* stream[2],
                                                        bool invertible,
                                                        const bf16* betta);

template <>
void launch_layerNorm_backward_fused_add<half>(const half* out_grad1,
                                               const half* out_grad2,
                                               const half* vals_hat,
                                               const half* vars,
                                               const half* gamma,
                                               half* gamma_grad,
                                               half* betta_grad,
                                               half* inp_grad,
                                               int batch,
                                               int hidden_dim,
                                               queue* stream[2],
                                               bool invertible,
                                               const half* betta)
{
    int threads = THREADS;

    range<3> grid_dim(1, 1, hidden_dim / TILE_DIM);
    range<3> block_dim(1, TILE_DIM, TILE_DIM);

    // LayerNormBackward1<half><<<grid_dim, block_dim, 0, stream[0]>>>(
    //     out_grad1, vals_hat, gamma, betta, gamma_grad, betta_grad, batch,
    //     hidden_dim, invertible);
    stream[0]->submit([&](handler& cgh) {
        accessor<float, 2, access_mode::read_write, access::target::local> betta_buffer(
            range<2>(MAX_SG_NUM /*MAX_WARP_NUM*/, MAX_SG_NUM1), cgh);
        accessor<float, 2, access_mode::read_write, access::target::local> gamma_buffer(
            range<2>(MAX_SG_NUM /*MAX_WARP_NUM*/, MAX_SG_NUM1), cgh);
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             LayerNormBackward1<half>(out_grad1,
                                                      vals_hat,
                                                      gamma,
                                                      betta,
                                                      gamma_grad,
                                                      betta_grad,
                                                      batch,
                                                      hidden_dim,
                                                      invertible,
                                                      item_ct1,
                                                      betta_buffer.get_pointer(),
                                                      gamma_buffer.get_pointer());
                         });
    });

    range<3> grid_dim2(1, 1, batch);

    if (hidden_dim > 8192 && hidden_dim <= 16384)
        threads <<= 1;
    else if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 2;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 3;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    range<3> block_dim2(1, 1, threads / 2);
    stream[1]->submit([&](handler& cgh) {
        accessor<float, 1, access_mode::read_write, access::target::local> partialSum_acc_ct1(
            range<1>(MAX_SG_NUM /*MAX_WARP_NUM*/), cgh);

        cgh.parallel_for(nd_range<3>(grid_dim2 * block_dim2, block_dim2),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             LayerNormBackward2<true>(out_grad1,
                                                      out_grad2,
                                                      vals_hat,
                                                      gamma,
                                                      betta,
                                                      vars,
                                                      inp_grad,
                                                      invertible,
                                                      hidden_dim / 2,
                                                      item_ct1,
                                                      partialSum_acc_ct1.get_pointer());
                         });
    });
}

template <typename T>
void launch_layerNorm_backward_fused_add(const T* out_grad1,
                                         const T* out_grad2,
                                         const T* X_data,
                                         const T* vars,
                                         const T* means,
                                         const T* gamma,
                                         T* gamma_grad,
                                         T* betta_grad,
                                         T* inp_grad,
                                         int batch,
                                         int hidden_dim,
                                         queue* stream[2])
{
    int threads = THREADS;

    range<3> grid_dim(1, 1, hidden_dim / TILE_DIM);
    range<3> block_dim(1, TILE_DIM, TILE_DIM);

    // LayerNormBackward1<float><<<grid_dim, block_dim, 0, stream[0]>>>(
    //     out_grad1, X_data, vars, means, gamma_grad, betta_grad, batch,
    //     hidden_dim);
    stream[0]->submit([&](handler& cgh) {
        accessor<float, 2, access_mode::read_write, access::target::local> betta_buffer(
            range<2>(MAX_SG_NUM /*MAX_WARP_NUM*/, MAX_SG_NUM1), cgh);
        accessor<float, 2, access_mode::read_write, access::target::local> gamma_buffer(
            range<2>(MAX_SG_NUM /*MAX_WARP_NUM*/, MAX_SG_NUM1), cgh);

        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             LayerNormBackward1(out_grad1,
                                                X_data,
                                                vars,
                                                means,
                                                gamma_grad,
                                                betta_grad,
                                                batch,
                                                hidden_dim,
                                                item_ct1,
                                                betta_buffer.get_pointer(),
                                                gamma_buffer.get_pointer());
                         });
    });

    range<3> grid_dim2(1, 1, batch);

    if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 1;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 2;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    range<3> block_dim2(1, 1, threads);
    stream[1]->submit([&](handler& cgh) {
        accessor<float, 1, access_mode::read_write, access::target::local> partialSum_acc_ct1(
            range<1>(MAX_SG_NUM /*MAX_WARP_NUM*/), cgh);

        cgh.parallel_for(nd_range<3>(grid_dim2 * block_dim2, block_dim2),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             LayerNormBackward2<true>(out_grad1,
                                                      out_grad2,
                                                      X_data,
                                                      gamma,
                                                      vars,
                                                      means,
                                                      inp_grad,
                                                      hidden_dim,
                                                      item_ct1,
                                                      partialSum_acc_ct1.get_pointer());
                         });
    });
}

template void launch_layerNorm_backward_fused_add<float>(const float* out_grad1,
                                                         const float* out_grad2,
                                                         const float* X_data,
                                                         const float* vars,
                                                         const float* means,
                                                         const float* gamma,
                                                         float* gamma_grad,
                                                         float* betta_grad,
                                                         float* inp_grad,
                                                         int batch,
                                                         int hidden_dim,
                                                         queue* stream[2]);
template void launch_layerNorm_backward_fused_add<bf16>(const bf16* out_grad1,
                                                        const bf16* out_grad2,
                                                        const bf16* X_data,
                                                        const bf16* vars,
                                                        const bf16* means,
                                                        const bf16* gamma,
                                                        bf16* gamma_grad,
                                                        bf16* betta_grad,
                                                        bf16* inp_grad,
                                                        int batch,
                                                        int hidden_dim,
                                                        queue* stream[2]);
template <>
void launch_layerNorm_backward_fused_add<half>(const half* out_grad1,
                                               const half* out_grad2,
                                               const half* X_data,
                                               const half* vars,
                                               const half* means,
                                               const half* gamma,
                                               half* gamma_grad,
                                               half* betta_grad,
                                               half* inp_grad,
                                               int batch,
                                               int hidden_dim,
                                               queue* stream[2])
{
    int threads = THREADS;

    range<3> grid_dim(1, 1, hidden_dim / TILE_DIM);
    range<3> block_dim(1, TILE_DIM, TILE_DIM);

    // LayerNormBackward1<half><<<grid_dim, block_dim, 0, stream[0]>>>(
    //     out_grad1, X_data, vars, means, gamma_grad, betta_grad, batch,
    //     hidden_dim);
    stream[0]->submit([&](handler& cgh) {
        accessor<float, 2, access_mode::read_write, access::target::local> betta_buffer(
            range<2>(MAX_SG_NUM /*MAX_WARP_NUM*/, MAX_SG_NUM1), cgh);
        accessor<float, 2, access_mode::read_write, access::target::local> gamma_buffer(
            range<2>(MAX_SG_NUM /*MAX_WARP_NUM*/, MAX_SG_NUM1), cgh);
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             LayerNormBackward1<half>(out_grad1,
                                                      X_data,
                                                      vars,
                                                      means,
                                                      gamma_grad,
                                                      betta_grad,
                                                      batch,
                                                      hidden_dim,
                                                      item_ct1,
                                                      betta_buffer.get_pointer(),
                                                      gamma_buffer.get_pointer());
                         });
    });

    range<3> grid_dim2(1, 1, batch);

    if (hidden_dim > 8192 && hidden_dim <= 16384)
        threads <<= 1;
    else if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 2;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 3;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    range<3> block_dim2(1, 1, threads / 2);
    stream[1]->submit([&](handler& cgh) {
        accessor<float, 1, access_mode::read_write, access::target::local> partialSum_acc_ct1(
            range<1>(MAX_SG_NUM /*MAX_WARP_NUM*/), cgh);

        cgh.parallel_for(nd_range<3>(grid_dim2 * block_dim2, block_dim2),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             LayerNormBackward2<true>(out_grad1,
                                                      out_grad2,
                                                      X_data,
                                                      gamma,
                                                      vars,
                                                      means,
                                                      inp_grad,
                                                      hidden_dim / 2,
                                                      item_ct1,
                                                      partialSum_acc_ct1.get_pointer());
                         });
    });
}
