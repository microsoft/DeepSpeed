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

const int unroll_factor = 4;

void dropout_kernel(const int N,
                    const float ratio,
                    float* out,
                    const float* Xdata,
                    uint8_t* mask,
                    const std::pair<uint64_t, uint64_t>& seed,
                    nd_item<3> item_ct1)
{
    const float scale = 1. / (1. - ratio);
    size_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    oneapi::mkl::rng::device::philox4x32x10<4> engine(seed.first, {idx * 4, seed.second});
    oneapi::mkl::rng::device::uniform<> distr;

    DPCPP_1D_KERNEL_LOOP(j, N / unroll_factor)
    {
        float4 rand = oneapi::mkl::rng::device::generate(distr, engine);
        uint8_t m[unroll_factor];

        m[0] = (uint8_t)(rand.x() > ratio);
        m[1] = (uint8_t)(rand.y() > ratio);
        m[2] = (uint8_t)(rand.z() > ratio);
        m[3] = (uint8_t)(rand.w() > ratio);

        int i = j * unroll_factor;

        mask[i] = (uint8_t)m[0];
        mask[i + 1] = (uint8_t)m[1];
        mask[i + 2] = (uint8_t)m[2];
        mask[i + 3] = (uint8_t)m[3];

        out[i] = Xdata[i] * scale * m[0];
        out[i + 1] = Xdata[i + 1] * scale * m[1];
        out[i + 2] = Xdata[i + 2] * scale * m[2];
        out[i + 3] = Xdata[i + 3] * scale * m[3];
    }
    int high_index = ((((N / unroll_factor) - 1) / item_ct1.get_local_range().get(2) + 1) *
                      (unroll_factor * item_ct1.get_local_range().get(2))) +
                     item_ct1.get_local_id(2);
    if (N > high_index) {
        float4 rand = oneapi::mkl::rng::device::generate(distr, engine);
        float* rand_data = &(rand.x());
        int k = 0;
        for (int i = high_index; i < N; i++) {
            uint8_t m = (uint8_t)(rand_data[k++] > ratio);
            out[i] = Xdata[i] * scale * m;
            mask[i] = m;
        }
    }
}

void dropout_kernel(const int N,
                    const float ratio,
                    bf16* out,
                    const bf16* Xdata,
                    uint8_t* mask,
                    const std::pair<uint64_t, uint64_t>& seed,
                    nd_item<3> item_ct1)
{
    const float scale = 1. / (1. - ratio);
    size_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    oneapi::mkl::rng::device::philox4x32x10<4> engine(seed.first, {idx * 4, seed.second});
    oneapi::mkl::rng::device::uniform<> distr;

    ushort* out_cast = reinterpret_cast<ushort*>(out);
    const ushort* Xdata_cast = reinterpret_cast<const ushort*>(Xdata);

    DPCPP_1D_KERNEL_LOOP(j, N / unroll_factor)
    {
        float4 rand = oneapi::mkl::rng::device::generate(distr, engine);
        uint8_t m[unroll_factor];

        m[0] = (uint8_t)(rand.x() > ratio);
        m[1] = (uint8_t)(rand.y() > ratio);
        m[2] = (uint8_t)(rand.z() > ratio);
        m[3] = (uint8_t)(rand.w() > ratio);

        int i = j * unroll_factor;

        mask[i] = (uint8_t)m[0];
        mask[i + 1] = (uint8_t)m[1];
        mask[i + 2] = (uint8_t)m[2];
        mask[i + 3] = (uint8_t)m[3];

        out_cast[i] = bf16(float(Xdata_cast[i]) * scale * m[0]);
        out_cast[i + 1] = bf16(float(Xdata_cast[i + 1]) * scale * m[1]);
        out_cast[i + 2] = bf16(float(Xdata_cast[i + 2]) * scale * m[2]);
        out_cast[i + 3] = bf16(float(Xdata_cast[i + 3]) * scale * m[3]);
    }
    int high_index = ((((N / unroll_factor) - 1) / item_ct1.get_local_range().get(2) + 1) *
                      (unroll_factor * item_ct1.get_local_range().get(2))) +
                     item_ct1.get_local_id(2);
    if (N > high_index) {
        float4 rand = oneapi::mkl::rng::device::generate(distr, engine);
        float* rand_data = &(rand.x());
        int k = 0;
        for (int i = high_index; i < N; i++) {
            uint8_t m = (uint8_t)(rand_data[k++] > ratio);
            out_cast[i] = bf16(float(Xdata_cast[i]) * scale * m);
            mask[i] = m;
        }
    }
}

void dropout_kernel(const int N,
                    const float ratio,
                    half* out,
                    const half* Xdata,
                    uint8_t* mask,
                    const std::pair<uint64_t, uint64_t>& seed,
                    nd_item<3> item_ct1)
{
    const float scale = 1. / (1. - ratio);

    size_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    oneapi::mkl::rng::device::philox4x32x10<4> engine(seed.first, {idx * 4, seed.second});
    oneapi::mkl::rng::device::uniform<> distr;

#ifdef __STOCHASTIC_MODE__

    const half2 h_scale = vec<float, 2>{scale}.convert<float>();
    const float2* x_cast = reinterpret_cast<const float2*>(Xdata);
    float2* out_cast = reinterpret_cast<float2*>(out);
    uint32_t* mask_cast = reinterpret_cast<uint32_t*>(mask);

    uint32_t m_32;
    uint8_t* m = reinterpret_cast<uint8_t*>(&m_32);

    float2 result_f;
    half2* result_h = reinterpret_cast<half2*>(&result_f);
    half2 mask_h[2];
    float2 mask_f[2];

    DPCPP_1D_KERNEL_LOOP(j, N / unroll_factor)
    {
        float2 x_f = x_cast[j];
        half2* x_h = reinterpret_cast<half2*>(&x_f);

        float4 rand = oneapi::mkl::rng::device::generate(distr, engine);

        m[0] = (uint8_t)(rand.x > ratio);
        m[1] = (uint8_t)(rand.y > ratio);
        m[2] = (uint8_t)(rand.z > ratio);
        m[3] = (uint8_t)(rand.w > ratio);

        float* mask_f_data = &mask_f[0].x;
#pragma unroll
        for (int i = 0; i < unroll_factor; i++) mask_f_data[i] = (float)(m[i]);

        mask_h[0] = mask_f[0].convert<half>();
        mask_h[1] = mask_f[1].convert<half> 9;

        result_h[0] = x_h[0] * h_scale * mask_h[0];
        result_h[1] = x_h[1] * h_scale * mask_h[1];

        out_cast[j] = result_f;

        mask_cast[j] = m_32;
    }

#else

    DPCPP_1D_KERNEL_LOOP(j, N / unroll_factor)
    {
        int i = j * unroll_factor;

        const half2* vals_half = reinterpret_cast<const half2*>(Xdata + i);
        float2 vals_half_f[2];
        vals_half_f[0] = vals_half[0].convert<float, rounding_mode::automatic>();
        vals_half_f[1] = vals_half[1].convert<float, rounding_mode::automatic>();

        uint8_t m[unroll_factor];
        float4 rand = oneapi::mkl::rng::device::generate(distr, engine);
        m[0] = (uint8_t)(rand.x() > ratio);
        m[1] = (uint8_t)(rand.y() > ratio);
        m[2] = (uint8_t)(rand.z() > ratio);
        m[3] = (uint8_t)(rand.w() > ratio);

        out[i] = vec<float, 1>{vals_half_f[0].x() * scale * m[0]}
                     .convert<half, rounding_mode::automatic>()[0];
        out[i + 1] = vec<float, 1>{vals_half_f[0].y() * scale * m[1]}
                         .convert<half, rounding_mode::automatic>()[0];
        out[i + 2] = vec<float, 1>{vals_half_f[1].x() * scale * m[2]}
                         .convert<half, rounding_mode::automatic>()[0];
        out[i + 3] = vec<float, 1>{vals_half_f[1].y() * scale * m[3]}
                         .convert<half, rounding_mode::automatic>()[0];

        mask[i] = m[0];
        mask[i + 1] = m[1];
        mask[i + 2] = m[2];
        mask[i + 3] = m[3];
    }

#endif
    int high_index = ((((N / unroll_factor) - 1) / item_ct1.get_local_range().get(2) + 1) *
                      (unroll_factor * item_ct1.get_local_range().get(2))) +
                     item_ct1.get_local_id(2);
    if (N > high_index) {
        float4 rand = oneapi::mkl::rng::device::generate(distr, engine);
        float* rand_data = &(rand.x());
        int k = 0;
        for (int i = high_index; i < N; i++) {
            uint8_t m = (uint8_t)(rand_data[k++] > ratio);
            out[i] = vec<float, 1>{(float)Xdata[i] * scale * m}
                         .convert<half, rounding_mode::automatic>()[0];
            mask[i] = m;
        }
    }
}

void dropout_kernel_bwd(const int N,
                        const float ratio,
                        const float* Xdata,
                        float* out,
                        uint8_t* mask,
                        const std::pair<uint64_t, uint64_t>& seed,
                        nd_item<3> item_ct1)
{
    const float scale = 1. / (1. - ratio);
    DPCPP_1D_KERNEL_LOOP(j, N / unroll_factor)
    {
        int i = j * unroll_factor;

        out[i] = mask[i] * Xdata[i] * scale;
        out[i + 1] = mask[i + 1] * Xdata[i + 1] * scale;
        out[i + 2] = mask[i + 2] * Xdata[i + 2] * scale;
        out[i + 3] = mask[i + 3] * Xdata[i + 3] * scale;
    }
    int high_index = ((((N / unroll_factor) - 1) / item_ct1.get_local_range().get(2) + 1) *
                      (unroll_factor * item_ct1.get_local_range().get(2))) +
                     item_ct1.get_local_id(2);
    if (N > high_index) {
        for (int i = high_index; i < N; i++) { out[i] = mask[i] * Xdata[i] * scale; }
    }
}

void dropout_kernel_bwd(const int N,
                        const float ratio,
                        const bf16* Xdata,
                        bf16* out,
                        uint8_t* mask,
                        const std::pair<uint64_t, uint64_t>& seed,
                        nd_item<3> item_ct1)
{
    const float scale = 1. / (1. - ratio);

    const ushort* Xdata_cast = reinterpret_cast<const ushort*>(Xdata);
    ushort* out_cast = reinterpret_cast<ushort*>(out);

    DPCPP_1D_KERNEL_LOOP(j, N / unroll_factor)
    {
        int i = j * unroll_factor;

        out_cast[i] = bf16(mask[i] * float(Xdata_cast[i]) * scale);
        out_cast[i + 1] = bf16(mask[i + 1] * float(Xdata_cast[i + 1]) * scale);
        out_cast[i + 2] = bf16(mask[i + 2] * float(Xdata_cast[i + 2]) * scale);
        out_cast[i + 3] = bf16(mask[i + 3] * float(Xdata_cast[i + 3]) * scale);
    }
    int high_index = ((((N / unroll_factor) - 1) / item_ct1.get_local_range().get(2) + 1) *
                      (unroll_factor * item_ct1.get_local_range().get(2))) +
                     item_ct1.get_local_id(2);
    if (N > high_index) {
        for (int i = high_index; i < N; i++) {
            out_cast[i] = bf16(mask[i] * float(Xdata_cast[i]) * scale);
        }
    }
}

void dropout_kernel_bwd(const int N,
                        const float ratio,
                        const half* Xdata,
                        half* out,
                        uint8_t* mask,
                        const std::pair<uint64_t, uint64_t>& seed,
                        nd_item<3> item_ct1)
{
    const float scale = 1. / (1. - ratio);

#ifdef __STOCHASTIC_MODE__

    const half2 h_scale = vec<float, 2>{scale}.convert<half>();

    const float2* x_cast = reinterpret_cast<const float2*>(Xdata);
    float2* out_cast = reinterpret_cast<float2*>(out);
    uint32_t* mask_cast = reinterpret_cast<uint32_t*>(mask);

    DPCPP_1D_KERNEL_LOOP(j, N / unroll_factor)
    {
        float2 x_f = x_cast[j];
        half2* x_h = reinterpret_cast<half2*>(&x_f);

        uint32_t m_32 = mask_cast[j];
        uint8_t* m = (uint8_t*)&m_32;

        half2 mask_h[2];
        float2 mask_f[2];

        float* mask_f_data = &mask_f[0].x;
#pragma unroll
        for (int i = 0; i < unroll_factor; i++) mask_f_data[i] = (float)(m[i]);

#pragma unroll
        for (int i = 0; i < 2; i++) mask_h[i] = __float22half2_rn(mask_f[i]);

        float2 result_f;
        half2* result_h = reinterpret_cast<half2*>(&result_f);

        result_h[0] = x_h[0] * h_scale * mask_h[0];
        result_h[1] = x_h[1] * h_scale * mask_h[1];

        out_cast[j] = result_f;
    }

#else

    const half h_scale = vec<float, 1>{scale}.convert<half, rounding_mode::automatic>()[0];
    const half h_zero = vec<float, 1>{0.0}.convert<half, rounding_mode::automatic>()[0];

    DPCPP_1D_KERNEL_LOOP(j, N / unroll_factor)
    {
        int i = j * unroll_factor;

        const half2* vals_half = reinterpret_cast<const half2*>(Xdata + i);

        uint8_t* m = mask + i;

        float2 vals_half_f[2];

        vals_half_f[0] = vals_half[0].convert<float, rounding_mode::automatic>();
        vals_half_f[1] = vals_half[1].convert<float, rounding_mode::automatic>();

        out[i] = vec<float, 1>{vals_half_f[0].x() * scale * m[0]}
                     .convert<half, rounding_mode::automatic>()[0];
        out[i + 1] = vec<float, 1>{vals_half_f[0].y() * scale * m[1]}
                         .convert<half, rounding_mode::automatic>()[0];
        out[i + 2] = vec<float, 1>{vals_half_f[1].x() * scale * m[2]}
                         .convert<half, rounding_mode::automatic>()[0];
        out[i + 3] = vec<float, 1>{vals_half_f[1].y() * scale * m[3]}
                         .convert<half, rounding_mode::automatic>()[0];
    }

#endif
    int high_index = ((((N / unroll_factor) - 1) / item_ct1.get_local_range().get(2) + 1) *
                      (unroll_factor * item_ct1.get_local_range().get(2))) +
                     item_ct1.get_local_id(2);
    if (N > high_index) {
        for (int i = high_index; i < N; i++) {
            out[i] = vec<float, 1>{(float)Xdata[i] * scale * mask[i]}
                         .convert<half, rounding_mode::automatic>()[0];
        }
    }
}

template <typename T>
void launch_dropout(T* out,
                    const T* vals,
                    uint8_t* mask,
                    int total_count,
                    int dim,
                    float ratio,
                    queue* stream,
                    bool bwd)
{
    /*
     * dropout.Forward
     */
    assert(unroll_factor == 4);

    range<3> grid_dim = range<3>(1, 1, DS_GET_BLOCKS(total_count / unroll_factor));
    range<3> block_dim = range<3>(1, 1, DS_CUDA_NUM_THREADS);

    if (dim > 512) {
        block_dim[2] >>= 1;
        grid_dim[2] <<= 1;
    }
    uint64_t inc = total_count / grid_dim[2] / block_dim[2];
    std::pair<uint64_t, uint64_t> seed = SyclContext::Instance().IncrementOffset(inc);
    if (bwd)
        stream->submit([&](handler& cgh) {
            cgh.parallel_for(
                nd_range<3>(grid_dim * block_dim, block_dim), [=](nd_item<3> item_ct1) {
                    dropout_kernel_bwd(total_count, ratio, vals, out, mask, seed, item_ct1);
                });
        });
    else
        stream->submit([&](handler& cgh) {
            cgh.parallel_for(
                nd_range<3>(grid_dim * block_dim, block_dim), [=](nd_item<3> item_ct1) {
                    dropout_kernel(total_count, ratio, out, vals, mask, seed, item_ct1);
                });
        });
}

template void launch_dropout(float* out,
                             const float* vals,
                             uint8_t* mask,
                             int total_count,
                             int dim,
                             float ratio,
                             queue* stream,
                             bool);
template void launch_dropout(bf16* out,
                             const bf16* vals,
                             uint8_t* mask,
                             int total_count,
                             int dim,
                             float ratio,
                             queue* stream,
                             bool);
template void launch_dropout(half* out,
                             const half* vals,
                             uint8_t* mask,
                             int total_count,
                             int dim,
                             float ratio,
                             queue* stream,
                             bool);

void dropout_grad_kernel(const int N,
                         const float scale,
                         float* Xdata,
                         uint8_t* mask,
                         nd_item<3> item_ct1)
{
    DPCPP_1D_KERNEL_LOOP(i, N) { Xdata[i] *= scale * mask[i]; }
}

void dropout_grad_kernel(const int N,
                         const float scale,
                         bf16* Xdata,
                         uint8_t* mask,
                         nd_item<3> item_ct1)
{
    ushort* Xdata_cast = reinterpret_cast<ushort*>(Xdata);
    DPCPP_1D_KERNEL_LOOP(i, N) { Xdata_cast[i] = bf16(float(Xdata_cast[i]) * scale * mask[i]); }
}

void dropout_grad_kernel(const int N,
                         const float scale,
                         half* Xdata,
                         uint8_t* mask,
                         nd_item<3> item_ct1)
{
    const half2 h_scale = float2{scale, scale}.convert<half, rounding_mode::rte>();
    float2* x_cast = reinterpret_cast<float2*>(Xdata);
    uint32_t* mask_cast = reinterpret_cast<uint32_t*>(mask);

    DPCPP_1D_KERNEL_LOOP(j, N / unroll_factor)
    {
        float2 x_data = x_cast[j];
        uint32_t m_32 = mask_cast[j];
        uint8_t* m = (uint8_t*)&m_32;

        float2 result_f;
        half2* result_h = reinterpret_cast<half2*>(&result_f);

#ifdef __STOCHASTIC_MODE__

        half2* x_data_h = reinterpret_cast<half2*>(&x_data);
        half2 mask_h[2];
        float2 mask_f[2];

        float* mask_f_data = &mask_f[0].x;
#pragma unroll
        for (int i = 0; i < unroll_factor; i++) *(mask_f_data++) = (float)(m[i]);

        mask_h[0] = __float22half2_rn(mask_f[0]);
        mask_h[1] = __float22half2_rn(mask_f[1]);

        result_h[0] = x_data_h[0] * h_scale * mask_h[0];
        result_h[1] = x_data_h[1] * h_scale * mask_h[1];

#else

        half* x_data_h = reinterpret_cast<half*>(&x_data);
        float2 result[2];

        result[0].x() = (float)x_data_h[0] * scale * m[0];
        result[0].y() = (float)x_data_h[1] * scale * m[1];
        result[1].x() = (float)x_data_h[2] * scale * m[2];
        result[1].y() = (float)x_data_h[3] * scale * m[3];

        result_h[0] = result[0].convert<half, rounding_mode::rte>();
        result_h[1] = result[1].convert<half, rounding_mode::rte>();

#endif
        x_cast[j] = result_f;
    }
    int high_index = ((((N / unroll_factor) - 1) / item_ct1.get_local_range().get(2) + 1) *
                      (unroll_factor * item_ct1.get_local_range().get(2))) +
                     item_ct1.get_local_id(2);
    if (N > high_index) {
        for (int i = high_index; i < N; i++) {
            Xdata[i] = vec<float, 1>{(float)Xdata[i] * scale * mask[i]}
                           .convert<half, rounding_mode::automatic>()[0];
        }
    }
}

template <typename T>
void launch_dropout_grad(T* vals, uint8_t* mask, int total_count, float ratio, queue* stream)
{
    /*
     * Dropout.Backward0
     */
    assert(unroll_factor == 4);

    const float scale = 1. / (1. - ratio);
    range<3> grid_dim = range<3>(1, 1, DS_GET_BLOCKS(total_count / unroll_factor));
    range<3> block_dim = range<3>(1, 1, DS_CUDA_NUM_THREADS);
    stream->submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim), [=](nd_item<3> item_ct1) {
            dropout_grad_kernel(total_count, scale, vals, mask, item_ct1);
        });
    });
}

template void launch_dropout_grad(float* vals,
                                  uint8_t* mask,
                                  int total_count,
                                  float ratio,
                                  queue* stream);
template void launch_dropout_grad(bf16* vals,
                                  uint8_t* mask,
                                  int total_count,
                                  float ratio,
                                  queue* stream);
template void launch_dropout_grad(half* vals,
                                  uint8_t* mask,
                                  int total_count,
                                  float ratio,
                                  queue* stream);

void dropout_grad_kernel(const int N,
                         const float scale,
                         const float* Xdata,
                         float* out,
                         uint8_t* mask,
                         nd_item<3> item_ct1)
{
    DPCPP_1D_KERNEL_LOOP(i, N) { out[i] = Xdata[i] * scale * mask[i]; }
}

void dropout_grad_kernel(const int N,
                         const float scale,
                         const bf16* Xdata,
                         bf16* out,
                         uint8_t* mask,
                         nd_item<3> item_ct1)
{
    const ushort* Xdata_cast = reinterpret_cast<const ushort*>(Xdata);
    ushort* out_cast = reinterpret_cast<ushort*>(out);
    DPCPP_1D_KERNEL_LOOP(i, N) { out_cast[i] = bf16(float(Xdata_cast[i]) * scale * mask[i]); }
}

void dropout_grad_kernel(const int N,
                         const float scale,
                         const half* Xdata,
                         half* out,
                         uint8_t* mask,
                         nd_item<3> item_ct1)
{
    const float2* x_cast = reinterpret_cast<const float2*>(Xdata);
    float2* out_cast = reinterpret_cast<float2*>(out);
    const uint32_t* mask_cast = reinterpret_cast<const uint32_t*>(mask);

    float2 result_f;
    half2* result_h = reinterpret_cast<half2*>(&result_f);

    DPCPP_1D_KERNEL_LOOP(j, N / unroll_factor)
    {
        float2 x_data = x_cast[j];
        uint32_t m_32 = mask_cast[j];
        uint8_t* m = (uint8_t*)&m_32;

        half* x_data_h = reinterpret_cast<half*>(&x_data);
        float2 result[2];

        result[0].x() = (float)x_data_h[0] * scale * m[0];
        result[0].y() = (float)x_data_h[1] * scale * m[1];
        result[1].x() = (float)x_data_h[2] * scale * m[2];
        result[1].y() = (float)x_data_h[3] * scale * m[3];

        result_h[0] = result[0].convert<half, rounding_mode::rte>();
        result_h[1] = result[1].convert<half, rounding_mode::rte>();

        out_cast[j] = result_f;
    }
    int high_index = ((((N / unroll_factor) - 1) / item_ct1.get_local_range().get(2) + 1) *
                      (unroll_factor * item_ct1.get_local_range().get(2))) +
                     item_ct1.get_local_id(2);
    if (N > high_index) {
        for (int i = high_index; i < N; i++) {
            out[i] = vec<float, 1>{(float)Xdata[i] * scale * mask[i]}
                         .convert<half, rounding_mode::automatic>()[0];
        }
    }
}

template <typename T>
void launch_dropout_grad(T* vals_out,
                         const T* vals,
                         uint8_t* mask,
                         int total_count,
                         float ratio,
                         queue* stream)
{
    /*
     * Dropout.Backward1
     */
    assert(unroll_factor == 4);

    const float scale = 1. / (1. - ratio);
    range<3> grid_dim = range<3>(1, 1, DS_GET_BLOCKS(total_count / unroll_factor));
    range<3> block_dim = range<3>(1, 1, DS_CUDA_NUM_THREADS);
    stream->submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim), [=](nd_item<3> item_ct1) {
            dropout_grad_kernel(total_count, scale, vals, vals_out, mask, item_ct1);
        });
    });
}
template void launch_dropout_grad(float* vals_out,
                                  const float* vals,
                                  uint8_t* mask,
                                  int total_count,
                                  float ratio,
                                  queue* stream);
template void launch_dropout_grad(bf16* vals_out,
                                  const bf16* vals,
                                  uint8_t* mask,
                                  int total_count,
                                  float ratio,
                                  queue* stream);
template void launch_dropout_grad(half* vals_out,
                                  const half* vals,
                                  uint8_t* mask,
                                  int total_count,
                                  float ratio,
                                  queue* stream);

/*
 * not called in transformer kernel Shi Yuankun 2021/10/21
 */
void dropout_kernel(const int N,
                    const int dim,
                    const float ratio,
                    const float* bias,
                    float* Xdata,
                    uint8_t* mask,
                    const std::pair<uint64_t, uint64_t>& seed,
                    nd_item<3> item_ct1)
{
    const float scale = 1. / (1. - ratio);
    size_t idx =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
    int tid = item_ct1.get_local_id(2) % (dim / unroll_factor);

    oneapi::mkl::rng::device::philox4x32x10<4> engine(seed.first, {idx * 4, seed.second});
    oneapi::mkl::rng::device::uniform<> distr;

    float4* Xdata_cast = reinterpret_cast<float4*>(Xdata);
    uint32_t* mask_32 = reinterpret_cast<uint32_t*>(mask);
    const float4* bias_cast = reinterpret_cast<const float4*>(bias);

    DPCPP_1D_KERNEL_LOOP(j, N)
    {
        float4 rand = oneapi::mkl::rng::device::generate(distr, engine);
        uint32_t m_32;
        uint8_t* m = (uint8_t*)&m_32;

        m[0] = (uint8_t)(rand.x() > ratio);
        m[1] = (uint8_t)(rand.y() > ratio);
        m[2] = (uint8_t)(rand.z() > ratio);
        m[3] = (uint8_t)(rand.w() > ratio);

        float4 x_data = Xdata_cast[j];
        float4 b_data = bias_cast[j % (dim / unroll_factor)];

        x_data.x() += b_data.x();
        x_data.y() += b_data.y();
        x_data.z() += b_data.z();
        x_data.w() += b_data.w();

        x_data.x() = x_data.x() * scale * m[0];
        x_data.y() = x_data.y() * scale * m[1];
        x_data.z() = x_data.z() * scale * m[2];
        x_data.w() = x_data.w() * scale * m[3];

        mask_32[j] = m_32;
        Xdata_cast[j] = x_data;
    }
    int high_index = ((((N / unroll_factor) - 1) / item_ct1.get_local_range().get(2) + 1) *
                      (unroll_factor * item_ct1.get_local_range().get(2))) +
                     item_ct1.get_local_id(2);
    if (N > high_index) {
        float4 rand = oneapi::mkl::rng::device::generate(distr, engine);
        float* rand_data = &(rand.x());
        int k = 0;
        for (int i = high_index; i < N; i++) {
            float x_data = Xdata[i] + bias[i % dim];
            uint8_t m = (uint8_t)(rand_data[k++] > ratio);
            Xdata[i] = x_data * scale * m;
            mask[i] = m;
        }
    }
}

void dropout_kernel(const int N,
                    const int dim,
                    const float ratio,
                    const half* bias,
                    half* Xdata,
                    uint8_t* mask,
                    const std::pair<uint64_t, uint64_t>& seed,
                    nd_item<3> item_ct1)
{
    const float scale = 1. / (1. - ratio);
    size_t idx =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
    int tid = item_ct1.get_local_id(2) % (dim / unroll_factor);

    oneapi::mkl::rng::device::philox4x32x10<4> engine(seed.first, {idx * 4, seed.second});
    oneapi::mkl::rng::device::uniform<> distr;

    float2* Xdata_cast = reinterpret_cast<float2*>(Xdata);
    uint32_t* mask_32 = reinterpret_cast<uint32_t*>(mask);
    const float2* bias_cast = reinterpret_cast<const float2*>(bias);

    DPCPP_1D_KERNEL_LOOP(j, N)
    {
        float4 rand = oneapi::mkl::rng::device::generate(distr, engine);

        float2 data_f;
        half2* data_h = reinterpret_cast<half2*>(&data_f);

        float2 bias_f;
        half2* bias_h = reinterpret_cast<half2*>(&bias_f);

        data_f = Xdata_cast[j];
        bias_f = bias_cast[j % (dim / unroll_factor)];

        float2 data_h_0 = data_h[0].convert<float, rounding_mode::automatic>();
        float2 data_h_1 = data_h[1].convert<float, rounding_mode::automatic>();

        float2 bias_h_0 = bias_h[0].convert<float, rounding_mode::automatic>();
        float2 bias_h_1 = bias_h[1].convert<float, rounding_mode::automatic>();

        data_h_0.x() += bias_h_0.x();
        data_h_0.y() += bias_h_0.y();
        data_h_1.x() += bias_h_1.x();
        data_h_1.y() += bias_h_1.y();

        uint32_t m_32;
        uint8_t* m = (uint8_t*)&m_32;

        m[0] = (uint8_t)(rand.x() > ratio);
        m[1] = (uint8_t)(rand.y() > ratio);
        m[2] = (uint8_t)(rand.z() > ratio);
        m[3] = (uint8_t)(rand.w() > ratio);

        data_h_0.x() =
            vec<float, 1>{data_h_0.x() * scale * m[0]}.convert<half, rounding_mode::automatic>()[0];
        data_h_0.y() =
            vec<float, 1>{data_h_0.y() * scale * m[1]}.convert<half, rounding_mode::automatic>()[0];
        data_h_1.x() =
            vec<float, 1>{data_h_1.x() * scale * m[2]}.convert<half, rounding_mode::automatic>()[0];
        data_h_1.y() =
            vec<float, 1>{data_h_1.y() * scale * m[3]}.convert<half, rounding_mode::automatic>()[0];

        float2 result_f;
        half2* result_h = reinterpret_cast<half2*>(&result_f);

        result_h[0] = data_h_0.convert<half, rounding_mode::rte>();
        result_h[1] = data_h_1.convert<half, rounding_mode::rte>();

        Xdata_cast[j] = result_f;
        mask_32[j] = m_32;
    }
    int high_index = ((((N / unroll_factor) - 1) / item_ct1.get_local_range().get(2) + 1) *
                      (unroll_factor * item_ct1.get_local_range().get(2))) +
                     item_ct1.get_local_id(2);
    if (N > high_index) {
        float4 rand = oneapi::mkl::rng::device::generate(distr, engine);
        float* rand_data = &(rand.x());
        int k = 0;
        for (int i = high_index; i < N; i++) {
            float x_data = (float)Xdata[i] + (float)bias[i % dim];
            uint8_t m = (uint8_t)(rand_data[k++] > ratio);
            Xdata[i] =
                vec<float, 1>{x_data * scale * m}.convert<half, rounding_mode::automatic>()[0];
            mask[i] = m;
        }
    }
}

template <typename T>
void launch_dropout(T* out,
                    const T* bias,
                    uint8_t* mask,
                    int batch,
                    int dim,
                    float ratio,
                    queue* stream)
{
    assert(unroll_factor == 4);

    int total_count = batch * dim / unroll_factor;

    range<3> grid_dim = range<3>(1, 1, DS_GET_BLOCKS(total_count));
    range<3> block_dim = range<3>(1, 1, DS_CUDA_NUM_THREADS);

    uint64_t inc = (batch * dim) / grid_dim[2] / block_dim[2];
    std::pair<uint64_t, uint64_t> seed = SyclContext::Instance().IncrementOffset(inc);
    stream->submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim), [=](nd_item<3> item_ct1) {
            dropout_kernel(total_count, dim, ratio, bias, out, mask, seed, item_ct1);
        });
    });
}

template void launch_dropout(float*,
                             const float* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             queue* stream);
template void launch_dropout(half*,
                             const half* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             queue* stream);

void dropout_kernel(const int N,
                    const int dim,
                    const float ratio,
                    const float* input,
                    const float* residual,
                    const float* bias,
                    float* out,
                    uint8_t* mask,
                    const std::pair<uint64_t, uint64_t>& seed,
                    nd_item<3> item_ct1)
{
    const float scale = 1. / (1. - ratio);
    size_t idx =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
    int tid = item_ct1.get_local_id(2) % (dim / unroll_factor);

    oneapi::mkl::rng::device::philox4x32x10<4> engine(seed.first, {idx * 4, seed.second});
    oneapi::mkl::rng::device::uniform<> distr;

    float4* out_cast = reinterpret_cast<float4*>(out);
    uint32_t* mask_32 = reinterpret_cast<uint32_t*>(mask);

    const float4* bias_cast = reinterpret_cast<const float4*>(bias);
    const float4* residual_cast = reinterpret_cast<const float4*>(residual);
    const float4* input_cast = reinterpret_cast<const float4*>(input);

    DPCPP_1D_KERNEL_LOOP(j, N)
    {
        float4 rand = oneapi::mkl::rng::device::generate(distr, engine);

        uint32_t m_32;
        uint8_t* m = (uint8_t*)&m_32;

        m[0] = (uint8_t)(rand.x() > ratio);
        m[1] = (uint8_t)(rand.y() > ratio);
        m[2] = (uint8_t)(rand.z() > ratio);
        m[3] = (uint8_t)(rand.w() > ratio);

        float4 out_data;
        float4 b_data = bias_cast[j % (dim / unroll_factor)];
        float4 res_data = residual_cast[j];
        float4 inp_data = input_cast[j];

        out_data.x() = (b_data.x() + inp_data.x());
        out_data.y() = (b_data.y() + inp_data.y());
        out_data.z() = (b_data.z() + inp_data.z());
        out_data.w() = (b_data.w() + inp_data.w());

        out_data.x() = out_data.x() * scale * m[0];
        out_data.y() = out_data.y() * scale * m[1];
        out_data.z() = out_data.z() * scale * m[2];
        out_data.w() = out_data.w() * scale * m[3];

        out_data.x() += res_data.x();
        out_data.y() += res_data.y();
        out_data.z() += res_data.z();
        out_data.w() += res_data.w();

        mask_32[j] = m_32;
        out_cast[j] = out_data;
    }
    int high_index = ((((N / unroll_factor) - 1) / item_ct1.get_local_range().get(2) + 1) *
                      (unroll_factor * item_ct1.get_local_range().get(2))) +
                     item_ct1.get_local_id(2);
    if (N > high_index) {
        float4 rand = oneapi::mkl::rng::device::generate(distr, engine);
        float* rand_data = &(rand.x());
        int k = 0;
        for (int i = high_index; i < N; i++) {
            float x_data = input[i] + bias[i % dim];
            uint8_t m = (uint8_t)(rand_data[k++] > ratio);
            x_data = x_data * scale * m;
            x_data += residual[i];

            out[i] = x_data;
            mask[i] = m;
        }
    }
}

void dropout_kernel(const int N,
                    const int dim,
                    const float ratio,
                    const bf16* input,
                    const bf16* residual,
                    const bf16* bias,
                    bf16* out,
                    uint8_t* mask,
                    const std::pair<uint64_t, uint64_t>& seed,
                    nd_item<3> item_ct1)
{
    const float scale = 1. / (1. - ratio);
    size_t idx =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
    int tid = item_ct1.get_local_id(2) % (dim / unroll_factor);

    oneapi::mkl::rng::device::philox4x32x10<4> engine(seed.first, {idx * 4, seed.second});
    oneapi::mkl::rng::device::uniform<> distr;

    ushort4* out_cast = reinterpret_cast<ushort4*>(out);
    uint32_t* mask_32 = reinterpret_cast<uint32_t*>(mask);

    const ushort4* bias_cast = reinterpret_cast<const ushort4*>(bias);
    const ushort4* residual_cast = reinterpret_cast<const ushort4*>(residual);
    const ushort4* input_cast = reinterpret_cast<const ushort4*>(input);

    DPCPP_1D_KERNEL_LOOP(j, N)
    {
        float4 rand = oneapi::mkl::rng::device::generate(distr, engine);

        uint32_t m_32;
        uint8_t* m = (uint8_t*)&m_32;

        m[0] = (uint8_t)(rand.x() > ratio);
        m[1] = (uint8_t)(rand.y() > ratio);
        m[2] = (uint8_t)(rand.z() > ratio);
        m[3] = (uint8_t)(rand.w() > ratio);

        float4 out_data;
        float4 b_data = {
            float(bias_cast[j % (dim / unroll_factor)].x()),
            float(bias_cast[j % (dim / unroll_factor)].y()),
            float(bias_cast[j % (dim / unroll_factor)].z()),
            float(bias_cast[j % (dim / unroll_factor)].w()),
        };
        float4 res_data = {float(residual_cast[j].x()),
                           float(residual_cast[j].y()),
                           float(residual_cast[j].z()),
                           float(residual_cast[j].w())};
        float4 inp_data = {float(input_cast[j].x()),
                           float(input_cast[j].y()),
                           float(input_cast[j].z()),
                           float(input_cast[j].w())};

        out_data.x() = (b_data.x() + inp_data.x());
        out_data.y() = (b_data.y() + inp_data.y());
        out_data.z() = (b_data.z() + inp_data.z());
        out_data.w() = (b_data.w() + inp_data.w());

        out_data.x() = out_data.x() * scale * m[0];
        out_data.y() = out_data.y() * scale * m[1];
        out_data.z() = out_data.z() * scale * m[2];
        out_data.w() = out_data.w() * scale * m[3];

        out_data.x() += res_data.x();
        out_data.y() += res_data.y();
        out_data.z() += res_data.z();
        out_data.w() += res_data.w();

        mask_32[j] = m_32;
        out_cast[j] = {
            bf16(out_data.x()), bf16(out_data.y()), bf16(out_data.z()), bf16(out_data.w())};
    }
    int high_index = ((((N / unroll_factor) - 1) / item_ct1.get_local_range().get(2) + 1) *
                      (unroll_factor * item_ct1.get_local_range().get(2))) +
                     item_ct1.get_local_id(2);
    if (N > high_index) {
        ushort* out_cast = reinterpret_cast<ushort*>(out);
        const ushort* bias_cast = reinterpret_cast<const ushort*>(bias);
        const ushort* residual_cast = reinterpret_cast<const ushort*>(residual);
        const ushort* input_cast = reinterpret_cast<const ushort*>(input);
        float4 rand = oneapi::mkl::rng::device::generate(distr, engine);
        float* rand_data = &(rand.x());
        int k = 0;
        for (int i = high_index; i < N; i++) {
            float x_data = float(input_cast[i]) + float(bias_cast[i % dim]);
            uint8_t m = (uint8_t)(rand_data[k++] > ratio);
            x_data = x_data * scale * m;
            x_data += float(residual_cast[i]);

            out_cast[i] = bf16(x_data);
            mask[i] = m;
        }
    }
}

void dropout_kernel(const int N,
                    const int dim,
                    const float ratio,
                    const half* input,
                    const half* residual,
                    const half* bias,
                    half* out,
                    uint8_t* mask,
                    const std::pair<uint64_t, uint64_t>& seed,
                    nd_item<3> item_ct1)
{
    const float scale = 1. / (1. - ratio);
    size_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    int tid = item_ct1.get_local_id(2) % (dim / unroll_factor);

    oneapi::mkl::rng::device::philox4x32x10<4> engine(seed.first, {idx * 4, seed.second});
    oneapi::mkl::rng::device::uniform<> distr;

    float2* out_cast = reinterpret_cast<float2*>(out);
    uint32_t* mask_32 = reinterpret_cast<uint32_t*>(mask);

    const float2* bias_cast = reinterpret_cast<const float2*>(bias);
    const float2* residual_cast = reinterpret_cast<const float2*>(residual);
    const float2* input_cast = reinterpret_cast<const float2*>(input);

    DPCPP_1D_KERNEL_LOOP(j, N)
    {
        float4 rand = oneapi::mkl::rng::device::generate(distr, engine);

        float2 data_f;
        half2* data_h = reinterpret_cast<half2*>(&data_f);

        float2 bias_f;
        half2* bias_h = reinterpret_cast<half2*>(&bias_f);

        float2 residual_f;
        half2* residual_h = reinterpret_cast<half2*>(&residual_f);

        float2 input_f;
        half2* input_h = reinterpret_cast<half2*>(&input_f);

        bias_f = bias_cast[j % (dim / unroll_factor)];
        residual_f = residual_cast[j];
        input_f = input_cast[j];

        float2 data_h_0 = data_h[0].convert<float, rounding_mode::automatic>();
        float2 data_h_1 = data_h[1].convert<float, rounding_mode::automatic>();

        float2 bias_h_0 = bias_h[0].convert<float, rounding_mode::automatic>();
        float2 bias_h_1 = bias_h[1].convert<float, rounding_mode::automatic>();

        float2 residual_h_0 = residual_h[0].convert<float, rounding_mode::automatic>();
        float2 residual_h_1 = residual_h[1].convert<float, rounding_mode::automatic>();

        float2 input_h_0 = input_h[0].convert<float, rounding_mode::automatic>();
        float2 input_h_1 = input_h[1].convert<float, rounding_mode::automatic>();

        data_h_0.x() = (bias_h_0.x() + input_h_0.x());
        data_h_0.y() = (bias_h_0.y() + input_h_0.y());
        data_h_1.x() = (bias_h_1.x() + input_h_1.x());
        data_h_1.y() = (bias_h_1.y() + input_h_1.y());

        uint32_t m_32;
        uint8_t* m = (uint8_t*)&m_32;

        m[0] = (uint8_t)(rand.x() > ratio);
        m[1] = (uint8_t)(rand.y() > ratio);
        m[2] = (uint8_t)(rand.z() > ratio);
        m[3] = (uint8_t)(rand.w() > ratio);

        data_h_0.x() =
            vec<float, 1>{data_h_0.x() * scale * m[0]}.convert<half, rounding_mode::automatic>()[0];
        data_h_0.y() =
            vec<float, 1>{data_h_0.y() * scale * m[1]}.convert<half, rounding_mode::automatic>()[0];
        data_h_1.x() =
            vec<float, 1>{data_h_1.x() * scale * m[2]}.convert<half, rounding_mode::automatic>()[0];
        data_h_1.y() =
            vec<float, 1>{data_h_1.y() * scale * m[3]}.convert<half, rounding_mode::automatic>()[0];

        data_h_0.x() += residual_h_0.x();
        data_h_0.y() += residual_h_0.y();
        data_h_1.x() += residual_h_1.x();
        data_h_1.y() += residual_h_1.y();

        float2 result_f;
        half2* result_h = reinterpret_cast<half2*>(&result_f);

        result_h[0] = data_h_0.convert<half, rounding_mode::rte>();
        result_h[1] = data_h_1.convert<half, rounding_mode::rte>();

        out_cast[j] = result_f;
        mask_32[j] = m_32;
    }
    int high_index = ((((N / unroll_factor) - 1) / item_ct1.get_local_range().get(2) + 1) *
                      (unroll_factor * item_ct1.get_local_range().get(2))) +
                     item_ct1.get_local_id(2);
    if (N > high_index) {
        float4 rand = oneapi::mkl::rng::device::generate(distr, engine);
        float* rand_data = &(rand.x());
        int k = 0;
        for (int i = high_index; i < N; i++) {
            float x_data = (float)input[i] + (float)bias[i % dim];
            uint8_t m = (uint8_t)(rand_data[k++] > ratio);
            x_data = x_data * scale * m;
            x_data += (float)residual[i];

            out[i] = vec<float, 1>{x_data}.convert<half, rounding_mode::automatic>()[0];
            mask[i] = m;
        }
    }
}

template <typename T>
void launch_dropout(T* out,
                    const T* input,
                    const T* residual,
                    const T* bias,
                    uint8_t* mask,
                    int batch,
                    int dim,
                    float ratio,
                    queue* stream)
{
    assert(unroll_factor == 4);

    int total_count = batch * dim / unroll_factor;
    range<3> grid_dim = range<3>(1, 1, DS_GET_BLOCKS(total_count));
    range<3> block_dim = range<3>(1, 1, DS_CUDA_NUM_THREADS);

    uint64_t inc = (batch * dim) / grid_dim[2] / block_dim[2];
    std::pair<uint64_t, uint64_t> seed = SyclContext::Instance().IncrementOffset(inc);

    stream->submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim), [=](nd_item<3> item_ct1) {
            dropout_kernel(
                total_count, dim, ratio, input, residual, bias, out, mask, seed, item_ct1);
        });
    });
}

template void launch_dropout(float*,
                             const float*,
                             const float* residual,
                             const float* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             queue* stream);
template void launch_dropout(bf16*,
                             const bf16*,
                             const bf16* residual,
                             const bf16* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             queue* stream);
template void launch_dropout(half*,
                             const half*,
                             const half* residual,
                             const half* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             queue* stream);
