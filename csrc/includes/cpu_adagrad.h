// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#define NOMINMAX  // Windows idiosyncrasy
                  // https://stackoverflow.com/questions/4913922/possible-problems-with-nominmax-on-visual-c

#include <stdio.h>
#include <cassert>
#include "simd.h"

#if defined(__ENABLE_CUDA__)
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include "cuda.h"
#include "custom_cuda_layers.h"
#define DEVICE_FP16_DTYPE __half
#ifdef BF16_AVAILABLE
#include <cuda_bf16.h>
#define DEVICE_BF16_DTYPE __nv_bfloat16
#endif
#elif defined(__ENABLE_CANN__)
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#define DEVICE_FP16_DTYPE c10::Half
#define DEVICE_BF16_DTYPE c10::BFloat16
#else
#include <cmath>
#define DEVICE_FP16_DTYPE c10::Half
#define DEVICE_BF16_DTYPE c10::BFloat16
#endif

#define STEP(SPAN)                                             \
    template <typename T, typename ds_device_precision_t>      \
    void Step_##SPAN(T* _params,                           \
                     T* grads,                             \
                     float* _exp_avg_sq,                       \
                     size_t _param_size,                       \
                     ds_device_precision_t* dev_param = nullptr);

class Adagrad_Optimizer {
public:
    Adagrad_Optimizer(float alpha = 1e-2, float eps = 1e-8, float weight_decay = 0)
        : _alpha(alpha), _eps(eps), _weight_decay(weight_decay)
    {
#if defined(__ENABLE_CUDA__)
        cudaMallocHost((void**)_doubled_buffer, TILE * sizeof(float));
        cudaMallocHost((void**)(_doubled_buffer + 1), TILE * sizeof(float));

        _streams[0] = TrainingContext::Instance().GetCurrentStream();
        _streams[1] = TrainingContext::Instance().GetNewStream();
        _buf_index = false;
#elif defined(__ENABLE_CANN__)
        aclrtMallocHost((void**)_doubled_buffer, TILE * sizeof(float));
        aclrtMallocHost((void**)(_doubled_buffer + 1), TILE * sizeof(float));

        _buf_index = false;
#endif
    }
    ~Adagrad_Optimizer()
    {
#if defined(__ENABLE_CUDA__)
        cudaFreeHost(_doubled_buffer[0]);
        cudaFreeHost(_doubled_buffer[1]);
#elif defined(__ENABLE_CANN__)
        aclrtFreeHost(_doubled_buffer[0]);
        aclrtFreeHost(_doubled_buffer[1]);
#endif
    }
#if defined(__AVX512__) or defined(__AVX256__)
    template <int span, typename T, typename ds_device_precision_t>
    void Step_AVX(size_t* rounded_size,
                  T* _params,
                  T* grads,
                  float* _exp_avg_sq,
                  size_t param_size,
                  ds_device_precision_t* dev_param = nullptr);
#endif
    STEP(1)
    STEP(4)
    STEP(8)
#if defined(__ENABLE_CUDA__)
    inline void SynchronizeStreams()
    {
        for (int i = 0; i < 2; i++) cudaStreamSynchronize(_streams[i]);
    }
#elif defined(__ENABLE_CANN__)
    inline void SynchronizeStreams()
    {
        for (int i = 0; i < 2; i++) aclrtSynchronizeStream(_streams[i].stream());
    }
#endif
    inline void IncrementStep(size_t step)
    {
        _step++;
        if (_step != step) { _step = step; }
    }
    inline void update_state(float lr, float epsilon, float weight_decay)
    {
        _alpha = lr;
        _eps = epsilon;
        _weight_decay = weight_decay;
    }

private:
    float _alpha;
    float _eps;
    float _weight_decay;

    float _betta1_t;
    float _betta2_t;
    size_t _step;

#if defined(__ENABLE_CUDA__)
    bool _buf_index;
    float* _doubled_buffer[2];
    cudaStream_t _streams[2];
#elif defined(__ENABLE_CANN__)
    float* _doubled_buffer[2];
    c10_npu::NPUStream _streams[2] = {c10_npu::getCurrentNPUStream(),
                                      c10_npu::getNPUStreamFromPool()};
    bool _buf_index;
#endif
};

#if defined(__AVX512__) or defined(__AVX256__)
template <int span, typename T, typename ds_device_precision_t>
void Adagrad_Optimizer::Step_AVX(size_t* rounded_size,
                                 T* _params,
                                 T* grads,
                                 float* _exp_avg_sq,
                                 size_t _param_size,
                                 ds_device_precision_t* dev_params)
{
#if !defined(__AVX512__)
    if (std::is_same_v<T, c10::BFloat16>) { return; }
#endif
    size_t new_rounded_size = 0;
    AVX_Data eps_4;
    eps_4.data = SIMD_SET(_eps);

    float step_size = -1 * _alpha;
    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    AVX_Data weight_decay4;
    if (_weight_decay > 0) weight_decay4.data = SIMD_SET(_weight_decay);
    new_rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * span);
    for (size_t t = 0; t < new_rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > new_rounded_size) copy_size = new_rounded_size - t;
        size_t offset = copy_size + t;
#if defined(__ENABLE_CUDA__)
        if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#elif defined(__ENABLE_CANN__)
        if ((t / TILE) >= 2) { aclrtSynchronizeStream(_streams[_buf_index].stream()); }
#endif
#pragma omp parallel for
        for (size_t i = t; i < offset; i += SIMD_WIDTH * span) {
            AVX_Data grad_4[span];
            simd_load<span, T>(grad_4, grads + i);

            AVX_Data momentum_4[span];
            simd_load<span, float>(momentum_4, grads + i);

            AVX_Data variance_4[span];
            simd_load<span, float>(variance_4, _exp_avg_sq + i);

            AVX_Data param_4[span];
            simd_load<span, T>(param_4, _params + i);

            if (_weight_decay > 0) { simd_fma<span>(grad_4, param_4, weight_decay4, grad_4); }

            simd_fma<span>(variance_4, grad_4, grad_4, variance_4);
            simd_sqrt<span>(grad_4, variance_4);
            simd_add<span>(grad_4, grad_4, eps_4);
            simd_div<span>(grad_4, momentum_4, grad_4);
            simd_fma<span>(param_4, grad_4, step_size_4, param_4);

            simd_store<span, T>(_params + i, param_4);
#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
            if (dev_params) { simd_store<span, T>((T*)(_doubled_buffer[_buf_index] + (i - t)), param_4); }
#endif
            simd_store<span, float>(_exp_avg_sq + i, variance_4);
        }
#if defined(__ENABLE_CUDA__)
        if (dev_params) {
            if (sizeof(T) == 2)
                launch_param_update_half(
                    _doubled_buffer[_buf_index], dev_params + t, copy_size, _streams[_buf_index]);
            else
                launch_param_update(
                    _doubled_buffer[_buf_index], dev_params + t, copy_size, _streams[_buf_index]);

            _buf_index = !_buf_index;
        }
#elif defined(__ENABLE_CANN__)
        if (dev_params) {
            size_t memcpy_size = copy_size * sizeof(_doubled_buffer[_buf_index][0]);
            if (sizeof(T) == 2) memcpy_size /= 2;
            aclrtMemcpy(dev_params + t,
                        memcpy_size,
                        _doubled_buffer[_buf_index],
                        memcpy_size,
                        aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE);

            _buf_index = !_buf_index;
        }
#endif
    }
    *rounded_size = new_rounded_size;
}
#endif
