// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#define NOMINMAX  // Windows idiosyncrasy
                  // https://stackoverflow.com/questions/4913922/possible-problems-with-nominmax-on-visual-c

#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <cassert>
#include <algorithm>
#include <tuple>
#include "simd.h"

#if defined(__ENABLE_CUDA__)
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include "cuda.h"
#include "custom_cuda_layers.h"
typedef __half ds_half_precision_t;
#elif defined(__ENABLE_CANN__)
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
typedef c10::Half ds_half_precision_t;
#else
#include <cmath>
typedef unsigned short ds_half_precision_t;
#endif

#define STEP(SPAN)                                              \
    std::pair<float, float> Step_##SPAN(float* _params,         \
                     float* grads,                              \
                     float* _exp_avg,                           \
                     float* _exp_avg_sq,                        \
                     size_t _param_size,                        \
                     ds_half_precision_t* dev_param = nullptr,  \
                     bool half_precision_param = false,         \
                     bool half_precision_opt_state = false,     \
                     float avg_scaling = 1,                     \
                     float avg_sq_scaling = 1,                  \
                     float new_avg_scaling = 1,                 \
                     float new_avg_sq_scaling = 1);

class Adam_Optimizer {
public:
    Adam_Optimizer(float alpha = 1e-3,
                   float betta1 = 0.9,
                   float betta2 = 0.999,
                   float eps = 1e-8,
                   float weight_decay = 0,
                   bool adamw_mode = true,
                   bool use_cuda_cache = false)
        : _alpha(alpha),
          _betta1(betta1),
          _betta2(betta2),
          _eps(eps),
          _weight_decay(weight_decay),
          _betta1_t(1.0),
          _betta2_t(1.0),
          _step(0),
          _adamw_mode(adamw_mode),
          _use_cuda_cache(use_cuda_cache)
    {

#if defined(__AVX512__)
    printf("We will use AVX512\n");
#endif
#if defined(__AVX256__)
    printf("We will use AVX2\n");
#endif
    _doubled_buffer[0] = _doubled_buffer[1] = nullptr;
#if defined(__ENABLE_CUDA__)
    _streams[0] = TrainingContext::Instance().GetCurrentStream();
    _streams[1] = TrainingContext::Instance().GetNewStream();
    _buf_index = false;
    if (_use_cuda_cache) {
        cudaMallocHost((void**)_doubled_buffer, TILE * sizeof(float));
        cudaMallocHost((void**)(_doubled_buffer + 1), TILE * sizeof(float));
        printf("CPU Adam will use %.3fGB CUDA memory\n",  (float) TILE * sizeof(float) / 1024.0 / 1024.0);
        }
#elif defined(__ENABLE_CANN__)
    if (_use_cuda_cache) {
        aclrtMallocHost((void**)_doubled_buffer, TILE * sizeof(float));
        aclrtMallocHost((void**)(_doubled_buffer + 1), TILE * sizeof(float));
        }
#endif
    }
    ~Adam_Optimizer()
    {
    if (_use_cuda_cache) {
#if defined(__ENABLE_CUDA__)
        cudaFreeHost(_doubled_buffer[0]);
        cudaFreeHost(_doubled_buffer[1]);
#elif defined(__ENABLE_CANN__)
        aclrtFreeHost(_doubled_buffer[0]);
        aclrtFreeHost(_doubled_buffer[1]);
#endif
        }
    }

#if defined(__AVX512__) or defined(__AVX256__) or defined(__AVX2__)
    template <int span>
    std::pair<float, float> Step_AVX(size_t* rounded_size,
                  float* _params,
                  float* grads,
                  float* _exp_avg,
                  float* _exp_avg_sq,
                  size_t param_size,
                  ds_half_precision_t* dev_param = nullptr,
                  bool half_precision = false,
                  bool half_precision_opt_state = false,
                  float avg_scaling = 1, 
                  float avg_sq_scaling = 1, 
                  float new_avg_scaling = 1, 
                  float new_avg_sq_scaling = 1);
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
    inline void IncrementStep(size_t step, float beta1, float beta2)
    {
        if (beta1 != _betta1 || beta2 != _betta2) {
            _step = step;
            _betta1 = beta1;
            _betta2 = beta2;
            _betta1_t = std::pow(_betta1, step);
            _betta2_t = std::pow(_betta2, step);
        } else {
            _step++;
            if (_step != step) {
                _betta1_t = std::pow(_betta1, step);
                _betta2_t = std::pow(_betta2, step);
                _step = step;
            } else {
                _betta1_t *= _betta1;
                _betta2_t *= _betta2;
            }
        }
    }
    inline void update_state(float lr, float epsilon, float weight_decay, bool bias_correction)
    {
        _alpha = lr;
        _eps = epsilon;
        _weight_decay = weight_decay;

        _bias_correction1 = 1.0f;
        _bias_correction2 = 1.0f;
        if (bias_correction == 1) {
            _bias_correction1 = 1 - _betta1_t;
            _bias_correction2 = 1 / sqrt(1 - _betta2_t);
        }
    }
    inline bool use_cuda_cache(void) { return _use_cuda_cache; }

private:
    float _alpha;
    float _betta1;
    float _betta2;
    float _eps;
    float _weight_decay;

    float _betta1_t;
    float _betta2_t;
    size_t _step;

    float _bias_correction1;
    float _bias_correction2;

    bool _adamw_mode;
    bool _use_cuda_cache;

#if defined(__ENABLE_CUDA__)
    float* _doubled_buffer[2];
    cudaStream_t _streams[2];
    bool _buf_index;
#elif defined(__ENABLE_CANN__)
    float* _doubled_buffer[2];
    c10_npu::NPUStream _streams[2] = {c10_npu::getCurrentNPUStream(),
                                      c10_npu::getNPUStreamFromPool()};
    bool _buf_index;
#endif
};

#if defined(__AVX512__) or defined(__AVX256__)
template <int span>
std::pair<float, float> Adam_Optimizer::Step_AVX(size_t* rounded_size,
                              float* _params,
                              float* grads,
                              float* _exp_avg,
                              float* _exp_avg_sq,
                              size_t _param_size,
                              ds_half_precision_t* dev_params,
                              bool half_precision_param,
                              bool half_precision_opt_state,
                              float avg_scaling, 
                              float avg_sq_scaling, 
                              float new_avg_scaling, 
                              float new_avg_sq_scaling)
{
    size_t new_rounded_size = 0;
    int rshft_param = half_precision_param ? 1 : 0;
    int rshft_opt_state = half_precision_opt_state ? 1 : 0;

    AVX_Data betta1_4;
    betta1_4.data = SIMD_SET(_betta1);
    AVX_Data betta2_4;
    betta2_4.data = SIMD_SET(_betta2);


    AVX_Data avg_scaling_4;
    avg_scaling_4.data = SIMD_SET(avg_scaling);
    AVX_Data avg_sq_scaling_4;
    avg_sq_scaling_4.data = SIMD_SET(avg_sq_scaling);

    AVX_Data new_avg_scaling_4;
    new_avg_scaling_4.data = SIMD_SET(new_avg_scaling);
    AVX_Data new_avg_sq_scaling_4;
    new_avg_sq_scaling_4.data = SIMD_SET(new_avg_sq_scaling);

    int OMP_MAX_THREADS = omp_get_max_threads();
    float max_avg[OMP_MAX_THREADS], max_avg_sq[OMP_MAX_THREADS];
    std::fill(max_avg, max_avg + OMP_MAX_THREADS, 0.0f);
    std::fill(max_avg_sq, max_avg_sq + OMP_MAX_THREADS, 0.0f);

    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;
    AVX_Data betta1_minus1_4;
    betta1_minus1_4.data = SIMD_SET(betta1_minus1);
    AVX_Data betta2_minus1_4;
    betta2_minus1_4.data = SIMD_SET(betta2_minus1);

    AVX_Data bias2_sqrt;
    bias2_sqrt.data = SIMD_SET(_bias_correction2);

    AVX_Data eps_4;
    eps_4.data = SIMD_SET(_eps);

    float step_size = -1 * _alpha / _bias_correction1;
    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    float w_decay = -1 * _alpha * _weight_decay;
    AVX_Data weight_decay4;
    if (_weight_decay > 0)
        weight_decay4.data = (_adamw_mode ? SIMD_SET(w_decay) : SIMD_SET(_weight_decay));
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

#pragma omp parallel shared(max_avg, max_avg_sq)
        {
        int idx = omp_get_thread_num();
#pragma omp for
        for (size_t i = t; i < offset; i += SIMD_WIDTH * span) {
            AVX_Data param_4[span];
            simd_load<span>(param_4, _params + (i >> rshft_param), half_precision_param);

            AVX_Data grad_4[span];
            simd_load<span>(grad_4, grads + (i >> rshft_param), half_precision_param);

            AVX_Data momentum_4[span];
            simd_load<span>(momentum_4, _exp_avg + (i >> rshft_opt_state), half_precision_opt_state);

            AVX_Data variance_4[span];
            simd_load<span>(variance_4, _exp_avg_sq + (i >> rshft_opt_state), half_precision_opt_state);

            if (_weight_decay > 0 && !_adamw_mode) {
                simd_fma<span>(grad_4, param_4, weight_decay4, grad_4);
            }

            simd_mul<span>(momentum_4, momentum_4, betta1_4);
            simd_div<span>(momentum_4, momentum_4, avg_scaling_4);

            simd_fma<span>(momentum_4, grad_4, betta1_minus1_4, momentum_4);
            simd_mul<span>(variance_4, variance_4, betta2_4);
            simd_div<span>(variance_4, variance_4, avg_sq_scaling_4);

            simd_mul<span>(grad_4, grad_4, grad_4);
            simd_fma<span>(variance_4, grad_4, betta2_minus1_4, variance_4);
            simd_sqrt<span>(grad_4, variance_4);
            simd_fma<span>(grad_4, grad_4, bias2_sqrt, eps_4);
            simd_div<span>(grad_4, momentum_4, grad_4);

            if (_weight_decay > 0 && _adamw_mode) {
                simd_fma<span>(param_4, param_4, weight_decay4, param_4);
            }

            simd_fma<span>(param_4, grad_4, step_size_4, param_4);

            simd_store<span>(_params + (i >> rshft_param), param_4, half_precision_param);
#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
            if (dev_params) {
                simd_store<span>(_doubled_buffer[_buf_index] + (i - t), param_4, half_precision_param);
            }
#endif

            max_avg[idx] = std::max(max_avg[idx], simd_max_abs<span>(momentum_4));
            max_avg_sq[idx] = std::max(max_avg_sq[idx], simd_max<span>(variance_4));
            
            simd_mul<span>(momentum_4, momentum_4, new_avg_scaling_4);
            simd_mul<span>(variance_4, variance_4, new_avg_sq_scaling_4);

            simd_store<span>(_exp_avg + (i >> rshft_opt_state), momentum_4, half_precision_opt_state);
            simd_store<span>(_exp_avg_sq + (i >> rshft_opt_state), variance_4, half_precision_opt_state);

        }
#if defined(__ENABLE_CUDA__)
        if (dev_params) {
            if (half_precision_param)
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
            if (half_precision_param) memoryCopySize /= 2;
            aclrtMemcpy(dev_params + t,
                        memcpy_size,
                        _doubled_buffer[_buf_index],
                        memcpy_size,
                        aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE);

            _buf_index = !_buf_index;
        }
#endif
    }
    }
    *rounded_size = new_rounded_size;
    auto ret =  std::make_pair(*std::max_element(max_avg, max_avg + OMP_MAX_THREADS), *std::max_element(max_avg_sq, max_avg_sq + OMP_MAX_THREADS));
    return ret;
}
#endif

int create_adam_optimizer(int optimizer_id,
                          float alpha = 1e-3,
                          float betta1 = 0.9,
                          float betta2 = 0.999,
                          float eps = 1e-8,
                          float weight_decay = 0,
                          bool adamw_mode = true,
                          bool should_log = false,
                          bool use_cuda_cache = false);

std::pair<float, float> ds_adam_step(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float epsilon,
                 float weight_decay,
                 bool bias_correction,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg,
                 torch::Tensor& exp_avg_sq,
                 float avg_scaling, 
                 float avg_sq_scaling, 
                 float new_avg_scaling, 
                 float new_avg_sq_scaling);

std::pair<float, float> ds_adam_step_plus_copy(int optimizer_id,
                           size_t step,
                           float lr,
                           float beta1,
                           float beta2,
                           float epsilon,
                           float weight_decay,
                           bool bias_correction,
                           torch::Tensor& params,
                           torch::Tensor& grads,
                           torch::Tensor& exp_avg,
                           torch::Tensor& exp_avg_sq,
                           torch::Tensor& gpu_params,
                           float avg_scaling, 
                           float avg_sq_scaling, 
                           float new_avg_scaling, 
                           float new_avg_sq_scaling);

int destroy_adam_optimizer(int optimizer_id);
