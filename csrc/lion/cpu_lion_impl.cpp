// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include "cpu_lion.h"

#if defined(__ENABLE_CUDA__)
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"
#include "custom_cuda_layers.h"
#endif

static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

// C++ interface

template <typename T, typename ds_device_precision_t>
void Lion_Optimizer::Step_1(T* _params,
                            T* grads,
                            float* _exp_avg,
                            size_t _param_size,
                            ds_device_precision_t* dev_params,
                            bool half_precision)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<1>(
        &rounded_size, _params, grads, _exp_avg, _param_size, dev_params);
#endif
    if (_param_size > rounded_size) {
        float betta1_minus1 = 1 - _betta1;
        float betta2_minus1 = 1 - _betta2;

        float alpha = _alpha;
        float after_decay = 1 - alpha * _weight_decay;

        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#if defined(__ENABLE_CUDA__)
            if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#elif defined(__ENABLE_CANN__)
            if ((t / TILE) >= 2) { aclrtSynchronizeStream(_streams[_buf_index].stream()); }
#endif
#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = (float)grads[k];
                float param = (float)_params[k];
                float momentum = _exp_avg[k];
                float tmp = momentum * _betta1;
                tmp = grad * betta1_minus1 + tmp;
                // Rely on portable C++ methods to manipulate the sign bit of a floating-point
                // number.
                tmp = -std::copysignf(alpha, tmp);
                if (_weight_decay > 0) {
                    param = param * after_decay + tmp;
                } else {
                    param = param + tmp;
                }
                momentum = momentum * _betta2;
                momentum = grad * betta2_minus1 + momentum;
#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
                if (dev_params) _doubled_buffer[_buf_index][k - t] = param;
#endif
                _params[k] = (T)param;
                _exp_avg[k] = momentum;
            }
#if defined(__ENABLE_CUDA__)
            if (dev_params) {
                launch_param_update(
                    _doubled_buffer[_buf_index], dev_params + t, (copy_size), _streams[_buf_index]);

                _buf_index = !_buf_index;
            }
#elif defined(__ENABLE_CANN__)
            if (dev_params) {
                size_t memcpy_size = copy_size * sizeof(_doubled_buffer[_buf_index][0]);
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
}

template <typename T, typename ds_device_precision_t>
void Lion_Optimizer::Step_4(T* _params,
                            T* grads,
                            float* _exp_avg,
                            size_t _param_size,
                            ds_device_precision_t* dev_params)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<4>(
        &rounded_size, _params, grads, _exp_avg, _param_size, dev_params);
#endif
    if (_param_size > rounded_size)
        Step_1((_params + rounded_size),
                  (grads + rounded_size),
                  (_exp_avg + rounded_size),
                  (_param_size - rounded_size),
                  (dev_params != nullptr ? (dev_params + rounded_size) : dev_params));
}

int create_lion_optimizer(int optimizer_id,
                          float alpha,
                          float betta1,
                          float betta2,
                          float weight_decay,
                          bool should_log)
{
    auto opt = std::make_shared<Lion_Optimizer>(alpha, betta1, betta2, weight_decay);

    s_optimizers[optimizer_id] = opt;

    if (should_log) {
        std::string avx_type = "";
#if defined(__AVX512__)
        avx_type = "AVX512";
#else
#if defined(__AVX256__)
        avx_type = "AVX2";
#else
        avx_type = "scalar";
#endif
#endif

        printf("Lion Optimizer #%d is created with %s arithmetic capability.\n",
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, betas=(%f, %f), weight_decay=%f\n",
               alpha,
               betta1,
               betta2,
               weight_decay);
    }

    return 0;
}

template <typename T, typename ds_device_precision_t>
void Lion_Optimizer::Step_8(T* _params,
                            T* grads,
                            float* _exp_avg,
                            size_t _param_size,
                            ds_device_precision_t* dev_params)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<8(
        &rounded_size, _params, grads, _exp_avg, _param_size, dev_params);
#endif
    if (_param_size > rounded_size)
        Step_4((_params + rounded_size),
                  (grads + rounded_size),
                  (_exp_avg + rounded_size),
                  (_param_size - rounded_size),
                  (dev_params != nullptr ? (dev_params + rounded_size) : dev_params));
}

int ds_lion_step(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float weight_decay,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg)
{
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();

    // assert(params.options().dtype() == grads.options().dtype());

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();

    std::shared_ptr<Lion_Optimizer> opt =
        std::static_pointer_cast<Lion_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, weight_decay);

    if (params.options().dtype() == at::kHalf)
        opt->Step_8((c10::Half*)params_ptr, (c10::Half*)grads_ptr, exp_avg_ptr, params_c.numel(), (DEVICE_FP16_DTYPE*)nullptr);
    else if (params.options().dtype() == at::kBFloat16)
        opt->Step_8(
            (c10::BFloat16*)params_ptr, (c10::BFloat16*)grads_ptr, exp_avg_ptr, params_c.numel(), (DEVICE_FP16_DTYPE*)nullptr);
    else
        opt->Step_8(params_ptr, grads_ptr, exp_avg_ptr, params_c.numel(), (DEVICE_FP16_DTYPE*)nullptr);

#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
    opt->SynchronizeStreams();
#endif
    return 0;
}

int ds_lion_step_plus_copy(int optimizer_id,
                           size_t step,
                           float lr,
                           float beta1,
                           float beta2,
                           float weight_decay,
                           torch::Tensor& params,
                           torch::Tensor& grads,
                           torch::Tensor& exp_avg,
                           torch::Tensor& gpu_params)
{
#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
    auto params_c = params.contiguous();
    auto gpu_params_c = gpu_params.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto grads_c = grads.contiguous();

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();

    std::shared_ptr<Lion_Optimizer> opt =
        std::static_pointer_cast<Lion_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, weight_decay);

    if (params.options().dtype() == at::kHalf)
        opt->Step_8(
            (c10::Half*)params_ptr, (c10::Half*)grads_ptr, exp_avg_ptr, params_c.numel(), (DEVICE_FP16_DTYPE*)gpu_params_c.data_ptr());
    else if (params.options().dtype() == at::kBFloat16)
#if defined(DEVICE_BF16_DTYPE)
        opt->Step_8(
            (c10::BFloat16*)params_ptr, (c10::BFloat16*)grads_ptr, exp_avg_ptr, params_c.numel(), (DEVICE_BF16_DTYPE*)gpu_params_c.data_ptr());
#else   
        throw std::runtime_error("BF16 not suppoted on device");
#endif
    else
        if (gpu_params_c.options().dtype() == at::kHalf)
            opt->Step_8(
                params_ptr, grads_ptr, exp_avg_ptr, params_c.numel(), (DEVICE_FP16_DTYPE*)gpu_params_c.data_ptr());
        else
#if defined(DEVICE_BF16_DTYPE)
            opt->Step_8(
                params_ptr, grads_ptr, exp_avg_ptr, params_c.numel(), (DEVICE_BF16_DTYPE*)gpu_params_c.data_ptr());
#else   
            throw std::runtime_error("BF16 not suppoted on device");
#endif

    opt->SynchronizeStreams();
#else
    assert(false);
#endif
    return 0;
}

int destroy_lion_optimizer(int optimizer_id)
{
    s_optimizers.erase(optimizer_id);

    return 0;
}
