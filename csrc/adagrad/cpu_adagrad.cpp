// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "cpu_adagrad.h"
#include <torch/extension.h>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
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
void Adagrad_Optimizer::Step_1(T* _params,
                               T* grads,
                               float* _exp_avg_sq,
                               size_t _param_size,
                               ds_device_precision_t* dev_params)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<1>(
        &rounded_size, _params, grads, _exp_avg_sq, _param_size, dev_params);
#endif
    if (_param_size > rounded_size) {
        float step_size = -1 * _alpha;
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
                float momentum = grads[k];
                float variance = _exp_avg_sq[k];
                if (_weight_decay > 0) { grad = param * _weight_decay + grad; }

                variance += grad * grad;

                grad = sqrt(variance);
                grad += _eps;
                grad = momentum / grad;
                param = grad * step_size + param;
#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
                if (dev_params) _doubled_buffer[_buf_index][k - t] = param;
#endif
                _params[k] = (T)param;
                // STORE UPDATE TERM TO GRAD'S MEMORY
                grads[k] = grad * step_size;
                _exp_avg_sq[k] = variance;
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
void Adagrad_Optimizer::Step_4(T* _params,
                               T* grads,
                               float* _exp_avg_sq,
                               size_t _param_size,
                               ds_device_precision_t* dev_params)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<4>(
        &rounded_size, _params, grads, _exp_avg_sq, _param_size, dev_params);
#endif
    if (_param_size > rounded_size)
        Step_1((_params + rounded_size),
                (grads + rounded_size),
                (_exp_avg_sq + rounded_size),
                (_param_size - rounded_size),
                (dev_params != nullptr ? (dev_params + rounded_size) : dev_params));
}

int create_adagrad_optimizer(int optimizer_id,
                             float alpha = 1e-2,
                             float eps = 1e-8,
                             float weight_decay = 0,
                             bool should_log = false)
{
    auto opt = std::make_shared<Adagrad_Optimizer>(alpha, eps, weight_decay);

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

        printf("Adagrad Optimizer #%d is created with %s arithmetic capability.\n",
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, weight_decay=%f\n", alpha, weight_decay);
    }

    return 0;
}

template <typename T, typename ds_device_precision_t>
void Adagrad_Optimizer::Step_8(T* _params,
                               T* grads,
                               float* _exp_avg_sq,
                               size_t _param_size,
                               ds_device_precision_t* dev_params)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<8>(
        &rounded_size, _params, grads, _exp_avg_sq, _param_size, dev_params);
#endif
    if (_param_size > rounded_size)
        Step_4((_params + rounded_size),
                  (grads + rounded_size),
                  (_exp_avg_sq + rounded_size),
                  (_param_size - rounded_size),
                  (dev_params != nullptr ? (dev_params + rounded_size) : dev_params));
}

int ds_adagrad_step(int optimizer_id,
                    size_t step,
                    float lr,
                    float epsilon,
                    float weight_decay,
                    torch::Tensor& params,
                    torch::Tensor& grads,
                    torch::Tensor& exp_avg_sq)
{
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Adagrad_Optimizer> opt =
        std::static_pointer_cast<Adagrad_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step);
    opt->update_state(lr, epsilon, weight_decay);
    if (params.options().dtype() == at::kHalf)
        opt->Step_8(
            (c10::Half*)params_ptr, (c10::Half*)grads_ptr, exp_avg_sq_ptr, params_c.numel(), (DEVICE_FP16_DTYPE*)nullptr);
    else if (params.options().dtype() == at::kBFloat16)
        opt->Step_8(
            (c10::BFloat16*)params_ptr, (c10::BFloat16*)grads_ptr, exp_avg_sq_ptr, params_c.numel(), (DEVICE_FP16_DTYPE*)nullptr);
    else
        opt->Step_8(params_ptr, grads_ptr, exp_avg_sq_ptr, params_c.numel(), (DEVICE_FP16_DTYPE*)nullptr);

#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
    opt->SynchronizeStreams();
#endif
    return 0;
}

int ds_adagrad_step_plus_copy(int optimizer_id,
                              size_t step,
                              float lr,
                              float epsilon,
                              float weight_decay,
                              torch::Tensor& params,
                              torch::Tensor& grads,
                              torch::Tensor& exp_avg_sq,
                              torch::Tensor& gpu_params)
{
#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
    auto params_c = params.contiguous();
    auto gpu_params_c = gpu_params.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();
    auto grads_c = grads.contiguous();

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Adagrad_Optimizer> opt =
        std::static_pointer_cast<Adagrad_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step);
    opt->update_state(lr, epsilon, weight_decay);

    if (params.options().dtype() == at::kHalf)
        opt->Step_8(
            (c10::Half*)params_ptr, (c10::Half*)grads_ptr, exp_avg_sq_ptr, params_c.numel(), (DEVICE_FP16_DTYPE*)gpu_params_c.data_ptr());
    else if (params.options().dtype() == at::kBFloat16)
#if defined(DEVICE_BF16_DTYPE)
        opt->Step_8(
            (c10::BFloat16*)params_ptr, (c10::BFloat16*)grads_ptr, exp_avg_sq_ptr, params_c.numel(), (DEVICE_BF16_DTYPE*)gpu_params_c.data_ptr());
#else   
        throw std::runtime_error("BF16 not suppoted on device");
#endif
    else
        if (gpu_params_c.options().dtype() == at::kHalf)
            opt->Step_8(params_ptr,
                        grads_ptr,
                        exp_avg_sq_ptr,
                        params_c.numel(),
                        (DEVICE_FP16_DTYPE*)gpu_params_c.data_ptr());
        else
#if defined(DEVICE_BF16_DTYPE)
            opt->Step_8(params_ptr,
                        grads_ptr,
                        exp_avg_sq_ptr,
                        params_c.numel(),
                        (DEVICE_BF16_DTYPE*)gpu_params_c.data_ptr());
#else   
            throw std::runtime_error("BF16 not suppoted on device");
#endif

    opt->SynchronizeStreams();
#else
    assert(false);
#endif
    return 0;
}

int destroy_adagrad_optimizer(int optimizer_id)
{
    s_optimizers.erase(optimizer_id);

    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("adagrad_update", &ds_adagrad_step, "DeepSpeed CPU Adagrad update (C++)");
    m.def("adagrad_update_copy",
          &ds_adagrad_step_plus_copy,
          "DeepSpeed CPU Adagrad update and param copy (C++)");
    m.def("create_adagrad", &create_adagrad_optimizer, "DeepSpeed CPU Adagrad (C++)");
    m.def("destroy_adagrad", &destroy_adagrad_optimizer, "DeepSpeed CPU Adagrad destroy (C++)");
}
