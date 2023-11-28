// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>
#include <cassert>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include "cpu_adam.h"

#if defined(__ENABLE_CUDA__)
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"
#include "custom_cuda_layers.h"
#endif

static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

// C++ interface

std::pair<float, float> Adam_Optimizer::Step_1(float* _params,
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
    size_t rounded_size = 0;
    std::pair<float, float> ret = std::make_pair(0.0f, 0.0f);
    
#if defined(__AVX512__) or defined(__AVX256__)
    auto avx_ret = Step_AVX<1>(&rounded_size,
                _params,
                grads,
                _exp_avg,
                _exp_avg_sq,
                _param_size,
                dev_params,
                half_precision_param,
                half_precision_opt_state,
                avg_scaling,
                avg_sq_scaling,
                new_avg_scaling,
                new_avg_sq_scaling);
    ret = std::max(ret, avx_ret);
#endif

    if (_param_size > rounded_size) {
        float betta1_minus1 = 1 - _betta1, betta2_minus1 = 1 - _betta2;
        float step_size = -1 * _alpha / _bias_correction1, w_decay = -1 * _alpha * _weight_decay;
        float betta1 = _betta1, betta2 = _betta2;
        
        ds_half_precision_t *grads_cast_h, *params_cast_h, *exp_avg_cast_h, *exp_avg_sq_cast_h; 

        if (half_precision_param) {
            params_cast_h = reinterpret_cast<ds_half_precision_t*>(_params);
            grads_cast_h = reinterpret_cast<ds_half_precision_t*>(grads);
        }

        if (half_precision_opt_state) {
            exp_avg_cast_h = reinterpret_cast<ds_half_precision_t*>(_exp_avg);
            exp_avg_sq_cast_h = reinterpret_cast<ds_half_precision_t*>(_exp_avg_sq);
        }

        const int OMP_MAX_THREADS = omp_get_max_threads();
        float tmp_max_avg[OMP_MAX_THREADS], tmp_max_var[OMP_MAX_THREADS];

        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#if defined(__ENABLE_CUDA__)
            if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#elif defined(__ENABLE_CANN__)
            if ((t / TILE) >= 2) { aclrtSynchronizeStream(_streams[_buf_index].stream()); }
#endif  
            std::fill(tmp_max_avg, tmp_max_avg + OMP_MAX_THREADS, 0.0f);
            std::fill(tmp_max_var, tmp_max_var + OMP_MAX_THREADS, 0.0f);

#pragma omp parallel shared(tmp_max_avg, tmp_max_var)
            {
            int idx = omp_get_thread_num();
#pragma omp for
            for (size_t k = t; k < offset; k++) {
                float grad = half_precision_param ? (float)grads_cast_h[k] : grads[k];
                float param = half_precision_param ? (float)params_cast_h[k] : _params[k];
                float momentum = half_precision_opt_state ? (float)exp_avg_cast_h[k] : _exp_avg[k];
                float variance = half_precision_opt_state ? (float)exp_avg_sq_cast_h[k] : _exp_avg_sq[k];
                if (_weight_decay > 0 && !_adamw_mode) { grad = param * _weight_decay + grad; }
                momentum = momentum * betta1 / avg_scaling;
                momentum = grad * betta1_minus1 + momentum;

                variance = variance * betta2 / avg_sq_scaling;
                grad = grad * grad;
                variance = grad * betta2_minus1 + variance;

                tmp_max_avg[idx] = std::max(tmp_max_avg[idx], std::fabs(momentum));
                tmp_max_var[idx] = std::max(tmp_max_var[idx], variance);

                grad = sqrt(variance);
                grad = grad * _bias_correction2 + _eps;
                grad = momentum / grad;
                if (_weight_decay > 0 && _adamw_mode) { param += w_decay * param; }
                param = grad * step_size + param;
#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
                if (dev_params) _doubled_buffer[_buf_index][k - t] = param;
#endif
                if (half_precision_param)
                    params_cast_h[k] = (ds_half_precision_t)param;
                else
                    _params[k] = param;
                if (half_precision_opt_state) {
                    exp_avg_cast_h[k] = (ds_half_precision_t) (momentum * new_avg_scaling);
                    exp_avg_sq_cast_h[k] = (ds_half_precision_t) (variance * new_avg_sq_scaling);
                }
                else {
                    _exp_avg[k] = momentum;
                    _exp_avg_sq[k] = variance;
                }
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
        ret = std::max(ret, std::make_pair(
                                *std::max_element(tmp_max_avg, tmp_max_avg + OMP_MAX_THREADS),
                                *std::max_element(tmp_max_var, tmp_max_var + OMP_MAX_THREADS)));
        }
    }
    return ret;
}

std::pair<float, float> Adam_Optimizer::Step_4(float* _params,
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
    size_t rounded_size = 0;
    std::pair<float, float> ret = std::make_pair(0.0f, 0.0f);

#if defined(__AVX512__) or defined(__AVX256__)
    
    auto ret_avx = Step_AVX<4>(&rounded_size,
                _params,
                grads,
                _exp_avg,
                _exp_avg_sq,
                _param_size,
                dev_params,
                half_precision_param,
                half_precision_opt_state,
                avg_scaling,
                avg_sq_scaling, 
                new_avg_scaling, 
                new_avg_sq_scaling);
    ret = std::max(ret, ret_avx);

#endif
    if (_param_size > rounded_size) {
        auto ret_remain = Step_1((_params + (rounded_size >> static_cast<int> (half_precision_param))),
               (grads + (rounded_size >> static_cast<int> (half_precision_param))),
               (_exp_avg + (rounded_size >> static_cast<int> (half_precision_opt_state))),
               (_exp_avg_sq + (rounded_size >> static_cast<int> (half_precision_opt_state))),
               (_param_size - rounded_size),
               (dev_params != nullptr ? (dev_params + rounded_size) : dev_params),
               half_precision_param,
               half_precision_opt_state,
               avg_scaling,
               avg_sq_scaling, 
               new_avg_scaling, 
               new_avg_sq_scaling);
        ret = std::max(ret, ret_remain);
    }
    return ret;
}

int create_adam_optimizer(int optimizer_id,
                          float alpha,
                          float betta1,
                          float betta2,
                          float eps,
                          float weight_decay,
                          bool adamw_mode,
                          bool should_log,
                          bool use_cuda_cache)
{
    auto opt = std::make_shared<Adam_Optimizer>(alpha, betta1, betta2, eps, weight_decay, adamw_mode, use_cuda_cache);

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

        printf("Adam Optimizer #%d is created with %s arithmetic capability.\n",
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, betas=(%f, %f), weight_decay=%f, adam_w=%d\n",
               alpha,
               betta1,
               betta2,
               weight_decay,
               (int)adamw_mode);
    }

    return 0;
}

std::pair<float, float> Adam_Optimizer::Step_8(float* _params,
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
    size_t rounded_size = 0;
    std::pair<float, float> ret = std::make_pair(0.0f, 0.0f);

#if defined(__AVX512__) or defined(__AVX256__)
    auto ret_avx = Step_AVX<8>(&rounded_size,
                _params,
                grads,
                _exp_avg,
                _exp_avg_sq,
                _param_size,
                dev_params,
                half_precision_param,
                half_precision_opt_state, 
                avg_scaling,
                avg_sq_scaling, 
                new_avg_scaling, 
                new_avg_sq_scaling);
    ret = std::max(ret, ret_avx);
#endif
    if (_param_size > rounded_size) {
        auto ret_remain = Step_4((_params + (rounded_size >> static_cast<int> (half_precision_param))),
               (grads + (rounded_size >> static_cast<int> (half_precision_param))),
               (_exp_avg + (rounded_size >> static_cast<int> (half_precision_opt_state))),
               (_exp_avg_sq + (rounded_size >> static_cast<int> (half_precision_opt_state))),
               (_param_size - rounded_size),
               (dev_params != nullptr ? (dev_params + rounded_size) : dev_params),
               half_precision_param,
               half_precision_opt_state,
               avg_scaling,
               avg_sq_scaling, 
               new_avg_scaling, 
               new_avg_sq_scaling);
        ret = std::max(ret, ret_remain);
    }
    return ret;
}

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
                 float new_avg_sq_scaling)
{
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    assert(params.options().dtype() == grads.options().dtype());
    assert(exp_avg.options().dtype() == exp_avg_sq.options().dtype());

    bool half_precision_param = (params.options().dtype() == at::kHalf);
    bool half_precision_opt_state = (exp_avg.options().dtype() == at::kHalf);

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);

    auto ret = opt->Step_8(params_ptr,
                grads_ptr,
                exp_avg_ptr,
                exp_avg_sq_ptr,
                params_c.numel(),
                nullptr,
                half_precision_param,
                half_precision_opt_state,
                avg_scaling,
                avg_sq_scaling, 
                new_avg_scaling, 
                new_avg_sq_scaling);

#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
    opt->SynchronizeStreams();
#endif
    return ret;
}

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
                           torch::Tensor& device_params,
                           float avg_scaling,
                           float avg_sq_scaling, 
                           float new_avg_scaling, 
                           float new_avg_sq_scaling)
{
#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();
    auto device_params_c = device_params.contiguous();

    assert(params.options().dtype() == grads.options().dtype());
    assert(exp_avg.options().dtype() == exp_avg_sq.options().dtype());
    assert(device_params.options().dtype() == at::kHalf);

    bool half_precision_param = (params.options().dtype() == at::kHalf);
    bool half_precision_opt_state = (exp_avg.options().dtype() == at::kHalf);

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    ds_half_precision_t* device_params_ptr = (ds_half_precision_t*)device_params_c.data_ptr();

    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);

    assert(opt->use_cuda_cache());
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);
    auto ret = opt->Step_8(params_ptr,
                grads_ptr,
                exp_avg_ptr,
                exp_avg_sq_ptr,
                params_c.numel(),
                device_params_ptr,
                half_precision_param,
                half_precision_opt_state,
                avg_scaling,
                avg_sq_scaling, 
                new_avg_scaling, 
                new_avg_sq_scaling);

    opt->SynchronizeStreams();
#else
    assert(false);
#endif
    return ret;
}

int destroy_adam_optimizer(int optimizer_id)
{
    s_optimizers.erase(optimizer_id);

    return 0;
}
