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

static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

// C++ interface

void Step_1(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            float lr,
                            float betta1,
                            float betta2,
                            float eps,
                            float weight_decay,
                            float bias_correction1,
                            float bias_correction2,
                            bool half_precision,
                            bool adamw_mode)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<1>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq, _param_size, lr, betta1, betta2, eps, weight_decay, bias_correction1, bias_correction2, half_precision, adamw_mode);
#endif
    if (_param_size > rounded_size) {
        float betta1_minus1 = 1 - betta1;
        float betta2_minus1 = 1 - betta2;

        float step_size = -1 * lr / bias_correction1;
        float w_decay = -1 * lr * weight_decay;
        ds_half_precision_t* grads_cast_h;
        ds_half_precision_t* params_cast_h;
        if (half_precision) {
            grads_cast_h = reinterpret_cast<ds_half_precision_t*>(grads);
            params_cast_h = reinterpret_cast<ds_half_precision_t*>(_params);
        }

        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = half_precision ? (float)grads_cast_h[k] : grads[k];
                float param = half_precision ? (float)params_cast_h[k] : _params[k];
                float momentum = _exp_avg[k];
                float variance = _exp_avg_sq[k];
                if (weight_decay > 0 && !adamw_mode) { grad = param * weight_decay + grad; }
                momentum = momentum * betta1;
                momentum = grad * betta1_minus1 + momentum;

                variance = variance * betta2;
                grad = grad * grad;
                variance = grad * betta2_minus1 + variance;

                grad = sqrt(variance);
                grad = grad * bias_correction2 + eps;
                grad = momentum / grad;
                if (weight_decay > 0 && adamw_mode) { param += w_decay * param; }
                param = grad * step_size + param;
                if (half_precision)
                    params_cast_h[k] = (ds_half_precision_t)param;
                else
                    _params[k] = param;
                _exp_avg[k] = momentum;
                _exp_avg_sq[k] = variance;
            }
        }
    }
}

void Step_4(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            float lr,
                            float betta1,
                            float betta2,
                            float eps,
                            float weight_decay,
                            float bias_correction1,
                            float bias_correction2,
                            bool half_precision,
                            bool adamw_mode)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<4>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq, _param_size, lr, betta1, betta2, eps, weight_decay, bias_correction1, bias_correction2, half_precision, adamw_mode);
#endif
    if (_param_size > rounded_size)
        Step_1((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               lr, betta1, betta2, eps, weight_decay,
               bias_correction1, bias_correction2,
               half_precision, adamw_mode);
}

void Step_8(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            float lr,
                            float betta1,
                            float betta2,
                            float eps,
                            float weight_decay,
                            float bias_correction1,
                            float bias_correction2,
                            bool half_precision,
                            bool adamw_mode)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<8>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq, _param_size, lr, betta1, betta2, eps, weight_decay, bias_correction1, bias_correction2, half_precision, adamw_mode);
#endif
    if (_param_size > rounded_size)
        Step_4((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               lr, betta1, betta2, eps, weight_decay,
               bias_correction1, bias_correction2,
               half_precision, adamw_mode);
}

int ds_adam_step(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float epsilon,
                 float weight_decay,
                 bool bias_correction,
                 bool adam_mode,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg,
                 torch::Tensor& exp_avg_sq)
{
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    // assert(params.options().dtype() == grads.options().dtype());

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    float bias_correction1 = 1.0f, bias_correction2= 1.0f;
    if (bias_correction == 1) {
        bias_correction1 = 1.0 - std::pow(beta1, step);
        bias_correction2 = 1 / sqrt(1.0 - std::pow(beta2, step));
    }
    Step_8(params_ptr,
                grads_ptr,
                exp_avg_ptr,
                exp_avg_sq_ptr,
                params_c.numel(),
                lr, beta1, beta2, epsilon, weight_decay,
                bias_correction1, bias_correction2,
                (params.options().dtype() == at::kHalf),
                adam_mode);

    return 0;
}

void multi_tensor_adam(int chunk_size,
                       at::Tensor noop_flag,
                       std::vector<std::vector<at::Tensor>> tensor_lists, /*gpmv*/
                       const float lr,
                       const float beta1,
                       const float beta2,
                       const float epsilon,
                       const int step,
                       const int mode,
                       const int bias_correction,
                       const float weight_decay)
{
    for (int i = 0; i < tensor_lists[0].size(); i++) {
        ds_adam_step(0,
                     step,
                     lr,
                     beta1,
                     beta2,
                     epsilon,
                     weight_decay,
                     bias_correction,
                     mode,
                     tensor_lists[1][i],
                     tensor_lists[0][i],
                     tensor_lists[2][i],
                     tensor_lists[3][i]);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("multi_tensor_adam",
          &multi_tensor_adam,
          "Compute and apply gradient update to parameters for Adam optimizer");
}
