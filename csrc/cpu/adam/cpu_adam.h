// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#define NOMINMAX  // Windows idiosyncrasy
                  // https://stackoverflow.com/questions/4913922/possible-problems-with-nominmax-on-visual-c

#include <stdio.h>
#include <cassert>
#include "simd.h"

#include <cmath>
typedef unsigned short ds_half_precision_t;

#define STEP(SPAN)                                \
    void Step_##SPAN(float* _params,              \
                     float* grads,                \
                     float* _exp_avg,             \
                     float* _exp_avg_sq,          \
                     size_t _param_size,          \
                     float lr,                    \
                     float betta1,                \
                     float betta2,                \
                     float eps,                   \
                     float weight_decay,          \
                     float bias_correction1,      \
                     float bias_correction2,      \
                     bool half_precision = false, \
                     bool adam_mode = true);

#if defined(__AVX512__) or defined(__AVX256__)
template <int span>
void Step_AVX(size_t* rounded_size,
              float* _params,
              float* grads,
              float* _exp_avg,
              float* _exp_avg_sq,
              size_t param_size,
              float lr,
              float betta1,
              float betta2,
              float eps,
              float weight_decay,
              float bias_correction1,
              float bias_correction2,
              bool half_precision,
              bool adamw_mode);
#endif
STEP(1)
STEP(4)
STEP(8)

#if defined(__AVX512__) or defined(__AVX256__)
template <int span>
void Step_AVX(size_t* rounded_size,
              float* _params,
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
    size_t new_rounded_size = 0;
    int rshft = half_precision ? 1 : 0;

    AVX_Data betta1_4;
    betta1_4.data = SIMD_SET(betta1);
    AVX_Data betta2_4;
    betta2_4.data = SIMD_SET(betta2);

    float betta1_minus1 = 1 - betta1;
    float betta2_minus1 = 1 - betta2;
    AVX_Data betta1_minus1_4;
    betta1_minus1_4.data = SIMD_SET(betta1_minus1);
    AVX_Data betta2_minus1_4;
    betta2_minus1_4.data = SIMD_SET(betta2_minus1);

    AVX_Data bias2_sqrt;
    bias2_sqrt.data = SIMD_SET(bias_correction2);

    AVX_Data eps_4;
    eps_4.data = SIMD_SET(eps);

    float step_size = -1 * lr / bias_correction1;
    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    float w_decay = -1 * lr * weight_decay;
    AVX_Data weight_decay4;
    if (weight_decay > 0)
        weight_decay4.data = (adamw_mode ? SIMD_SET(w_decay) : SIMD_SET(weight_decay));
    new_rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * span);
    for (size_t t = 0; t < new_rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > new_rounded_size) copy_size = new_rounded_size - t;
        size_t offset = copy_size + t;
#pragma omp parallel for
        for (size_t i = t; i < offset; i += SIMD_WIDTH * span) {
            AVX_Data grad_4[span];
            simd_load<span>(grad_4, grads + (i >> rshft), half_precision);

            AVX_Data momentum_4[span];
            simd_load<span>(momentum_4, _exp_avg + i, false);

            AVX_Data variance_4[span];
            simd_load<span>(variance_4, _exp_avg_sq + i, false);

            AVX_Data param_4[span];
            simd_load<span>(param_4, _params + (i >> rshft), half_precision);

            if (weight_decay > 0 && !adamw_mode) {
                simd_fma<span>(grad_4, param_4, weight_decay4, grad_4);
            }

            simd_mul<span>(momentum_4, momentum_4, betta1_4);
            simd_fma<span>(momentum_4, grad_4, betta1_minus1_4, momentum_4);
            simd_mul<span>(variance_4, variance_4, betta2_4);
            simd_mul<span>(grad_4, grad_4, grad_4);
            simd_fma<span>(variance_4, grad_4, betta2_minus1_4, variance_4);
            simd_sqrt<span>(grad_4, variance_4);
            simd_fma<span>(grad_4, grad_4, bias2_sqrt, eps_4);
            simd_div<span>(grad_4, momentum_4, grad_4);

            if (weight_decay > 0 && adamw_mode) {
                simd_fma<span>(param_4, param_4, weight_decay4, param_4);
            }

            simd_fma<span>(param_4, grad_4, step_size_4, param_4);

            simd_store<span>(_params + (i >> rshft), param_4, half_precision);
            simd_store<span>(_exp_avg + i, momentum_4, false);
            simd_store<span>(_exp_avg_sq + i, variance_4, false);
        }
    }
    *rounded_size = new_rounded_size;
}
#endif
