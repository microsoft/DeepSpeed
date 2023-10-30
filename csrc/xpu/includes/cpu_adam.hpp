// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#if (__x86_64__ || __i386__)
#include <cpuid.h>
#include <x86intrin.h>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#endif

#include <stdio.h>
#include <cassert>
#include <oneapi/mkl.hpp>
#include "context.hpp"

#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>

#include <cmath>

#define TILE (128 * 1024 * 1024)

#if defined(__AVX512__)
#define SIMD_STORE(a, d) _mm512_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm512_loadu_ps(x)
#define SIMD_SET(x) _mm512_set1_ps(x)
#define SIMD_MUL(x, y) _mm512_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm512_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm512_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm512_div_ps(x, y)
#define SIMD_WIDTH 16
#else
#if defined(__AVX256__)
#define SIMD_STORE(a, d) _mm256_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm256_loadu_ps(x)
#define SIMD_SET(x) _mm256_set1_ps(x)
#define SIMD_MUL(x, y) _mm256_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm256_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm256_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm256_div_ps(x, y)
#define SIMD_WIDTH 8
#endif
#endif

class Adam_Optimizer {
public:
    Adam_Optimizer(float alpha = 1e-3,
                   float betta1 = 0.9,
                   float betta2 = 0.999,
                   float eps = 1e-8,
                   float weight_decay = 0,
                   bool adamw_mode = true)
        : _alpha(alpha),
          _betta1(betta1),
          _betta2(betta2),
          _eps(eps),
          _weight_decay(weight_decay),
          _betta1_t(1.0),
          _betta2_t(1.0),
          _step(0),
          _buf_index(false),
          _adamw_mode(adamw_mode)
    {
        _streams[0] = ::SyclContext::Instance().GetCurrentStream();
        _streams[1] = ::SyclContext::Instance().GetNewStream();
        sycl::queue& q_ct1 = *_streams[0];

        *_doubled_buffer = sycl::malloc_host<float>(TILE, q_ct1);
        *(_doubled_buffer + 1) = sycl::malloc_host<float>(TILE, q_ct1);
    }
    ~Adam_Optimizer()
    {
        sycl::queue& q_ct1 = *_streams[0];
        sycl::free(_doubled_buffer[0], q_ct1);
        sycl::free(_doubled_buffer[1], q_ct1);
    }
    void Step(float* _params,
              float* grads,
              float* _exp_avg,
              float* _exp_avg_sq,
              size_t param_size,
              sycl::half* dev_param = nullptr,
              bool half_precision = false);
    void Step_4(float* _params,
                float* grads,
                float* _exp_avg,
                float* _exp_avg_sa,
                size_t param_size,
                sycl::half* dev_param = nullptr,
                bool half_precision = false);
    void Step_8(float* _params,
                float* grads,
                float* _exp_avg,
                float* _exp_avg_sq,
                size_t _param_size,
                sycl::half* dev_params = nullptr,
                bool half_precision = false);
    inline void SynchronizeStreams()
    {
        for (int i = 0; i < 2; i++) _streams[i]->wait();
    }
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

private:
#if defined(__AVX512__) or defined(__AVX256__)
    union AVX_Data {
#if defined(__AVX512__)
        __m512 data;
#else
        __m256 data;
#endif
        // float data_f[16];
    };
#endif

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

    float* _doubled_buffer[2];
    bool _buf_index;
    bool _adamw_mode;

    sycl::queue* _streams[2];
};
