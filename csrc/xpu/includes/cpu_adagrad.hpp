#pragma once
#if (__x86_64__ || __i386__)
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include <cpuid.h>
#include <x86intrin.h>
#endif

#include <stdio.h>
#include <cassert>
#include "context.hpp"
#include <oneapi/mkl.hpp>

#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>

#include <cmath>
#define STEP(SPAN)                                \
    void Step_##SPAN(float* _params,              \
                     float* grads,                \
                     float* _exp_avg_sq,          \
                     size_t _param_size,          \
                     sycl::half* dev_param = nullptr, \
                     bool half_precision = false);

#define TILE (128 * 1024 * 1024)


class Adagrad_Optimizer {
public:
    Adagrad_Optimizer(float alpha = 1e-2, float eps = 1e-8, float weight_decay = 0)
        : _alpha(alpha), _eps(eps), _weight_decay(weight_decay), _buf_index(false)
    {
        _streams[0] = ::SyclContext::Instance().GetCurrentStream();
        _streams[1] = ::SyclContext::Instance().GetNewStream();
        sycl::queue& q_ct1 = *_streams[0];

        *_doubled_buffer = sycl::malloc_host<float>(TILE, q_ct1);
        *(_doubled_buffer + 1) = sycl::malloc_host<float>(TILE, q_ct1);
    }
    ~Adagrad_Optimizer()
    {
            sycl::queue& q_ct1 = *_streams[0];
            sycl::free(_doubled_buffer[0], q_ct1);
            sycl::free(_doubled_buffer[1], q_ct1);
    }

    STEP(1)
    STEP(4)
    STEP(8)
    inline void SynchronizeStreams()
    {
        for (int i = 0; i < 2; i++) _streams[i]->wait();
    }
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

    float* _doubled_buffer[2];
    bool _buf_index;

    sycl::queue* _streams[2];
};
