#include "cpu_adagrad.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <omp.h>
#include <torch/extension.h>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"
#include "custom_cuda_layers.h"

static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

#define ROUND_DOWN(size, step) ((size) & ~((step)-1))

// C++ interface

void Adagrad_Optimizer::Step(float* _params,
                             float* grads,
                             float* _exp_avg_sq,
                             size_t _param_size,
                             __half* dev_params,
                             bool half_precision)
{
    float step_size = -1 * _alpha;
    size_t rounded_size = 0;

#if defined(__AVX512__) or defined(__AVX256__)

    AVX_Data eps_4;
    eps_4.data = SIMD_SET(_eps);

    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    AVX_Data weight_decay4;
    if (_weight_decay > 0) weight_decay4.data = SIMD_SET(_weight_decay);
    rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH);

    for (size_t t = 0; t < rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
        size_t offset = copy_size + t;
        if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }

#pragma omp parallel for
        for (size_t i = t; i < offset; i += SIMD_WIDTH) {
            AVX_Data grad_4;
            grad_4.data = SIMD_LOAD2(grads + i, half_precision);

            AVX_Data momentum_4;
            momentum_4.data = SIMD_LOAD(grads + i);

            AVX_Data variance_4;
            variance_4.data = SIMD_LOAD(_exp_avg_sq + i);

            AVX_Data param_4;
            param_4.data = SIMD_LOAD2(_params + i, half_precision);

            if (_weight_decay > 0) {
                grad_4.data = SIMD_FMA(param_4.data, weight_decay4.data, grad_4.data);
            }

            variance_4.data = SIMD_FMA(grad_4.data, grad_4.data, variance_4.data);

            grad_4.data = SIMD_SQRT(variance_4.data);
            grad_4.data = SIMD_ADD(grad_4.data, eps_4.data);
            grad_4.data = SIMD_DIV(momentum_4.data, grad_4.data);

            param_4.data = SIMD_FMA(grad_4.data, step_size_4.data, param_4.data);

            SIMD_STORE2(_params + i, param_4.data, half_precision);

            if (dev_params)
                SIMD_STORE2(_doubled_buffer[_buf_index] + (i - t), param_4.data, half_precision);

            SIMD_STORE(_exp_avg_sq + i, variance_4.data);
        }
        if (dev_params) {
            if (half_precision)
                launch_param_update_half(
                    _doubled_buffer[_buf_index], dev_params + t, copy_size, _streams[_buf_index]);
            else
                launch_param_update(
                    _doubled_buffer[_buf_index], dev_params + t, copy_size, _streams[_buf_index]);

            _buf_index = !_buf_index;
        }
    }

#endif

    if (_param_size > rounded_size) {
        __half* grads_cast_h;
        __half* params_cast_h;
        if (half_precision) {
            grads_cast_h = reinterpret_cast<__half*>(grads);
            params_cast_h = reinterpret_cast<__half*>(_params);
        }
        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
            if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = half_precision ? (float)grads_cast_h[k] : grads[k];
                float param = half_precision ? (float)params_cast_h[k] : _params[k];
                float momentum = grads[k];
                float variance = _exp_avg_sq[k];
                if (_weight_decay > 0) { grad = param * _weight_decay + grad; }

                variance += grad * grad;

                grad = sqrt(variance);
                grad += _eps;
                grad = momentum / grad;
                param = grad * step_size + param;
                if (dev_params) _doubled_buffer[_buf_index][k - t] = param;

                if (half_precision)
                    params_cast_h[k] = (__half)param;
                else
                    _params[k] = param;
                // STORE UPDATE TERM TO GRAD'S MEMORY
                grads[k] = grad * step_size;
                _exp_avg_sq[k] = variance;
            }
            if (dev_params) {
                launch_param_update(
                    _doubled_buffer[_buf_index], dev_params + t, (copy_size), _streams[_buf_index]);
                _buf_index = !_buf_index;
            }
        }
    }
}

void Adagrad_Optimizer::Step_4(float* _params,
                               float* grads,
                               float* _exp_avg_sq,
                               size_t _param_size,
                               __half* dev_params,
                               bool half_precision)
{
    size_t rounded_size = 0;

#if defined(__AVX512__) or defined(__AVX256__)

    AVX_Data eps_4;
    eps_4.data = SIMD_SET(_eps);

    float step_size = -1 * _alpha;
    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    AVX_Data weight_decay4;
    if (_weight_decay > 0) weight_decay4.data = SIMD_SET(_weight_decay);
    rounded_size = ROUND_DOWN(_param_size, (SIMD_WIDTH << 2));

    for (size_t t = 0; t < rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
        size_t offset = copy_size + t;
        if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#pragma omp parallel for
        for (size_t i = t; i < offset; i += (SIMD_WIDTH << 2)) {
            AVX_Data grad_4[4];
            grad_4[0].data = SIMD_LOAD2(grads + i, half_precision);
            grad_4[1].data = SIMD_LOAD2(grads + i + SIMD_WIDTH, half_precision);
            grad_4[2].data = SIMD_LOAD2(grads + i + (SIMD_WIDTH << 1), half_precision);
            grad_4[3].data = SIMD_LOAD2(grads + i + SIMD_WIDTH * 3, half_precision);

            AVX_Data momentum_4[4];
            momentum_4[0].data = SIMD_LOAD(grads + i);
            momentum_4[1].data = SIMD_LOAD(grads + i + SIMD_WIDTH);
            momentum_4[2].data = SIMD_LOAD(grads + i + (SIMD_WIDTH << 1));
            momentum_4[3].data = SIMD_LOAD(grads + i + SIMD_WIDTH * 3);

            AVX_Data variance_4[4];
            variance_4[0].data = SIMD_LOAD(_exp_avg_sq + i);
            variance_4[1].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH);
            variance_4[2].data = SIMD_LOAD(_exp_avg_sq + i + (SIMD_WIDTH << 1));
            variance_4[3].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH * 3);

            AVX_Data param_4[4];
            param_4[0].data = SIMD_LOAD2(_params + i, half_precision);
            param_4[1].data = SIMD_LOAD2(_params + i + SIMD_WIDTH, half_precision);
            param_4[2].data = SIMD_LOAD2(_params + i + (SIMD_WIDTH << 1), half_precision);
            param_4[3].data = SIMD_LOAD2(_params + i + SIMD_WIDTH * 3, half_precision);

            if (_weight_decay > 0) {
                grad_4[0].data = SIMD_FMA(param_4[0].data, weight_decay4.data, grad_4[0].data);
                grad_4[1].data = SIMD_FMA(param_4[1].data, weight_decay4.data, grad_4[1].data);
                grad_4[2].data = SIMD_FMA(param_4[2].data, weight_decay4.data, grad_4[2].data);
                grad_4[3].data = SIMD_FMA(param_4[3].data, weight_decay4.data, grad_4[3].data);
            }

            variance_4[0].data = SIMD_FMA(grad_4[0].data, grad_4[0].data, variance_4[0].data);
            variance_4[1].data = SIMD_FMA(grad_4[1].data, grad_4[1].data, variance_4[1].data);
            variance_4[2].data = SIMD_FMA(grad_4[2].data, grad_4[2].data, variance_4[2].data);
            variance_4[3].data = SIMD_FMA(grad_4[3].data, grad_4[3].data, variance_4[3].data);

            grad_4[0].data = SIMD_SQRT(variance_4[0].data);
            grad_4[1].data = SIMD_SQRT(variance_4[1].data);
            grad_4[2].data = SIMD_SQRT(variance_4[2].data);
            grad_4[3].data = SIMD_SQRT(variance_4[3].data);

            grad_4[0].data = SIMD_ADD(grad_4[0].data, eps_4.data);
            grad_4[1].data = SIMD_ADD(grad_4[1].data, eps_4.data);
            grad_4[2].data = SIMD_ADD(grad_4[2].data, eps_4.data);
            grad_4[3].data = SIMD_ADD(grad_4[3].data, eps_4.data);

            grad_4[0].data = SIMD_DIV(momentum_4[0].data, grad_4[0].data);
            grad_4[1].data = SIMD_DIV(momentum_4[1].data, grad_4[1].data);
            grad_4[2].data = SIMD_DIV(momentum_4[2].data, grad_4[2].data);
            grad_4[3].data = SIMD_DIV(momentum_4[3].data, grad_4[3].data);

            param_4[0].data = SIMD_FMA(grad_4[0].data, step_size_4.data, param_4[0].data);
            param_4[1].data = SIMD_FMA(grad_4[1].data, step_size_4.data, param_4[1].data);
            param_4[2].data = SIMD_FMA(grad_4[2].data, step_size_4.data, param_4[2].data);
            param_4[3].data = SIMD_FMA(grad_4[3].data, step_size_4.data, param_4[3].data);

            SIMD_STORE2(_params + i, param_4[0].data, half_precision);
            SIMD_STORE2(_params + i + SIMD_WIDTH, param_4[1].data, half_precision);
            SIMD_STORE2(_params + i + (SIMD_WIDTH << 1), param_4[2].data, half_precision);
            SIMD_STORE2(_params + i + SIMD_WIDTH * 3, param_4[3].data, half_precision);

            if (dev_params) {
                SIMD_STORE2(_doubled_buffer[_buf_index] + (i - t), param_4[0].data, half_precision);
                SIMD_STORE2(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH,
                            param_4[1].data,
                            half_precision);
                SIMD_STORE2(_doubled_buffer[_buf_index] + (i - t) + (SIMD_WIDTH << 1),
                            param_4[2].data,
                            half_precision);
                SIMD_STORE2(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH * 3,
                            param_4[3].data,
                            half_precision);
            }

            SIMD_STORE(_exp_avg_sq + i, variance_4[0].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH, variance_4[1].data);
            SIMD_STORE(_exp_avg_sq + i + (SIMD_WIDTH << 1), variance_4[2].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH * 3, variance_4[3].data);
        }

        if (dev_params) {
            if (half_precision)
                launch_param_update_half(
                    _doubled_buffer[_buf_index], dev_params + t, copy_size, _streams[_buf_index]);
            else
                launch_param_update(
                    _doubled_buffer[_buf_index], dev_params + t, copy_size, _streams[_buf_index]);

            _buf_index = !_buf_index;
        }
    }
#endif
    if (_param_size > rounded_size)
        Step((_params + rounded_size),
             (grads + rounded_size),
             (_exp_avg_sq + rounded_size),
             (_param_size - rounded_size),
             (dev_params != nullptr ? (dev_params + rounded_size) : dev_params),
             half_precision);
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

void Adagrad_Optimizer::Step_8(float* _params,
                               float* grads,
                               float* _exp_avg_sq,
                               size_t _param_size,
                               __half* dev_params,
                               bool half_precision)
{
    size_t rounded_size = 0;

#if defined(__AVX512__) or defined(__AVX256__)

#endif
    AVX_Data eps_4;
    eps_4.data = SIMD_SET(_eps);

    float step_size = -1 * _alpha;
    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    AVX_Data weight_decay4;
    if (_weight_decay > 0) weight_decay4.data = SIMD_SET(_weight_decay);
    rounded_size = ROUND_DOWN(_param_size, (SIMD_WIDTH << 3));

    for (size_t t = 0; t < rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
        size_t offset = copy_size + t;
        if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#pragma omp parallel for
        for (size_t i = t; i < offset; i += (SIMD_WIDTH << 3)) {
            AVX_Data grad_4[8];
            grad_4[0].data = SIMD_LOAD2(grads + i, half_precision);
            grad_4[1].data = SIMD_LOAD2(grads + i + SIMD_WIDTH, half_precision);
            grad_4[2].data = SIMD_LOAD2(grads + i + (SIMD_WIDTH << 1), half_precision);
            grad_4[3].data = SIMD_LOAD2(grads + i + SIMD_WIDTH * 3, half_precision);
            grad_4[4].data = SIMD_LOAD2(grads + i + (SIMD_WIDTH << 2), half_precision);
            grad_4[5].data = SIMD_LOAD2(grads + i + SIMD_WIDTH * 5, half_precision);
            grad_4[6].data = SIMD_LOAD2(grads + i + SIMD_WIDTH * 6, half_precision);
            grad_4[7].data = SIMD_LOAD2(grads + i + SIMD_WIDTH * 7, half_precision);

            AVX_Data momentum_4[8];
            momentum_4[0].data = SIMD_LOAD(grads + i);
            momentum_4[1].data = SIMD_LOAD(grads + i + SIMD_WIDTH);
            momentum_4[2].data = SIMD_LOAD(grads + i + (SIMD_WIDTH << 1));
            momentum_4[3].data = SIMD_LOAD(grads + i + SIMD_WIDTH * 3);
            momentum_4[4].data = SIMD_LOAD(grads + i + (SIMD_WIDTH << 2));
            momentum_4[5].data = SIMD_LOAD(grads + i + SIMD_WIDTH * 5);
            momentum_4[6].data = SIMD_LOAD(grads + i + SIMD_WIDTH * 6);
            momentum_4[7].data = SIMD_LOAD(grads + i + SIMD_WIDTH * 7);

            AVX_Data variance_4[8];
            variance_4[0].data = SIMD_LOAD(_exp_avg_sq + i);
            variance_4[1].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH);
            variance_4[2].data = SIMD_LOAD(_exp_avg_sq + i + (SIMD_WIDTH << 1));
            variance_4[3].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH * 3);
            variance_4[4].data = SIMD_LOAD(_exp_avg_sq + i + (SIMD_WIDTH << 2));
            variance_4[5].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH * 5);
            variance_4[6].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH * 6);
            variance_4[7].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH * 7);

            AVX_Data param_4[8];
            param_4[0].data = SIMD_LOAD2(_params + i, half_precision);
            param_4[1].data = SIMD_LOAD2(_params + i + SIMD_WIDTH, half_precision);
            param_4[2].data = SIMD_LOAD2(_params + i + (SIMD_WIDTH << 1), half_precision);
            param_4[3].data = SIMD_LOAD2(_params + i + SIMD_WIDTH * 3, half_precision);
            param_4[4].data = SIMD_LOAD2(_params + i + (SIMD_WIDTH << 2), half_precision);
            param_4[5].data = SIMD_LOAD2(_params + i + SIMD_WIDTH * 5, half_precision);
            param_4[6].data = SIMD_LOAD2(_params + i + SIMD_WIDTH * 6, half_precision);
            param_4[7].data = SIMD_LOAD2(_params + i + SIMD_WIDTH * 7, half_precision);

            if (_weight_decay > 0) {
                grad_4[0].data = SIMD_FMA(param_4[0].data, weight_decay4.data, grad_4[0].data);
                grad_4[1].data = SIMD_FMA(param_4[1].data, weight_decay4.data, grad_4[1].data);
                grad_4[2].data = SIMD_FMA(param_4[2].data, weight_decay4.data, grad_4[2].data);
                grad_4[3].data = SIMD_FMA(param_4[3].data, weight_decay4.data, grad_4[3].data);
                grad_4[4].data = SIMD_FMA(param_4[4].data, weight_decay4.data, grad_4[4].data);
                grad_4[5].data = SIMD_FMA(param_4[5].data, weight_decay4.data, grad_4[5].data);
                grad_4[6].data = SIMD_FMA(param_4[6].data, weight_decay4.data, grad_4[6].data);
                grad_4[7].data = SIMD_FMA(param_4[7].data, weight_decay4.data, grad_4[7].data);
            }

            variance_4[0].data = SIMD_FMA(grad_4[0].data, grad_4[0].data, variance_4[0].data);
            variance_4[1].data = SIMD_FMA(grad_4[1].data, grad_4[1].data, variance_4[1].data);
            variance_4[2].data = SIMD_FMA(grad_4[2].data, grad_4[2].data, variance_4[2].data);
            variance_4[3].data = SIMD_FMA(grad_4[3].data, grad_4[3].data, variance_4[3].data);
            variance_4[4].data = SIMD_FMA(grad_4[4].data, grad_4[4].data, variance_4[4].data);
            variance_4[5].data = SIMD_FMA(grad_4[5].data, grad_4[5].data, variance_4[5].data);
            variance_4[6].data = SIMD_FMA(grad_4[6].data, grad_4[6].data, variance_4[6].data);
            variance_4[7].data = SIMD_FMA(grad_4[7].data, grad_4[7].data, variance_4[7].data);

            grad_4[0].data = SIMD_SQRT(variance_4[0].data);
            grad_4[1].data = SIMD_SQRT(variance_4[1].data);
            grad_4[2].data = SIMD_SQRT(variance_4[2].data);
            grad_4[3].data = SIMD_SQRT(variance_4[3].data);
            grad_4[4].data = SIMD_SQRT(variance_4[4].data);
            grad_4[5].data = SIMD_SQRT(variance_4[5].data);
            grad_4[6].data = SIMD_SQRT(variance_4[6].data);
            grad_4[7].data = SIMD_SQRT(variance_4[7].data);

            grad_4[0].data = SIMD_ADD(grad_4[0].data, eps_4.data);
            grad_4[1].data = SIMD_ADD(grad_4[1].data, eps_4.data);
            grad_4[2].data = SIMD_ADD(grad_4[2].data, eps_4.data);
            grad_4[3].data = SIMD_ADD(grad_4[3].data, eps_4.data);
            grad_4[4].data = SIMD_ADD(grad_4[4].data, eps_4.data);
            grad_4[5].data = SIMD_ADD(grad_4[5].data, eps_4.data);
            grad_4[6].data = SIMD_ADD(grad_4[6].data, eps_4.data);
            grad_4[7].data = SIMD_ADD(grad_4[7].data, eps_4.data);

            grad_4[0].data = SIMD_DIV(momentum_4[0].data, grad_4[0].data);
            grad_4[1].data = SIMD_DIV(momentum_4[1].data, grad_4[1].data);
            grad_4[2].data = SIMD_DIV(momentum_4[2].data, grad_4[2].data);
            grad_4[3].data = SIMD_DIV(momentum_4[3].data, grad_4[3].data);
            grad_4[4].data = SIMD_DIV(momentum_4[4].data, grad_4[4].data);
            grad_4[5].data = SIMD_DIV(momentum_4[5].data, grad_4[5].data);
            grad_4[6].data = SIMD_DIV(momentum_4[6].data, grad_4[6].data);
            grad_4[7].data = SIMD_DIV(momentum_4[7].data, grad_4[7].data);

            param_4[0].data = SIMD_FMA(grad_4[0].data, step_size_4.data, param_4[0].data);
            param_4[1].data = SIMD_FMA(grad_4[1].data, step_size_4.data, param_4[1].data);
            param_4[2].data = SIMD_FMA(grad_4[2].data, step_size_4.data, param_4[2].data);
            param_4[3].data = SIMD_FMA(grad_4[3].data, step_size_4.data, param_4[3].data);
            param_4[4].data = SIMD_FMA(grad_4[4].data, step_size_4.data, param_4[4].data);
            param_4[5].data = SIMD_FMA(grad_4[5].data, step_size_4.data, param_4[5].data);
            param_4[6].data = SIMD_FMA(grad_4[6].data, step_size_4.data, param_4[6].data);
            param_4[7].data = SIMD_FMA(grad_4[7].data, step_size_4.data, param_4[7].data);

            SIMD_STORE2(_params + i, param_4[0].data, half_precision);
            SIMD_STORE2(_params + i + SIMD_WIDTH, param_4[1].data, half_precision);
            SIMD_STORE2(_params + i + (SIMD_WIDTH << 1), param_4[2].data, half_precision);
            SIMD_STORE2(_params + i + SIMD_WIDTH * 3, param_4[3].data, half_precision);
            SIMD_STORE2(_params + i + (SIMD_WIDTH << 2), param_4[4].data, half_precision);
            SIMD_STORE2(_params + i + SIMD_WIDTH * 5, param_4[5].data, half_precision);
            SIMD_STORE2(_params + i + SIMD_WIDTH * 6, param_4[6].data, half_precision);
            SIMD_STORE2(_params + i + SIMD_WIDTH * 7, param_4[7].data, half_precision);

            if (dev_params) {
                SIMD_STORE2(_doubled_buffer[_buf_index] + (i - t), param_4[0].data, half_precision);
                SIMD_STORE2(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH,
                            param_4[1].data,
                            half_precision);
                SIMD_STORE2(_doubled_buffer[_buf_index] + (i - t) + (SIMD_WIDTH << 1),
                            param_4[2].data,
                            half_precision);
                SIMD_STORE2(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH * 3,
                            param_4[3].data,
                            half_precision);
                SIMD_STORE2(_doubled_buffer[_buf_index] + (i - t) + (SIMD_WIDTH << 2),
                            param_4[4].data,
                            half_precision);
                SIMD_STORE2(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH * 5,
                            param_4[5].data,
                            half_precision);
                SIMD_STORE2(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH * 6,
                            param_4[6].data,
                            half_precision);
                SIMD_STORE2(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH * 7,
                            param_4[7].data,
                            half_precision);
            }

            SIMD_STORE(_exp_avg_sq + i, variance_4[0].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH, variance_4[1].data);
            SIMD_STORE(_exp_avg_sq + i + (SIMD_WIDTH << 1), variance_4[2].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH * 3, variance_4[3].data);
            SIMD_STORE(_exp_avg_sq + i + (SIMD_WIDTH << 2), variance_4[4].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH * 5, variance_4[5].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH * 6, variance_4[6].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH * 7, variance_4[7].data);
        }
        if (dev_params) {
            if (half_precision)
                launch_param_update_half(
                    _doubled_buffer[_buf_index], dev_params + t, copy_size, _streams[_buf_index]);
            else
                launch_param_update(
                    _doubled_buffer[_buf_index], dev_params + t, copy_size, _streams[_buf_index]);
            _buf_index = !_buf_index;
        }
    }
    if (_param_size > rounded_size)
        Step_4((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               (dev_params != nullptr ? (dev_params + rounded_size) : dev_params),
               half_precision);
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
    opt->Step_8(params_ptr, grads_ptr, exp_avg_sq_ptr, params_c.size(0));

    opt->SynchronizeStreams();
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
    auto params_c = params.contiguous();
    auto gpu_params_c = gpu_params.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();
    auto grads_c = grads.contiguous();

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    __half* gpu_params_ptr = (__half*)gpu_params_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Adagrad_Optimizer> opt =
        std::static_pointer_cast<Adagrad_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step);
    opt->update_state(lr, epsilon, weight_decay);
    opt->Step_8(params_ptr,
                grads_ptr,
                exp_avg_sq_ptr,
                params_c.size(0),
                gpu_params_ptr,
                (params.options().dtype() == at::kHalf));

    opt->SynchronizeStreams();
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
