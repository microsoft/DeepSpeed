#include "cpu_adam.h"
#include <cuda_runtime_api.h>
#include <omp.h>
#include <torch/extension.h>
#include <x86intrin.h>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"
#include "custom_cuda_layers.h"

static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

// C++ interface

void Adam_Optimizer::Step(float* _params,
                          float* grads,
                          float* _exp_avg,
                          float* _exp_avg_sq,
                          size_t _param_size,
                          __half* dev_params)
{
    _betta1_t *= _betta1;
    _betta2_t *= _betta2;

    bool buf_index = 0;

    __m512 betta1_4 = _mm512_set1_ps(_betta1);
    __m512 betta2_4 = _mm512_set1_ps(_betta2);

    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;
    __m512 betta1_minus1_4 = _mm512_set1_ps(betta1_minus1);
    __m512 betta2_minus1_4 = _mm512_set1_ps(betta2_minus1);

    float bias_correction1 = 1 - _betta1_t;
    float bias_correction2 = 1 - _betta2_t;
    //__m512 bias_correction1_4 = _mm512_set1_ps(bias_correction1);
    __m512 bias_correction2_4 = _mm512_set1_ps(bias_correction2);

    __m512 eps_4 = _mm512_set1_ps(_eps);

    float step_size = -1 * _alpha / bias_correction1;
    __m512 step_size_4 = _mm512_set1_ps(step_size);

    __m512 bias2_sqrt = _mm512_sqrt_ps(bias_correction2_4);

    size_t tile = 0;

    for (size_t t = 0; t < _param_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > _param_size) copy_size = _param_size - t;
        size_t offset = copy_size + t;
#pragma omp parallel for
        for (size_t i = t; i < offset; i += SIMD_WIDTH) {
            __m512 grad_4 = _mm512_loadu_ps(grads + i);

            __m512 momntum_4 = _mm512_loadu_ps(_exp_avg + i);
            __m512 varianc_4 = _mm512_loadu_ps(_exp_avg_sq + i);

            __m512 param_4 = _mm512_loadu_ps(_params + i);

            if (_weight_decay > 0) {
                __m512 weight_decay4 = _mm512_set1_ps(_weight_decay);
                grad_4 = _mm512_fmadd_ps(param_4, weight_decay4, grad_4);
            }

            momntum_4 = _mm512_mul_ps(momntum_4, betta1_4);
            momntum_4 = _mm512_fmadd_ps(grad_4, betta1_minus1_4, momntum_4);

            varianc_4 = _mm512_mul_ps(varianc_4, betta2_4);
            grad_4 = _mm512_mul_ps(grad_4, grad_4);
            varianc_4 = _mm512_fmadd_ps(grad_4, betta2_minus1_4, varianc_4);

            grad_4 = _mm512_sqrt_ps(varianc_4) / bias2_sqrt;
            grad_4 = _mm512_add_ps(grad_4, eps_4);
            grad_4 = _mm512_div_ps(momntum_4, grad_4);

            param_4 = _mm512_fmadd_ps(grad_4, step_size_4, param_4);

            _mm512_storeu_ps(_params + i, param_4);
            _mm512_storeu_ps(_exp_avg + i, momntum_4);
            _mm512_storeu_ps(_exp_avg_sq + i, varianc_4);
        }
        if (dev_params) {
#pragma omp parallel for
            for (size_t j = 0; j < copy_size; j += 4) {
                _doubled_buffer[buf_index][j] = (__half)_params[t + j];
                _doubled_buffer[buf_index][j + 1] = (__half)_params[t + j + 1];
                _doubled_buffer[buf_index][j + 2] = (__half)_params[t + j + 2];
                _doubled_buffer[buf_index][j + 3] = (__half)_params[t + j + 3];
            }

            CUDA_CHECK(cudaMemcpyAsync(dev_params + t,
                                       _doubled_buffer[buf_index],
                                       copy_size * sizeof(__half),
                                       cudaMemcpyHostToDevice,
                                       Context::Instance().GetCurrentStream()));
            buf_index = !buf_index;
        }
    }
}

void Adam_Optimizer::Step_4(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            __half* dev_params)
{
    _betta1_t *= _betta1;
    _betta2_t *= _betta2;

    __m512 betta1_4 = _mm512_set1_ps(_betta1);
    __m512 betta2_4 = _mm512_set1_ps(_betta2);

    bool buf_index = 0;
    size_t tile = 0;

    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;
    __m512 betta1_minus1_4 = _mm512_set1_ps(betta1_minus1);
    __m512 betta2_minus1_4 = _mm512_set1_ps(betta2_minus1);

    float bias_correction1 = 1 - _betta1_t;
    float bias_correction2 = 1 - _betta2_t;
    //__m512 bias_correction1_4 = _mm512_set1_ps(bias_correction1);
    __m512 bias_correction2_4 = _mm512_set1_ps(bias_correction2);

    __m512 eps_4 = _mm512_set1_ps(_eps);

    float step_size = -1 * _alpha / bias_correction1;
    __m512 step_size_4 = _mm512_set1_ps(step_size);

    __m512 bias2_sqrt = _mm512_sqrt_ps(bias_correction2_4);

    for (size_t t = 0; t < _param_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > _param_size) copy_size = _param_size - t;
        size_t offset = copy_size + t;
#pragma omp parallel for
        for (size_t i = t; i < offset; i += (SIMD_WIDTH << 2)) {
            __m512 grad_4[4];
            grad_4[0] = _mm512_loadu_ps(grads + i);
            grad_4[1] = _mm512_loadu_ps(grads + i + SIMD_WIDTH);
            grad_4[2] = _mm512_loadu_ps(grads + i + (SIMD_WIDTH << 1));
            grad_4[3] = _mm512_loadu_ps(grads + i + SIMD_WIDTH * 3);

            __m512 momntum_4[4];
            momntum_4[0] = _mm512_loadu_ps(_exp_avg + i);
            momntum_4[1] = _mm512_loadu_ps(_exp_avg + i + SIMD_WIDTH);
            momntum_4[2] = _mm512_loadu_ps(_exp_avg + i + (SIMD_WIDTH << 1));
            momntum_4[3] = _mm512_loadu_ps(_exp_avg + i + SIMD_WIDTH * 3);

            __m512 varianc_4[4];
            varianc_4[0] = _mm512_loadu_ps(_exp_avg_sq + i);
            varianc_4[1] = _mm512_loadu_ps(_exp_avg_sq + i + SIMD_WIDTH);
            varianc_4[2] = _mm512_loadu_ps(_exp_avg_sq + i + (SIMD_WIDTH << 1));
            varianc_4[3] = _mm512_loadu_ps(_exp_avg_sq + i + SIMD_WIDTH * 3);

            __m512 param_4[4];
            param_4[0] = _mm512_loadu_ps(_params + i);
            param_4[1] = _mm512_loadu_ps(_params + i + SIMD_WIDTH);
            param_4[2] = _mm512_loadu_ps(_params + i + (SIMD_WIDTH << 1));
            param_4[3] = _mm512_loadu_ps(_params + i + SIMD_WIDTH * 3);

            if (_weight_decay > 0) {
                __m512 weight_decay4 = _mm512_set1_ps(_weight_decay);
                grad_4[0] = _mm512_fmadd_ps(param_4[0], weight_decay4, grad_4[0]);
                grad_4[1] = _mm512_fmadd_ps(param_4[1], weight_decay4, grad_4[1]);
                grad_4[2] = _mm512_fmadd_ps(param_4[2], weight_decay4, grad_4[2]);
                grad_4[3] = _mm512_fmadd_ps(param_4[3], weight_decay4, grad_4[3]);
            }

            momntum_4[0] = _mm512_mul_ps(momntum_4[0], betta1_4);
            momntum_4[0] = _mm512_fmadd_ps(grad_4[0], betta1_minus1_4, momntum_4[0]);
            momntum_4[1] = _mm512_mul_ps(momntum_4[1], betta1_4);
            momntum_4[1] = _mm512_fmadd_ps(grad_4[1], betta1_minus1_4, momntum_4[1]);
            momntum_4[2] = _mm512_mul_ps(momntum_4[2], betta1_4);
            momntum_4[2] = _mm512_fmadd_ps(grad_4[2], betta1_minus1_4, momntum_4[2]);
            momntum_4[3] = _mm512_mul_ps(momntum_4[3], betta1_4);
            momntum_4[3] = _mm512_fmadd_ps(grad_4[3], betta1_minus1_4, momntum_4[3]);

            varianc_4[0] = _mm512_mul_ps(varianc_4[0], betta2_4);
            varianc_4[1] = _mm512_mul_ps(varianc_4[1], betta2_4);
            varianc_4[2] = _mm512_mul_ps(varianc_4[2], betta2_4);
            varianc_4[3] = _mm512_mul_ps(varianc_4[3], betta2_4);
            grad_4[0] = _mm512_mul_ps(grad_4[0], grad_4[0]);
            grad_4[1] = _mm512_mul_ps(grad_4[1], grad_4[1]);
            grad_4[2] = _mm512_mul_ps(grad_4[2], grad_4[2]);
            grad_4[3] = _mm512_mul_ps(grad_4[3], grad_4[3]);
            varianc_4[0] = _mm512_fmadd_ps(grad_4[0], betta2_minus1_4, varianc_4[0]);
            varianc_4[1] = _mm512_fmadd_ps(grad_4[1], betta2_minus1_4, varianc_4[1]);
            varianc_4[2] = _mm512_fmadd_ps(grad_4[2], betta2_minus1_4, varianc_4[2]);
            varianc_4[3] = _mm512_fmadd_ps(grad_4[3], betta2_minus1_4, varianc_4[3]);

            grad_4[0] = _mm512_sqrt_ps(varianc_4[0]) / bias2_sqrt;
            grad_4[1] = _mm512_sqrt_ps(varianc_4[1]) / bias2_sqrt;
            grad_4[2] = _mm512_sqrt_ps(varianc_4[2]) / bias2_sqrt;
            grad_4[3] = _mm512_sqrt_ps(varianc_4[3]) / bias2_sqrt;

            grad_4[0] = _mm512_add_ps(grad_4[0], eps_4);
            grad_4[1] = _mm512_add_ps(grad_4[1], eps_4);
            grad_4[2] = _mm512_add_ps(grad_4[2], eps_4);
            grad_4[3] = _mm512_add_ps(grad_4[3], eps_4);
            grad_4[0] = _mm512_div_ps(momntum_4[0], grad_4[0]);
            grad_4[1] = _mm512_div_ps(momntum_4[1], grad_4[1]);
            grad_4[2] = _mm512_div_ps(momntum_4[2], grad_4[2]);
            grad_4[3] = _mm512_div_ps(momntum_4[3], grad_4[3]);

            param_4[0] = _mm512_fmadd_ps(grad_4[0], step_size_4, param_4[0]);
            param_4[1] = _mm512_fmadd_ps(grad_4[1], step_size_4, param_4[1]);
            param_4[2] = _mm512_fmadd_ps(grad_4[2], step_size_4, param_4[2]);
            param_4[3] = _mm512_fmadd_ps(grad_4[3], step_size_4, param_4[3]);

            _mm512_storeu_ps(_params + i, param_4[0]);
            _mm512_storeu_ps(_params + i + SIMD_WIDTH, param_4[1]);
            _mm512_storeu_ps(_params + i + (SIMD_WIDTH << 1), param_4[2]);
            _mm512_storeu_ps(_params + i + SIMD_WIDTH * 3, param_4[3]);

            _mm512_storeu_ps(_exp_avg + i, momntum_4[0]);
            _mm512_storeu_ps(_exp_avg + i + SIMD_WIDTH, momntum_4[1]);
            _mm512_storeu_ps(_exp_avg + i + (SIMD_WIDTH << 1), momntum_4[2]);
            _mm512_storeu_ps(_exp_avg + i + SIMD_WIDTH * 3, momntum_4[3]);

            _mm512_storeu_ps(_exp_avg_sq + i, varianc_4[0]);
            _mm512_storeu_ps(_exp_avg_sq + i + SIMD_WIDTH, varianc_4[1]);
            _mm512_storeu_ps(_exp_avg_sq + i + (SIMD_WIDTH << 1), varianc_4[2]);
            _mm512_storeu_ps(_exp_avg_sq + i + SIMD_WIDTH * 3, varianc_4[3]);
        }

        if (dev_params) {
#pragma omp parallel for
            for (size_t j = 0; j < copy_size; j += 4) {
                _doubled_buffer[buf_index][j] = (__half)_params[t + j];
                _doubled_buffer[buf_index][j + 1] = (__half)_params[t + j + 1];
                _doubled_buffer[buf_index][j + 2] = (__half)_params[t + j + 2];
                _doubled_buffer[buf_index][j + 3] = (__half)_params[t + j + 3];
            }

            CUDA_CHECK(cudaMemcpyAsync(dev_params + t,
                                       _doubled_buffer[buf_index],
                                       copy_size * sizeof(__half),
                                       cudaMemcpyHostToDevice,
                                       Context::Instance().GetCurrentStream()));
            buf_index = !buf_index;
        }
    }
}

int create_adam_optimizer(int optimizer_id,
                          float alpha = 1e-3,
                          float betta1 = 0.9,
                          float betta2 = 0.999,
                          float eps = 1e-8,
                          float weight_decay = 0)

{
    auto opt = std::make_shared<Adam_Optimizer>(alpha, betta1, betta2, eps, weight_decay);

    s_optimizers[optimizer_id] = opt;

    std::cout << "Adam Optimizer #" << optimizer_id << " is created." << std::endl;

    return 0;
}

void Adam_Optimizer::Step_8(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            __half* dev_params)
{
    _betta1_t *= _betta1;
    _betta2_t *= _betta2;

    __m512 betta1_4 = _mm512_set1_ps(_betta1);
    __m512 betta2_4 = _mm512_set1_ps(_betta2);

    bool buf_index = 0;

    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;
    __m512 betta1_minus1_4 = _mm512_set1_ps(betta1_minus1);
    __m512 betta2_minus1_4 = _mm512_set1_ps(betta2_minus1);

    float bias_correction1 = 1 - _betta1_t;
    float bias_correction2 = 1 - _betta2_t;
    //__m512 bias_correction1_4 = _mm512_set1_ps(bias_correction1);
    __m512 bias_correction2_4 = _mm512_set1_ps(bias_correction2);

    __m512 eps_4 = _mm512_set1_ps(_eps);

    float step_size = -1 * _alpha / bias_correction1;
    __m512 step_size_4 = _mm512_set1_ps(step_size);

    __m512 bias2_sqrt = _mm512_sqrt_ps(bias_correction2_4);

    for (size_t t = 0; t < _param_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > _param_size) copy_size = _param_size - t;
        size_t offset = copy_size + t;
#pragma omp parallel for
        for (size_t i = t; i < offset; i += (SIMD_WIDTH << 3)) {
            __m512 grad_4[8];
            grad_4[0] = _mm512_loadu_ps(grads + i);
            grad_4[1] = _mm512_loadu_ps(grads + i + SIMD_WIDTH);
            grad_4[2] = _mm512_loadu_ps(grads + i + (SIMD_WIDTH << 1));
            grad_4[3] = _mm512_loadu_ps(grads + i + SIMD_WIDTH * 3);
            grad_4[4] = _mm512_loadu_ps(grads + i + (SIMD_WIDTH << 2));
            grad_4[5] = _mm512_loadu_ps(grads + i + SIMD_WIDTH * 5);
            grad_4[6] = _mm512_loadu_ps(grads + i + SIMD_WIDTH * 6);
            grad_4[7] = _mm512_loadu_ps(grads + i + SIMD_WIDTH * 7);

            __m512 momntum_4[8];
            momntum_4[0] = _mm512_loadu_ps(_exp_avg + i);
            momntum_4[1] = _mm512_loadu_ps(_exp_avg + i + SIMD_WIDTH);
            momntum_4[2] = _mm512_loadu_ps(_exp_avg + i + (SIMD_WIDTH << 1));
            momntum_4[3] = _mm512_loadu_ps(_exp_avg + i + SIMD_WIDTH * 3);
            momntum_4[4] = _mm512_loadu_ps(_exp_avg + i + (SIMD_WIDTH << 2));
            momntum_4[5] = _mm512_loadu_ps(_exp_avg + i + SIMD_WIDTH * 5);
            momntum_4[6] = _mm512_loadu_ps(_exp_avg + i + SIMD_WIDTH * 6);
            momntum_4[7] = _mm512_loadu_ps(_exp_avg + i + SIMD_WIDTH * 7);

            __m512 varianc_4[8];
            varianc_4[0] = _mm512_loadu_ps(_exp_avg_sq + i);
            varianc_4[1] = _mm512_loadu_ps(_exp_avg_sq + i + SIMD_WIDTH);
            varianc_4[2] = _mm512_loadu_ps(_exp_avg_sq + i + (SIMD_WIDTH << 1));
            varianc_4[3] = _mm512_loadu_ps(_exp_avg_sq + i + SIMD_WIDTH * 3);
            varianc_4[4] = _mm512_loadu_ps(_exp_avg_sq + i + (SIMD_WIDTH << 2));
            varianc_4[5] = _mm512_loadu_ps(_exp_avg_sq + i + SIMD_WIDTH * 5);
            varianc_4[6] = _mm512_loadu_ps(_exp_avg_sq + i + SIMD_WIDTH * 6);
            varianc_4[7] = _mm512_loadu_ps(_exp_avg_sq + i + SIMD_WIDTH * 7);

            __m512 param_4[8];
            param_4[0] = _mm512_loadu_ps(_params + i);
            param_4[1] = _mm512_loadu_ps(_params + i + SIMD_WIDTH);
            param_4[2] = _mm512_loadu_ps(_params + i + (SIMD_WIDTH << 1));
            param_4[3] = _mm512_loadu_ps(_params + i + SIMD_WIDTH * 3);
            param_4[4] = _mm512_loadu_ps(_params + i + (SIMD_WIDTH << 2));
            param_4[5] = _mm512_loadu_ps(_params + i + SIMD_WIDTH * 5);
            param_4[6] = _mm512_loadu_ps(_params + i + SIMD_WIDTH * 6);
            param_4[7] = _mm512_loadu_ps(_params + i + SIMD_WIDTH * 7);

            if (_weight_decay > 0) {
                __m512 weight_decay4 = _mm512_set1_ps(_weight_decay);
                grad_4[0] = _mm512_fmadd_ps(param_4[0], weight_decay4, grad_4[0]);
                grad_4[1] = _mm512_fmadd_ps(param_4[1], weight_decay4, grad_4[1]);
                grad_4[2] = _mm512_fmadd_ps(param_4[2], weight_decay4, grad_4[2]);
                grad_4[3] = _mm512_fmadd_ps(param_4[3], weight_decay4, grad_4[3]);
                grad_4[4] = _mm512_fmadd_ps(param_4[4], weight_decay4, grad_4[4]);
                grad_4[5] = _mm512_fmadd_ps(param_4[5], weight_decay4, grad_4[5]);
                grad_4[6] = _mm512_fmadd_ps(param_4[6], weight_decay4, grad_4[6]);
                grad_4[7] = _mm512_fmadd_ps(param_4[7], weight_decay4, grad_4[7]);
            }

            momntum_4[0] = _mm512_mul_ps(momntum_4[0], betta1_4);
            momntum_4[0] = _mm512_fmadd_ps(grad_4[0], betta1_minus1_4, momntum_4[0]);
            momntum_4[1] = _mm512_mul_ps(momntum_4[1], betta1_4);
            momntum_4[1] = _mm512_fmadd_ps(grad_4[1], betta1_minus1_4, momntum_4[1]);
            momntum_4[2] = _mm512_mul_ps(momntum_4[2], betta1_4);
            momntum_4[2] = _mm512_fmadd_ps(grad_4[2], betta1_minus1_4, momntum_4[2]);
            momntum_4[3] = _mm512_mul_ps(momntum_4[3], betta1_4);
            momntum_4[3] = _mm512_fmadd_ps(grad_4[3], betta1_minus1_4, momntum_4[3]);
            momntum_4[4] = _mm512_mul_ps(momntum_4[4], betta1_4);
            momntum_4[4] = _mm512_fmadd_ps(grad_4[4], betta1_minus1_4, momntum_4[4]);
            momntum_4[5] = _mm512_mul_ps(momntum_4[5], betta1_4);
            momntum_4[5] = _mm512_fmadd_ps(grad_4[5], betta1_minus1_4, momntum_4[5]);
            momntum_4[6] = _mm512_mul_ps(momntum_4[6], betta1_4);
            momntum_4[6] = _mm512_fmadd_ps(grad_4[6], betta1_minus1_4, momntum_4[6]);
            momntum_4[7] = _mm512_mul_ps(momntum_4[7], betta1_4);
            momntum_4[7] = _mm512_fmadd_ps(grad_4[7], betta1_minus1_4, momntum_4[7]);

            varianc_4[0] = _mm512_mul_ps(varianc_4[0], betta2_4);
            varianc_4[1] = _mm512_mul_ps(varianc_4[1], betta2_4);
            varianc_4[2] = _mm512_mul_ps(varianc_4[2], betta2_4);
            varianc_4[3] = _mm512_mul_ps(varianc_4[3], betta2_4);
            varianc_4[4] = _mm512_mul_ps(varianc_4[4], betta2_4);
            varianc_4[5] = _mm512_mul_ps(varianc_4[5], betta2_4);
            varianc_4[6] = _mm512_mul_ps(varianc_4[6], betta2_4);
            varianc_4[7] = _mm512_mul_ps(varianc_4[7], betta2_4);
            grad_4[0] = _mm512_mul_ps(grad_4[0], grad_4[0]);
            grad_4[1] = _mm512_mul_ps(grad_4[1], grad_4[1]);
            grad_4[2] = _mm512_mul_ps(grad_4[2], grad_4[2]);
            grad_4[3] = _mm512_mul_ps(grad_4[3], grad_4[3]);
            grad_4[4] = _mm512_mul_ps(grad_4[4], grad_4[4]);
            grad_4[5] = _mm512_mul_ps(grad_4[5], grad_4[5]);
            grad_4[6] = _mm512_mul_ps(grad_4[6], grad_4[6]);
            grad_4[7] = _mm512_mul_ps(grad_4[7], grad_4[7]);
            varianc_4[0] = _mm512_fmadd_ps(grad_4[0], betta2_minus1_4, varianc_4[0]);
            varianc_4[1] = _mm512_fmadd_ps(grad_4[1], betta2_minus1_4, varianc_4[1]);
            varianc_4[2] = _mm512_fmadd_ps(grad_4[2], betta2_minus1_4, varianc_4[2]);
            varianc_4[3] = _mm512_fmadd_ps(grad_4[3], betta2_minus1_4, varianc_4[3]);
            varianc_4[4] = _mm512_fmadd_ps(grad_4[4], betta2_minus1_4, varianc_4[4]);
            varianc_4[5] = _mm512_fmadd_ps(grad_4[5], betta2_minus1_4, varianc_4[5]);
            varianc_4[6] = _mm512_fmadd_ps(grad_4[6], betta2_minus1_4, varianc_4[6]);
            varianc_4[7] = _mm512_fmadd_ps(grad_4[7], betta2_minus1_4, varianc_4[7]);

            grad_4[0] = _mm512_sqrt_ps(varianc_4[0]) / bias2_sqrt;
            grad_4[1] = _mm512_sqrt_ps(varianc_4[1]) / bias2_sqrt;
            grad_4[2] = _mm512_sqrt_ps(varianc_4[2]) / bias2_sqrt;
            grad_4[3] = _mm512_sqrt_ps(varianc_4[3]) / bias2_sqrt;
            grad_4[4] = _mm512_sqrt_ps(varianc_4[4]) / bias2_sqrt;
            grad_4[5] = _mm512_sqrt_ps(varianc_4[5]) / bias2_sqrt;
            grad_4[6] = _mm512_sqrt_ps(varianc_4[6]) / bias2_sqrt;
            grad_4[7] = _mm512_sqrt_ps(varianc_4[7]) / bias2_sqrt;

            grad_4[0] = _mm512_add_ps(grad_4[0], eps_4);
            grad_4[1] = _mm512_add_ps(grad_4[1], eps_4);
            grad_4[2] = _mm512_add_ps(grad_4[2], eps_4);
            grad_4[3] = _mm512_add_ps(grad_4[3], eps_4);
            grad_4[4] = _mm512_add_ps(grad_4[4], eps_4);
            grad_4[5] = _mm512_add_ps(grad_4[5], eps_4);
            grad_4[6] = _mm512_add_ps(grad_4[6], eps_4);
            grad_4[7] = _mm512_add_ps(grad_4[7], eps_4);
            grad_4[0] = _mm512_div_ps(momntum_4[0], grad_4[0]);
            grad_4[1] = _mm512_div_ps(momntum_4[1], grad_4[1]);
            grad_4[2] = _mm512_div_ps(momntum_4[2], grad_4[2]);
            grad_4[3] = _mm512_div_ps(momntum_4[3], grad_4[3]);
            grad_4[4] = _mm512_div_ps(momntum_4[4], grad_4[4]);
            grad_4[5] = _mm512_div_ps(momntum_4[5], grad_4[5]);
            grad_4[6] = _mm512_div_ps(momntum_4[6], grad_4[6]);
            grad_4[7] = _mm512_div_ps(momntum_4[7], grad_4[7]);

            param_4[0] = _mm512_fmadd_ps(grad_4[0], step_size_4, param_4[0]);
            param_4[1] = _mm512_fmadd_ps(grad_4[1], step_size_4, param_4[1]);
            param_4[2] = _mm512_fmadd_ps(grad_4[2], step_size_4, param_4[2]);
            param_4[3] = _mm512_fmadd_ps(grad_4[3], step_size_4, param_4[3]);
            param_4[4] = _mm512_fmadd_ps(grad_4[4], step_size_4, param_4[4]);
            param_4[5] = _mm512_fmadd_ps(grad_4[5], step_size_4, param_4[5]);
            param_4[6] = _mm512_fmadd_ps(grad_4[6], step_size_4, param_4[6]);
            param_4[7] = _mm512_fmadd_ps(grad_4[7], step_size_4, param_4[7]);

            _mm512_storeu_ps(_params + i, param_4[0]);
            _mm512_storeu_ps(_params + i + SIMD_WIDTH, param_4[1]);
            _mm512_storeu_ps(_params + i + (SIMD_WIDTH << 1), param_4[2]);
            _mm512_storeu_ps(_params + i + SIMD_WIDTH * 3, param_4[3]);
            _mm512_storeu_ps(_params + i + (SIMD_WIDTH << 2), param_4[4]);
            _mm512_storeu_ps(_params + i + SIMD_WIDTH * 5, param_4[5]);
            _mm512_storeu_ps(_params + i + SIMD_WIDTH * 6, param_4[6]);
            _mm512_storeu_ps(_params + i + SIMD_WIDTH * 7, param_4[7]);

            _mm512_storeu_ps(_doubled_buffer[buf_index] + (i - t), param_4[0]);
            _mm512_storeu_ps(_doubled_buffer[buf_index] + (i - t) + SIMD_WIDTH, param_4[1]);
            _mm512_storeu_ps(_doubled_buffer[buf_index] + (i - t) + (SIMD_WIDTH << 1), param_4[2]);
            _mm512_storeu_ps(_doubled_buffer[buf_index] + (i - t) + SIMD_WIDTH * 3, param_4[3]);
            _mm512_storeu_ps(_doubled_buffer[buf_index] + (i - t) + (SIMD_WIDTH << 2), param_4[4]);
            _mm512_storeu_ps(_doubled_buffer[buf_index] + (i - t) + SIMD_WIDTH * 5, param_4[5]);
            _mm512_storeu_ps(_doubled_buffer[buf_index] + (i - t) + SIMD_WIDTH * 6, param_4[6]);
            _mm512_storeu_ps(_doubled_buffer[buf_index] + (i - t) + SIMD_WIDTH * 7, param_4[7]);

            _mm512_storeu_ps(_exp_avg + i, momntum_4[0]);
            _mm512_storeu_ps(_exp_avg + i + SIMD_WIDTH, momntum_4[1]);
            _mm512_storeu_ps(_exp_avg + i + (SIMD_WIDTH << 1), momntum_4[2]);
            _mm512_storeu_ps(_exp_avg + i + SIMD_WIDTH * 3, momntum_4[3]);
            _mm512_storeu_ps(_exp_avg + i + (SIMD_WIDTH << 2), momntum_4[4]);
            _mm512_storeu_ps(_exp_avg + i + SIMD_WIDTH * 5, momntum_4[5]);
            _mm512_storeu_ps(_exp_avg + i + SIMD_WIDTH * 6, momntum_4[6]);
            _mm512_storeu_ps(_exp_avg + i + SIMD_WIDTH * 7, momntum_4[7]);

            _mm512_storeu_ps(_exp_avg_sq + i, varianc_4[0]);
            _mm512_storeu_ps(_exp_avg_sq + i + SIMD_WIDTH, varianc_4[1]);
            _mm512_storeu_ps(_exp_avg_sq + i + (SIMD_WIDTH << 1), varianc_4[2]);
            _mm512_storeu_ps(_exp_avg_sq + i + SIMD_WIDTH * 3, varianc_4[3]);
            _mm512_storeu_ps(_exp_avg_sq + i + (SIMD_WIDTH << 2), varianc_4[4]);
            _mm512_storeu_ps(_exp_avg_sq + i + SIMD_WIDTH * 5, varianc_4[5]);
            _mm512_storeu_ps(_exp_avg_sq + i + SIMD_WIDTH * 6, varianc_4[6]);
            _mm512_storeu_ps(_exp_avg_sq + i + SIMD_WIDTH * 7, varianc_4[7]);
        }
        if (dev_params) {
            launch_param_update(_doubled_buffer[buf_index],
                                dev_params + t,
                                copy_size,
                                Context::Instance().GetCurrentStream());
            buf_index = !buf_index;
        }
    }
}

int ds_adam_step(int optimizer_id,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg,
                 torch::Tensor& exp_avg_sq)
{
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);

    opt->Step_8(params_ptr, grads_ptr, exp_avg_ptr, exp_avg_sq_ptr, params_c.size(0));

    return 0;
}

int ds_adam_step_plus_copy(int optimizer_id,
                           torch::Tensor& params,
                           torch::Tensor& grads,
                           torch::Tensor& exp_avg,
                           torch::Tensor& exp_avg_sq,
                           torch::Tensor& gpu_params)
{
    auto params_c = params.contiguous();
    auto gpu_params_c = gpu_params.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();
    auto grads_c = grads.contiguous();

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    __half* gpu_params_ptr = (__half*)gpu_params_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);

    opt->Step_8(
        params_ptr, grads_ptr, exp_avg_ptr, exp_avg_sq_ptr, params_c.size(0), gpu_params_ptr);

    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("adam_update", &ds_adam_step, "DeepSpeed CPU Adam update (C++)");
    m.def("adam_update_copy",
          &ds_adam_step_plus_copy,
          "DeepSpeed CPU Adam update and param copy (C++)");
    m.def("create_adam", &create_adam_optimizer, "DeepSpeed CPU Adam (C++)");
}
