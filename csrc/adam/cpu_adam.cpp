#include "cpu_adam.h"
#include <cuda_runtime_api.h>
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
#include <math.h>

static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

#define ROUND_DOWN(size, step) ((size) & ~((step)-1))


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

    AVX_512 betta1_4;
    betta1_4.data = _mm512_set1_ps(_betta1);
    AVX_512 betta2_4;
    betta2_4.data = _mm512_set1_ps(_betta2);

    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;
    AVX_512 betta1_minus1_4;
    betta1_minus1_4.data = _mm512_set1_ps(betta1_minus1);
    AVX_512 betta2_minus1_4;
    betta2_minus1_4.data = _mm512_set1_ps(betta2_minus1);

    float bias_correction1 = 1 - _betta1_t;
    float bias_correction2 = 1 / sqrt(1 - _betta2_t);
    //AVX_512 bias_correction1_4 = _mm512_set1_ps(bias_correction1);
    AVX_512 bias2_sqrt ;
    bias2_sqrt.data = _mm512_set1_ps(bias_correction2);

    AVX_512 eps_4;
    eps_4.data = _mm512_set1_ps(_eps);

    float step_size = -1 * _alpha / bias_correction1;
    AVX_512 step_size_4;
    step_size_4.data = _mm512_set1_ps(step_size);

    size_t rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH);

    for (size_t t = 0; t < rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
        size_t offset = copy_size + t;
#pragma omp parallel for
        for (size_t i = t; i < offset; i += SIMD_WIDTH) {
            AVX_512 grad_4;
            grad_4.data = _mm512_loadu_ps(grads + i);

            AVX_512 momntum_4;
            momntum_4.data = _mm512_loadu_ps(_exp_avg + i);
            AVX_512 varianc_4;
            varianc_4.data = _mm512_loadu_ps(_exp_avg_sq + i);

            AVX_512 param_4;
            param_4.data = _mm512_loadu_ps(_params + i);

            if (_weight_decay > 0) {
                AVX_512 weight_decay4;
                weight_decay4.data = _mm512_set1_ps(_weight_decay);
                grad_4.data = _mm512_fmadd_ps(param_4.data, weight_decay4.data, grad_4.data);
            }

            momntum_4.data = _mm512_mul_ps(momntum_4.data, betta1_4.data);
            momntum_4.data = _mm512_fmadd_ps(grad_4.data, betta1_minus1_4.data, momntum_4.data);

            varianc_4.data = _mm512_mul_ps(varianc_4.data, betta2_4.data);
            grad_4.data = _mm512_mul_ps(grad_4.data, grad_4.data);
            varianc_4.data = _mm512_fmadd_ps(grad_4.data, betta2_minus1_4.data, varianc_4.data);

            grad_4.data = _mm512_sqrt_ps(varianc_4.data);
            grad_4.data = _mm512_fmadd_ps(grad_4.data, bias2_sqrt.data, eps_4.data);
            grad_4.data = _mm512_div_ps(momntum_4.data, grad_4.data);

            param_4.data = _mm512_fmadd_ps(grad_4.data, step_size_4.data, param_4.data);

            _mm512_storeu_ps(_params + i, param_4.data);
            
            if (dev_params)_mm512_storeu_ps(_doubled_buffer[_buf_index] + (i - t), param_4.data);

            _mm512_storeu_ps(_exp_avg + i, momntum_4.data);
            _mm512_storeu_ps(_exp_avg_sq + i, varianc_4.data);
        }
        if (dev_params) {/*
#pragma omp parallel for
            for (size_t j = 0; j < copy_size; j += 4) {
                _doubled_buffer[_buf_index][j] = (__half)_params[t + j];
                _doubled_buffer[_buf_index][j + 1] = (__half)_params[t + j + 1];
                _doubled_buffer[_buf_index][j + 2] = (__half)_params[t + j + 2];
                _doubled_buffer[_buf_index][j + 3] = (__half)_params[t + j + 3];
            }

            CUDA_CHECK(cudaMemcpyAsync(dev_params + t,
                                       _doubled_buffer[_buf_index],
                                       copy_size * sizeof(__half),
                                       cudaMemcpyHostToDevice,
                                       Context::Instance().GetCurrentStream()));*/
            launch_param_update(_doubled_buffer[_buf_index],
                                dev_params + t,
                                copy_size,
                                Context::Instance().GetCurrentStream());
            _buf_index = !_buf_index;
        }
    }

    if(_param_size > rounded_size)
    {
#pragma omp parallel for
        for (size_t k = rounded_size; k < _param_size; k++) 
        {
            float grad = grads[k];
            float param = _params[k];
            float momntum = _exp_avg[k];
            float varianc = _exp_avg_sq[k];
            if (_weight_decay > 0) {
                grad = param * _weight_decay + grad;
            }

            momntum *= momntum * _betta1;
            momntum = grad * betta1_minus1 + momntum;

            varianc = varianc * _betta2;
            grad = grad * grad;
            varianc = grad * betta2_minus1 + varianc;

            grad = sqrt(varianc);
            grad = grad * bias_correction2 + _eps;
            grad = momntum / grad;

            param = grad * step_size + param;
            if (dev_params) 
                _doubled_buffer[_buf_index][k - rounded_size] = (__half)param;

            _params[k] = param;
            _exp_avg[k] = momntum;
            _exp_avg_sq[k] = varianc;
        }
        if (dev_params) {
            launch_param_update(_doubled_buffer[_buf_index],
                                dev_params + rounded_size,
                                (_param_size - rounded_size),
                                Context::Instance().GetCurrentStream());
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

    AVX_512 betta1_4;
    betta1_4.data = _mm512_set1_ps(_betta1);
    AVX_512 betta2_4;
    betta2_4.data = _mm512_set1_ps(_betta2);

    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;
    AVX_512 betta1_minus1_4;
    betta1_minus1_4.data = _mm512_set1_ps(betta1_minus1);
    AVX_512 betta2_minus1_4;
    betta2_minus1_4.data = _mm512_set1_ps(betta2_minus1);

    float bias_correction1 = 1 - _betta1_t;
    float bias_correction2 = 1 / sqrt(1 - _betta2_t);
    //AVX_512 bias_correction1_4 = _mm512_set1_ps(bias_correction1);
    AVX_512 bias2_sqrt ;
    bias2_sqrt.data = _mm512_set1_ps(bias_correction2);

    AVX_512 eps_4;
    eps_4.data = _mm512_set1_ps(_eps);

    float step_size = -1 * _alpha / bias_correction1;
    AVX_512 step_size_4;
    step_size_4.data = _mm512_set1_ps(step_size);

    size_t rounded_size = ROUND_DOWN(_param_size, (SIMD_WIDTH << 2));

    for (size_t t = 0; t < rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
        size_t offset = copy_size + t;
#pragma omp parallel for
        for (size_t i = t; i < offset; i += (SIMD_WIDTH << 2)) {
            AVX_512 grad_4[4];
            grad_4[0].data = _mm512_loadu_ps(grads + i);
            grad_4[1].data = _mm512_loadu_ps(grads + i + SIMD_WIDTH);
            grad_4[2].data = _mm512_loadu_ps(grads + i + (SIMD_WIDTH << 1));
            grad_4[3].data = _mm512_loadu_ps(grads + i + SIMD_WIDTH * 3);

            AVX_512 momntum_4[4];
            momntum_4[0].data = _mm512_loadu_ps(_exp_avg + i);
            momntum_4[1].data = _mm512_loadu_ps(_exp_avg + i + SIMD_WIDTH);
            momntum_4[2].data = _mm512_loadu_ps(_exp_avg + i + (SIMD_WIDTH << 1));
            momntum_4[3].data = _mm512_loadu_ps(_exp_avg + i + SIMD_WIDTH * 3);

            AVX_512 varianc_4[4];
            varianc_4[0].data = _mm512_loadu_ps(_exp_avg_sq + i);
            varianc_4[1].data = _mm512_loadu_ps(_exp_avg_sq + i + SIMD_WIDTH);
            varianc_4[2].data = _mm512_loadu_ps(_exp_avg_sq + i + (SIMD_WIDTH << 1));
            varianc_4[3].data = _mm512_loadu_ps(_exp_avg_sq + i + SIMD_WIDTH * 3);

            AVX_512 param_4[4];
            param_4[0].data = _mm512_loadu_ps(_params + i);
            param_4[1].data = _mm512_loadu_ps(_params + i + SIMD_WIDTH);
            param_4[2].data = _mm512_loadu_ps(_params + i + (SIMD_WIDTH << 1));
            param_4[3].data = _mm512_loadu_ps(_params + i + SIMD_WIDTH * 3);

            if (_weight_decay > 0) {
                AVX_512 weight_decay4;
                weight_decay4.data = _mm512_set1_ps(_weight_decay);
                grad_4[0].data = _mm512_fmadd_ps(param_4[0].data, weight_decay4.data, grad_4[0].data);
                grad_4[1].data = _mm512_fmadd_ps(param_4[1].data, weight_decay4.data, grad_4[1].data);
                grad_4[2].data = _mm512_fmadd_ps(param_4[2].data, weight_decay4.data, grad_4[2].data);
                grad_4[3].data = _mm512_fmadd_ps(param_4[3].data, weight_decay4.data, grad_4[3].data);
            }

            momntum_4[0].data = _mm512_mul_ps(momntum_4[0].data, betta1_4.data);
            momntum_4[0].data = _mm512_fmadd_ps(grad_4[0].data, betta1_minus1_4.data, momntum_4[0].data);
            momntum_4[1].data = _mm512_mul_ps(momntum_4[1].data, betta1_4.data);
            momntum_4[1].data = _mm512_fmadd_ps(grad_4[1].data, betta1_minus1_4.data, momntum_4[1].data);
            momntum_4[2].data = _mm512_mul_ps(momntum_4[2].data, betta1_4.data);
            momntum_4[2].data = _mm512_fmadd_ps(grad_4[2].data, betta1_minus1_4.data, momntum_4[2].data);
            momntum_4[3].data = _mm512_mul_ps(momntum_4[3].data, betta1_4.data);
            momntum_4[3].data = _mm512_fmadd_ps(grad_4[3].data, betta1_minus1_4.data, momntum_4[3].data);

            varianc_4[0].data = _mm512_mul_ps(varianc_4[0].data, betta2_4.data);
            varianc_4[1].data = _mm512_mul_ps(varianc_4[1].data, betta2_4.data);
            varianc_4[2].data = _mm512_mul_ps(varianc_4[2].data, betta2_4.data);
            varianc_4[3].data = _mm512_mul_ps(varianc_4[3].data, betta2_4.data);
            grad_4[0].data = _mm512_mul_ps(grad_4[0].data, grad_4[0].data);
            grad_4[1].data = _mm512_mul_ps(grad_4[1].data, grad_4[1].data);
            grad_4[2].data = _mm512_mul_ps(grad_4[2].data, grad_4[2].data);
            grad_4[3].data = _mm512_mul_ps(grad_4[3].data, grad_4[3].data);
            varianc_4[0].data = _mm512_fmadd_ps(grad_4[0].data, betta2_minus1_4.data, varianc_4[0].data);
            varianc_4[1].data = _mm512_fmadd_ps(grad_4[1].data, betta2_minus1_4.data, varianc_4[1].data);
            varianc_4[2].data = _mm512_fmadd_ps(grad_4[2].data, betta2_minus1_4.data, varianc_4[2].data);
            varianc_4[3].data = _mm512_fmadd_ps(grad_4[3].data, betta2_minus1_4.data, varianc_4[3].data);

            grad_4[0].data = _mm512_sqrt_ps(varianc_4[0].data);
            grad_4[1].data = _mm512_sqrt_ps(varianc_4[1].data);
            grad_4[2].data = _mm512_sqrt_ps(varianc_4[2].data);
            grad_4[3].data = _mm512_sqrt_ps(varianc_4[3].data);

            grad_4[0].data = _mm512_fmadd_ps(grad_4[0].data, bias2_sqrt.data, eps_4.data);
            grad_4[1].data = _mm512_fmadd_ps(grad_4[1].data, bias2_sqrt.data, eps_4.data);
            grad_4[2].data = _mm512_fmadd_ps(grad_4[2].data, bias2_sqrt.data, eps_4.data);
            grad_4[3].data = _mm512_fmadd_ps(grad_4[3].data, bias2_sqrt.data, eps_4.data);
            grad_4[0].data = _mm512_div_ps(momntum_4[0].data, grad_4[0].data);
            grad_4[1].data = _mm512_div_ps(momntum_4[1].data, grad_4[1].data);
            grad_4[2].data = _mm512_div_ps(momntum_4[2].data, grad_4[2].data);
            grad_4[3].data = _mm512_div_ps(momntum_4[3].data, grad_4[3].data);

            param_4[0].data = _mm512_fmadd_ps(grad_4[0].data, step_size_4.data, param_4[0].data);
            param_4[1].data = _mm512_fmadd_ps(grad_4[1].data, step_size_4.data, param_4[1].data);
            param_4[2].data = _mm512_fmadd_ps(grad_4[2].data, step_size_4.data, param_4[2].data);
            param_4[3].data = _mm512_fmadd_ps(grad_4[3].data, step_size_4.data, param_4[3].data);

            _mm512_storeu_ps(_params + i, param_4[0].data);
            _mm512_storeu_ps(_params + i + SIMD_WIDTH, param_4[1].data);
            _mm512_storeu_ps(_params + i + (SIMD_WIDTH << 1), param_4[2].data);
            _mm512_storeu_ps(_params + i + SIMD_WIDTH * 3, param_4[3].data);

            if (dev_params) {
                _mm512_storeu_ps(_doubled_buffer[_buf_index] + (i - t), param_4[0].data);
                _mm512_storeu_ps(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH, param_4[1].data);
                _mm512_storeu_ps(_doubled_buffer[_buf_index] + (i - t) + (SIMD_WIDTH << 1), param_4[2].data);
                _mm512_storeu_ps(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH * 3, param_4[3].data);
            }

            _mm512_storeu_ps(_exp_avg + i, momntum_4[0].data);
            _mm512_storeu_ps(_exp_avg + i + SIMD_WIDTH, momntum_4[1].data);
            _mm512_storeu_ps(_exp_avg + i + (SIMD_WIDTH << 1), momntum_4[2].data);
            _mm512_storeu_ps(_exp_avg + i + SIMD_WIDTH * 3, momntum_4[3].data);

            _mm512_storeu_ps(_exp_avg_sq + i, varianc_4[0].data);
            _mm512_storeu_ps(_exp_avg_sq + i + SIMD_WIDTH, varianc_4[1].data);
            _mm512_storeu_ps(_exp_avg_sq + i + (SIMD_WIDTH << 1), varianc_4[2].data);
            _mm512_storeu_ps(_exp_avg_sq + i + SIMD_WIDTH * 3, varianc_4[3].data);
        }

        if (dev_params) {/*
#pragma omp parallel for
            for (size_t j = 0; j < copy_size; j += 4) {
                _doubled_buffer[_buf_index][j] = (__half)_params[t + j];
                _doubled_buffer[_buf_index][j + 1] = (__half)_params[t + j + 1];
                _doubled_buffer[_buf_index][j + 2] = (__half)_params[t + j + 2];
                _doubled_buffer[_buf_index][j + 3] = (__half)_params[t + j + 3];
            }

            CUDA_CHECK(cudaMemcpyAsync(dev_params + t,
                                       _doubled_buffer[_buf_index],
                                       copy_size * sizeof(__half),
                                       cudaMemcpyHostToDevice,
                                       Context::Instance().GetCurrentStream()));
            */
            launch_param_update(_doubled_buffer[_buf_index],
                                dev_params + t,
                                copy_size,
                                Context::Instance().GetCurrentStream());
            _buf_index = !_buf_index;
        }
    }
    if(_param_size > rounded_size)
        Step((_params + rounded_size),
             (grads + rounded_size),
             (_exp_avg + rounded_size),
             (_exp_avg_sq + rounded_size),
             (_param_size - rounded_size),
             (dev_params != nullptr ? (dev_params + rounded_size) : dev_params));
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

    AVX_512 betta1_4;
    betta1_4.data = _mm512_set1_ps(_betta1);
    AVX_512 betta2_4;
    betta2_4.data = _mm512_set1_ps(_betta2);

    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;
    AVX_512 betta1_minus1_4;
    betta1_minus1_4.data = _mm512_set1_ps(betta1_minus1);
    AVX_512 betta2_minus1_4;
    betta2_minus1_4.data = _mm512_set1_ps(betta2_minus1);

    float bias_correction1 = 1 - _betta1_t;
    float bias_correction2 = 1 / sqrt(1 - _betta2_t);
    //AVX_512 bias_correction1_4 = _mm512_set1_ps(bias_correction1);
    AVX_512 bias2_sqrt ;
    bias2_sqrt.data = _mm512_set1_ps(bias_correction2);

    AVX_512 eps_4;
    eps_4.data = _mm512_set1_ps(_eps);

    float step_size = -1 * _alpha / bias_correction1;
    AVX_512 step_size_4;
    step_size_4.data = _mm512_set1_ps(step_size);

    size_t rounded_size = ROUND_DOWN(_param_size, (SIMD_WIDTH << 3));

    for (size_t t = 0; t < rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
        size_t offset = copy_size + t;
#pragma omp parallel for
        for (size_t i = t; i < offset; i += (SIMD_WIDTH << 3)) {
            AVX_512 grad_4[8];
            grad_4[0].data = _mm512_loadu_ps(grads + i);
            grad_4[1].data = _mm512_loadu_ps(grads + i + SIMD_WIDTH);
            grad_4[2].data = _mm512_loadu_ps(grads + i + (SIMD_WIDTH << 1));
            grad_4[3].data = _mm512_loadu_ps(grads + i + SIMD_WIDTH * 3);
            grad_4[4].data = _mm512_loadu_ps(grads + i + (SIMD_WIDTH << 2));
            grad_4[5].data = _mm512_loadu_ps(grads + i + SIMD_WIDTH * 5);
            grad_4[6].data = _mm512_loadu_ps(grads + i + SIMD_WIDTH * 6);
            grad_4[7].data = _mm512_loadu_ps(grads + i + SIMD_WIDTH * 7);

            AVX_512 momntum_4[8];
            momntum_4[0].data = _mm512_loadu_ps(_exp_avg + i);
            momntum_4[1].data = _mm512_loadu_ps(_exp_avg + i + SIMD_WIDTH);
            momntum_4[2].data = _mm512_loadu_ps(_exp_avg + i + (SIMD_WIDTH << 1));
            momntum_4[3].data = _mm512_loadu_ps(_exp_avg + i + SIMD_WIDTH * 3);
            momntum_4[4].data = _mm512_loadu_ps(_exp_avg + i + (SIMD_WIDTH << 2));
            momntum_4[5].data = _mm512_loadu_ps(_exp_avg + i + SIMD_WIDTH * 5);
            momntum_4[6].data = _mm512_loadu_ps(_exp_avg + i + SIMD_WIDTH * 6);
            momntum_4[7].data = _mm512_loadu_ps(_exp_avg + i + SIMD_WIDTH * 7);

            AVX_512 varianc_4[8];
            varianc_4[0].data = _mm512_loadu_ps(_exp_avg_sq + i);
            varianc_4[1].data = _mm512_loadu_ps(_exp_avg_sq + i + SIMD_WIDTH);
            varianc_4[2].data = _mm512_loadu_ps(_exp_avg_sq + i + (SIMD_WIDTH << 1));
            varianc_4[3].data = _mm512_loadu_ps(_exp_avg_sq + i + SIMD_WIDTH * 3);
            varianc_4[4].data = _mm512_loadu_ps(_exp_avg_sq + i + (SIMD_WIDTH << 2));
            varianc_4[5].data = _mm512_loadu_ps(_exp_avg_sq + i + SIMD_WIDTH * 5);
            varianc_4[6].data = _mm512_loadu_ps(_exp_avg_sq + i + SIMD_WIDTH * 6);
            varianc_4[7].data = _mm512_loadu_ps(_exp_avg_sq + i + SIMD_WIDTH * 7);

            AVX_512 param_4[8];
            param_4[0].data = _mm512_loadu_ps(_params + i);
            param_4[1].data = _mm512_loadu_ps(_params + i + SIMD_WIDTH);
            param_4[2].data = _mm512_loadu_ps(_params + i + (SIMD_WIDTH << 1));
            param_4[3].data = _mm512_loadu_ps(_params + i + SIMD_WIDTH * 3);
            param_4[4].data = _mm512_loadu_ps(_params + i + (SIMD_WIDTH << 2));
            param_4[5].data = _mm512_loadu_ps(_params + i + SIMD_WIDTH * 5);
            param_4[6].data = _mm512_loadu_ps(_params + i + SIMD_WIDTH * 6);
            param_4[7].data = _mm512_loadu_ps(_params + i + SIMD_WIDTH * 7);

            if (_weight_decay > 0) {
                AVX_512 weight_decay4;
                weight_decay4.data = _mm512_set1_ps(_weight_decay);
                grad_4[0].data = _mm512_fmadd_ps(param_4[0].data, weight_decay4.data, grad_4[0].data);
                grad_4[1].data = _mm512_fmadd_ps(param_4[1].data, weight_decay4.data, grad_4[1].data);
                grad_4[2].data = _mm512_fmadd_ps(param_4[2].data, weight_decay4.data, grad_4[2].data);
                grad_4[3].data = _mm512_fmadd_ps(param_4[3].data, weight_decay4.data, grad_4[3].data);
                grad_4[4].data = _mm512_fmadd_ps(param_4[4].data, weight_decay4.data, grad_4[4].data);
                grad_4[5].data = _mm512_fmadd_ps(param_4[5].data, weight_decay4.data, grad_4[5].data);
                grad_4[6].data = _mm512_fmadd_ps(param_4[6].data, weight_decay4.data, grad_4[6].data);
                grad_4[7].data = _mm512_fmadd_ps(param_4[7].data, weight_decay4.data, grad_4[7].data);
            }

            momntum_4[0].data = _mm512_mul_ps(momntum_4[0].data, betta1_4.data);
            momntum_4[0].data = _mm512_fmadd_ps(grad_4[0].data, betta1_minus1_4.data, momntum_4[0].data);
            momntum_4[1].data = _mm512_mul_ps(momntum_4[1].data, betta1_4.data);
            momntum_4[1].data = _mm512_fmadd_ps(grad_4[1].data, betta1_minus1_4.data, momntum_4[1].data);
            momntum_4[2].data = _mm512_mul_ps(momntum_4[2].data, betta1_4.data);
            momntum_4[2].data = _mm512_fmadd_ps(grad_4[2].data, betta1_minus1_4.data, momntum_4[2].data);
            momntum_4[3].data = _mm512_mul_ps(momntum_4[3].data, betta1_4.data);
            momntum_4[3].data = _mm512_fmadd_ps(grad_4[3].data, betta1_minus1_4.data, momntum_4[3].data);
            momntum_4[4].data = _mm512_mul_ps(momntum_4[4].data, betta1_4.data);
            momntum_4[4].data = _mm512_fmadd_ps(grad_4[4].data, betta1_minus1_4.data, momntum_4[4].data);
            momntum_4[5].data = _mm512_mul_ps(momntum_4[5].data, betta1_4.data);
            momntum_4[5].data = _mm512_fmadd_ps(grad_4[5].data, betta1_minus1_4.data, momntum_4[5].data);
            momntum_4[6].data = _mm512_mul_ps(momntum_4[6].data, betta1_4.data);
            momntum_4[6].data = _mm512_fmadd_ps(grad_4[6].data, betta1_minus1_4.data, momntum_4[6].data);
            momntum_4[7].data = _mm512_mul_ps(momntum_4[7].data, betta1_4.data);
            momntum_4[7].data = _mm512_fmadd_ps(grad_4[7].data, betta1_minus1_4.data, momntum_4[7].data);

            varianc_4[0].data = _mm512_mul_ps(varianc_4[0].data, betta2_4.data);
            varianc_4[1].data = _mm512_mul_ps(varianc_4[1].data, betta2_4.data);
            varianc_4[2].data = _mm512_mul_ps(varianc_4[2].data, betta2_4.data);
            varianc_4[3].data = _mm512_mul_ps(varianc_4[3].data, betta2_4.data);
            varianc_4[4].data = _mm512_mul_ps(varianc_4[4].data, betta2_4.data);
            varianc_4[5].data = _mm512_mul_ps(varianc_4[5].data, betta2_4.data);
            varianc_4[6].data = _mm512_mul_ps(varianc_4[6].data, betta2_4.data);
            varianc_4[7].data = _mm512_mul_ps(varianc_4[7].data, betta2_4.data);
            grad_4[0].data = _mm512_mul_ps(grad_4[0].data, grad_4[0].data);
            grad_4[1].data = _mm512_mul_ps(grad_4[1].data, grad_4[1].data);
            grad_4[2].data = _mm512_mul_ps(grad_4[2].data, grad_4[2].data);
            grad_4[3].data = _mm512_mul_ps(grad_4[3].data, grad_4[3].data);
            grad_4[4].data = _mm512_mul_ps(grad_4[4].data, grad_4[4].data);
            grad_4[5].data = _mm512_mul_ps(grad_4[5].data, grad_4[5].data);
            grad_4[6].data = _mm512_mul_ps(grad_4[6].data, grad_4[6].data);
            grad_4[7].data = _mm512_mul_ps(grad_4[7].data, grad_4[7].data);
            varianc_4[0].data = _mm512_fmadd_ps(grad_4[0].data, betta2_minus1_4.data, varianc_4[0].data);
            varianc_4[1].data = _mm512_fmadd_ps(grad_4[1].data, betta2_minus1_4.data, varianc_4[1].data);
            varianc_4[2].data = _mm512_fmadd_ps(grad_4[2].data, betta2_minus1_4.data, varianc_4[2].data);
            varianc_4[3].data = _mm512_fmadd_ps(grad_4[3].data, betta2_minus1_4.data, varianc_4[3].data);
            varianc_4[4].data = _mm512_fmadd_ps(grad_4[4].data, betta2_minus1_4.data, varianc_4[4].data);
            varianc_4[5].data = _mm512_fmadd_ps(grad_4[5].data, betta2_minus1_4.data, varianc_4[5].data);
            varianc_4[6].data = _mm512_fmadd_ps(grad_4[6].data, betta2_minus1_4.data, varianc_4[6].data);
            varianc_4[7].data = _mm512_fmadd_ps(grad_4[7].data, betta2_minus1_4.data, varianc_4[7].data);

            grad_4[0].data = _mm512_sqrt_ps(varianc_4[0].data);
            grad_4[1].data = _mm512_sqrt_ps(varianc_4[1].data);
            grad_4[2].data = _mm512_sqrt_ps(varianc_4[2].data);
            grad_4[3].data = _mm512_sqrt_ps(varianc_4[3].data);
            grad_4[4].data = _mm512_sqrt_ps(varianc_4[4].data);
            grad_4[5].data = _mm512_sqrt_ps(varianc_4[5].data);
            grad_4[6].data = _mm512_sqrt_ps(varianc_4[6].data);
            grad_4[7].data = _mm512_sqrt_ps(varianc_4[7].data);

            grad_4[0].data = _mm512_fmadd_ps(grad_4[0].data, bias2_sqrt.data, eps_4.data);
            grad_4[1].data = _mm512_fmadd_ps(grad_4[1].data, bias2_sqrt.data, eps_4.data);
            grad_4[2].data = _mm512_fmadd_ps(grad_4[2].data, bias2_sqrt.data, eps_4.data);
            grad_4[3].data = _mm512_fmadd_ps(grad_4[3].data, bias2_sqrt.data, eps_4.data);
            grad_4[4].data = _mm512_fmadd_ps(grad_4[4].data, bias2_sqrt.data, eps_4.data);
            grad_4[5].data = _mm512_fmadd_ps(grad_4[5].data, bias2_sqrt.data, eps_4.data);
            grad_4[6].data = _mm512_fmadd_ps(grad_4[6].data, bias2_sqrt.data, eps_4.data);
            grad_4[7].data = _mm512_fmadd_ps(grad_4[7].data, bias2_sqrt.data, eps_4.data);
            grad_4[0].data = _mm512_div_ps(momntum_4[0].data, grad_4[0].data);
            grad_4[1].data = _mm512_div_ps(momntum_4[1].data, grad_4[1].data);
            grad_4[2].data = _mm512_div_ps(momntum_4[2].data, grad_4[2].data);
            grad_4[3].data = _mm512_div_ps(momntum_4[3].data, grad_4[3].data);
            grad_4[4].data = _mm512_div_ps(momntum_4[4].data, grad_4[4].data);
            grad_4[5].data = _mm512_div_ps(momntum_4[5].data, grad_4[5].data);
            grad_4[6].data = _mm512_div_ps(momntum_4[6].data, grad_4[6].data);
            grad_4[7].data = _mm512_div_ps(momntum_4[7].data, grad_4[7].data);

            param_4[0].data = _mm512_fmadd_ps(grad_4[0].data, step_size_4.data, param_4[0].data);
            param_4[1].data = _mm512_fmadd_ps(grad_4[1].data, step_size_4.data, param_4[1].data);
            param_4[2].data = _mm512_fmadd_ps(grad_4[2].data, step_size_4.data, param_4[2].data);
            param_4[3].data = _mm512_fmadd_ps(grad_4[3].data, step_size_4.data, param_4[3].data);
            param_4[4].data = _mm512_fmadd_ps(grad_4[4].data, step_size_4.data, param_4[4].data);
            param_4[5].data = _mm512_fmadd_ps(grad_4[5].data, step_size_4.data, param_4[5].data);
            param_4[6].data = _mm512_fmadd_ps(grad_4[6].data, step_size_4.data, param_4[6].data);
            param_4[7].data = _mm512_fmadd_ps(grad_4[7].data, step_size_4.data, param_4[7].data);

            _mm512_storeu_ps(_params + i, param_4[0].data);
            _mm512_storeu_ps(_params + i + SIMD_WIDTH, param_4[1].data);
            _mm512_storeu_ps(_params + i + (SIMD_WIDTH << 1), param_4[2].data);
            _mm512_storeu_ps(_params + i + SIMD_WIDTH * 3, param_4[3].data);
            _mm512_storeu_ps(_params + i + (SIMD_WIDTH << 2), param_4[4].data);
            _mm512_storeu_ps(_params + i + SIMD_WIDTH * 5, param_4[5].data);
            _mm512_storeu_ps(_params + i + SIMD_WIDTH * 6, param_4[6].data);
            _mm512_storeu_ps(_params + i + SIMD_WIDTH * 7, param_4[7].data);

            if (dev_params) {
                _mm512_storeu_ps(_doubled_buffer[_buf_index] + (i - t), param_4[0].data);
                _mm512_storeu_ps(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH, param_4[1].data);
                _mm512_storeu_ps(_doubled_buffer[_buf_index] + (i - t) + (SIMD_WIDTH << 1), param_4[2].data);
                _mm512_storeu_ps(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH * 3, param_4[3].data);
                _mm512_storeu_ps(_doubled_buffer[_buf_index] + (i - t) + (SIMD_WIDTH << 2), param_4[4].data);
                _mm512_storeu_ps(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH * 5, param_4[5].data);
                _mm512_storeu_ps(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH * 6, param_4[6].data);
                _mm512_storeu_ps(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH * 7, param_4[7].data);
            }

            _mm512_storeu_ps(_exp_avg + i, momntum_4[0].data);
            _mm512_storeu_ps(_exp_avg + i + SIMD_WIDTH, momntum_4[1].data);
            _mm512_storeu_ps(_exp_avg + i + (SIMD_WIDTH << 1), momntum_4[2].data);
            _mm512_storeu_ps(_exp_avg + i + SIMD_WIDTH * 3, momntum_4[3].data);
            _mm512_storeu_ps(_exp_avg + i + (SIMD_WIDTH << 2), momntum_4[4].data);
            _mm512_storeu_ps(_exp_avg + i + SIMD_WIDTH * 5, momntum_4[5].data);
            _mm512_storeu_ps(_exp_avg + i + SIMD_WIDTH * 6, momntum_4[6].data);
            _mm512_storeu_ps(_exp_avg + i + SIMD_WIDTH * 7, momntum_4[7].data);

            _mm512_storeu_ps(_exp_avg_sq + i, varianc_4[0].data);
            _mm512_storeu_ps(_exp_avg_sq + i + SIMD_WIDTH, varianc_4[1].data);
            _mm512_storeu_ps(_exp_avg_sq + i + (SIMD_WIDTH << 1), varianc_4[2].data);
            _mm512_storeu_ps(_exp_avg_sq + i + SIMD_WIDTH * 3, varianc_4[3].data);
            _mm512_storeu_ps(_exp_avg_sq + i + (SIMD_WIDTH << 2), varianc_4[4].data);
            _mm512_storeu_ps(_exp_avg_sq + i + SIMD_WIDTH * 5, varianc_4[5].data);
            _mm512_storeu_ps(_exp_avg_sq + i + SIMD_WIDTH * 6, varianc_4[6].data);
            _mm512_storeu_ps(_exp_avg_sq + i + SIMD_WIDTH * 7, varianc_4[7].data);
        }
        if (dev_params) {
            launch_param_update(_doubled_buffer[_buf_index],
                                dev_params + t,
                                copy_size,
                                Context::Instance().GetCurrentStream());
            _buf_index = !_buf_index;
        }
    }
    if(_param_size > rounded_size)
         Step_4((_params + rounded_size),
             (grads + rounded_size),
             (_exp_avg + rounded_size),
             (_exp_avg_sq + rounded_size),
             (_param_size - rounded_size),
             (dev_params != nullptr ? (dev_params + rounded_size) : dev_params));
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
