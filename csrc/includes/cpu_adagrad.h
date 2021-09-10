#pragma once

#include <cpuid.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <x86intrin.h>
#include <cassert>
#include "context.h"
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#define TILE (128 * 1024 * 1024)

#if defined(__AVX512__)
#define SIMD_STORE(a, d) _mm512_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm512_loadu_ps(x)
#define SIMD_SET(x) _mm512_set1_ps(x)
#define SIMD_ADD(x, y) _mm512_add_ps(x, y)
#define SIMD_MUL(x, y) _mm512_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm512_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm512_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm512_div_ps(x, y)
#define SIMD_WIDTH 16

#define SIMD_LOAD2(x, h) \
    ((h) ? _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)x)) : _mm512_loadu_ps(x))
#define SIMD_STORE2(x, d, h)                                                                      \
    ((h) ? _mm256_store_ps(x, _mm256_castsi256_ps(_mm512_cvtps_ph(d, _MM_FROUND_TO_NEAREST_INT))) \
         : _mm512_storeu_ps(x, d))

#define INTV __m256i
#else
#if defined(__AVX256__)
#define SIMD_STORE(a, d) _mm256_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm256_loadu_ps(x)
#define SIMD_SET(x) _mm256_set1_ps(x)
#define SIMD_ADD(x, y) _mm256_add_ps(x, y)
#define SIMD_MUL(x, y) _mm256_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm256_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm256_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm256_div_ps(x, y)
#define SIMD_WIDTH 8
#define SIMD_LOAD2(x, h) \
    ((h) ? _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)x)) : _mm256_loadu_ps(x))

#define SIMD_STORE2(x, d, h)                                                                \
    ((h) ? _mm_store_ps(x, _mm_castsi128_ps(_mm256_cvtps_ph(d, _MM_FROUND_TO_NEAREST_INT))) \
         : _mm256_storeu_ps(x, d))

#define INTV __m128i

#endif
#endif

class Adagrad_Optimizer {
public:
    Adagrad_Optimizer(float alpha = 1e-2,
                      float eps = 1e-8,
                      float weight_decay = 0)
        : _alpha(alpha),
          _eps(eps),
          _weight_decay(weight_decay),
          _buf_index(false)
    {
        cudaMallocHost((void**)_doubled_buffer, TILE * sizeof(float));
        cudaMallocHost((void**)(_doubled_buffer + 1), TILE * sizeof(float));

        _streams[0] = Context::Instance().GetCurrentStream();
        _streams[1] = Context::Instance().GetNewStream();
    }
    ~Adagrad_Optimizer()
    {
        cudaFreeHost(_doubled_buffer[0]);
        cudaFreeHost(_doubled_buffer[1]);
    }
    void Step(float* _params,
              float* grads,
              float* _exp_avg_sq,
              size_t param_size,
              __half* dev_param = nullptr,
              bool half_precision = false);
    void Step_4(float* _params,
                float* grads,
                float* _exp_avg_sa,
                size_t param_size,
                __half* dev_param = nullptr,
                bool half_precision = false);
    void Step_8(float* _params,
                float* grads,
                float* _exp_avg_sq,
                size_t _param_size,
                __half* dev_param = nullptr,
                bool half_precision = false);
    inline void SynchronizeStreams()
    {
        for (int i = 0; i < 2; i++) cudaStreamSynchronize(_streams[i]);
    }
    inline void IncrementStep(size_t step)
    {
        _step++;
        if (_step != step) {
            _step = step;
        }
    }
    inline void update_state(float lr, float epsilon, float weight_decay)
    {
        _alpha = lr;
        _eps = epsilon;
        _weight_decay = weight_decay;
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
    float _eps;
    float _weight_decay;

    float _betta1_t;
    float _betta2_t;
    size_t _step;

    float* _doubled_buffer[2];
    bool _buf_index;

    cudaStream_t _streams[2];
};
