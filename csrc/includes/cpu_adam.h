#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
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

#define SIMD_WIDTH 16

#define TILE (1024 * 1024 * 1024)

class Adam_Optimizer {
public:
    Adam_Optimizer(float alpha = 1e-3,
                   float betta1 = 0.9,
                   float betta2 = 0.999,
                   float eps = 1e-8,
                   float weight_decay = 0)
        : _alpha(alpha),
          _betta1(betta1),
          _betta2(betta2),
          _eps(eps),
          _weight_decay(weight_decay),
          _betta1_t(1.0),
          _betta2_t(1.0),
          _buf_index(false)
    {
        cudaMallocHost((void**)_doubled_buffer, TILE * sizeof(float));
        cudaMallocHost((void**)(_doubled_buffer + 1), TILE * sizeof(float));
    }
    ~Adam_Optimizer()
    {
        cudaFreeHost(_doubled_buffer[0]);
        cudaFreeHost(_doubled_buffer[1]);
    }
    void Step(float* _params,
              float* grads,
              float* _exp_avg,
              float* _exp_avg_sq,
              size_t param_size,
              __half* dev_param = nullptr);
    void Step_4(float* _params,
                float* grads,
                float* _exp_avg,
                float* _exp_avg_sa,
                size_t param_size,
                __half* dev_param = nullptr);
    void Step_8(float* _params,
                float* grads,
                float* _exp_avg,
                float* _exp_avg_sq,
                size_t _param_size,
                __half* dev_params = nullptr);

private:
    union AVX_512 {
        __m512 data;
        // float data_f[16];
    };

    float _alpha;
    float _betta1;
    float _betta2;
    float _eps;
    float _weight_decay;

    float _betta1_t;
    float _betta2_t;

    float* _doubled_buffer[2];
    bool _buf_index;
};
