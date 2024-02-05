/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include <math.h>
#include "hip/hip_runtime.h"
#include "memory_access_utils.h"
#include "activation_type.h"
#include "conversion_utils.h"
#include "ds_kernel_utils.h"

#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#endif


#define WIDTH     1024
#define HEIGHT    1024

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  16
#define THREADS_PER_BLOCK_Z  1

// Default activation function will error out
template <ActivationType ActType>
DS_D_INLINE float act_fn(float val);

template <ActivationType ActType>
float act_fn_cpu(float val);

template <>
DS_D_INLINE float act_fn<ActivationType::IDENTITY>(float val)
{
    return val;
}

template <>
DS_D_INLINE float act_fn<ActivationType::RELU>(float val)
{
    return val > 0.0f ? val : 0.0f;
}

template <>
DS_D_INLINE float act_fn<ActivationType::GELU>(float val)
{
    constexpr float sqrt_param = 0.79788456080286535587989211986876f;
    constexpr float mul_param = 0.044715f;
    return val * 0.5f * (1.0f + tanhf(sqrt_param * (val + mul_param * val * val * val)));
}

template <>
DS_D_INLINE float act_fn<ActivationType::SILU>(float val)
{
    return val / (1.0f + expf(-val));
}

// CPU checker functions
template <>
float act_fn_cpu<ActivationType::IDENTITY>(float val)
{
    return val;
}

template <>
float act_fn_cpu<ActivationType::RELU>(float val)
{
    return val > 0.0f ? val : 0.0f;
}

template <>
float act_fn_cpu<ActivationType::GELU>(float val)
{
    constexpr float sqrt_param = 0.79788456080286535587989211986876f;
    constexpr float mul_param = 0.044715f;
    return val * 0.5f * (1.0f + tanhf(sqrt_param * (val + mul_param * val * val * val)));
}

template <>
float act_fn_cpu<ActivationType::SILU>(float val)
{
    return val / (1.0f + expf(-val));
}

#define ACT_TYPE_SWITCH(ACT_TYPE, ...)                                \
    if (ACT_TYPE == ActivationType::IDENTITY) {                       \
        constexpr ActivationType act_fn_t = ActivationType::IDENTITY; \
        return __VA_ARGS__();                                         \
    } else if (ACT_TYPE == ActivationType::RELU) {                    \
        constexpr ActivationType act_fn_t = ActivationType::RELU;     \
        return __VA_ARGS__();                                         \
    } else if (ACT_TYPE == ActivationType::GELU) {                    \
        constexpr ActivationType act_fn_t = ActivationType::GELU;     \
        return __VA_ARGS__();                                         \
    } else if (ACT_TYPE == ActivationType::SILU) {                    \
        constexpr ActivationType act_fn_t = ActivationType::SILU;     \
        return __VA_ARGS__();                                         \
    } else {                                                          \
        assert(false);                                                \
    }

namespace bias_act {

constexpr int access_size = 16;
constexpr int threads = 512;
constexpr int unroll = 4;

}  // namespace bias_act

template <typename T, ActivationType ActType>
__global__ void bias_activation_kernel(T* activation,
                                       const T* bias,
                                       const int32_t rows,
                                       const int32_t cols)
{
    constexpr int vector_T = bias_act::access_size / sizeof(T);

    const int32_t thread_offset = threadIdx.x * vector_T;
    const int32_t block_offset = blockIdx.x * vector_T * bias_act::unroll * bias_act::threads;
    const int32_t base_offset = block_offset + thread_offset;

    const int32_t thread_stride = bias_act::threads * vector_T;

#pragma unroll
    for (int i = 0; i < bias_act::unroll; i++) {
        const int32_t iter_offset = base_offset + i * thread_stride;

        const int32_t row = iter_offset / cols;

        T buffer[vector_T];
        T bias_buffer[vector_T];

        if (row < rows) {
            const int32_t col = iter_offset % cols;

            mem_access::load_global<bias_act::access_size>(buffer, activation + iter_offset);
            mem_access::load_global<bias_act::access_size>(
                bias_buffer, bias + col, bias != nullptr);

#pragma unroll
            for (int j = 0; j < vector_T; j++) {
                float val = conversion::to<float>(buffer[j]) + conversion::to<float>(bias_buffer[j]);
                buffer[j] = conversion::to<T>(act_fn<ActType>(val));
            }

            mem_access::store_global<bias_act::access_size>(activation + iter_offset, buffer);
        }
    }
}

using namespace std;

int main() {

    float* hostOrigVals;
    float* hostActivations;

    float* deviceA;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;

    cout << "hip Device prop succeeded " << endl ;

    int i;
    int errors;

    const int32_t n_rows = 128;
    const int32_t n_cols = 128;

    constexpr int32_t elems_per_block =
        bias_act::threads * bias_act::unroll * bias_act::access_size / sizeof(float);
    const int32_t total_elems = n_rows * n_cols;

    const int32_t blocks = (total_elems + elems_per_block - 1) / elems_per_block;

    hostOrigVals = (float*)malloc(total_elems * sizeof(float));
    hostActivations = (float*)malloc(total_elems * sizeof(float));

    // initialize the input data
    for (i = 0; i < total_elems; i++) {
        hostOrigVals[i] = (float)i*1.15f;
        hostActivations[i] = (float)i*1.15f;
        //hostOrigVals[i] = (float)pow(-1, i % 2)*(float)i*1.15f;
        //hostActivations[i] = (float)pow(-1, i % 2)*(float)i*1.15f;
        printf("before kernel [%i], %.6f\n", i, hostActivations[i]);
    }

    HIP_ASSERT(hipMalloc((void**)&deviceA, total_elems * sizeof(float)));

    HIP_ASSERT(hipMemcpy(deviceA, hostActivations, total_elems*sizeof(float), hipMemcpyHostToDevice));

    // IDENTITY - passes
    // RELU - passes
    // GELU - passes
    // SILU - passes
	constexpr ActivationType activation_type = ActivationType::GELU;

    // const dim3 grid(blocks);
    // const dim3 block(bias_act::threads);

    // __global__ void bias_activation_kernel(T* activation,
    //                                        const T* bias,
    //                                        const int32_t rows,
    //                                        const int32_t cols)
    //ACT_TYPE_SWITCH(activation_type, [&] {
        bias_activation_kernel<float, activation_type><<<dim3(blocks), dim3(bias_act::threads), 0, 0>>>(deviceA, nullptr, n_rows, n_cols);
    //});
            //<<<dim3(blocks), dim3(bias_act::threads), 0, 0>>>(activation, bias, n_rows, n_cols);

    // hipLaunchKernelGGL(vectoradd_float, 
    //                 dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
    //                 dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
    //                 0, 0,
    //                 deviceA ,deviceB ,deviceC ,WIDTH ,HEIGHT);

    HIP_ASSERT(hipMemcpy(hostActivations, deviceA, total_elems*sizeof(float), hipMemcpyDeviceToHost));

    for (i = 0; i < total_elems; i++) {
        printf("after kernel [%i], %.6f\n", i, hostActivations[i]);
    }

    // verify the results
    errors = 0;
    for (i = 0; i < total_elems; i++) {
        if (hostActivations[i] != act_fn_cpu<activation_type>(hostOrigVals[i])) {
            errors++;
        }
    }
    if (errors!=0) {
        printf("FAILED: %d errors\n",errors);
    } else {
        printf ("PASSED!\n");
    }

    HIP_ASSERT(hipFree(deviceA));

    free(hostOrigVals);
    free(hostActivations);

    //hipResetDefaultAccelerator();

    return errors;
}
