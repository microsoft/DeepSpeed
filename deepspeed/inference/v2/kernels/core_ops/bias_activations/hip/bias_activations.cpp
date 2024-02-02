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
#include "hip/hip_runtime.h"
#include "" // TODO: Add mem_access utilities here
#include "activation_type.h" // TODO: Add source file 


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

// __global__ void 
// vectoradd_float(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, int width, int height) 
// 
//   {
//  
//       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
//       int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
// 
//       int i = y * width + x;
//       if ( i < (width * height)) {
//         a[i] = b[i] + c[i];
//       }
// 
// 
// 
//   }
// 
// #if 0
// __kernel__ void vectoradd_float(float* a, const float* b, const float* c, int width, int height) {
// 
//   
//   int x = blockDimX * blockIdx.x + threadIdx.x;
//   int y = blockDimY * blockIdy.y + threadIdx.y;
// 
//   int i = y * width + x;
//   if ( i < (width * height)) {
//     a[i] = b[i] + c[i];
//   }
// }
// #endif

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
                float val =
                    conversion::to<float>(buffer[j]) + conversion::to<float>(bias_buffer[j]);
                buffer[j] = conversion::to<T>(act_fn<ActType>(val));
            }

            mem_access::store_global<bias_act::access_size>(activation + iter_offset, buffer);
        }
    }
}

using namespace std;

int main() {
  
  float* hostA;
  float* hostB;
  float* hostC;

  float* deviceA;
  float* deviceB;
  float* deviceC;

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  cout << " System minor " << devProp.minor << endl;
  cout << " System major " << devProp.major << endl;
  cout << " agent prop name " << devProp.name << endl;

  cout << "hip Device prop succeeded " << endl ;

  int i;
  int errors;

  hostA = (float*)malloc(NUM * sizeof(float));
  hostB = (float*)malloc(NUM * sizeof(float));
  hostC = (float*)malloc(NUM * sizeof(float));
  
  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = (float)i;
    hostC[i] = (float)i*100.0f;
  }
  
  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceC, NUM * sizeof(float)));
  
  HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM*sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(deviceC, hostC, NUM*sizeof(float), hipMemcpyHostToDevice));

    const int32_t n_rows = 128;
    const int32_t n_cols = 128;

  constexpr int32_t elems_per_block =
      bias_act::threads * bias_act::unroll * bias_act::access_size / sizeof(T);
  const int32_t total_elems = n_rows * n_cols;
  
  const int32_t blocks = (total_elems + elems_per_block - 1) / elems_per_block;
  
  // const dim3 grid(blocks);
  // const dim3 block(bias_act::threads);
  
  // ACT_TYPE_SWITCH(activation_type, [&] {
  //     bias_activation_kernel<T, act_fn_t>
  //         <<<grid, block, 0, stream>>>(activation, bias, n_rows, n_cols);
  // });

  // __global__ void bias_activation_kernel(T* activation,
  //                                        const T* bias,
  //                                        const int32_t rows,
  //                                        const int32_t cols)
    ACT_TYPE_SWITCH(activation_type, [&] {
        //hipLaunchKernelGGL(bias_activation_kernel<float, act_fn_t>,
        //                dim3(blocks),            // TODO: Update
        //                dim3(bias_act::threads), // TODO: Update
        //                0, 0,
        //                activation, bias, n_rows, n_cols);
        //
        bias_activation_kernel<float, act_fn_t>
            <<<dim3(blocks), dim3(bias_act::threads), 0, 0>>>(activation, bias, n_rows, n_cols);
    });

  // hipLaunchKernelGGL(vectoradd_float, 
  //                 dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
  //                 dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
  //                 0, 0,
  //                 deviceA ,deviceB ,deviceC ,WIDTH ,HEIGHT);

  HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM*sizeof(float), hipMemcpyDeviceToHost));

  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != (hostB[i] + hostC[i])) {
      errors++;
    }
  }
  if (errors!=0) {
    printf("FAILED: %d errors\n",errors);
  } else {
      printf ("PASSED!\n");
  }

  HIP_ASSERT(hipFree(deviceA));
  HIP_ASSERT(hipFree(deviceB));
  HIP_ASSERT(hipFree(deviceC));

  free(hostA);
  free(hostB);
  free(hostC);

  //hipResetDefaultAccelerator();

  return errors;
}

