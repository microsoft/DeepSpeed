/*
Copyright 2022 The Microsoft DeepSpeed Team

Centralized header file for preprocessor macros and constants
used throughout the codebase.
*/

#pragma once

#include <cuda.h>

#define DS_HD_INLINE __host__ __device__ __forceinline__
#define DS_D_INLINE __device__ __forceinline__

/*
Inference Data Type will be defined with INFERENCE_DATA_TYPE by the
op builder. If it is not defined, compilation will crash. This is by
design to ensure that the inference data type is always explicitly and
intentionally defined.
*/
#ifndef INFERENCE_DATA_TYPE
static_assert(false, "INFERENCE_DATA_TYPE must be defined");
#endif

typedef INFERENCE_DATA_TYPE inference_data_t;

#ifdef __HIP_PLATFORM_HCC__

// constexpr variant of warpSize for templating
constexpr int hw_warp_size = 64;
#define HALF_PRECISION_AVAILABLE = 1
#include <hip/hip_cooperative_groups.h>

#else  // !__HIP_PLATFORM_HCC__

// constexpr variant of warpSize for templating
constexpr int hw_warp_size = 32;

#if __CUDA_ARCH__ >= 530
#define HALF_PRECISION_AVAILABLE = 1
#define PTX_AVAILABLE
#endif  // __CUDA_ARCH__ >= 530

#if __CUDA_ARCH__ >= 800
#define ASYNC_COPY_AVAILABLE
#define BF16_AVAILABLE
#include <cuda_bf16.h>
#elif

#if INFERENCE_DATA_TYPE == __nv_bfloat16
#define DISABLE_KERNEL_BUILD
#endif

#endif  // __CUDA_ARCH__ >= 800

#include <cooperative_groups.h>
#include <cuda_fp16.h>

#endif  //__HIP_PLATFORM_HCC__

inline int next_pow2(const int val)
{
    int rounded_val = val - 1;
    rounded_val |= rounded_val >> 1;
    rounded_val |= rounded_val >> 2;
    rounded_val |= rounded_val >> 4;
    rounded_val |= rounded_val >> 8;
    return rounded_val + 1;
}

template <typename T>
class Pack;

template <>
class Pack<__half> {
public:
    using type = __half2;
};

#ifdef BF16_AVAILABLE
template <>
class Pack<__nv_bfloat16> {
public:
    using type = __nv_bfloat162;
};
#endif

template <>
class Pack<float> {
public:
    using type = float2;
};

template <typename T>
using Packed = typename Pack<T>::type;
