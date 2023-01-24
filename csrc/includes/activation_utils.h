/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#pragma once

#include "conversion_utils.h"
#include "ds_kernel_utils.h"

#ifdef HALF_PRECISION_AVAILABLE
#include "cuda_fp16.h"
#endif

#ifdef BF16_AVAILABLE
#include "cuda_bf16.h"
#endif

namespace activation {

enum Type {
    // For pass through
    Identity = 0,

    // NewGelu: OpenAI style GELU calculation
    GELU = 1,

    // Gelu in original BERT repo
    OldGELU = 2,

    // ReLU
    ReLU = 3,

    // Sigmoid activation
    Sigmoid = 4,

    // Sigmoid linear unit, also known as swish
    SiLU = 5,
};

/*
This function enables easier templating of functions for various
types and activations functions together. In general, the objective
is to ensure that we don't write the same kernel schedule multiple
times when only varying on activation function.

Usage:
template<typename T, activation::Type ActFn>
__global__ void someKernel(args) {
    // do things with inputs to produce a value
    T activated_val = activation::func<ActFn>(unactivated_val);
}
*/

/* Floating point implementations */
template <Type ActFn>
DS_D_INLINE float func(float val);

template <Type ActFn>
DS_D_INLINE float2 func(float2 val)
{
    val.x = func<ActFn>(val.x);
    val.y = func<ActFn>(val.y);
    return val;
}

// Identity
template <>
DS_D_INLINE float func<Type::Identity>(float val)
{
    return val;
}

// GELU implementation
template <>
DS_D_INLINE float func<Type::GELU>(float val)
{
    constexpr float sqrt_param = 0.797884583473205566406f;
    constexpr float mul_param = 0.044715f;
    return val * 0.5f * (1.0f + tanhf(sqrt_param * (val + mul_param * val * val * val)));
}

// Old GELU implementation
template <>
DS_D_INLINE float func<Type::OldGELU>(float val)
{
    // 1 / sqrt(2)
    constexpr float rsqrt_2 = 0.707106769084930419922;
    return val * 0.5f * (1.0f + erff(val * rsqrt_2));
}

// ReLU
template <>
DS_D_INLINE float func<Type::ReLU>(float val)
{
    return max(val, 0.f);
}

// Sigmoid
template <>
DS_D_INLINE float func<Type::Sigmoid>(float val)
{
    // log_2(e), this is a slight approximation since exp2f is faster
    constexpr float log_2_e = 1.44269502162933349609f;
    return 1.0f / (1.0f + exp2f(-val * log_2_e));
}

// Swish/SiLU
template <>
DS_D_INLINE float func<Type::SiLU>(float val)
{
    return val * func<Type::Sigmoid>(val);
}

/* Half precision implementations */
#ifdef HALF_PRECISION_AVAILABLE

template <Type ActFn>
DS_D_INLINE __half func(__half val)
{
    float up_cast = conversion::to<float>(val);
    up_cast = func<ActFn>(up_cast);
    return conversion::to<__half>(up_cast);
}

template <>
DS_D_INLINE __half func<Type::Identity>(__half val)
{
    return val;
}

template <>
DS_D_INLINE __half func<Type::ReLU>(__half val)
{
    const __half_raw zero_raw = {0};
    const __half zero(zero_raw);

    return (val > zero) ? val : zero;
}

template <Type ActFn>
DS_D_INLINE __half2 func(__half2 val)
{
    float2 up_cast = conversion::to<float2>(val);
    up_cast = func<ActFn>(up_cast);
    return conversion::to<__half2>(val);
}

template <>
DS_D_INLINE __half2 func<Type::Identity>(__half2 val)
{
    return val;
}

template <>
DS_D_INLINE __half2 func<Type::ReLU>(__half2 val)
{
    // TODO(cmikeh2): Use __hmax2 where available
    const __half_raw zero_raw = {0};
    const __half zero(zero_raw);

    __half2 ret_val;
    ret_val.x = (val.x > zero) ? val.x : zero;
    ret_val.y = (val.y > zero) ? val.y : zero;
    return ret_val;
}
#endif

/* BF16 precision implementations */
#ifdef BF16_AVAILABLE

template <Type ActFn>
DS_D_INLINE __nv_bfloat16 func(__nv_bfloat16 val)
{
    float up_cast = conversion::to<float>(val);
    up_cast = func<ActFn>(up_cast);
    return conversion::to<__nv_bfloat16>(up_cast);
}

template <>
DS_D_INLINE __nv_bfloat16 func<Type::Identity>(__nv_bfloat16 val)
{
    return val;
}

template <>
DS_D_INLINE __nv_bfloat16 func<Type::ReLU>(__nv_bfloat16 val)
{
    const __nv_bfloat16_raw zero_raw = {0};
    const __nv_bfloat16 zero(zero_raw);
    return __hmax(val, zero);
}

template <Type ActFn>
DS_D_INLINE __nv_bfloat162 func(__nv_bfloat162 val)
{
    float2 up_cast = conversion::to<float2>(val);
    up_cast = func<ActFn>(up_cast);
    return conversion::to<__half2>(val);
}

template <>
DS_D_INLINE __nv_bfloat162 func<Type::Identity>(__nv_bfloat162 val)
{
    return val;
}

template <>
DS_D_INLINE __nv_bfloat162 func<Type::ReLU>(__nv_bfloat162 val)
{
    const __nv_bfloat162_raw zeros_raw = {0, 0};
    const __nv_bfloat162 zeros(zeros_raw);
    return __hmax2(val, zeros);
}
#endif

}  // namespace activation
