/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#pragma once

#include <stdint.h>
#include <sycl/half_type.hpp>

#include "compatible.hpp"

namespace conversion {

// Basic primitive for constructing conversions
// sycl cannot call recursive func
template <typename TO, typename FROM>
inline TO to(FROM val)
{
    return static_cast<TO>(val);
}

#ifdef __SYCL_DEVICE_ONLY__
template <>
inline double to(int64_t val)
{
    return __imf_ll2double_rn(val);
}
template <>
inline double to(int32_t val)
{
    return __imf_int2double_rn(val);
}
template <>
inline double to(int16_t val)
{
    return __imf_int2double_rn(val);
}
template <>
inline double to(int8_t val)
{
    return __imf_int2double_rn(val);
}
template <>
inline double to(uint64_t val)
{
    return __imf_ull2double_rn(val);
}
template <>
inline double to(uint32_t val)
{
    return __imf_uint2double_rn(val);
}
template <>
inline double to(uint16_t val)
{
    return __imf_uint2double_rn(val);
}
template <>
inline double to(uint8_t val)
{
    return __imf_uint2double_rn(val);
}
#endif

#ifdef __SYCL_DEVICE_ONLY__
template <>
inline float to(double val)
{
    return __imf_double2float_rn(val);
}
template <>
inline float to(int64_t val)
{
    return __imf_ll2float_rn(val);
}
template <>
inline float to(int32_t val)
{
    return __imf_int2float_rn(val);
}
template <>
inline float to(int16_t val)
{
    return __imf_int2float_rn(val);
}
template <>
inline float to(int8_t val)
{
    return __imf_int2float_rn(val);
}
template <>
inline float to(uint64_t val)
{
    return __imf_ull2float_rn(val);
}
template <>
inline float to(uint32_t val)
{
    return __imf_uint2float_rn(val);
}
template <>
inline float to(uint16_t val)
{
    return __imf_uint2float_rn(val);
}
template <>
inline float to(uint8_t val)
{
    return __imf_uint2float_rn(val);
}
#endif

/*********************  To Float2 Conversions *********************/
template <>
inline float2 to(half2 val)
{
    return val.convert<float>();
}

// TODO: ushort as bf16 replacement for bf16 is not compatible with sycl::vec
template <>
inline float2 to(sycl::ushort2 val)
{
    float2 tmp;
    tmp[0] = (float)val[0];
    tmp[1] = (float)val[1];
    return tmp;
}

#ifdef BF16_AVAILABLE
template <>
inline float2 to(bf162 val)
{
    float2 tmp;
    tmp[0] = (float)val[0];
    tmp[1] = (float)val[1];
    return tmp;
}
#endif

/*********************  To Half Conversions *********************/
#ifdef BF16_AVAILABLE
// No direct conversion
template <>
inline half to(bf16 val)
{
    return to<half>(to<float>(val));
}
#endif

/*********************  To Half2 Conversions *********************/
template <>
inline half2 to(float2 val)
{
    return val.convert<half, rounding_mode::rtn>();
}
template <>
inline half2 to(float val)
{
    half2 tmp;
    tmp[0] = to<half>(val);
    tmp[1] = to<half>(val);
    return tmp;
}

#ifdef BF16_AVAILABLE
// No direct conversion
template <>
inline half2 to(bf162 val)
{
    return to<half2>(to<float2>(val));
}
#endif

/*********************  To BF162 Conversions *********************/
// TODO: use ushort as vec<bf16> replacement
template <>
inline sycl::ushort2 to(float2 val)
{
    sycl::ushort2 tmp;
    tmp[0] = val[0];
    tmp[1] = val[1];
    return tmp;
}

#ifdef BF16_AVAILABLE
template <>
inline bf162 to(float2 val)
{
    bf162 tmp;
    tmp[0] = val[0];
    tmp[1] = val[1];
    return tmp;
}
template <>
inline bf162 to(float val)
{
    bf162 tmp;
    tmp[0] = to<bf16>(val);
    tmp[1] = to<bf16>(val);
    return tmp;
}
template <>
inline bf162 to(half2 val)
{
    auto tmp = to<float>(val);
    return to<bf162>(tmp);
}
#endif

/*********************  To INT64_T Conversions *********************/
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline int64_t to(double val)
{
    return __imf_double2ll_rn(val);
}
template <>
inline int64_t to(float val)
{
    return __imf_float2ll_rn(val);
}
#endif
template <>
inline int64_t to(half val)
{
    return to<int64_t>(to<float>(val));
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
inline int64_t to(bf16 val)
{
    return to<int64_t>(to<float>(val));
}
#endif

/*********************  To INT32_T Conversions *********************/
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline int32_t to(double val)
{
    return __imf_double2int_rn(val);
}
template <>
inline int32_t to(float val)
{
    return __imf_float2int_rn(val);
}
#endif
template <>
inline int32_t to(half val)
{
    return to<int32_t>(to<float>(val));
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
inline int32_t to(bf16 val)
{
    return to<int32_t>(to<float>(val));
}
#endif

/*********************  To INT16_T Conversions *********************/
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline int16_t to(double val)
{
    return __imf_double2int_rn(val);
}
template <>
inline int16_t to(float val)
{
    return __imf_float2int_rn(val);
}
#endif
template <>
inline int16_t to(half val)
{
    return to<int16_t>(to<float>(val));
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
inline int16_t to(bf16 val)
{
    return to<int16_t>(to<float>(val));
}
#endif

/*********************  To INT8_T Conversions *********************/
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline int8_t to(double val)
{
    return __imf_double2int_rn(val);
}
template <>
inline int8_t to(float val)
{
    return __imf_float2int_rn(val);
}
#endif
template <>
inline int8_t to(half val)
{
    return to<int8_t>(to<float>(val));
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
inline int8_t to(bf16 val)
{
    return to<int8_t>(to<float>(val));
}
#endif

/*********************  To UINT64_T Conversions *********************/
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline uint64_t to(double val)
{
    return __imf_double2ull_rn(val);
}
template <>
inline uint64_t to(float val)
{
    return __imf_float2ull_rn(val);
}
#endif
template <>
inline uint64_t to(half val)
{
    return to<uint64_t>(to<float>(val));
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
inline uint64_t to(bf16 val)
{
    return to<uint64_t>(to<float>(val));
}
#endif

/*********************  To UINT32_T Conversions *********************/
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline uint32_t to(double val)
{
    return __imf_double2uint_rn(val);
}
template <>
inline uint32_t to(float val)
{
    return __imf_float2uint_rn(val);
}
#endif
template <>
inline uint32_t to(half val)
{
    return to<uint32_t>(to<float>(val));
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
inline uint32_t to(bf16 val)
{
    return to<uint32_t>(to<float>(val));
}
#endif

/*********************  To UINT16_T Conversions *********************/
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline uint16_t to(double val)
{
    return __imf_double2uint_rn(val);
}
template <>
inline uint16_t to(float val)
{
    return __imf_float2uint_rn(val);
}
#endif
template <>
inline uint16_t to(half val)
{
    return to<uint16_t>(to<float>(val));
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
inline uint16_t to(bf16 val)
{
    return to<uint16_t>(to<float>(val));
}
#endif

/*********************  To UINT8_T Conversions *********************/
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline uint8_t to(double val)
{
    return __imf_double2uint_rn(val);
}
template <>
inline uint8_t to(float val)
{
    return __imf_float2uint_rn(val);
}
#endif
template <>
inline uint8_t to(half val)
{
    return to<uint8_t>(to<float>(val));
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
inline uint8_t to(bf16 val)
{
    return to<uint8_t>(to<float>(val));
}
#endif

}  // namespace conversion
