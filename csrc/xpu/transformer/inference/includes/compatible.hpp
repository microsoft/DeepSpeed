#pragma once

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
using namespace sycl;
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
using namespace cl::sycl;
#else
#error "Unsupported compiler"
#endif

#define __global__
#define __device__

#define MAX_WARP_NUM 32
#define WARP_SIZE 32

constexpr int hw_warp_size = 32;
constexpr int warpSize = 32;

using bf16 = sycl::ext::oneapi::bfloat16;
using fp16 = sycl::half;

using float4 = sycl::vec<float, 4>;
using float2 = sycl::vec<float, 2>;
using half4 = sycl::vec<fp16, 4>;
using half2 = sycl::vec<fp16, 2>;
using bf164 = sycl::vec<bf16, 4>;
using bf162 = sycl::vec<bf16, 2>;
using uint4 = sycl::vec<uint, 4>;
using uint2 = sycl::vec<uint, 2>;

inline int next_pow2(const int val)
{
    int rounded_val = val - 1;
    rounded_val |= rounded_val >> 1;
    rounded_val |= rounded_val >> 2;
    rounded_val |= rounded_val >> 4;
    rounded_val |= rounded_val >> 8;
    return rounded_val + 1;
}

template <typename T, typename Group, typename... Args>
std::enable_if_t<std::is_trivially_destructible<T>::value && sycl::detail::is_group<Group>::value,
                 sycl::local_ptr<typename std::remove_extent<T>::type>>
    __SYCL_ALWAYS_INLINE __group_local_memory(Group g, Args&&... args)
{
    (void)g;
#ifdef __SYCL_DEVICE_ONLY__
    __attribute__((opencl_local))
    std::uint8_t* AllocatedMem = __sycl_allocateLocalMemory(sizeof(T), alignof(T));

    if constexpr (!std::is_trivial_v<T>) {
        id<3> Id = __spirv::initLocalInvocationId<3, id<3>>();
        if (Id == id<3>(0, 0, 0)) new (AllocatedMem) T(std::forward<Args>(args)...);
        sycl::detail::workGroupBarrier();
    }
    return reinterpret_cast<__attribute__((opencl_local)) typename std::remove_extent<T>::type*>(
        AllocatedMem);
#else
    // Silence unused variable warning
    [&args...] {}();
    throw sycl::exception(sycl::errc::feature_not_supported,
        "sycl_ext_oneapi_local_memory extension is not supported on host device");
#endif
}
