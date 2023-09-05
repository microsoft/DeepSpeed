/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#pragma once

#include "compatible.hpp"

/////////////////////////////// Memory Access Utils ///////////////////////////////
namespace mem_access {

enum class LoadPolicy {
    CacheAll,       // Cache at all levels
    CacheGlobal,    // Cache at L2 only
    CacheStreaming  // Cache with evict first policy
};

enum class StorePolicy {
    Writeback,      // Cache in L1, write-back on eviction
    CacheGlobal,    // Bypass L1, write-back on eviction
    CacheStreaming  // Allocate cache line with evict first policy
};

template <int AccessSize, LoadPolicy policy = LoadPolicy::CacheAll>
inline void load_global(void* dst, const void* src);

template <int AccessSize, LoadPolicy policy = LoadPolicy::CacheAll>
inline void load_global(void* dst, const void* src, bool do_access);

// Shared accesses have no cache policy
template <int AccessSize>
inline void load_shared(void* dst, const void* src);

template <int AccessSize>
inline void load_shared(void* dst, const void* src, bool do_access);

template <int AccessSize, StorePolicy policy = StorePolicy::Writeback>
inline void store_global(void* dst, const void* src);

// Shared accesses have no cache policy
template <int AccessSize>
inline void store_shared(void* dst, const void* src);


// Util for tracking pipeline buffers
// TODO: Evaluate whether this should also be guarded by ASYNC_COPY_AVAILABLE
template <int max>
class BufferTracker {
public:
    int current_state;

    inline BufferTracker() : current_state(0) {}

    inline int get()
    {
        int return_val = current_state++;
        current_state = (current_state == max ? 0 : current_state);
        return return_val;
    }
};

inline uint32_t lane_id()
{
    auto pos = sycl::ext::oneapi::experimental::this_nd_item<1>();
    return pos.get_local_id(0) & (warpSize - 1);  // Portable
}

/////////// Load Global ///////////
template <>
inline void load_global<16>(void* dst, const void* src)
{
    uint4* data = reinterpret_cast<uint4*>(dst);
    const uint4* src_cast = reinterpret_cast<const uint4*>(src);
    data[0] = src_cast[0];
}

template <>
inline void load_global<16>(void* dst, const void* src, bool do_access)
{
    uint4* data = reinterpret_cast<uint4*>(dst);
    const uint4* src_cast = reinterpret_cast<const uint4*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x() = 0;
        data[0].y() = 0;
        data[0].z() = 0;
        data[0].w() = 0;
    }
}

template <>
inline void load_global<16, LoadPolicy::CacheGlobal>(void* dst, const void* src)
{
    uint4* data = reinterpret_cast<uint4*>(dst);
    const uint4* src_cast = reinterpret_cast<const uint4*>(src);
    data[0] = src_cast[0];
}

template <>
inline void load_global<16, LoadPolicy::CacheGlobal>(void* dst,
                                                                         const void* src,
                                                                         bool do_access)
{
    uint4* data = reinterpret_cast<uint4*>(dst);
    const uint4* src_cast = reinterpret_cast<const uint4*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x() = 0;
        data[0].y() = 0;
        data[0].z() = 0;
        data[0].w() = 0;
    }
}

template <>
inline void load_global<16, LoadPolicy::CacheStreaming>(void* dst,
                                                                            const void* src)
{
    uint4* data = reinterpret_cast<uint4*>(dst);
    const uint4* src_cast = reinterpret_cast<const uint4*>(src);
    data[0] = src_cast[0];
}

template <>
inline void load_global<16, LoadPolicy::CacheStreaming>(void* dst,
                                                                            const void* src,
                                                                            bool do_access)
{
    uint4* data = reinterpret_cast<uint4*>(dst);
    const uint4* src_cast = reinterpret_cast<const uint4*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x() = 0;
        data[0].y() = 0;
        data[0].z() = 0;
        data[0].w() = 0;
    }
}

template <>
inline void load_global<8>(void* dst, const void* src)
{
    uint2* data = reinterpret_cast<uint2*>(dst);
    const uint2* src_cast = reinterpret_cast<const uint2*>(src);
    data[0] = src_cast[0];
}

template <>
inline void load_global<8>(void* dst, const void* src, bool do_access)
{
    uint2* data = reinterpret_cast<uint2*>(dst);
    const uint2* src_cast = reinterpret_cast<const uint2*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x() = 0;
        data[0].y() = 0;
    }
}

template <>
inline void load_global<8, LoadPolicy::CacheGlobal>(void* dst, const void* src)
{
    uint2* data = reinterpret_cast<uint2*>(dst);
    const uint2* src_cast = reinterpret_cast<const uint2*>(src);
    data[0] = src_cast[0];
}

template <>
inline void load_global<8, LoadPolicy::CacheGlobal>(void* dst,
                                                                        const void* src,
                                                                        bool do_access)
{
    uint2* data = reinterpret_cast<uint2*>(dst);
    const uint2* src_cast = reinterpret_cast<const uint2*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x() = 0;
        data[0].y() = 0;
    }
}

template <>
inline void load_global<8, LoadPolicy::CacheStreaming>(void* dst,
                                                                           const void* src)
{
    uint2* data = reinterpret_cast<uint2*>(dst);
    const uint2* src_cast = reinterpret_cast<const uint2*>(src);
    data[0] = src_cast[0];
}

template <>
inline void load_global<8, LoadPolicy::CacheStreaming>(void* dst,
                                                                           const void* src,
                                                                           bool do_access)
{
    uint2* data = reinterpret_cast<uint2*>(dst);
    const uint2* src_cast = reinterpret_cast<const uint2*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x() = 0;
        data[0].y() = 0;
    }
}

template <>
inline void load_global<4>(void* dst, const void* src)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    data[0] = src_cast[0];
}

template <>
inline void load_global<4>(void* dst, const void* src, bool do_access)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
}

template <>
inline void load_global<4, LoadPolicy::CacheGlobal>(void* dst, const void* src)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    data[0] = src_cast[0];
}

template <>
inline void load_global<4, LoadPolicy::CacheGlobal>(void* dst,
                                                                        const void* src,
                                                                        bool do_access)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
}

template <>
inline void load_global<4, LoadPolicy::CacheStreaming>(void* dst,
                                                                           const void* src)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    data[0] = src_cast[0];
}

template <>
inline void load_global<4, LoadPolicy::CacheStreaming>(void* dst,
                                                                           const void* src,
                                                                           bool do_access)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
}

template <>
inline void load_global<2>(void* dst, const void* src)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    data[0] = src_cast[0];
}

template <>
inline void load_global<2>(void* dst, const void* src, bool do_access)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
}

template <>
inline void load_global<2, LoadPolicy::CacheGlobal>(void* dst, const void* src)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    data[0] = src_cast[0];
}

template <>
inline void load_global<2, LoadPolicy::CacheGlobal>(void* dst,
                                                                        const void* src,
                                                                        bool do_access)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
}

template <>
inline void load_global<2, LoadPolicy::CacheStreaming>(void* dst,
                                                                           const void* src)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    data[0] = src_cast[0];
}

template <>
inline void load_global<2, LoadPolicy::CacheStreaming>(void* dst,
                                                                           const void* src,
                                                                           bool do_access)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
}

template <>
inline void load_shared<16>(void* dst, const void* src)
{
    uint4* data = reinterpret_cast<uint4*>(dst);
    const uint4* src_cast = reinterpret_cast<const uint4*>(src);
    data[0] = src_cast[0];
}

template <>
inline void load_shared<16>(void* dst, const void* src, bool do_access)
{
    uint4* data = reinterpret_cast<uint4*>(dst);
    const uint4* src_cast = reinterpret_cast<const uint4*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x() = 0;
        data[0].y() = 0;
        data[0].z() = 0;
        data[0].w() = 0;
    }
}

template <>
inline void load_shared<8>(void* dst, const void* src)
{
    uint2* data = reinterpret_cast<uint2*>(dst);
    const uint2* src_cast = reinterpret_cast<const uint2*>(src);
    data[0] = src_cast[0];
}

template <>
inline void load_shared<8>(void* dst, const void* src, bool do_access)
{
    uint2* data = reinterpret_cast<uint2*>(dst);
    const uint2* src_cast = reinterpret_cast<const uint2*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x() = 0;
        data[0].y() = 0;
    }
}

template <>
inline void load_shared<4>(void* dst, const void* src)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    data[0] = src_cast[0];
}

template <>
inline void load_shared<4>(void* dst, const void* src, bool do_access)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
}

/////////// Store Global ///////////

template <>
inline void store_global<16>(void* dst, const void* src)
{
    const uint4* data = reinterpret_cast<const uint4*>(src);
    uint4* dst_cast = reinterpret_cast<uint4*>(dst);
    dst_cast[0] = data[0];
}

template <>
inline void store_global<16, StorePolicy::CacheGlobal>(void* dst,
                                                                           const void* src)
{
    const uint4* data = reinterpret_cast<const uint4*>(src);
    uint4* dst_cast = reinterpret_cast<uint4*>(dst);
    dst_cast[0] = data[0];
}

template <>
inline void store_global<16, StorePolicy::CacheStreaming>(void* dst,
                                                                              const void* src)
{
    const uint4* data = reinterpret_cast<const uint4*>(src);
    uint4* dst_cast = reinterpret_cast<uint4*>(dst);
    dst_cast[0] = data[0];
}

template <>
inline void store_global<8>(void* dst, const void* src)
{
    const uint2* data = reinterpret_cast<const uint2*>(src);
    uint2* dst_cast = reinterpret_cast<uint2*>(dst);
    dst_cast[0] = data[0];
}

template <>
inline void store_global<8, StorePolicy::CacheGlobal>(void* dst,
                                                                          const void* src)
{
    const uint2* data = reinterpret_cast<const uint2*>(src);
    uint2* dst_cast = reinterpret_cast<uint2*>(dst);
    dst_cast[0] = data[0];
}

template <>
inline void store_global<8, StorePolicy::CacheStreaming>(void* dst,
                                                                             const void* src)
{
    const uint2* data = reinterpret_cast<const uint2*>(src);
    uint2* dst_cast = reinterpret_cast<uint2*>(dst);
    dst_cast[0] = data[0];
}

template <>
inline void store_global<4>(void* dst, const void* src)
{
    const int32_t* data = reinterpret_cast<const int32_t*>(src);
    int32_t* dst_cast = reinterpret_cast<int32_t*>(dst);
    dst_cast[0] = data[0];
}

template <>
inline void store_global<4, StorePolicy::CacheGlobal>(void* dst,
                                                                          const void* src)
{
    const int32_t* data = reinterpret_cast<const int32_t*>(src);
    int32_t* dst_cast = reinterpret_cast<int32_t*>(dst);
    dst_cast[0] = data[0];
}

template <>
inline void store_global<4, StorePolicy::CacheStreaming>(void* dst,
                                                                             const void* src)
{
    const int32_t* data = reinterpret_cast<const int32_t*>(src);
    int32_t* dst_cast = reinterpret_cast<int32_t*>(dst);
    dst_cast[0] = data[0];
}

/////////// Store Shared ///////////

template <>
inline void store_shared<16>(void* dst, const void* src)
{
    const uint4* data = reinterpret_cast<const uint4*>(src);
    uint4* dst_cast = reinterpret_cast<uint4*>(dst);
    dst_cast[0] = data[0];
}

template <>
inline void store_shared<8>(void* dst, const void* src)
{
    const uint2* data = reinterpret_cast<const uint2*>(src);
    uint2* dst_cast = reinterpret_cast<uint2*>(dst);
    dst_cast[0] = data[0];
}

template <>
inline void store_shared<4>(void* dst, const void* src)
{
    const int32_t* data = reinterpret_cast<const int32_t*>(src);
    int32_t* dst_cast = reinterpret_cast<int32_t*>(dst);
    dst_cast[0] = data[0];
}

}  // namespace mem_access
