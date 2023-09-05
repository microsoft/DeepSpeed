/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#pragma once

#include "conversion_utils.hpp"
#include "compatible.hpp"
#include "memory_access_utils.hpp"

namespace reduce {

enum class ROpType {
    // Addition
    Add,

    // Maximum reduction
    Max,

    // Minimum reduction
    Min,
};

constexpr int max_threads = 1024;
constexpr int max_warps = max_threads / hw_warp_size;


template <ROpType Op, int warp_bound = max_warps>
void block(sycl::group<2>& tb, 
           sycl::sub_group& warp, 
           float& val);

template <ROpType Op1, ROpType Op2, int warp_bound = max_warps>
void block(sycl::group<2>& tb,
           sycl::sub_group& warp,
           float& val1,
           float& val2);

template <ROpType Op1, ROpType Op2, ROpType Op3, int warp_bound = max_warps>
void block(sycl::group<2>& tb,
                       sycl::sub_group& warp,
                       float& val1,
                       float& val2,
                       float& val3);

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int warp_bound = max_warps>
void block(sycl::group<2>& tb,
           sycl::sub_group& warp,
           float& val1,
           float& val2,
           float& val3,
           float& val4);

/*
The partitioned block is a special case of the above where in the warps of a threadblock are
partitioned into separate independent reductions. For example, I might have an 8 warp thread block
in which each pair of warps is processing an independent piece of data. I would then reduce that
data with the something like the following:
``` cpp
float max_val;
reduce::partitioned_block<rop::Max, 2>(tb, warp, max_val);
```
After which, each pair of warps would have coherent data with each other. Note, this API will not
provide correct results if the number of warps per partition is not a power of 2.
*/
template <ROpType Op, int num_threads>
void partitioned_block(sycl::group<2>& tb,
                                   sycl::sub_group& warp,
                                   float& val);

template <ROpType Op1, ROpType Op2, int num_threads>
void partitioned_block(sycl::group<2>& tb,
                                   sycl::sub_group& warp,
                                   float& val1,
                                   float& val2);

template <ROpType Op1, ROpType Op2, ROpType Op3, int num_threads>
void partitioned_block(sycl::group<2>& tb,
                                   sycl::sub_group& warp,
                                   float& val1,
                                   float& val2,
                                   float& val3);

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int num_threads>
void partitioned_block(sycl::group<2>& tb,
                                   sycl::sub_group& warp,
                                   float& val1,
                                   float& val2,
                                   float& val3,
                                   float& val4);

/*
Single element reduction primitives. Used inside serial collection
loops.

Example usage:
using rop = reduce::OpType;
float min = init<rop::Min>();
for (int i = 0; i < 4; i++) {
    min = reduce::element<rop::Min>(min, data[i]);
}
*/

template <ROpType Op, typename T>
T element(const T lhs, const T rhs);

template <ROpType OType, typename T = float>
T init();

/********************** Internal reduction APIs **********************/

/*
Single element "reductions". TODO(cmikeh2): this sort of "op" concept
should be refactored into its own implementation at some point. This interface
may be easily expanded for new types/operations, but the typical reductions
we need are covered with min/max/add on float.

NOTE: there is no mean reduction because that relies on knowledge of how
many values were already reduced into each scalar. Implementing this on top
of reduce should be straightforward (can just wrap the sum reduction) and
would be a good extension of the header.
*/

/* Float element reduce implementations */
template <>
float element<ROpType::Add>(const float lhs, const float rhs)
{
    return lhs + rhs;
}

template <>
float element<ROpType::Max>(const float lhs, const float rhs)
{
    return fmaxf(lhs, rhs);
}

template <>
float element<ROpType::Min>(const float lhs, const float rhs)
{
    return fminf(lhs, rhs);
}

/* sycl::half element reduce implementation */
template <>
sycl::half element<ROpType::Add>(const sycl::half lhs, const sycl::half rhs)
{
    return lhs + rhs;
}

template <>
sycl::half element<ROpType::Max>(const sycl::half lhs, const sycl::half rhs)
{
    return (lhs > rhs) ? lhs : rhs;
}

template <>
sycl::half element<ROpType::Min>(const sycl::half lhs, const sycl::half rhs)
{
    return (lhs < rhs) ? lhs : rhs;
}

/* sycl::half2 element reduce implementation */
template <>
sycl::half2 element<ROpType::Add>(const sycl::half2 lhs, const sycl::half2 rhs)
{
    return lhs + rhs;
}

template <>
sycl::half2 element<ROpType::Max>(const sycl::half2 lhs, const sycl::half2 rhs)
{
    sycl::half2 ret_val;
    ret_val[0] = (lhs[0] > rhs[0]) ? lhs[0] : rhs[0];
    ret_val[1] = (lhs[1] > rhs[1]) ? lhs[1] : rhs[1];
    return ret_val;
}

template <>
sycl::half2 element<ROpType::Min>(const sycl::half2 lhs, const sycl::half2 rhs)
{
    sycl::half2 ret_val;
    ret_val[0] = (lhs[0] < rhs[0]) ? lhs[0] : rhs[0];
    ret_val[1] = (lhs[1] < rhs[1]) ? lhs[1] : rhs[1];
    return ret_val;
}

/*
Reduction initialization primitives
*/
template <>
float init<ROpType::Add>()
{
    return 0.0f;
}

template <>
float init<ROpType::Min>()
{
    // Positive infinity
    return INFINITY;
}

template <>
float init<ROpType::Max>()
{
    // Negative infinity
    return -INFINITY;
}

template <>
sycl::half init<ROpType::Add>()
{
    constexpr sycl::half zero = 0.0;
    return sycl::half(zero);
}

template <>
sycl::half init<ROpType::Min>()
{
    constexpr sycl::half inf = std::numeric_limits<sycl::half>::infinity();
    return sycl::half(inf);
}

template <>
sycl::half init<ROpType::Max>()
{
    constexpr sycl::half neg_inf = -std::numeric_limits<sycl::half>::infinity();
    return sycl::half(neg_inf);
}

template <>
sycl::half2 init<ROpType::Add>()
{
    return {0.0, 0.0};
}

template <>
sycl::half2 init<ROpType::Min>()
{
    return {std::numeric_limits<sycl::half>::infinity(), std::numeric_limits<sycl::half>::infinity()};
}

template <>
sycl::half2 init<ROpType::Max>()
{
    return {-std::numeric_limits<sycl::half>::infinity(), -std::numeric_limits<sycl::half>::infinity()};
}

template <ROpType Op, typename T>
void init(T* data)
{
    data[0] = init<Op, T>();
}

template <ROpType Op1, ROpType Op2, typename T>
void init(T* data)
{
    data[0] = init<Op1, T>();
    data[1] = init<Op2, T>();
}

template <ROpType Op1, ROpType Op2, ROpType Op3, typename T>
void init(T* data)
{
    data[0] = init<Op1, T>();
    data[1] = init<Op2, T>();
    data[2] = init<Op3, T>();
}

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, typename T>
void init(T* data)
{
    data[0] = init<Op1, T>();
    data[1] = init<Op2, T>();
    data[2] = init<Op3, T>();
    data[3] = init<Op4, T>();
}

/*
Warp reduction primitives

`reduction_width` is an unsafe template parameter, that is that
when using `reduction_width` < hw_warp_size the warp is partitioned
into `hw_warp_size` / `reduction_width` groups of partial sums.

If someone can figure out how to use variadic templates in a reasonable way
here (fold is C++17 only and I don't think helps and recursion feels like
huge overkill that harms readability) that would be wonderful.
*/

template <ROpType Op, int reduce_width = hw_warp_size>
void _warp(sycl::sub_group& warp, float* data)
{
#pragma unroll
    for (int i = 1; i < reduce_width; i *= 2) {
        data[0] = element<Op>(data[0], warp.shuffle_xor(data[0], i));
    }
}

template <ROpType Op1, ROpType Op2, int reduce_width = hw_warp_size>
void _warp(sycl::sub_group& warp, float* data)
{
#pragma unroll
    for (int i = 1; i < reduce_width; i *= 2) {
        data[0] = element<Op1>(data[0], warp.shuffle_xor(data[0], i));
        data[1] = element<Op2>(data[1], warp.shuffle_xor(data[1], i));
    }
}

template <ROpType Op1, ROpType Op2, ROpType Op3, int reduce_width = hw_warp_size>
void _warp(sycl::sub_group& warp, float* data)
{
#pragma unroll
    for (int i = 1; i < reduce_width; i *= 2) {
        data[0] = element<Op1>(data[0], warp.shuffle_xor(data[0], i));
        data[1] = element<Op2>(data[1], warp.shuffle_xor(data[1], i));
        data[2] = element<Op3>(data[2], warp.shuffle_xor(data[2], i));
    }
}

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int reduce_width = hw_warp_size>
void _warp(sycl::sub_group& warp, float* data)
{
#pragma unroll
    for (int i = 1; i < reduce_width; i *= 2) {
        data[0] = element<Op1>(data[0], warp.shuffle_xor(data[0], i));
        data[1] = element<Op2>(data[1], warp.shuffle_xor(data[1], i));
        data[2] = element<Op3>(data[2], warp.shuffle_xor(data[2], i));
        data[3] = element<Op4>(data[3], warp.shuffle_xor(data[3], i));
    }
}

/*
Implementation for primary block reduction that serves both `block` and
`partitioned_block`.

`local_warp_rank` refers to the warp's location within the partition, so
for an unpartitioned threadblock this will be equivalent to
`warp_arg.meta_group_rank()`.

Similarly, the warp offset is the `local_warp_rank` of the warp with the
lowest rank in the partition. In the case of an 8 warp block with a
4 warp reduction, this would map to [0, 0, 0, 0, 4, 4, 4, 4].

Partition size is the number of warps per partition (equal to the thread
block in the default case). This enables us to only perform the warp reduction
when able to.
*/
template <int total_warps, ROpType... Ops>
void _block(sycl::group<2>& tb,
                        sycl::sub_group& warp_arg,
                        float* data,
                        int warp_offset)
{
    constexpr int elems = sizeof...(Ops);
    // Separated for now in case this no longer is true
    constexpr int bytes = sizeof(float);
    // Unused when `partition_size == 1` or total_warps == 1
    auto reduce_buffer = __group_local_memory<float[max_warps * elems]>(tb);

    // Always perform warp-scope reduction
    _warp<Ops...>(warp_arg, data);

    // If max_warps == 1 let's skip the runtime check
    if (warp_arg.get_group_range().size() > 1 && total_warps != 1) {
        if (warp_arg.get_local_id() == 0) {
#pragma unroll
            for (int i = 0; i < elems; i++) {
                mem_access::store_shared<bytes>(
                    reduce_buffer + elems * warp_arg.get_group_id() + i, data + i);
            }
        }

        // Synchronization inside block-uniform conditional is safe
        sycl::group_barrier(tb, tb.fence_scope);

        if (warp_arg.get_group_id() == 0) {
            if (warp_arg.get_local_id() < warp_arg.get_group_range()) {
#pragma unroll
                for (int i = 0; i < elems; i++) {
                    mem_access::load_shared<bytes>(
                        data + i, reduce_buffer + elems * warp_arg.get_local_id() + i);
                }
            } else {
                init<Ops...>(data);
            }

            _warp<Ops..., total_warps>(warp_arg, data);

#pragma unroll
            for (int i = 0; i < elems; i++) {
                mem_access::store_shared<bytes>(reduce_buffer + elems * warp_arg.get_local_id() + i,
                                                data + i);
            }
        }

        // Synchronization inside block-uniform conditional is safe
        sycl::group_barrier(tb, tb.fence_scope);

#pragma unroll
        for (int i = 0; i < elems; i++) {
            mem_access::load_shared<bytes>(data + i,
                                           reduce_buffer + warp_arg.get_group_id() * elems + i);
        }
    }
}

/*
Main API implementations. For the most part, they just convert the individual
variables into arrays, which makes working with them easier with a single
implementation. In theory, we could use the `_block` implementation as another
option, but the nature of using a pointer is a little less safe and this allows
us to obfuscate the details of the partitioned implementation.
*/
template <ROpType Op, int warp_bound>
void block(sycl::group<2>& tb, 
           sycl::sub_group& warp, float& val)
{
    _block<warp_bound, Op>(tb, warp, &val, 0);
}

template <ROpType Op1, ROpType Op2, int warp_bound>
void block(sycl::group<2>& tb,
                       sycl::sub_group& warp,
                       float& val1,
                       float& val2)
{
    float data[2] = {val1, val2};
    _block<warp_bound, Op1, Op2>(tb, warp, data, 0);
    val1 = data[0];
    val2 = data[1];
}

template <ROpType Op1, ROpType Op2, ROpType Op3, int warp_bound>
void block(sycl::group<2>& tb,
                       sycl::sub_group& warp,
                       float& val1,
                       float& val2,
                       float& val3)
{
    float data[3] = {val1, val2, val3};
    _block<warp_bound, Op1, Op2, Op3>(tb, warp, data, 0);
    val1 = data[0];
    val2 = data[1];
    val3 = data[2];
}

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int warp_bound>
void block(sycl::group<2>& tb,
           sycl::sub_group& warp,
           float& val1,
           float& val2,
           float& val3,
           float& val4)
{
    float data[4] = {val1, val2, val3, val4};
    _block<warp_bound, Op1, Op2, Op3, Op4>(tb, warp, data, 0);
    val1 = data[0];
    val2 = data[1];
    val3 = data[2];
    val4 = data[3];
}

/*
Note: for the partitioned blocks, the implementation does not support non-power of 2 blocks in order
to shorten block scale reduction length.
*/
template <ROpType Op, int num_threads>
void partitioned_block(sycl::group<2>& tb,
                       sycl::sub_group& warp,
                       float& val)
{
    if (num_threads <= hw_warp_size) {
        _warp<Op, num_threads>(warp, &val);
    } else {
        constexpr int num_warps = num_threads / hw_warp_size;
        const int warp_offset = warp.get_group_id() & ~(num_warps - 1);
        _block<num_warps, Op>(tb, warp, &val, warp_offset);
    }
}

template <ROpType Op1, ROpType Op2, int num_threads>
void partitioned_block(sycl::group<2>& tb,
                                   sycl::sub_group& warp,
                                   float& val1,
                                   float& val2)
{
    float data[2] = {val1, val2};

    if (num_threads <= hw_warp_size) {
        _warp<Op1, Op2, num_threads>(warp, data);
    } else {
        constexpr int num_warps = num_threads / hw_warp_size;
        const int warp_offset = warp.get_group_id() & ~(num_warps - 1);
        _block<num_warps, Op1, Op2>(tb, warp, data, warp_offset);
    }

    val1 = data[0];
    val2 = data[1];
}

template <ROpType Op1, ROpType Op2, ROpType Op3, int num_threads>
void partitioned_block(sycl::group<2>& tb,
                                   sycl::sub_group& warp,
                                   float& val1,
                                   float& val2,
                                   float& val3)
{
    float data[3] = {val1, val2, val3};

    if (num_threads <= hw_warp_size) {
        _warp<Op1, Op2, Op3, num_threads>(warp, data);
    } else {
        constexpr int num_warps = num_threads / hw_warp_size;
        const int warp_offset = warp.get_group_id() & ~(num_warps - 1);
        _block<num_warps, Op1, Op2, Op3>(tb, warp, data, warp_offset);
    }

    val1 = data[0];
    val2 = data[1];
    val3 = data[2];
}

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int num_threads>
void partitioned_block(sycl::group<2>& tb,
                       sycl::sub_group& warp,
                       float& val1,
                       float& val2,
                       float& val3,
                       float& val4)
{
    float data[4] = {val1, val2, val3, val4};

    if (num_threads <= hw_warp_size) {
        _warp<Op1, Op2, Op3, Op4, num_threads>(warp, data);
    } else {
        constexpr int num_warps = num_threads / hw_warp_size;
        const int warp_offset = warp.get_group_id() & ~(num_warps - 1);
        _block<num_warps, Op1, Op2, Op3, Op4>(tb, warp, data, warp_offset);
    }

    val1 = data[0];
    val2 = data[1];
    val3 = data[2];
    val4 = data[3];
}

}  // namespace reduce
