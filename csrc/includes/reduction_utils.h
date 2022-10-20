/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#pragma once

#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;

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

/*
High level API. The API takes in a set of operations and variables
and performs that reduction operation on that variable. The reductions
of each of the arguments are completely independent of each other (
i.e., the val1-op1 combination has no impact on val2-op2).

TODO(cmikeh2): In theory, we might be able to do this sequentially with
device functions and rely on the assembler correctly behaving. My initial
instinct is this won't work, but if it does it would reduce implementation
cost significantly.

TODO(cmikeh2): We need to support sub-block reductions. The warp intrinsic
currently supports this (more incidentally than anything else). It is not
uncommon in something like softmax or a fused attention kernel to map multiple
reductions to a thread block, but each reduction itself is only scoped
to part of the threads (i.e block size = 512, 128 threads per reduction).
*/
template <ROpType Op, int warp_bound = max_warps>
DS_D_INLINE void block(cg::thread_block& tb, cg::thread_block_tile<hw_warp_size>& warp, float& val);

template <ROpType Op1, ROpType Op2, int warp_bound = max_warps>
DS_D_INLINE void block(cg::thread_block& tb,
                       cg::thread_block_tile<hw_warp_size>& warp,
                       float& val1,
                       float& val2);

template <ROpType Op1, ROpType Op2, ROpType Op3, int warp_bound = max_warps>
DS_D_INLINE void block(cg::thread_block& tb,
                       cg::thread_block_tile<hw_warp_size>& warp,
                       float& val1,
                       float& val2,
                       float& val3);

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int warp_bound = max_warps>
DS_D_INLINE void block(cg::thread_block& tb,
                       cg::thread_block_tile<hw_warp_size>& warp,
                       float& val1,
                       float& val2,
                       float& val3,
                       float& val4);

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
template <ROpType op_type>
DS_D_INLINE float elem_reduce(const float lhs, const float rhs);

/* Element reduce implementations */
template <>
DS_D_INLINE float elem_reduce<ROpType::Add>(const float lhs, const float rhs)
{
    return lhs + rhs;
}

template <>
DS_D_INLINE float elem_reduce<ROpType::Max>(const float lhs, const float rhs)
{
    return fmaxf(lhs, rhs);
}

template <>
DS_D_INLINE float elem_reduce<ROpType::Min>(const float lhs, const float rhs)
{
    return fminf(lhs, rhs);
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
DS_D_INLINE void _warp(cg::thread_block_tile<hw_warp_size>& warp, float* data)
{
#pragma unroll
    for (int i = 1; i < reduce_width; i *= 2) {
        data[0] = elem_reduce<Op>(data[0], warp.shfl_xor(data[0], i));
    }
}

template <ROpType Op1, ROpType Op2, int reduce_width = hw_warp_size>
DS_D_INLINE void _warp(cg::thread_block_tile<hw_warp_size>& warp, float* data)
{
#pragma unroll
    for (int i = 1; i < reduce_width; i *= 2) {
        data[0] = elem_reduce<Op1>(data[0], warp.shfl_xor(data[0], i));
        data[1] = elem_reduce<Op2>(data[1], warp.shfl_xor(data[1], i));
    }
}

template <ROpType Op1, ROpType Op2, ROpType Op3, int reduce_width = hw_warp_size>
DS_D_INLINE void _warp(cg::thread_block_tile<hw_warp_size>& warp, float* data)
{
#pragma unroll
    for (int i = 1; i < reduce_width; i *= 2) {
        data[0] = elem_reduce<Op1>(data[0], warp.shfl_xor(data[0], i));
        data[1] = elem_reduce<Op2>(data[1], warp.shfl_xor(data[1], i));
        data[2] = elem_reduce<Op3>(data[2], warp.shfl_xor(data[2], i));
    }
}

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int reduce_width = hw_warp_size>
DS_D_INLINE void _warp(cg::thread_block_tile<hw_warp_size>& warp, float* data)
{
#pragma unroll
    for (int i = 1; i < reduce_width; i *= 2) {
        data[0] = elem_reduce<Op1>(data[0], warp.shfl_xor(data[0], i));
        data[1] = elem_reduce<Op2>(data[1], warp.shfl_xor(data[1], i));
        data[2] = elem_reduce<Op3>(data[2], warp.shfl_xor(data[2], i));
        data[3] = elem_reduce<Op4>(data[3], warp.shfl_xor(data[3], i));
    }
}

/*
Implementation for primary block reduction
*/
template <int total_warps, ROpType... Ops>
DS_D_INLINE void _block(cg::thread_block& tb,
                        cg::thread_block_tile<hw_warp_size>& warp_arg,
                        float* data)
{
    constexpr int elems = sizeof...(Ops);
    // Separated for now in case this no longer is true
    constexpr int bytes = sizeof(float);
    // Unused when `warp_arg.meta_group_size() == 1` or total_warps == 1
    __shared__ float reduce_buffer[total_warps * elems];

    // Always perform warp-scope reduction
    _warp<Ops...>(warp_arg, data);

    // If max_warps == 1 let's skip the runtime check
    if (warp_arg.meta_group_size() > 1 && total_warps != 1) {
        if (warp_arg.thread_rank() == 0) {
#pragma unroll
            for (int i = 0; i < elems; i++) {
                mem_access::store_shared<bytes>(
                    reduce_buffer + i * total_warps + warp_arg.meta_group_rank(), data + i);
            }
        }

        // Synchronization inside block-uniform conditional is safe
        tb.sync();

        if (warp_arg.meta_group_rank() == 0) {
#pragma unroll
            for (int i = 0; i < elems; i++) {
                mem_access::load_shared<bytes>(
                    data + i,
                    reduce_buffer + i * max_warps + warp_arg.thread_rank(),
                    warp_arg.thread_rank() < warp_arg.meta_group_size());
            }

            _warp<Ops..., max_warps>(warp_arg, data);

            if (warp_arg.thread_rank() == 0) {
#pragma unroll
                for (int i = 0; i < elems; i++) {
                    mem_access::store_shared<bytes>(reduce_buffer + i, data + i);
                }
            }
        }

        // Synchronization inside block-uniform conditional is safe
        tb.sync();

#pragma unroll
        for (int i = 0; i < elems; i++) {
            mem_access::load_shared<bytes>(data + i, reduce_buffer + i);
        }
    }
}

/*
Main API implementations. For the most part, they just convert the individual
variables into arrays, which makes working with them easier with a single
implementation. In theory, we could use the `_block` implementation as another
option, but the nature of using a pointer is a little less safe.
*/
template <ROpType Op, int warp_bound>
DS_D_INLINE void block(cg::thread_block& tb, cg::thread_block_tile<hw_warp_size>& warp, float& val)
{
    _block<warp_bound, Op>(tb, warp, val);
}

template <ROpType Op1, ROpType Op2, int warp_bound>
DS_D_INLINE void block(cg::thread_block& tb,
                       cg::thread_block_tile<hw_warp_size>& warp,
                       float& val1,
                       float& val2)
{
    float data[2] = {val1, val2};
    _block<warp_bound, Op1, Op2>(tb, warp, data);
    val1 = data[0];
    val2 = data[1];
}

template <ROpType Op1, ROpType Op2, ROpType Op3, int warp_bound>
DS_D_INLINE void block(cg::thread_block& tb,
                       cg::thread_block_tile<hw_warp_size>& warp,
                       float& val1,
                       float& val2,
                       float& val3)
{
    float data[3] = {val1, val2, val3};
    _block<warp_bound, Op1, Op2, Op3>(tb, warp, data);
    val1 = data[0];
    val2 = data[1];
    val3 = data[2];
}

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int warp_bound>
DS_D_INLINE void block(cg::thread_block& tb,
                       cg::thread_block_tile<hw_warp_size>& warp,
                       float& val1,
                       float& val2,
                       float& val3,
                       float& val4)
{
    float data[4] = {val1, val2, val3, val4};
    _block<warp_bound, Op1, Op2, Op3, Op4>(tb, warp, data);
    val1 = data[0];
    val2 = data[1];
    val3 = data[2];
    val4 = data[3];
}

}  // namespace reduce
