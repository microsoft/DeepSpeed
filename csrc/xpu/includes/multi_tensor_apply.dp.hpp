/* Copyright 2020 The Microsoft DeepSpeed Team
   Copyright NVIDIA/apex
   This file is adapted from fused adam in NVIDIA/apex, commit a109f85
*/
#pragma once

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include "compat.h"
#include "context.hpp"

#include <assert.h>
#include <xpu/Stream.h>

// #include <iostream>

// This header is the one-stop shop for all your multi-tensor apply needs.

constexpr int depth_to_max_tensors[5] = {110, 64, 48, 36, 30};
constexpr int depth_to_max_blocks[5] = {320, 320, 320, 320, 320};

template <int n>
struct TensorListMetadata {
    void* addresses[n][depth_to_max_tensors[n - 1]];
    int sizes[depth_to_max_tensors[n - 1]];
    unsigned char block_to_tensor[depth_to_max_blocks[n - 1]];
    int block_to_chunk[depth_to_max_blocks[n - 1]];  // I fear this needs to be a
                                                     // full int.
    int start_tensor_this_launch;
};

template <typename T>
SYCL_EXTERNAL void AdamFunctor(sycl::nd_item<1> item_ct1,
                               int chunk_size,
                               int* noop_gmem,
                               const int tensor_loc,
                               const int chunk_idx,
                               int n,
                               T* g,
                               T* p,
                               T* m,
                               T* v,
                               const float beta1,
                               const float beta2,
                               const float beta1_correction,
                               const float beta2_correction,
                               const float epsilon,
                               const float lr,
                               const int mode,
                               const float decay);

template <int depth, typename T>
void multi_tensor_apply(int block_size,
                        int chunk_size,
                        const at::Tensor& noop_flag,
                        const std::vector<std::vector<at::Tensor>>& tensor_lists,
                        const float beta1,
                        const float beta2,
                        const float beta_correction1,
                        const float beta_correction2,
                        const float epsilon,
                        const float lr,
                        const int mode,
                        const float decay)
{
    TORCH_CHECK(tensor_lists.size() == depth, "tensor_lists.size() != depth");
    int len0 = tensor_lists[0].size();
    TORCH_CHECK(len0 > 0, "tensor_lists[0].size() is not > 0");
    auto ref_device = tensor_lists[0][0].device();
    TORCH_CHECK(ref_device.type() == at::kXPU, "expected input to be on XPU");
    for (int l = 0; l < tensor_lists.size(); l++)  // No range-based for because I need indices
    {
        TORCH_CHECK(tensor_lists[l].size() == len0, "Size mismatch among tensor lists");
        for (int t = 0; t < tensor_lists[l].size(); t++) {
            bool contiguous_memory = tensor_lists[l][t].is_contiguous();
#ifdef VERSION_GE_1_5
            contiguous_memory = (contiguous_memory ||
                                 tensor_lists[l][t].is_contiguous(at::MemoryFormat::ChannelsLast));
#endif
            TORCH_CHECK(contiguous_memory, "A tensor was not contiguous.");
            TORCH_CHECK(tensor_lists[l][t].device() == ref_device,
                        "A tensor was not on the same device as the first tensor");
            TORCH_CHECK(tensor_lists[l][t].numel() == tensor_lists[0][t].numel(), "Size mismatch");
        }
    }

    int ntensors = tensor_lists[0].size();

    TensorListMetadata<depth> tl;

    sycl::queue* stream = SyclContext::Instance().GetCurrentStream();

    tl.start_tensor_this_launch = 0;
    int loc_block_info = 0;
    int loc_tensor_info = 0;
    for (int t = 0; t < ntensors; t++) {
        tl.sizes[loc_tensor_info] = tensor_lists[0][t].numel();
        for (int d = 0; d < depth; d++)
            tl.addresses[d][loc_tensor_info] = tensor_lists[d][t].data_ptr();
        loc_tensor_info++;

        int chunks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;

        for (int chunk = 0; chunk < chunks_this_tensor; chunk++) {
            // std::cout << chunks_this_tensor << std::endl;
            tl.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
            tl.block_to_chunk[loc_block_info] = chunk;
            loc_block_info++;

            bool tensors_full = (loc_tensor_info == depth_to_max_tensors[depth - 1] &&
                                 chunk == chunks_this_tensor - 1);
            bool blocks_full = (loc_block_info == depth_to_max_blocks[depth - 1]);
            bool last_chunk = (t == ntensors - 1 && chunk == chunks_this_tensor - 1);
            if (tensors_full || blocks_full || last_chunk) {
                int* data_ptr = noop_flag.DATA_PTR<int>();
                sycl::buffer<unsigned char, 1> block_to_tensor_buf(&(tl.block_to_tensor[0]), {320});
                sycl::buffer<int, 1> block_to_chunk_buf(&(tl.block_to_chunk[0]), {320});
                sycl::buffer<void*, 2> addresses_buf(&(tl.addresses[0][0]), {4, 36});
                sycl::buffer<int, 1> sizes_buf(&(tl.sizes[0]), {36});
                sycl::buffer<int, 1> data_buf(data_ptr, noop_flag.numel());
                stream->submit([&](sycl::handler& cgh) {
                    sycl::accessor tl_block_to_tensor(block_to_tensor_buf, cgh, sycl::read_only);
                    sycl::accessor tl_block_to_chunk(block_to_chunk_buf, cgh, sycl::read_only);
                    sycl::accessor tl_addresses(addresses_buf, cgh, sycl::read_only);
                    sycl::accessor tl_sizes(sizes_buf, cgh, sycl::read_only);
                    sycl::accessor data_acc(data_buf, cgh, sycl::read_only);
                    cgh.parallel_for(sycl::nd_range<1>(loc_block_info * block_size, block_size),
                                     [=](sycl::nd_item<1> item_ct1) {
                                         int tensor_loc = tl_block_to_tensor[item_ct1.get_group(0)];
                                         int chunk_idx = tl_block_to_chunk[item_ct1.get_group(0)];
                                         int n = tl_sizes[tensor_loc];
                                         T* g = (T*)tl_addresses[0][tensor_loc];
                                         T* p = (T*)tl_addresses[1][tensor_loc];
                                         T* m = (T*)tl_addresses[2][tensor_loc];
                                         T* v = (T*)tl_addresses[3][tensor_loc];

                                         AdamFunctor<T>(item_ct1,
                                                        chunk_size,
                                                        data_acc.get_pointer(),
                                                        tensor_loc,
                                                        chunk_idx,
                                                        n,
                                                        g,
                                                        p,
                                                        m,
                                                        v,
                                                        beta1,
                                                        beta2,
                                                        beta_correction1,
                                                        beta_correction2,
                                                        epsilon,
                                                        lr,
                                                        mode,
                                                        decay);
                                     });
                });

                // Reset.  The control flow possibilities here make my brain hurt.
                loc_block_info = 0;
                if (chunk == chunks_this_tensor - 1) {
                    loc_tensor_info = 0;
                    tl.start_tensor_this_launch = t + 1;
                } else {
                    tl.sizes[0] = tl.sizes[loc_tensor_info - 1];
                    for (int d = 0; d < depth; d++)
                        tl.addresses[d][0] = tl.addresses[d][loc_tensor_info - 1];
                    loc_tensor_info = 1;
                    tl.start_tensor_this_launch = t;
                }
            }
        }
    }
}
