// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <ATen/record_function.h>
#include <c10/core/Stream.h>
#include <ipex.h>
#include <torch/extension.h>
#include <torch/library.h>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include <cassert>
#include <ext/oneapi/bfloat16.hpp>
#include <iostream>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include <vector>
using bf16 = sycl::ext::oneapi::bfloat16;

#define WARP_SIZE 32
#define ONEMKL_OP_T oneapi::mkl::transpose::trans
#define ONEMKL_OP_N oneapi::mkl::transpose::nontrans

#define DPCPP_1D_KERNEL_LOOP(i, n) \
    for (size_t(i) = item_ct1.get_global_id(2); (i) < (n); (i) += item_ct1.get_global_range(2))

#define DPCPP_2D_KERNEL_LOOP(i, n, j, m)                                                       \
    for (size_t i = item_ct1.get_global_id(2); (i) < (n); (i) += item_ct1.get_global_range(2)) \
        for (size_t j = item_ct1.get_global_id(1); (j) < (m); (j) += item_ct1.get_global_range(1))

#define DS_CUDA_NUM_THREADS 512
#define DS_MAXIMUM_NUM_BLOCKS 262144

inline int DS_GET_BLOCKS(const int N)
{
    return (std::max)(
        (std::min)((N + DS_CUDA_NUM_THREADS - 1) / DS_CUDA_NUM_THREADS, DS_MAXIMUM_NUM_BLOCKS),
        // Use at least 1 block, since CUDA does not allow empty block
        1);
}

class SyclContext {
public:
    SyclContext()
    try : _workspace(nullptr), _seed(42), _curr_offset(0) {
        auto type_ = c10::DeviceType::XPU;
        c10::impl::VirtualGuardImpl impl(type_);
        auto device_ = c10::Device(type_);
        c10::Stream dpcpp_stream = impl.getStream(device_);
        _gen = new oneapi::mkl::rng::philox4x32x10(xpu::get_queue_from_stream(dpcpp_stream), 123);
        if ((_onemklQ = &xpu::get_queue_from_stream(dpcpp_stream), 0) != 0) {
            auto message = std::string("Fail to create onemkl queue.");
            std::cerr << message << std::endl;
            throw std::runtime_error(message);
        }
    } catch (sycl::exception const& exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
                  << std::endl;
        std::exit(1);
    }

    virtual ~SyclContext()
    {
        _onemklQ = nullptr;
        free(_gen);

        auto type_ = c10::DeviceType::XPU;
        c10::impl::VirtualGuardImpl impl(type_);
        auto device_ = c10::Device(type_);
        c10::Stream dpcpp_stream = impl.getStream(device_);
        sycl::free(_workspace, xpu::get_queue_from_stream(dpcpp_stream));
    }

    static SyclContext& Instance()
    {
        static SyclContext _ctx;
        return _ctx;
    }

    void SetWorkSpace(void* workspace)
    {
        if (!workspace) { throw std::runtime_error("Workspace is null."); }
        _workspace = workspace;
    }

    void* GetWorkSpace() { return _workspace; }

    sycl::queue* GetCurrentStream()
    {
        // get current pytorch stream.
        // return &xpu::dpcpp::getCurrentDPCPPStream().dpcpp_queue();

        auto type_ = c10::DeviceType::XPU;
        c10::impl::VirtualGuardImpl impl(type_);
        auto device_ = c10::Device(type_);
        c10::Stream dpcpp_stream = impl.getStream(device_);
        return &xpu::get_queue_from_stream(dpcpp_stream);
    }

    sycl::queue* GetNewStream()
    {
        auto type_ = c10::DeviceType::XPU;
        c10::impl::VirtualGuardImpl impl(type_);
        auto device_ = c10::Device(type_);
        c10::Stream dpcpp_stream = impl.getStream(device_);
        c10::Stream stream = impl.getStreamFromGlobalPool(device_, /*isHighPriority=*/false);

        return &xpu::get_queue_from_stream(dpcpp_stream);
    }

    sycl::queue* GetOneMKLQ() { return _onemklQ; }

    std::pair<uint64_t, uint64_t> IncrementOffset(uint64_t offset_inc)
    {
        uint64_t offset = _curr_offset;
        _curr_offset += offset_inc;
        // set _GPT_DEBUG_ and fix seed to avoid randomness
#ifdef _GPT_DEBUG_
        return std::pair<uint64_t, uint64_t>(_seed, 0);
#else
        return std::pair<uint64_t, uint64_t>(_seed, offset);
#endif
    }

    void SetSeed(uint64_t new_seed) { _seed = new_seed; }

    void TestGemmFP16(bool test_gemm, int batch_size, int seq_len, int head_num, int size_per_head)
    {
        // avoid rerun.
        if (_gemm_algos.size() > 0) return;

        // Use default algo.
        _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
        _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
        _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
        _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
        _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
    }

    const std::vector<std::array<int, 3>>& GetGemmAlgos() const { return _gemm_algos; }

private:
    oneapi::mkl::rng::philox4x32x10* _gen;
    sycl::queue* _onemklQ;
    void* _workspace;
    uint64_t _seed;
    uint64_t _curr_offset;
    std::vector<std::array<int, 3>> _gemm_algos;
};
