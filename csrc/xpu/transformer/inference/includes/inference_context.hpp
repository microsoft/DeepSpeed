#pragma once

#include <ATen/record_function.h>
#include <c10/core/Stream.h>
#include <ipex.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <cassert>
#include <iostream>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include <vector>

#include "compatible.hpp"

#define MEGABYTE (1024 * 1024)
#define GIGABYTE (1024 * 1024 * 1024)

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

class InferenceContext {
public:
    InferenceContext()
    try : _workspace(nullptr), _seed(42), _curr_offset(0), _workSpaceSize(0), _max_seq_len(0) {
        auto type_ = c10::DeviceType::XPU;
        c10::impl::VirtualGuardImpl impl(type_);
        auto device_ = c10::Device(type_);
        c10::Stream stream = impl.getStream(device_);
        _gen = new oneapi::mkl::rng::philox4x32x10(xpu::get_queue_from_stream(stream), 123);
        if ((_onemklQ = xpu::get_queue_from_stream(stream), 0) != 0) {
            auto message = std::string("Fail to create onemkl queue.");
            std::cerr << message << std::endl;
            throw std::runtime_error(message);
        }
    } catch (sycl::exception const& exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
                  << std::endl;
        std::exit(1);
    }

    virtual ~InferenceContext()
    {
        free(_gen);

        auto type_ = c10::DeviceType::XPU;
        c10::impl::VirtualGuardImpl impl(type_);
        auto device_ = c10::Device(type_);
        c10::Stream stream = impl.getStream(device_);
        sycl::free(_workspace, xpu::get_queue_from_stream(stream));
    }

    static InferenceContext& Instance()
    {
        static InferenceContext _ctx;
        return _ctx;
    }

    void GenWorkSpace(const unsigned& num_layers,
                      const unsigned& num_heads,
                      const size_t& batch_size,
                      const size_t& prompt_len,
                      const size_t& hidden_dim,
                      const unsigned& mp_size,
                      const bool& external_cache,
                      const size_t& elem_size,
                      const unsigned& rank,
                      unsigned max_out_tokens,
                      unsigned min_out_tokens)
    {
        size_t total_size;

        // Flash attention requires padded heads and we'll conservatively allocate
        // for that here. Flash attention is only enabled for head size <= 128 right now
        const int head_size = hidden_dim / num_heads;
        const int padded_head_size = head_size <= 32 ? 32 : (head_size <= 64 ? 64 : 128);
        const int effective_head_size = (head_size > 128) ? head_size : padded_head_size;

        size_t activation_size = 16 * (num_heads * effective_head_size) * batch_size;
        // Other sequence length dimension is added when the final workSpaceSize is calculated
        size_t temp_size = batch_size * num_heads * max_out_tokens * 2;
        size_t cache_size =
            num_layers * batch_size * ((num_heads * effective_head_size) / mp_size) * 2;

        _max_seq_len = (size_t)max_out_tokens;
        size_t workSpaceSize = ((external_cache ? (activation_size + temp_size)
                                                : (activation_size + temp_size + cache_size))) *
                               _max_seq_len * elem_size;
        temp_size *= _max_seq_len * elem_size;

        if (_max_seq_len < min_out_tokens) {
            printf(
                "Allocatable workspace available (%ld tokens) is less than minimum requested "
                "workspace (%d tokens)\n",
                _max_seq_len,
                min_out_tokens);
            throw std::runtime_error("Workspace can't be allocated, not enough memory");
        }

        auto current_queue = this->GetCurrentStream();
        if (!_workspace) {
            assert(_workspace == nullptr);
            _workspace = sycl::malloc_device(workSpaceSize, current_queue);
        } else if (_workSpaceSize < workSpaceSize) {
            sycl::free(_workspace, current_queue);
            _workspace = sycl::malloc_device(workSpaceSize, current_queue);
        }
        if (rank == 0 && !_workspace)
            printf(
                "------------------------------------------------------\n"
                "Requested memory: %f (GigaBytes) \n"
                "Setting maximum total tokens (input + output) to %lu \n"
                "------------------------------------------------------\n",
                (float)workSpaceSize / GIGABYTE,
                _max_seq_len);

        if (!_workspace) { throw std::runtime_error("Workspace is null."); }
        _workSpaceSize = workSpaceSize;
    }

    void SetWorkSpace(void* workspace)
    {
        if (!workspace) { throw std::runtime_error("Workspace is null."); }
        _workspace = workspace;
    }

    void* GetWorkSpace() { return _workspace; }

    sycl::queue GetCurrentStream()
    {
        auto type_ = c10::DeviceType::XPU;
        c10::impl::VirtualGuardImpl impl(type_);
        auto device_ = c10::Device(type_);
        c10::Stream stream = impl.getStream(device_);
        return xpu::get_queue_from_stream(stream);
    }

    // This could be problematic
    sycl::queue GetNewStream()
    {
        auto type_ = c10::DeviceType::XPU;
        c10::impl::VirtualGuardImpl impl(type_);
        auto device_ = c10::Device(type_);
        c10::Stream stream = impl.getStream(device_);

        return xpu::get_queue_from_stream(stream);
    }

    sycl::queue GetOneMKLQ() { return _onemklQ; }

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
    sycl::queue _onemklQ;
    void* _workspace;
    uint64_t _seed;
    uint64_t _curr_offset;

    size_t _workSpaceSize;
    size_t _max_seq_len;

    std::vector<std::array<int, 3>> _gemm_algos;
};
