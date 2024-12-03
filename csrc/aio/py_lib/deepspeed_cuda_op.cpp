// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "deepspeed_cuda_op.h"

using namespace std;

cuda_op_desc_t::cuda_op_desc_t(
    const bool host_copy,
    const torch::Tensor& buffer,
    const torch::Tensor& dst_buffer,
    const int op_id,
    const int intra_op_parallelism,
    const bool validate)
    : io_op_desc_t(host_copy,
                   buffer.contiguous(),
                   op_id,
                   0,
                   "",
                   0,
                   intra_op_parallelism,
                   validate,
                   0),
      _xfer_type(host_copy ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost),
      _dst_buffer(dst_buffer.contiguous()),
      _device(buffer.get_device())
{
    check_cudaruntimecall(cudaSetDevice(_device));
    // Check correct devices for read or write op
    if (host_copy) {
        assert(_buffer.is_cpu());
    } else {
        assert(_dst_buffer.is_cpu());
    }
    // Check equal buffer sizes
    assert(_buffer.nbytes() == _dst_buffer.nbytes());
}

char* cuda_op_desc_t::data_ptr() const { return (char*)_buffer.data_ptr(); }

void cuda_op_desc_t::validate()
{
    // pass
    return;
}

void cuda_op_desc_t::finish()
{
    return;
}

void cuda_op_desc_t::run(const int tid,
                        std::unique_ptr<aio_context>& aio_ctxt,
                        deepspeed_aio_config_t* aio_config)
{
    assert(tid < _intra_op_parallelism);
    cudaStream_t stream;

    const auto buffer_base_offset = _num_bytes_per_thread * tid;
    auto src = (char*)_buffer.data_ptr() + buffer_base_offset;
    auto dst = (char*)_dst_buffer.data_ptr() + buffer_base_offset;
    check_cudaruntimecall(cudaSetDevice(_device));
    check_cudaruntimecall(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    //check_cudaruntimecall(cudaMemcpyAsync((void*)dst, (void*)src, _num_bytes_per_thread, _xfer_type, stream));
    check_cudaruntimecall(cudaMemcpyAsync(dst, src, _num_bytes_per_thread, _xfer_type, stream));
    check_cudaruntimecall(cudaStreamSynchronize(stream));
    check_cudaruntimecall(cudaStreamDestroy(stream));
    return;
}
