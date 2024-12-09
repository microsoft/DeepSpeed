// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "deepspeed_cuda_op.h"

using namespace std;

cuda_op_desc_t::cuda_op_desc_t(
    std::shared_ptr<struct io_op_desc_t> sub_op,
    const torch::Tensor& gpu_buffer,
    const int op_id,
    const int intra_op_parallelism,
    const bool validate)
    : io_op_desc_t(sub_op->_read_op,
                   gpu_buffer.contiguous(),
                   op_id,
                   0,
                   "",
                   0,
                   intra_op_parallelism,
                   validate,
                   0),
      _xfer_type(sub_op->_read_op ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost),
      _sub_op(sub_op),
      _device(gpu_buffer.get_device())
{
    assert(_sub_op->_buffer.is_cpu());
    // Check equal buffer sizes
    assert(_sub_op->_buffer.nbytes() == _buffer.nbytes());
}

char* cuda_op_desc_t::data_ptr() const { return (char*)_buffer.data_ptr(); }

void cuda_op_desc_t::validate()
{
    _sub_op->validate();
    return;
}

void cuda_op_desc_t::finish()
{
    _sub_op->finish();
    return;
}

void cuda_op_desc_t::run(const int tid,
                        std::unique_ptr<aio_context>& aio_ctxt,
                        deepspeed_aio_config_t* aio_config)
{
    assert(tid < _intra_op_parallelism);
    cudaStream_t stream;
    check_cudaruntimecall(cudaSetDevice(_device));
    check_cudaruntimecall(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const auto buffer_base_offset = _num_bytes_per_thread * tid;
    char* src;
    char* dst;

    if (_read_op) {
        src = (char*)_sub_op->_buffer.data_ptr() + buffer_base_offset;
        dst = (char*)_buffer.data_ptr() + buffer_base_offset;
        _sub_op->run(tid, aio_ctxt, aio_config);
    } else {
        src = (char*)_buffer.data_ptr() + buffer_base_offset;
        dst = (char*)_sub_op->_buffer.data_ptr() + buffer_base_offset;
    }

    //check_cudaruntimecall(cudaMemcpyAsync((void*)dst, (void*)src, _num_bytes_per_thread, _xfer_type, stream));
    check_cudaruntimecall(cudaMemcpyAsync(dst, src, _num_bytes_per_thread, _xfer_type, stream));
    check_cudaruntimecall(cudaStreamSynchronize(stream));
    check_cudaruntimecall(cudaStreamDestroy(stream));

    if (!_read_op) {
        _sub_op->run(tid, aio_ctxt, aio_config);
    }

    return;
}
