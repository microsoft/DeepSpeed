// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "deepspeed_cuda_op.h"

using namespace std;

cuda_op_desc_t::cuda_op_desc_t(
    bool host_copy,
    const torch::Tensor& cpu_buffer,
    const torch::Tensor& gpu_buffer,
    const int op_id,
    const int intra_op_parallelism,
    const bool validate)
    : io_op_desc_t(host_copy,
                   cpu_buffer,
                   op_id,
                   -1, // no fd
                   "",
                   0,
                   intra_op_parallelism,
                   validate,
                   0),
      _xfer_type(host_copy ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost),
      _gpu_buffer(gpu_buffer),
      _device(gpu_buffer.get_device())
{
    // Check equal buffer sizes
    assert(_gpu_buffer.nbytes() == _buffer.nbytes());
}

char* cuda_op_desc_t::data_ptr() const { return (char*)_buffer.data_ptr(); }

void cuda_op_desc_t::validate()
{
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
    check_cudaruntimecall(cudaSetDevice(_device));
    check_cudaruntimecall(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const auto buffer_base_offset = _num_bytes_per_thread * tid;
    char* src;
    char* dst;


    // dst first, src second
    if (_read_op) {
        src = (char*)_buffer.data_ptr() + buffer_base_offset;
        dst = (char*)_gpu_buffer.data_ptr() + buffer_base_offset;;
    } else {
        dst = (char*)_buffer.data_ptr() + buffer_base_offset;
        src = (char*)_gpu_buffer.data_ptr() + buffer_base_offset;;
    }
    check_cudaruntimecall(cudaMemcpyAsync(dst, src, _num_bytes_per_thread, _xfer_type, stream));
    check_cudaruntimecall(cudaStreamSynchronize(stream));
    check_cudaruntimecall(cudaStreamDestroy(stream));
    return;
}
