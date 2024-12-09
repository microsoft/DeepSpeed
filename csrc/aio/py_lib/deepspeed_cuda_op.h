// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <memory>
#include "deepspeed_aio_op_desc.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define check_cudaruntimecall(fn)                                                         \
    do {                                                                                  \
        cudaError_t res = fn;                                                             \
        if (res != cudaSuccess) {                                                         \
            const char* str = cudaGetErrorName(res);                                      \
            std::cerr << "cuda runtime api call failed " << #fn << __LINE__ << ":" << str \
                      << std::endl;                                                       \
            std::cerr << "EXITING program!!!" << std::endl;                               \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

struct cuda_op_desc_t : io_op_desc_t {
    cudaMemcpyKind _xfer_type;
    std::shared_ptr<struct io_op_desc_t> _sub_op;
    const int64_t _device;

    cuda_op_desc_t(std::shared_ptr<struct io_op_desc_t> sub_op,
                   const torch::Tensor& gpu_buffer,
                   const int op_id,
                   const int intra_op_parallelism,
                   const bool validate);

    void run(const int tid,
             std::unique_ptr<aio_context>& aio_ctxt,
             deepspeed_aio_config_t* aio_config);

    char* data_ptr() const;

    void validate();

    void finish();
};
