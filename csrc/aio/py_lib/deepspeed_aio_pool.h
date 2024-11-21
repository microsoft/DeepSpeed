// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <memory>
#include "deepspeed_aio_thread.h"

struct deepspeed_aio_pool_t {
    const int _pid;
    const int _done_count;

    std::vector<std::shared_ptr<struct deepspeed_aio_thread_t>> _thread_contexts;

    deepspeed_aio_pool_t(const int pid, const int done_count);
    ~deepspeed_aio_pool_t();


    void submit_pool_work(std::shared_ptr<struct io_op_desc_t> scheduled_op);

    std::shared_ptr<struct io_op_desc_t> pool_work_done();

    void stop_threads();
};
