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

    std::vector<std::shared_ptr<struct deepspeed_aio_thread_t>> _thread_contexts;

    std::map<std::shared_ptr<struct io_op_desc_t>,int> _complete_map;
    std::mutex _complete_map_mutex;

    deepspeed_aio_pool_t(const int pid);
    ~deepspeed_aio_pool_t();


    void submit_pool_work(std::shared_ptr<struct io_op_desc_t> scheduled_op);

    void stop_threads();
};
