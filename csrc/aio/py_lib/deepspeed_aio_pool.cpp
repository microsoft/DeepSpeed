// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepspeed_aio_pool.h"

using namespace std;

deepspeed_aio_pool_t::deepspeed_aio_pool_t(const int pid)
    : _pid(pid),
      _complete_map({})
{
}

deepspeed_aio_pool_t::~deepspeed_aio_pool_t() {}

void deepspeed_aio_pool_t::submit_pool_work(std::shared_ptr<struct io_op_desc_t> scheduled_op) {
    for (auto& ctxt : _thread_contexts) {
        {
            std::lock_guard<std::mutex> lock(ctxt->_work_sync._mutex);
            ctxt->_work_queue.push(scheduled_op);
        }
        ctxt->_work_sync._cond_var.notify_one();
    }
    {
        std::lock_guard<std::mutex> lock(_complete_map_mutex);
        _complete_map[scheduled_op] = 0;
    }
};

void deepspeed_aio_pool_t::stop_threads()
{
    for (auto& ctxt : _thread_contexts) {
        {
            std::lock_guard<std::mutex> lock(ctxt->_work_sync._mutex);
            ctxt->_time_to_exit = true;
        }
        ctxt->_work_sync._cond_var.notify_one();
    }
}
