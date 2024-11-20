// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepspeed_aio_pool.h"

using namespace std;

deepspeed_aio_pool_t::deepspeed_aio_pool_t(const int pid, const int done_count)
    : _pid(pid),
      _done_count(done_count)
{
}

deepspeed_aio_pool_t::~deepspeed_aio_pool_t() {_intra_op_count.clear();}

void deepspeed_aio_pool_t::submit_pool_work(std::shared_ptr<struct io_op_desc_t> scheduled_op) {
    for (auto& ctxt : _thread_contexts) {
        {
            std::lock_guard<std::mutex> lock(ctxt->_work_sync._mutex);
            ctxt->_work_queue.push(scheduled_op);
        }
        ctxt->_work_sync._cond_var.notify_one();
    }
    _intra_op_count[scheduled_op] = 0;
};

std::shared_ptr<struct io_op_desc_t> deepspeed_aio_pool_t::pool_work_done() {
    std::shared_ptr<struct io_op_desc_t> completed_op = nullptr;
    for (auto& ctxt : _thread_contexts) {
        std::unique_lock<std::mutex> lock(ctxt->_complete_sync._mutex, std::try_to_lock);
        if (lock.owns_lock() ) {
            if (!ctxt->_complete_queue.empty()) {
                completed_op = ctxt->_complete_queue.front();
                ctxt->_complete_queue.pop();
                _intra_op_count[completed_op]++;
                if (_intra_op_count[completed_op] >= _done_count) {
                    lock.unlock();
                    _intra_op_count.erase(completed_op);
                    return completed_op;
                }
                completed_op = nullptr;
            }
            lock.unlock();
        }
    }
    return completed_op;
};

void deepspeed_aio_pool_t::stop_threads()
{
    assert(_intra_op_count.empty());
    for (auto& ctxt : _thread_contexts) {
        {
            std::lock_guard<std::mutex> lock(ctxt->_work_sync._mutex);
            ctxt->_time_to_exit = true;
        }
        ctxt->_work_sync._cond_var.notify_one();
    }
}
