// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepspeed_aio_thread.h"

using namespace std;

deepspeed_aio_thread_t::deepspeed_aio_thread_t(
    const int tid, 
    const int num_intra_ops, 
    std::map<std::shared_ptr<struct io_op_desc_t>,int>& complete_map,
    std::mutex& complete_map_mutex,
    std::queue<std::shared_ptr<struct io_op_desc_t>>& complete_queue,
    struct thread_sync_t& complete_queue_sync,
    deepspeed_aio_config_t& aio_config)
    : _tid(tid),
      _done_count(num_intra_ops),
      _complete_map(complete_map),
      _complete_map_mutex(complete_map_mutex),
      _complete_queue(complete_queue),
      _complete_queue_sync(complete_queue_sync),
      _aio_config(aio_config),
      _aio_ctxt(new aio_context(aio_config._block_size, aio_config._queue_depth)),
      _time_to_exit(false)
{
}

deepspeed_aio_thread_t::~deepspeed_aio_thread_t() {}

void deepspeed_aio_thread_t::run()
{
    while (true) {
        std::shared_ptr<struct io_op_desc_t> next_io_op = nullptr;
        bool io_op_done = false;

        {
            std::unique_lock<std::mutex> lock(_work_sync._mutex);
            _work_sync._cond_var.wait(lock,
                                      [this] { return (!_work_queue.empty() || _time_to_exit); });
            if (!_work_queue.empty()) {
                next_io_op = _work_queue.front();
                _work_queue.pop();
            }
        }

        if (next_io_op) {
            next_io_op->run(_tid, _aio_ctxt, &_aio_config);

            {
                std::lock_guard<std::mutex> lock(_complete_map_mutex);
                int& count = _complete_map[next_io_op];
                ++count;
                if (count == _done_count) { 
                    _complete_map.erase(next_io_op);
                    io_op_done = true;
                }
            }
        }

        if (io_op_done) {
            {
                std::lock_guard<std::mutex> lock(_complete_queue_sync._mutex);
                _complete_queue.push(next_io_op);
            }
            _complete_queue_sync._cond_var.notify_one();
        }

        if (_time_to_exit) { break; }
    }
}
