// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepspeed_py_io_handle.h"
#include <cstdlib>
#include <chrono>

using namespace std;

static void _start_aio_thread(std::shared_ptr<struct deepspeed_aio_thread_t> ctxt) { ctxt->run(); }

deepspeed_io_handle_t::deepspeed_io_handle_t(const int block_size,
                                             const int queue_depth,
                                             const bool single_submit,
                                             const bool overlap_events,
                                             const int intra_op_parallelism,
                                             const int inter_op_parallelism)
    : _aio_ctxt(new aio_context(block_size, queue_depth)),
      _single_submit(single_submit),
      _overlap_events(overlap_events),
      _intra_op_parallelism(intra_op_parallelism),
      _inter_op_parallelism(inter_op_parallelism),
      _aio_config(block_size, queue_depth, single_submit, overlap_events, false),
      _num_pending_ops(0),
      _op_ids(0),
      _pinned_tensor_mgr(new deepspeed_pin_tensor_t())
{
    for (auto n = 0; n < inter_op_parallelism; ++n) {
        auto pool = std::make_shared<deepspeed_aio_pool_t>(n);
        for (auto i = 0; i < intra_op_parallelism; ++i) {
            auto ctxt = std::make_shared<deepspeed_aio_thread_t>(i, intra_op_parallelism,
                                                                 pool->_complete_map, pool->_complete_map_mutex,
                                                                 _complete_queue, _complete_queue_sync,
                                                                 _aio_config);
            pool->_thread_contexts.push_back(ctxt);
            _threads.push_back(std::thread(_start_aio_thread, ctxt));
        }
        _pool_work_map[pool] = 0;
    }
}

deepspeed_io_handle_t::~deepspeed_io_handle_t()
{
    _stop_threads();
    for (auto& thr : _threads) { thr.join(); }
}

const int deepspeed_io_handle_t::get_block_size() const
{
    return _aio_ctxt ? _aio_ctxt->_block_size : -1;
}

const int deepspeed_io_handle_t::get_queue_depth() const
{
    return _aio_ctxt ? _aio_ctxt->_queue_depth : -1;
}

const bool deepspeed_io_handle_t::get_single_submit() const { return _single_submit; }

const bool deepspeed_io_handle_t::get_overlap_events() const { return _overlap_events; }

const int deepspeed_io_handle_t::get_intra_op_parallelism() const { return _intra_op_parallelism; }

const int deepspeed_io_handle_t::get_inter_op_parallelism() const { return _inter_op_parallelism; }


int deepspeed_io_handle_t::read(torch::Tensor& buffer,
                                const char* filename,
                                const bool validate,
                                const int64_t file_offset)
{
    const auto start_time = std::chrono::high_resolution_clock::now();

    assert(_aio_ctxt);

    int64_t num_file_bytes;
    if (-1 == get_file_size(filename, num_file_bytes)) {
        const auto error_code = errno;
        report_file_error(filename, " fstat for read", error_code);
        return -1;
    }
    assert(static_cast<int64_t>(buffer.nbytes()) == num_file_bytes);

    const auto fd = open_file(filename, true);
    if (fd == -1) { return -1; }

    auto read_buffer = (char*)buffer.data_ptr();
    std::unique_ptr<io_xfer_ctxt> xfer_ctxt(
        new io_xfer_ctxt(fd, file_offset, 0, num_file_bytes, read_buffer));

    if (_aio_config._overlap_events) {
        do_aio_operation_overlap(true, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
    } else {
        do_aio_operation_sequential(true, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
    }

    close(fd);
    const std::chrono::duration<double> aio_time =
        std::chrono::high_resolution_clock::now() - start_time;

    if (validate) { validate_aio_operation(true, filename, read_buffer, num_file_bytes); }
    const std::chrono::duration<double> fn_time =
        std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Elapsed time(usec): " << "aio = " << aio_time.count() * 1e6
              << " call = " << fn_time.count() * 1e6 << std::endl;
    return 0;
}

int deepspeed_io_handle_t::write(const torch::Tensor& buffer,
                                 const char* filename,
                                 const bool validate,
                                 const int64_t file_offset)
{
    assert(_aio_ctxt);

    const auto start_time = std::chrono::high_resolution_clock::now();

    const auto fd = open_file(filename, false);
    if (fd == -1) { return -1; }

    auto write_buffer = (char*)buffer.data_ptr();
    const auto num_write_bytes = static_cast<int64_t>(buffer.nbytes());
    std::unique_ptr<io_xfer_ctxt> xfer_ctxt(
        new io_xfer_ctxt(fd, file_offset, 0, num_write_bytes, write_buffer));

    if (_aio_config._overlap_events) {
        do_aio_operation_overlap(false, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
    } else {
        do_aio_operation_sequential(false, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
    }
    const std::chrono::duration<double> aio_time =
        std::chrono::high_resolution_clock::now() - start_time;

    close(fd);

    if (validate) { validate_aio_operation(false, filename, write_buffer, num_write_bytes); }

    const std::chrono::duration<double> fn_time =
        std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Elapsed time(usec): " << "aio = " << aio_time.count() * 1e6
              << " call = " << fn_time.count() * 1e6 << std::endl;
    return 0;
}

void deepspeed_io_handle_t::_stop_threads()
{
    assert(0 == _num_pending_ops);
    for (const auto& [pool, work] : _pool_work_map) {
        pool->stop_threads();
    }
}

void deepspeed_io_handle_t::_schedule_aio_work(std::shared_ptr<struct io_op_desc_t> scheduled_op)
{
    // find pool w/least number of outstanding jobs
    auto minIt = *std::min_element(_pool_work_map.begin(), _pool_work_map.end(), 
                [](const auto& p1, const auto& p2) {
                        return p1.second < p2.second;
                            });
    auto& ctxt = minIt.first;
    ctxt->submit_pool_work(scheduled_op);
    _pool_work_map[ctxt]++;
    _op_pool_map[scheduled_op] = ctxt;
    _num_pending_ops++;
}

std::vector<std::shared_ptr<struct io_op_desc_t>> deepspeed_io_handle_t::_wait_for_aio_work()
{
    std::vector<std::shared_ptr<struct io_op_desc_t>> completed_ops;
    std::shared_ptr<struct io_op_desc_t> op = nullptr;
    std::unique_lock<std::mutex> lock(_complete_queue_sync._mutex);
    _complete_queue_sync._cond_var.wait(lock,
                                        [this] { return !_complete_queue.empty(); });
    while (!_complete_queue.empty()) {
        op = _complete_queue.front();
        _complete_queue.pop();
        assert(op != nullptr);
        auto& pool = _op_pool_map[op];
        _pool_work_map[pool]--;
        _op_pool_map.erase(op);
        completed_ops.push_back(op);
    }
    return completed_ops;
}

int deepspeed_io_handle_t::wait()
{
    assert(_num_pending_ops > 0);
    auto num_completed_ops = 0;
    while (_num_pending_ops > 0) {
        auto completed_ops = _wait_for_aio_work();
        for (auto& op : completed_ops){
            if (op->_validate) { op->validate(); }
            op->finish();
            close(op->_fd);
            --_num_pending_ops;
            ++num_completed_ops;
        }
    }
    return num_completed_ops;
}

std::vector<int> deepspeed_io_handle_t::get_completion()
{
    assert(_num_pending_ops > 0);
    std::vector<int> completed_ids;
    auto completed_ops = _wait_for_aio_work();
    for (auto& op : completed_ops){
        if (op->_validate) { op->validate(); }
        op->finish();
        close(op->_fd);
        --_num_pending_ops;
        completed_ids.push_back(op->_op_id);
    }
    return completed_ids;
}

bool deepspeed_io_handle_t::_is_valid_parallel_aio_op(const bool read_op, const int64_t num_bytes)
{
    const auto op_string = read_op ? "Read" : "Write";
    if (num_bytes % get_intra_op_parallelism()) {
        std::cout << "deepspeed_aio failure: parallel " << op_string << " num_bytes = " << num_bytes
                  << " not divisible by thread count = " << get_intra_op_parallelism() << std::endl;
        return false;
    }

    return true;
}

std::shared_ptr<struct io_op_desc_t> deepspeed_io_handle_t::_create_io_op_desc(
    const bool read_op,
    const torch::Tensor& buffer,
    const int op_id,
    const int fd,
    const char* filename,
    const int64_t file_num_bytes,
    const bool validate,
    const int64_t file_offset)
{
    return std::make_shared<cpu_op_desc_t>(read_op,
                                           buffer,
                                           _pinned_tensor_mgr,
                                           op_id,
                                           fd,
                                           filename,
                                           file_num_bytes,
                                           _intra_op_parallelism,
                                           validate,
                                           file_offset);
}

int deepspeed_io_handle_t::pread(const torch::Tensor& buffer,
                                 const char* filename,
                                 const bool validate,
                                 const bool async,
                                 const int64_t file_offset)
{
    int64_t num_file_bytes;
    if (-1 == get_file_size(filename, num_file_bytes)) {
        const auto error_code = errno;
        report_file_error(filename, " fstat for read", error_code);
        return -1;
    }

    // buffer can exceed file size to enable 4k alignment
    const auto buffer_bytes = static_cast<int64_t>(buffer.nbytes());
    assert((num_file_bytes % _intra_op_parallelism) == 0);

    if (!_is_valid_parallel_aio_op(true, buffer_bytes)) { return -1; }

    const auto fd = open_file(filename, true);
    if (fd == -1) { return -1; }

    // TODO: op id rollover, use op pointer value?
    _op_ids++;
    auto scheduled_op =
        _create_io_op_desc(true, buffer, _op_ids, fd, filename, num_file_bytes, validate, file_offset);

    _schedule_aio_work(scheduled_op);

    if (async) { return _op_ids; } // Return op number

    return wait();
}

int deepspeed_io_handle_t::pwrite(const torch::Tensor& buffer,
                                  const char* filename,
                                  const bool validate,
                                  const bool async,
                                  const int64_t file_offset)
{
    const auto num_write_bytes = static_cast<int64_t>(buffer.nbytes());
    assert((num_write_bytes % _intra_op_parallelism) == 0);

    if (!_is_valid_parallel_aio_op(false, num_write_bytes)) { return -1; }

    const auto fd = open_file(filename, false);
    if (fd == -1) { return -1; }

    _op_ids++;
    auto scheduled_op =
        _create_io_op_desc(false, buffer, _op_ids, fd, filename, num_write_bytes, validate, file_offset);

    _schedule_aio_work(scheduled_op);

    if (async) { return _op_ids; }

    return wait();
}

int deepspeed_io_handle_t::pcopy(bool read_op,
                                 torch::Tensor& cpu_buffer,
                                 const torch::Tensor& gpu_buffer,
                                 const char* filename,
                                 const bool validate,
                                 const bool async,
                                 const int64_t file_offset)
{
    if (read_op) {
        int64_t num_file_bytes;
        if (-1 == get_file_size(filename, num_file_bytes)) {
            const auto error_code = errno;
            report_file_error(filename, " fstat for read", error_code);
            return -1;
        }
    }
    // buffer can exceed file size to enable 4k alignment
    const auto buffer_bytes = static_cast<int64_t>(cpu_buffer.nbytes());
    assert((buffer_bytes % _intra_op_parallelism) == 0);

    if (!_is_valid_parallel_aio_op(true, buffer_bytes)) { return -1; }

    const auto fd = open_file(filename, true);
    if (fd == -1) { return -1; }

    _op_ids++;

    auto sub_op =
        _create_io_op_desc(read_op, cpu_buffer, _op_ids, fd, filename, buffer_bytes, validate, file_offset);

    auto cuda_op = 
        std::make_shared<cuda_op_desc_t>(sub_op, gpu_buffer, _op_ids, _intra_op_parallelism, validate);

    _schedule_aio_work(cuda_op);

    if (async) { return _op_ids; }

    return wait();
}



int deepspeed_io_handle_t::sync_pread(torch::Tensor& buffer,
                                      const char* filename,
                                      const int64_t file_offset)
{
    return pread(buffer, filename, false, false, file_offset);
}

int deepspeed_io_handle_t::sync_pwrite(const torch::Tensor& buffer,
                                       const char* filename,
                                       const int64_t file_offset)
{
    return pwrite(buffer, filename, false, false, file_offset);
}

int deepspeed_io_handle_t::async_pread(torch::Tensor& buffer,
                                       const char* filename,
                                       const int64_t file_offset)
{
    return pread(buffer, filename, false, true, file_offset);
}

int deepspeed_io_handle_t::async_pwrite(const torch::Tensor& buffer,
                                        const char* filename,
                                        const int64_t file_offset)
{
    return pwrite(buffer, filename, false, true, file_offset);
}

at::Tensor deepspeed_io_handle_t::new_cpu_locked_tensor(const int64_t num_elem,
                                                        const torch::Tensor& example_tensor)
{
    return _pinned_tensor_mgr->alloc(num_elem, example_tensor.scalar_type());
}

bool deepspeed_io_handle_t::free_cpu_locked_tensor(torch::Tensor& locked_tensor)
{
    return _pinned_tensor_mgr->free(locked_tensor);
}
