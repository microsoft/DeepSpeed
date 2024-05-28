// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepspeed_py_aio_handle.h"
#include <fstream>
#include <string>
#include <cstdlib>

using namespace std;

bool deepspeed_aio_handle_t::s_cuFile_init = false;

static void _start_aio_thread(std::shared_ptr<struct deepspeed_aio_thread_t> ctxt) { ctxt->run(); }

static std::shared_ptr<struct io_op_desc_t> _create_io_op_desc(const bool read_op,
                                                               const torch::Tensor& buffer,
                                                               const int fd,
                                                               const char* filename,
                                                               const long long int file_num_bytes,
                                                               const int num_threads,
                                                               const bool validate,
                                                               const bool use_gds)
{
    if (buffer.is_cuda() && use_gds) {
        return std::make_shared<gds_op_desc_t>(
            read_op, buffer, fd, filename, file_num_bytes, num_threads, validate);
    } else {
        return std::make_shared<cpu_op_desc_t>(
            read_op, buffer, fd, filename, file_num_bytes, num_threads, validate);
    }
}

deepspeed_aio_handle_t::deepspeed_aio_handle_t(const int block_size,
                                               const int queue_depth,
                                               const bool single_submit,
                                               const bool overlap_events,
                                               const bool use_gds,
                                               const int num_threads)
    : _aio_ctxt(new aio_context(block_size, queue_depth)),
      _single_submit(single_submit),
      _overlap_events(overlap_events),
      _use_gds(use_gds),
      _num_threads(num_threads),
      _aio_config(block_size, queue_depth, single_submit, overlap_events, false),
      _num_pending_ops(0),
      _pinned_tensor_mgr(new deepspeed_pin_tensor_t())
{

    if (!deepspeed_aio_handle_t::s_cuFile_init && use_gds) {
        std::string depthStr = std::to_string(queue_depth);
        std::string threadsStr = std::to_string(num_threads);
        std::string json1 = R"({"execution": {"max_io_queue_depth": )"+depthStr+", ";
        std::string json2 = R"("max_request_parallelism": )"+threadsStr+", ";
        std::string json3 = R"("max_io_threads": )"+threadsStr+", ";
        std::string json4 = R"("parallel_io": true, "min_io_threshold_size_kb": 8192}})";
        std::ofstream outFile("local_cufile.json");
        if (outFile.is_open()){
            outFile << json1 + json2 + json3 + json4;
            outFile.close();
        } else { std::cerr<<"Can't open local cufile" << std::endl;exit(EXIT_FAILURE);}
        putenv("CUFILE_ENV_PATH_JSON=$PWD/local_cufile.json");
        cuFileDriverOpen();
        cudaCheckError();
        size_t direct_io_size = (size_t)block_size / 1024;
        CUfileError_t status = cuFileDriverSetMaxDirectIOSize(direct_io_size);
        if (status.err != CU_FILE_SUCCESS) {
            std::cerr << "file register error:" << cuFileGetErrorString(status) << std::endl;
            exit(EXIT_FAILURE);
        }
        deepspeed_aio_handle_t::s_cuFile_init = true;
        // GDS threads handled internally
        _thread_contexts.push_back(std::make_shared<deepspeed_aio_thread_t>(0, _aio_config));
        _num_threads = 1;
    } else { // CPU OP
        for (auto i = 0; i < num_threads; ++i) {
            _thread_contexts.push_back(std::make_shared<deepspeed_aio_thread_t>(i, _aio_config));
        }
    }

    for (auto& ctxt : _thread_contexts) {
        _threads.push_back(std::thread(_start_aio_thread, ctxt));
    }
}

deepspeed_aio_handle_t::~deepspeed_aio_handle_t()
{
    _stop_threads();
    for (auto& thr : _threads) { thr.join(); }
    if (_use_gds) {cuFileDriverClose();}
}

const int deepspeed_aio_handle_t::get_block_size() const
{
    return _aio_ctxt ? _aio_ctxt->_block_size : -1;
}

const int deepspeed_aio_handle_t::get_queue_depth() const
{
    return _aio_ctxt ? _aio_ctxt->_queue_depth : -1;
}

const bool deepspeed_aio_handle_t::get_single_submit() const { return _single_submit; }

const bool deepspeed_aio_handle_t::get_overlap_events() const { return _overlap_events; }

const bool deepspeed_aio_handle_t::get_use_gds() const { return _use_gds; }

const int deepspeed_aio_handle_t::get_thread_count() const { return _num_threads; }

int deepspeed_aio_handle_t::read(torch::Tensor& buffer, const char* filename, const bool validate)
{
    const auto start_time = std::chrono::high_resolution_clock::now();

    assert(_aio_ctxt);

    long long num_file_bytes;
    if (-1 == get_file_size(filename, num_file_bytes)) {
        const auto error_code = errno;
        report_file_error(filename, " fstat for read", error_code);
        return -1;
    }
    assert(static_cast<long long int>(buffer.nbytes()) == num_file_bytes);

    const auto fd = open_file(filename, true);
    if (fd == -1) { return -1; }

    auto read_buffer = (char*)buffer.data_ptr();
    std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(fd, 0, num_file_bytes, read_buffer));

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
    std::cout << "Elapsed time(usec): "
              << "aio = " << aio_time.count() * 1e6 << " call = " << fn_time.count() * 1e6
              << std::endl;
    return 0;
}

int deepspeed_aio_handle_t::write(const torch::Tensor& buffer,
                                  const char* filename,
                                  const bool validate)
{
    assert(_aio_ctxt);

    const auto start_time = std::chrono::high_resolution_clock::now();

    const auto fd = open_file(filename, false);
    if (fd == -1) { return -1; }

    auto write_buffer = (char*)buffer.data_ptr();
    const auto num_write_bytes = static_cast<long long int>(buffer.nbytes());
    std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(fd, 0, num_write_bytes, write_buffer));

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
    std::cout << "Elapsed time(usec): "
              << "aio = " << aio_time.count() * 1e6 << " call = " << fn_time.count() * 1e6
              << std::endl;
    return 0;
}

void deepspeed_aio_handle_t::_schedule_aio_work(std::shared_ptr<struct io_op_desc_t> scheduled_op)
{
    for (auto& ctxt : _thread_contexts) {
        {
            std::lock_guard<std::mutex> lock(ctxt->_work_sync._mutex);
            ctxt->_work_queue.push(scheduled_op);
        }
        ctxt->_work_sync._cond_var.notify_one();
    }
    _num_pending_ops++;
}

std::shared_ptr<struct io_op_desc_t> deepspeed_aio_handle_t::_wait_for_aio_work()
{
    std::shared_ptr<struct io_op_desc_t> completed_op = nullptr;
    for (auto& ctxt : _thread_contexts) {
        std::unique_lock<std::mutex> lock(ctxt->_complete_sync._mutex);
        ctxt->_complete_sync._cond_var.wait(lock,
                                            [ctxt] { return !ctxt->_complete_queue.empty(); });
        completed_op = ctxt->_complete_queue.front();
        ctxt->_complete_queue.pop();
    }
    return completed_op;
}

void deepspeed_aio_handle_t::_stop_threads()
{
    assert(0 == _num_pending_ops);
    for (auto& ctxt : _thread_contexts) {
        {
            std::lock_guard<std::mutex> lock(ctxt->_work_sync._mutex);
            ctxt->_time_to_exit = true;
        }
        ctxt->_work_sync._cond_var.notify_one();
    }
}

int deepspeed_aio_handle_t::wait()
{
    assert(_num_pending_ops > 0);
    auto num_completed_ops = 0;

    while (_num_pending_ops > 0) {
        auto completed_op = _wait_for_aio_work();

        if (completed_op->_validate) { completed_op->validate(); }

        completed_op->fini();

        close(completed_op->_fd);

        --_num_pending_ops;
        ++num_completed_ops;
    }

    return num_completed_ops;
}

bool deepspeed_aio_handle_t::_is_valid_parallel_aio_op(const bool read_op,
                                                       const long long int num_bytes)
{
    const auto op_string = read_op ? "Read" : "Write";
    if (num_bytes % get_thread_count()) {
        std::cout << "deepseed_aio failure: parallel " << op_string << " num_bytes = " << num_bytes
                  << " not divisible by thread count = " << get_thread_count() << std::endl;
        return false;
    }

    return true;
}

int deepspeed_aio_handle_t::pread(const torch::Tensor& buffer,
                                  const char* filename,
                                  const bool validate,
                                  const bool async)
{
    long long num_file_bytes;
    if (-1 == get_file_size(filename, num_file_bytes)) {
        const auto error_code = errno;
        report_file_error(filename, " fstat for read", error_code);
        return -1;
    }
    const auto buffer_bytes = static_cast<long long int>(buffer.nbytes());
    if (buffer_bytes != num_file_bytes) {
        std::cout << filename << ": buffer nbytes != file bytes " << buffer_bytes
                  << " != " << num_file_bytes << std::endl;
    }
    assert(static_cast<long long int>(buffer.nbytes()) == num_file_bytes);
    assert((num_file_bytes % _num_threads) == 0);

    if (!_is_valid_parallel_aio_op(true, num_file_bytes)) { return -1; }

    const auto fd = open_file(filename, true);
    if (fd == -1) { return -1; }

    auto scheduled_op = _create_io_op_desc(
        true, buffer, fd, filename, num_file_bytes, _num_threads, validate, _use_gds);

    _schedule_aio_work(scheduled_op);

    if (async) { return 0; }

    return wait();
}

int deepspeed_aio_handle_t::pwrite(const torch::Tensor& buffer,
                                   const char* filename,
                                   const bool validate,
                                   const bool async)
{

    const auto num_write_bytes = static_cast<long long int>(buffer.nbytes());
    assert((num_write_bytes % _num_threads) == 0);

    if (!_is_valid_parallel_aio_op(false, num_write_bytes)) { return -1; }

    const auto fd = open_file(filename, false);
    if (fd == -1) { return -1; }

    auto scheduled_op = _create_io_op_desc(
        false, buffer, fd, filename, num_write_bytes, _num_threads, validate, _use_gds);

    _schedule_aio_work(scheduled_op);

    if (async) { return 0; }

    return wait();
}

int deepspeed_aio_handle_t::sync_pread(torch::Tensor& buffer, const char* filename)
{
    return pread(buffer, filename, false, false);
}

int deepspeed_aio_handle_t::sync_pwrite(const torch::Tensor& buffer, const char* filename)
{
    return pwrite(buffer, filename, false, false);
}

int deepspeed_aio_handle_t::async_pread(torch::Tensor& buffer, const char* filename)
{
    return pread(buffer, filename, false, true);
}

int deepspeed_aio_handle_t::async_pwrite(const torch::Tensor& buffer, const char* filename)
{
    return pwrite(buffer, filename, false, true);
}

at::Tensor deepspeed_aio_handle_t::new_cpu_locked_tensor(const size_t num_elem,
                                                         const torch::Tensor& example_tensor)
{
    return _pinned_tensor_mgr->alloc(num_elem, example_tensor.scalar_type());
}

bool deepspeed_aio_handle_t::free_cpu_locked_tensor(torch::Tensor& locked_tensor)
{
    return _pinned_tensor_mgr->free(locked_tensor);
}

int deepspeed_aio_handle_t::new_device_locked_tensor(const torch::Tensor& buffer)
{
    return register_buffer(buffer);
}

int deepspeed_aio_handle_t::free_device_locked_tensor(const torch::Tensor& buffer)
{
    return deregister_buffer(buffer);
}
