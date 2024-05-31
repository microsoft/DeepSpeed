#pragma once

#include <iostream>

using namespace std;

class DeepSpeedAIOBase {
public:
    virtual void aio_read(torch::Tensor& buffer, const char* filename, const bool validate) = 0;
    virtual void aio_write(const torch::Tensor& buffer, const char* filename, const bool validate) = 0;
    virtual void deepspeed_memcpy(torch::Tensor& dest, const torch::Tensor& src) = 0;
    virtual int get_block_size() const = 0;
    virtual int get_queue_depth() const = 0;
    virtual bool get_single_submit() const = 0;
    virtual bool get_overlap_events() const = 0;
    virtual int get_thread_count() const = 0;
    virtual void read(torch::Tensor& buffer, const char* filename, const bool validate) = 0;
    virtual void write(const torch::Tensor& buffer, const char* filename, const bool validate) = 0;
    virtual void pread(const torch::Tensor& buffer, const char* filename, const bool validate, const bool async) = 0;
    virtual void pwrite(const torch::Tensor& buffer, const char* filename, const bool validate, const bool async) = 0;
    virtual void sync_pread(torch::Tensor& buffer, const char* filename) = 0;
    virtual void sync_pwrite(const torch::Tensor& buffer, const char* filename) = 0;
    virtual void async_pread(torch::Tensor& buffer, const char* filename) = 0;
    virtual void async_pwrite(const torch::Tensor& buffer, const char* filename) = 0;
    virtual void new_cpu_locked_tensor(const size_t num_elem, const torch::Tensor& example_tensor) = 0;
    virtual void free_cpu_locked_tensor(torch::Tensor& tensor) = 0;
    virtual void wait() = 0;
};