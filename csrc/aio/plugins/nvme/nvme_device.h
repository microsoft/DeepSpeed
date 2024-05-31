
// nvme_device.h
#pragma once

#include "../include/deepspeed_aio_base.h" // created deepspeed_aio_base.h in py_lib
#include "deepspeed_py_aio.h"
#include <iostream>
#include "deepspeed_py_aio_handle.h"
#include "deepspeed_py_copy.h"
#include "deepspeed_py_aio.h"

class NVMEDevice : public DeepSpeedAIOBase {
public:
    NVMEDevice() {
        aio_handle = std::make_unique<deepspeed_aio_handle_t>(get_block_size(),
                                                               get_queue_depth(),
                                                               get_single_submit(),
                                                               get_overlap_events(),
                                                               get_thread_count());
    }

    ~NVMEDevice() = default;

    void aio_read(torch::Tensor& buffer, const char* filename, const bool validate) override {
        DeepSpeedAIO::deepspeed_py_aio_read(buffer, filename, get_block_size(), get_queue_depth(), get_single_submit(), get_overlap_events(), validate);
    }

    void aio_write(const torch::Tensor& buffer, const char* filename, const bool validate) override {
        DeepSpeedAIO::deepspeed_py_aio_write(buffer, filename, get_block_size(), get_queue_depth(), get_single_submit(), get_overlap_events(), validate);
    }

    void deepspeed_memcpy(torch::Tensor& dest, const torch::Tensor& src) override {
        DeepSpeedCopy::deepspeed_py_memcpy(dest, src);
    }

    int get_block_size() const override {
        return aio_handle->get_block_size();
    }

    int get_queue_depth() const override {
        return aio_handle->get_queue_depth();
    }

    bool get_single_submit() const override {
        return aio_handle->get_single_submit();
    }

    bool get_overlap_events() const override {
        return aio_handle->get_overlap_events();
    }

    int get_thread_count() const override {
        return aio_handle->get_thread_count();
    }

    void read(torch::Tensor& buffer, const char* filename, const bool validate) override {
        aio_handle->read(buffer, filename, validate);
    }

    void write(const torch::Tensor& buffer, const char* filename, const bool validate) override {
        aio_handle->write(buffer, filename, validate);
    }

    void pread(const torch::Tensor& buffer, const char* filename, const bool validate, const bool async) override {
        aio_handle->pread(buffer, filename, validate, async);
    }

    void pwrite(const torch::Tensor& buffer, const char* filename, const bool validate, const bool async) override {
        aio_handle->pwrite(buffer, filename, validate, async);
    }

    void sync_pread(torch::Tensor& buffer, const char* filename) override {
        aio_handle->sync_pread(buffer, filename);
    }

    void sync_pwrite(const torch::Tensor& buffer, const char* filename) override {
        aio_handle->sync_pwrite(buffer, filename);
    }

    void async_pread(torch::Tensor& buffer, const char* filename) override {
        aio_handle->async_pread(buffer, filename);
    }

    void async_pwrite(const torch::Tensor& buffer, const char* filename) override {
        aio_handle->async_pwrite(buffer, filename);
    }

    void new_cpu_locked_tensor(const size_t num_elem, const torch::Tensor& example_tensor) override {
        aio_handle->new_cpu_locked_tensor(num_elem, example_tensor);
    }

    void free_cpu_locked_tensor(torch::Tensor& tensor) override {
        aio_handle->free_cpu_locked_tensor(tensor);
    }

    void wait() override {
        aio_handle->wait();
    }

private:
    std::unique_ptr<deepspeed_aio_handle_t> aio_handle;
};





























// #include "csrc\aio\py_lib\deepspeed_aio_base.h"
// #include <iostream>
// #include "deepspeed_py_aio_handle.h"
// #include "deepspeed_py_copy.h"

// class NVMEDevice : public DeepSpeedAIOBase {
// public:
//     void aio_read() override {
//         std::cout << "NVMe aio_read operation" << std::endl;
//     }

//     void aio_write() override {
//         std::cout << "NVMe aio_write operation" << std::endl;
//     }

//     void deepspeed_memcpy() override {
//         std::cout << "NVMe deepspeed_memcpy operation" << std::endl;
//     }

//     int get_block_size() const override {
//         return 4096;
//     }

//     int get_queue_depth() const override {
//         return 128;
//     }

//     bool get_single_submit() const override {
//         return true;
//     }

//     bool get_overlap_events() const override {
//         return true;
//     }

//     int get_thread_count() const override {
//         return 4;
//     }

//     void read() override {
//         std::cout << "NVMe read operation" << std::endl;
//     }

//     void write() override {
//         std::cout << "NVMe write operation" << std::endl;
//     }

//     void pread() override {
//         std::cout << "NVMe pread operation" << std::endl;
//     }

//     void pwrite() override {
//         std::cout << "NVMe pwrite operation" << std::endl;
//     }

//     void sync_pread() override {
//         std::cout << "NVMe sync_pread operation" << std::endl;
//     }

//     void sync_pwrite() override {
//         std::cout << "NVMe sync_pwrite operation" << std::endl;
//     }

//     void async_pread() override {
//         std::cout << "NVMe async_pread operation" << std::endl;
//     }

//     void async_pwrite() override {
//         std::cout << "NVMe async_pwrite operation" << std::endl;
//     }

//     void new_cpu_locked_tensor() override {
//         std::cout << "NVMe new_cpu_locked_tensor operation" << std::endl;
//     }

//     void free_cpu_locked_tensor() override {
//         std::cout << "NVMe free_cpu_locked_tensor operation" << std::endl;
//     }

//     void wait() override {
//         std::cout << "NVMe wait operation" << std::endl;
//     }
// };


