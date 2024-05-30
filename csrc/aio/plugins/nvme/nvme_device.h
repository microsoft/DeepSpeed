
#include "../include/deepspeed_aio_base.h"
#include <iostream>
#include "deepspeed_py_aio_handle.h"
#include "deepspeed_py_copy.h"

class NVMEDevice : public DeepSpeedAIOBase {
public:
    void aio_read() override {
        std::cout << "NVMe aio_read operation" << std::endl;
    }

    void aio_write() override {
        std::cout << "NVMe aio_write operation" << std::endl;
    }

    void deepspeed_memcpy() override {
        std::cout << "NVMe deepspeed_memcpy operation" << std::endl;
    }

    int get_block_size() const override {
        return 4096;
    }

    int get_queue_depth() const override {
        return 128;
    }

    bool get_single_submit() const override {
        return true;
    }

    bool get_overlap_events() const override {
        return true;
    }

    int get_thread_count() const override {
        return 4;
    }

    void read() override {
        std::cout << "NVMe read operation" << std::endl;
    }

    void write() override {
        std::cout << "NVMe write operation" << std::endl;
    }

    void pread() override {
        std::cout << "NVMe pread operation" << std::endl;
    }

    void pwrite() override {
        std::cout << "NVMe pwrite operation" << std::endl;
    }

    void sync_pread() override {
        std::cout << "NVMe sync_pread operation" << std::endl;
    }

    void sync_pwrite() override {
        std::cout << "NVMe sync_pwrite operation" << std::endl;
    }

    void async_pread() override {
        std::cout << "NVMe async_pread operation" << std::endl;
    }

    void async_pwrite() override {
        std::cout << "NVMe async_pwrite operation" << std::endl;
    }

    void new_cpu_locked_tensor() override {
        std::cout << "NVMe new_cpu_locked_tensor operation" << std::endl;
    }

    void free_cpu_locked_tensor() override {
        std::cout << "NVMe free_cpu_locked_tensor operation" << std::endl;
    }

    void wait() override {
        std::cout << "NVMe wait operation" << std::endl;
    }
};


