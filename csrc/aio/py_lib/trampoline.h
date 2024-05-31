#pragma once

#include <string>
#include <memory>
#include <dlfcn.h>
#include <iostream>
#include "../include/deepspeed_aio_base.h"

class Trampoline {
public:
    explicit Trampoline(const std::string& device_type) : device(nullptr), handle(nullptr) {
        load_device(device_type);
    }

    void load_device(const std::string& device_type) {
        if (device) {
            delete device;
        }

        if (handle) {
            dlclose(handle);
        }

        std::string lib_name = "lib" + device_type + "_device.so";
        handle = dlopen(lib_name.c_str(), RTLD_LAZY);
        if (!handle) {
            std::cerr << "Cannot open library: " << dlerror() << '\n';
            return;
        }

        typedef DeepSpeedAIOBase* (*create_t)();
        create_t create_device = (create_t) dlsym(handle, "create_device");
        const char* dlsym_error = dlerror();
        if (dlsym_error) {
            std::cerr << "Cannot load symbol create_device: " << dlsym_error << '\n';
            dlclose(handle);
            handle = nullptr;
            return;
        }

        device = create_device();
    }

    void aio_read(torch::Tensor& buffer, const char* filename, const bool validate) { if (device) device->aio_read(buffer, filename, validate); else std::cerr << "No device loaded for aio_read\n"; }
    void aio_write(const torch::Tensor& buffer, const char* filename, const bool validate) { if (device) device->aio_write(buffer, filename, validate); else std::cerr << "No device loaded for aio_write\n"; }
    void deepspeed_memcpy(torch::Tensor& dest, const torch::Tensor& src) { if (device) device->deepspeed_memcpy(dest, src); else std::cerr << "No device loaded for deepspeed_memcpy\n"; }

    int get_block_size() const { if (device) return device->get_block_size(); else { std::cerr << "No device loaded for get_block_size\n"; return -1; } }
    int get_queue_depth() const { if (device) return device->get_queue_depth(); else { std::cerr << "No device loaded for get_queue_depth\n"; return -1; } }
    bool get_single_submit() const { if (device) return device->get_single_submit(); else { std::cerr << "No device loaded for get_single_submit\n"; return false; } }
    bool get_overlap_events() const { if (device) return device->get_overlap_events(); else { std::cerr << "No device loaded for get_overlap_events\n"; return false; } }
    int get_thread_count() const { if (device) return device->get_thread_count(); else { std::cerr << "No device loaded for get_thread_count\n"; return -1; } }

    void read(torch::Tensor& buffer, const char* filename, const bool validate) { if (device) device->read(buffer, filename, validate); else std::cerr << "No device loaded for read\n"; }
    void write(const torch::Tensor& buffer, const char* filename, const bool validate) { if (device) device->write(buffer, filename, validate); else std::cerr << "No device loaded for write\n"; }
    void pread(const torch::Tensor& buffer, const char* filename, const bool validate, const bool async) { if (device) device->pread(buffer, filename, validate, async); else std::cerr << "No device loaded for pread\n"; }
    void pwrite(const torch::Tensor& buffer, const char* filename, const bool validate, const bool async) { if (device) device->pwrite(buffer, filename, validate, async); else std::cerr << "No device loaded for pwrite\n"; }

    void sync_pread(torch::Tensor& buffer, const char* filename) { if (device) device->sync_pread(buffer, filename); else std::cerr << "No device loaded for sync_pread\n"; }
    void sync_pwrite(const torch::Tensor& buffer, const char* filename) { if (device) device->sync_pwrite(buffer, filename); else std::cerr << "No device loaded for sync_pwrite\n"; }
    void async_pread(torch::Tensor& buffer, const char* filename) { if (device) device->async_pread(buffer, filename); else std::cerr << "No device loaded for async_pread\n"; }
    void async_pwrite(const torch::Tensor& buffer, const char* filename) { if (device) device->async_pwrite(buffer, filename); else std::cerr << "No device loaded for async_pwrite\n"; }

    void new_cpu_locked_tensor(const size_t num_elem, const torch::Tensor& example_tensor) { if (device) device->new_cpu_locked_tensor(num_elem, example_tensor); else std::cerr << "No device loaded for new_cpu_locked_tensor\n"; }
    void free_cpu_locked_tensor(torch::Tensor& tensor) { if (device) device->free_cpu_locked_tensor(tensor); else std::cerr << "No device loaded for free_cpu_locked_tensor\n"; }

    void wait() { if (device) device->wait(); else std::cerr << "No device loaded for wait\n"; }

    ~Trampoline() {
        if (device) {
            delete device;
        }
        if (handle) {
            dlclose(handle);
        }
    }

private:
    DeepSpeedAIOBase* device;
    void* handle;
};



































// #pragma once

// #include <string>
// #include <memory>
// #include <dlfcn.h>
// #include <iostream>
// #include "../include/deepspeed_aio_base.h"

// class Trampoline {
// public:
//     explicit Trampoline(const std::string& device_type) : device(nullptr), handle(nullptr) {
//         load_device(device_type);
//     }

//     void load_device(const std::string& device_type) {
//         if (device) {
//             delete device;
//         }

//         if (handle) {
//             dlclose(handle);
//         }

//         std::string lib_name = "lib" + device_type + "_device.so";
//         handle = dlopen(lib_name.c_str(), RTLD_LAZY);
//         if (!handle) {
//             std::cerr << "Cannot open library: " << dlerror() << '\n';
//             return;
//         }

//         typedef DeepSpeedAIOBase* (*create_t)();
//         create_t create_device = (create_t) dlsym(handle, "create_device");
//         const char* dlsym_error = dlerror();
//         if (dlsym_error) {
//             std::cerr << "Cannot load symbol create_device: " << dlsym_error << '\n';
//             dlclose(handle);
//             handle = nullptr;
//             return;
//         }

//         device = create_device();
//     }

//     void aio_read() { if (device) device->aio_read(); else std::cerr << "No device loaded for aio_read\n"; }
//     void aio_write() { if (device) device->aio_write(); else std::cerr << "No device loaded for aio_write\n"; }
//     void deepspeed_memcpy() { if (device) device->deepspeed_memcpy(); else std::cerr << "No device loaded for deepspeed_memcpy\n"; }

//     int get_block_size() const { if (device) return device->get_block_size(); else { std::cerr << "No device loaded for get_block_size\n"; return -1; } }
//     int get_queue_depth() const { if (device) return device->get_queue_depth(); else { std::cerr << "No device loaded for get_queue_depth\n"; return -1; } }
//     bool get_single_submit() const { if (device) return device->get_single_submit(); else { std::cerr << "No device loaded for get_single_submit\n"; return false; } }
//     bool get_overlap_events() const { if (device) return device->get_overlap_events(); else { std::cerr << "No device loaded for get_overlap_events\n"; return false; } }
//     int get_thread_count() const { if (device) return device->get_thread_count(); else { std::cerr << "No device loaded for get_thread_count\n"; return -1; } }

//     void read() { if (device) device->read(); else std::cerr << "No device loaded for read\n"; }
//     void write() { if (device) device->write(); else std::cerr << "No device loaded for write\n"; }
//     void pread() { if (device) device->pread(); else std::cerr << "No device loaded for pread\n"; }
//     void pwrite() { if (device) device->pwrite(); else std::cerr << "No device loaded for pwrite\n"; }

//     void sync_pread() { if (device) device->sync_pread(); else std::cerr << "No device loaded for sync_pread\n"; }
//     void sync_pwrite() { if (device) device->sync_pwrite(); else std::cerr << "No device loaded for sync_pwrite\n"; }
//     void async_pread() { if (device) device->async_pread(); else std::cerr << "No device loaded for async_pread\n"; }
//     void async_pwrite() { if (device) device->async_pwrite(); else std::cerr << "No device loaded for async_pwrite\n"; }

//     void new_cpu_locked_tensor() { if (device) device->new_cpu_locked_tensor(); else std::cerr << "No device loaded for new_cpu_locked_tensor\n"; }
//     void free_cpu_locked_tensor() { if (device) device->free_cpu_locked_tensor(); else std::cerr << "No device loaded for free_cpu_locked_tensor\n"; }

//     void wait() { if (device) device->wait(); else std::cerr << "No device loaded for wait\n"; }

//     ~Trampoline() {
//         if (device) {
//             delete device;
//         }
//         if (handle) {
//             dlclose(handle);
//         }
//     }

// private:
//     DeepSpeedAIOBase* device;
//     void* handle;
// };
