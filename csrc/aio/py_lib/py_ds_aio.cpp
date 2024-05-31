// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "deepspeed_aio_base.h"

namespace py = pybind11;

class DeepSpeedAIOTrampoline : public DeepSpeedAIOBase {
public:
    DeepSpeedAIOTrampoline() : device(nullptr) {
        load_device("nvme");
    }

    void load_device(const std::string& device_type) {
        if (device_type == "nvme") {
            device = new NVMEDevice();
        } else {
            std::cerr << "Unknown device type: " << device_type << std::endl;
        }
    }

    void aio_read(torch::Tensor& buffer, const char* filename, const bool validate) override {
        if (device) {
            device->aio_read(buffer, filename, validate);
        } else {
            std::cerr << "No device loaded" << std::endl;
        }
    }

    void aio_write(const torch::Tensor& buffer, const char* filename, const bool validate) override {
        if (device) {
            device->aio_write(buffer, filename, validate);
        } else {
            std::cerr << "No device loaded" << std::endl;
        }
    }

    void deepspeed_memcpy(torch::Tensor& dest, const torch::Tensor& src) override {
        if (device) {
            device->deepspeed_memcpy(dest, src);
        } else {
            std::cerr << "No device loaded" << std::endl;
        }
    }

    int get_block_size() const override {
        if (device) {
            return device->get_block_size();
        } else {
            std::cerr << "No device loaded" << std::endl;
            return -1;
        }
    }

    int get_queue_depth() const override {
        if (device) {
            return device->get_queue_depth();
        } else {
            std::cerr << "No device loaded" << std::endl;
            return -1;
        }
    }

    bool get_single_submit() const override {
        if (device) {
            return device->get_single_submit();
        } else {
            std::cerr << "No device loaded" << std::endl;
            return false;
        }
    }

    bool get_overlap_events() const override {
        if (device) {
            return device->get_overlap_events();
        } else {
            std::cerr << "No device loaded" << std::endl;
            return false;
        }
    }

    int get_thread_count() const override {
        if (device) {
            return device->get_thread_count();
        } else {
            std::cerr << "No device loaded" << std::endl;
            return -1;
        }
    }

    void read(torch::Tensor& buffer, const char* filename, const bool validate) override {
        if (device) {
            device->read(buffer, filename, validate);
        } else {
            std::cerr << "No device loaded" << std::endl;
        }
    }

    void write(const torch::Tensor& buffer, const char* filename, const bool validate) override {
        if (device) {
            device->write(buffer, filename, validate);
        } else {
            std::cerr << "No device loaded" << std::endl;
        }
    }

    void pread(const torch::Tensor& buffer, const char* filename, const bool validate, const bool async) override {
        if (device) {
            device->pread(buffer, filename, validate, async);
        } else {
            std::cerr << "No device loaded" << std::endl;
        }
    }

    void pwrite(const torch::Tensor& buffer, const char* filename, const bool validate, const bool async) override {
        if (device) {
            device->pwrite(buffer, filename, validate, async);
        } else {
            std::cerr << "No device loaded" << std::endl;
        }
    }

    void sync_pread(torch::Tensor& buffer, const char* filename) override {
        if (device) {
            device->sync_pread(buffer, filename);
        } else {
            std::cerr << "No device loaded" << std::endl;
        }
    }

    void sync_pwrite(const torch::Tensor& buffer, const char* filename) override {
        if (device) {
            device->sync_pwrite(buffer, filename);
        } else {
            std::cerr << "No device loaded" << std::endl;
        }
    }

    void async_pread(torch::Tensor& buffer, const char* filename) override {
        if (device) {
            device->async_pread(buffer, filename);
        } else {
            std::cerr << "No device loaded" << std::endl;
        }
    }

    void async_pwrite(const torch::Tensor& buffer, const char* filename) override {
        if (device) {
            device->async_pwrite(buffer, filename);
        } else {
            std::cerr << "No device loaded" << std::endl;
        }
    }

    void new_cpu_locked_tensor(const size_t num_elem, const torch::Tensor& example_tensor) override {
        if (device) {
            device->new_cpu_locked_tensor(num_elem, example_tensor);
        } else {
            std::cerr << "No device loaded" << std::endl;
        }
    }

    void free_cpu_locked_tensor(torch::Tensor& tensor) override {
        if (device) {
            device->free_cpu_locked_tensor(tensor);
        } else {
            std::cerr << "No device loaded" << std::endl;
        }
    }

    void wait() override {
        if (device) {
            device->wait();
        } else {
            std::cerr << "No device loaded" << std::endl;
        }
    }

    ~DeepSpeedAIOTrampoline() {
        if (device) {
            delete device;
        }
    }

private:
    DeepSpeedAIOBase* device;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<DeepSpeedAIOTrampoline>(m, "DeepSpeedAIOTrampoline")
        .def(py::init<>())
        .def("load_device", &DeepSpeedAIOTrampoline::load_device)
        .def("aio_read", &DeepSpeedAIOTrampoline::aio_read)
        .def("aio_write", &DeepSpeedAIOTrampoline::aio_write)
        .def("deepspeed_memcpy", &DeepSpeedAIOTrampoline::deepspeed_memcpy)
        .def("get_block_size", &DeepSpeedAIOTrampoline::get_block_size)
        .def("get_queue_depth", &DeepSpeedAIOTrampoline::get_queue_depth)
        .def("get_single_submit", &DeepSpeedAIOTrampoline::get_single_submit)
        .def("get_overlap_events", &DeepSpeedAIOTrampoline::get_overlap_events)
        .def("get_thread_count", &DeepSpeedAIOTrampoline::get_thread_count)
        .def("read", &DeepSpeedAIOTrampoline::read)
        .def("write", &DeepSpeedAIOTrampoline::write)
        .def("pread", &DeepSpeedAIOTrampoline::pread)
        .def("pwrite", &DeepSpeedAIOTrampoline::pwrite)
        .def("sync_pread", &DeepSpeedAIOTrampoline::sync_pread)
        .def("sync_pwrite", &DeepSpeedAIOTrampoline::sync_pwrite)
        .def("async_pread", &DeepSpeedAIOTrampoline::async_pread)
        .def("async_pwrite", &DeepSpeedAIOTrampoline::async_pwrite)
        .def("new_cpu_locked_tensor", &DeepSpeedAIOTrampoline::new_cpu_locked_tensor)
        .def("free_cpu_locked_tensor", &DeepSpeedAIOTrampoline::free_cpu_locked_tensor)
        .def("wait", &DeepSpeedAIOTrampoline::wait);
}




































// // Copyright (c) Microsoft Corporation.
// // SPDX-License-Identifier: Apache-2.0

// // DeepSpeed Team

// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include "deepspeed_aio_base.h"

// namespace py = pybind11;

// class DeepSpeedAIOTrampoline : public DeepSpeedAIOBase {
// public:
//     DeepSpeedAIOTrampoline() : device(nullptr) {
//         load_device("nvme");
//     }

//     void load_device(const std::string& device_type) {
//         if (device_type == "nvme") {
//             device = new NVMEDevice();
//         } else {
//             std::cerr << "Unknown device type: " << device_type << std::endl;
//         }
//     }

//     void aio_read() override {
//         if (device) {
//             device->aio_read();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//         }
//     }

//     void aio_write() override {
//         if (device) {
//             device->aio_write();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//         }
//     }

//     void deepspeed_memcpy() override {
//         if (device) {
//             device->deepspeed_memcpy();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//         }
//     }

//     int get_block_size() const override {
//         if (device) {
//             return device->get_block_size();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//             return -1;
//         }
//     }

//     int get_queue_depth() const override {
//         if (device) {
//             return device->get_queue_depth();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//             return -1;
//         }
//     }

//     bool get_single_submit() const override {
//         if (device) {
//             return device->get_single_submit();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//             return false;
//         }
//     }

//     bool get_overlap_events() const override {
//         if (device) {
//             return device->get_overlap_events();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//             return false;
//         }
//     }

//     int get_thread_count() const override {
//         if (device) {
//             return device->get_thread_count();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//             return -1;
//         }
//     }

//     void read() override {
//         if (device) {
//             device->read();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//         }
//     }

//     void write() override {
//         if (device) {
//             device->write();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//         }
//     }

//     void pread() override {
//         if (device) {
//             device->pread();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//         }
//     }

//     void pwrite() override {
//         if (device) {
//             device->pwrite();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//         }
//     }

//     void sync_pread() override {
//         if (device) {
//             device->sync_pread();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//         }
//     }

//     void sync_pwrite() override {
//         if (device) {
//             device->sync_pwrite();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//         }
//     }

//     void async_pread() override {
//         if (device) {
//             device->async_pread();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//         }
//     }

//     void async_pwrite() override {
//         if (device) {
//             device->async_pwrite();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//         }
//     }

//     void new_cpu_locked_tensor() override {
//         if (device) {
//             device->new_cpu_locked_tensor();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//         }
//     }

//     void free_cpu_locked_tensor() override {
//         if (device) {
//             device->free_cpu_locked_tensor();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//         }
//     }

//     void wait() override {
//         if (device) {
//             device->wait();
//         } else {
//             std::cerr << "No device loaded" << std::endl;
//         }
//     }

//     ~DeepSpeedAIOTrampoline() {
//         if (device) {
//             delete device;
//         }
//     }

// private:
//     DeepSpeedAIOBase* device;
// };

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     py::class_<DeepSpeedAIOTrampoline>(m, "DeepSpeedAIOTrampoline")
//         .def(py::init<>())
//         .def("load_device", &DeepSpeedAIOTrampoline::load_device)
//         .def("aio_read", &DeepSpeedAIOTrampoline::aio_read)
//         .def("aio_write", &DeepSpeedAIOTrampoline::aio_write)
//         .def("deepspeed_memcpy", &DeepSpeedAIOTrampoline::deepspeed_memcpy)
//         .def("get_block_size", &DeepSpeedAIOTrampoline::get_block_size)
//         .def("get_queue_depth", &DeepSpeedAIOTrampoline::get_queue_depth)
//         .def("get_single_submit", &DeepSpeedAIOTrampoline::get_single_submit)
//         .def("get_overlap_events", &DeepSpeedAIOTrampoline::get_overlap_events)
//         .def("get_thread_count", &DeepSpeedAIOTrampoline::get_thread_count)
//         .def("read", &DeepSpeedAIOTrampoline::read)
//         .def("write", &DeepSpeedAIOTrampoline::write)
//         .def("pread", &DeepSpeedAIOTrampoline::pread)
//         .def("pwrite", &DeepSpeedAIOTrampoline::pwrite)
//         .def("sync_pread", &DeepSpeedAIOTrampoline::sync_pread)
//         .def("sync_pwrite", &DeepSpeedAIOTrampoline::sync_pwrite)
//         .def("async_pread", &DeepSpeedAIOTrampoline::async_pread)
//         .def("async_pwrite", &DeepSpeedAIOTrampoline::async_pwrite)
//         .def("new_cpu_locked_tensor", &DeepSpeedAIOTrampoline::new_cpu_locked_tensor)
//         .def("free_cpu_locked_tensor", &DeepSpeedAIOTrampoline::free_cpu_locked_tensor)
//         .def("wait", &DeepSpeedAIOTrampoline::wait);
// }
