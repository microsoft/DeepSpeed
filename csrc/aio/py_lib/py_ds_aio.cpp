// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <torch/extension.h>
#include "deepspeed_py_aio_handle.h"
#include "deepspeed_py_copy.h"
using namespace pybind11::literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("aio_read", &deepspeed_py_aio_read, "DeepSpeed Asynchronous I/O Read");

    m.def("aio_write", &deepspeed_py_aio_write, "DeepSpeed Asynchronous I/O Write");

    m.def("deepspeed_memcpy", &deepspeed_py_memcpy, "DeepSpeed Memory Copy");

    py::class_<deepspeed_aio_handle_t>(m, "aio_handle")
        .def(py::init<const int, const int, const bool, const bool, const bool, const int>(),
             "AIO handle constructor",
             "block_size"_a,
             "queue_depth"_a,
             "single_submit"_a,
             "overlap_events"_a,
             "use_gds"_a,
             "num_threads"_a)

        .def("get_block_size", &deepspeed_aio_handle_t::get_block_size)
        .def("get_queue_depth", &deepspeed_aio_handle_t::get_queue_depth)
        .def("get_single_submit", &deepspeed_aio_handle_t::get_single_submit)
        .def("get_overlap_events", &deepspeed_aio_handle_t::get_overlap_events)
        .def("get_use_gds", &deepspeed_aio_handle_t::get_use_gds)
        .def("get_thread_count", &deepspeed_aio_handle_t::get_thread_count)

        .def("read", &deepspeed_aio_handle_t::read)
        .def("write", &deepspeed_aio_handle_t::write)

        .def("pread", &deepspeed_aio_handle_t::pread)
        .def("pwrite", &deepspeed_aio_handle_t::pwrite)

        .def("sync_pread", &deepspeed_aio_handle_t::sync_pread)
        .def("sync_pwrite", &deepspeed_aio_handle_t::sync_pwrite)
        .def("async_pread", &deepspeed_aio_handle_t::async_pread)
        .def("async_pwrite", &deepspeed_aio_handle_t::async_pwrite)

        .def("new_cpu_locked_tensor", &deepspeed_aio_handle_t::new_cpu_locked_tensor)
        .def("free_cpu_locked_tensor", &deepspeed_aio_handle_t::free_cpu_locked_tensor)
        .def("new_device_locked_tensor", &deepspeed_aio_handle_t::new_device_locked_tensor)
        .def("free_device_locked_tensor", &deepspeed_aio_handle_t::free_device_locked_tensor)

        .def("wait", &deepspeed_aio_handle_t::wait);
}
