// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <torch/extension.h>
#include "deepspeed_py_gds_handle.h"
using namespace pybind11::literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<deepspeed_gds_handle_t>(m, "gds_handle")
        .def(py::init<const int, const int, const bool, const bool, const int>(),
             "GDS handle constructor",
             "block_size"_a,
             "queue_depth"_a,
             "single_submit"_a,
             "overlap_events"_a,
             "num_threads"_a)

        .def("get_block_size", &deepspeed_gds_handle_t::get_block_size)
        .def("get_queue_depth", &deepspeed_gds_handle_t::get_queue_depth)
        .def("get_single_submit", &deepspeed_gds_handle_t::get_single_submit)
        .def("get_overlap_events", &deepspeed_gds_handle_t::get_overlap_events)
        .def("get_thread_count", &deepspeed_gds_handle_t::get_thread_count)

        .def("read", &deepspeed_gds_handle_t::read)
        .def("write", &deepspeed_gds_handle_t::write)

        .def("pread", &deepspeed_gds_handle_t::pread)
        .def("pwrite", &deepspeed_gds_handle_t::pwrite)

        .def("sync_pread", &deepspeed_gds_handle_t::sync_pread)
        .def("sync_pwrite", &deepspeed_gds_handle_t::sync_pwrite)
        .def("async_pread", &deepspeed_gds_handle_t::async_pread)
        .def("async_pwrite", &deepspeed_gds_handle_t::async_pwrite)

        .def("new_cpu_locked_tensor", &deepspeed_gds_handle_t::new_cpu_locked_tensor)
        .def("free_cpu_locked_tensor", &deepspeed_gds_handle_t::free_cpu_locked_tensor)

        .def("new_pinned_device_tensor", &deepspeed_gds_handle_t::new_pinned_device_tensor)
        .def("free_pinned_device_tensor", &deepspeed_gds_handle_t::free_pinned_device_tensor)
        .def("pin_device_tensor", &deepspeed_gds_handle_t::pin_device_tensor)
        .def("unpin_device_tensor", &deepspeed_gds_handle_t::unpin_device_tensor)

        .def("wait", &deepspeed_gds_handle_t::wait);
}
