// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>
void foo(void) {}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("foo", &foo, "Placeholder function"); }
