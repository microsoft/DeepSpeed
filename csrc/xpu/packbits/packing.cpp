// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>
#include <iostream>

// C++ interface


int packbits()
{
    std::cout << "mll packbits kernel!!>>>>>>>>>>>>>>>>>" << std::endl;
    return 0;
}

int unpackbits()
{
    std::cout << "mll unpackbits kernel!!>>>>>>>>>>>>>>>>>" << std::endl;
    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("packbits", &packbits, "DeepSpeed XPU packbits (C++)");
    m.def("unpackbits", &unpackbits, "DeepSpeed XPU unpackbits (C++)");
}
