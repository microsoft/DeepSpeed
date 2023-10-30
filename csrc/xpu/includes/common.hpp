// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once
#include <torch/extension.h>

#define CHECK_XPU(x) AT_ASSERTM(x.is_xpu(), #x " must be a XPU tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_XPU(x);      \
    CHECK_CONTIGUOUS(x)

template <typename T>
inline void print_nan(sycl::queue* stream, int bsz, const T* buf, char* name)
{
    T temp_tensor[10000];
    bool has_nan = false;
    stream->wait();
    stream->memcpy(temp_tensor, buf, bsz * sizeof(T));
    stream->wait();
    for (int i = 0; i < bsz; i++) {
        if (isnan(float(temp_tensor[i]))) { has_nan = true; }
    }
    printf("%s[%d](%p)%s --> ", name, bsz, buf, has_nan ? "has_nan" : "");
    for (int i = 0; i < bsz; i++) {
        if (isnan(float(temp_tensor[i]))) {
            printf("%d:nan ", i);
        } else {
            printf("%d:%f, ", i, float(temp_tensor[i]));
        }
    }
    printf("\n");
}
