// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <stdio.h>
#include <stdlib.h>

// #include <cooperative_groups.h>

#include "context.hpp"
#include "onemkl_wrappers.hpp"

#define THREADS 256

#define minus_infinity -1 * std::numeric_limits<float>::infinity()

#define FINAL_MASK 0xffffffff

template <typename T>
void launch_fused_add2(T* out,
                       const T* inp1,
                       const T* inp2,
                       int batch_size,
                       int seq_length,
                       int hidden_size,
                       sycl::queue* stream);

template <typename T>
void launch_fused_add4(T* out,
                       const T* inp1,
                       const T* inp2,
                       const T* inp3,
                       const T* inp4,
                       int batch_size,
                       int seq_length,
                       int hidden_size,
                       sycl::queue* stream);

template <typename T>
void launch_fused_add3(T* out,
                       const T* inp1,
                       const T* inp2,
                       const T* inp3,
                       int batch_size,
                       int seq_length,
                       int hidden_size,
                       sycl::queue* stream);
