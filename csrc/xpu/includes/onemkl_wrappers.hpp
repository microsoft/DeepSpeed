// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <assert.h>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include <oneapi/mkl.hpp>

#include <stdio.h>

int onemkl_gemm_ex(sycl::queue* handle,
                   oneapi::mkl::transpose transa,
                   oneapi::mkl::transpose transb,
                   int m,
                   int n,
                   int k,
                   const float alpha,
                   const float beta,
                   const float* A,
                   const float* B,
                   float* C);

int onemkl_gemm_ex(sycl::queue* handle,
                   oneapi::mkl::transpose transa,
                   oneapi::mkl::transpose transb,
                   int m,
                   int n,
                   int k,
                   const sycl::half alpha,
                   const sycl::half beta,
                   const sycl::half* A,
                   const sycl::half* B,
                   sycl::half* C);

int onemkl_strided_batched_gemm(sycl::queue* handle,
                                int m,
                                int n,
                                int k,
                                const float alpha,
                                const float beta,
                                const float* A,
                                const float* B,
                                float* C,
                                oneapi::mkl::transpose op_A,
                                oneapi::mkl::transpose op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                int algo = -1);

int onemkl_strided_batched_gemm(sycl::queue* handle,
                                int m,
                                int n,
                                int k,
                                const sycl::half alpha,
                                const sycl::half beta,
                                const sycl::half* A,
                                const sycl::half* B,
                                sycl::half* C,
                                oneapi::mkl::transpose op_A,
                                oneapi::mkl::transpose op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                int algo = 99);
