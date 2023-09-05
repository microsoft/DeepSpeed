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
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <stdio.h>

template <typename T>
int onemkl_matmul_ex(sycl::queue handle,
                     oneapi::mkl::transpose transa,
                     oneapi::mkl::transpose transb,
                     int m,
                     int n,
                     int k,
                     const float alpha,
                     const float beta,
                     const T* A,
                     const T* B,
                     T* C);

template <typename T>
int onemkl_strided_batched_gemm(sycl::queue handle,
                                oneapi::mkl::transpose transa,
                                oneapi::mkl::transpose transb,
                                int m,
                                int n,
                                int k,
                                const float alpha,
                                const float beta,
                                const T* A,
                                const T* B,
                                T* C,
                                int batch);
