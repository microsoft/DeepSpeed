#pragma once

#include <assert.h>
#ifndef __HIP_PLATFORM_HCC__
#include <mma.h>
#endif
#include <stdio.h>
#ifdef __hip_platform_hcc__
int cublas_gemm_ex(rocblas_handle handle,
                   rocblas_operation transa,
                   rocblas_operation transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const float* a,
                   const float* b,
                   float* c,
                   rocblas_gemm_algo algo = rocblas_gemm_algo_standard);
#else
int cublas_gemm_ex(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const float* A,
                   const float* B,
                   float* C,
                   cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT);
#endif

#ifdef __hip_platform_hcc__
int cublas_gemm_ex(rocblas_handle handle,
                   rocblas_operation transa,
                   rocblas_operation transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const __half* a,
                   const __half* b,
                   __half* c,
                   rocblas_gemm_algo algo = rocblas_gemm_algo_standard);
#else
int cublas_gemm_ex(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const __half* A,
                   const __half* B,
                   __half* C,
                   cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif

#ifdef __HIP_PLATFORM_HCC__
int cublas_strided_batched_gemm(rocblas_handle handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const float* A,
                                const float* B,
                                float* C,
                                rocblas_operation op_A,
                                rocblas_operation op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                rocblas_gemm_algo algo = rocblas_gemm_algo_standard);
#else
int cublas_strided_batched_gemm(cublasHandle_t handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const float* A,
                                const float* B,
                                float* C,
                                cublasOperation_t op_A,
                                cublasOperation_t op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT);
#endif

#ifdef __HIP_PLATFORM_HCC__
int cublas_strided_batched_gemm(rocblas_handle handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const __half* A,
                                const __half* B,
                                __half* C,
                                rocblas_operation op_A,
                                rocblas_operation op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                rocblas_gemm_algo algo = rocblas_gemm_algo_standard);
#else
int cublas_strided_batched_gemm(cublasHandle_t handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const __half* A,
                                const __half* B,
                                __half* C,
                                cublasOperation_t op_A,
                                cublasOperation_t op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
