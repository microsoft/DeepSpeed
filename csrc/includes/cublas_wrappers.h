#pragma once

#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifndef __HIP_PLATFORM_HCC__
#include <mma.h>
#endif
#include <stdio.h>
#include "gpu_lib/gpu_lib.h"

int cublas_gemm_ex(gpu_lib::blasHandle_t handle,
                   gpu_lib::blasOperation_t transa,
                   gpu_lib::blasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const float* A,
                   const float* B,
                   float* C,
                   gpu_lib::blasGemmAlgo_t = gpu_lib::BLAS_GEMM_DEFAULT);

int cublas_gemm_ex(gpu_lib::blasHandle_t handle,
                   gpu_lib::blasOperation_t transa,
                   gpu_lib::blasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const __half* A,
                   const __half* B,
                   __half* C,
                   gpu_lib::blasGemmAlgo_t = gpu_lib::BLAS_GEMM_DEFAULT_TENSOR_OP);

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
                                gpu_lib::blasGemmAlgo_t = gpu_lib::BLAS_GEMM_DEFAULT);

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
                                gpu_lib::blasGemmAlgo_t = gpu_lib::BLAS_GEMM_DEFAULT_TENSOR_OP);
