#include "cublas_wrappers.h"

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
                   gpu_lib::blasGemmAlgo_t algo)
{
    gpu_lib::blasStatus_t status = gpu_lib::blasGemmEx(handle,
                                                       transa,
                                                       transb,
                                                       m,
                                                       n,
                                                       k,
                                                       (const void*)alpha,
                                                       (const void*)A,
                                                       gpu_lib::DT_R_32F,
                                                       (transa == gpu_lib::BLAS_OP_N) ? m : k,
                                                       (const void*)B,
                                                       gpu_lib::DT_R_32F,
                                                       (transb == gpu_lib::BLAS_OP_N) ? k : n,
                                                       (const void*)beta,
                                                       C,
                                                       gpu_lib::DT_R_32F,
                                                       m,
                                                       gpu_lib::DT_R_32F,
                                                       algo);

    if (status != gpu_lib::BLAS_STATUS_SUCCESS) {
        fprintf(stderr,
                "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}

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
                   gpu_lib::blasGemmAlgo_t algo)
{
    gpu_lib::blasStatus_t status = cublasGemmEx(handle,
                                                transa,
                                                transb,
                                                m,
                                                n,
                                                k,
                                                (const void*)alpha,
                                                (const void*)A,
                                                gpu_lib::DT_R_16F,
                                                (transa == gpu_lib::BLAS_OP_N) ? m : k,
                                                (const void*)B,
                                                gpu_lib::DT_R_16F,
                                                (transb == gpu_lib::BLAS_OP_N) ? k : n,
                                                (const void*)beta,
                                                (void*)C,
                                                gpu_lib::DT_R_16F,
                                                m,
                                                gpu_lib::DT_R_32F,
                                                algo);
    if (status != gpu_lib::BLAS_STATUS_SUCCESS) {
        fprintf(stderr,
                "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}

int cublas_strided_batched_gemm(gpu_lib::blasHandle_t handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const float* A,
                                const float* B,
                                float* C,
                                gpu_lib::blasOperation_t op_A,
                                gpu_lib::blasOperation_t op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                gpu_lib::blasGemmAlgo_t algo) gpu_lib::blasStatus_t status =
    gpu_lib::blasGemmStridedBatchedEx(handle,
                                      op_A,
                                      op_B,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      gpu_lib::DT_R_32F,
                                      (op_A == gpu_lib::BLAS_OP_N) ? m : k,
                                      stride_A,
                                      B,
                                      gpu_lib::DT_R_32F,
                                      (op_B == gpu_lib::BLAS_OP_N) ? k : n,
                                      stride_B,
                                      beta,
                                      C,
                                      gpu_lib::DT_R_32F,
                                      m,
                                      stride_C,
                                      batch,
                                      gpu_lib::DT_R_32F,
                                      algo);

if (status != gpu_lib::BLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "!!!! kernel execution error. (batch: %d, m: %d, n: %d, k: %d, error: %d) \n",
            batch,
            m,
            n,
            k,
            (int)status);
    return EXIT_FAILURE;
}
return 0;
}

int cublas_strided_batched_gemm(gpu_lib::blasHandle_t handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const __half* A,
                                const __half* B,
                                __half* C,
                                gpu_lib::blasOperation_t op_A,
                                gpu_lib::blasOperation_t op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                gpu_lib::blasGemmAlgo_t algo)
{
    gpu_lib::blasStatus_t status =
        gpu_lib::blasGemmStridedBatchedEx(handle,
                                          op_A,
                                          op_B,
                                          m,
                                          n,
                                          k,
                                          alpha,
                                          A,
                                          gpu_lib::DT_R_16F,
                                          (op_A == gpu_lib::BLAS_OP_N) ? m : k,
                                          stride_A,
                                          B,
                                          gpu_lib::DT_R_16F,
                                          (op_B == gpu_lib::BLAS_OP_N) ? k : n,
                                          stride_B,
                                          beta,
                                          C,
                                          gpu_lib::DT_R_16F,
                                          m,
                                          stride_C,
                                          batch,
                                          gpu_lib::DT_R_32F,
                                          algo);
    if (status != gpu_lib::BLAS_STATUS_SUCCESS) {
        fprintf(stderr,
                "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }

    return 0;
}
