namespace rocm {
using blasHandle_t = rocblas_handle;
using blasOperation_t = rocblas_operation;
using blasGemmAlgo_t = rocblas_gemm_algo;
using blasStatus_t = rocblas_status;

rocblas_datatype DT_R_32F = rocblas_datatype_f32_r;
rocblas_datatype DT_R_16F = rocblas_datatype_f16_r;

rocblas_gemm_algo BLAS_GEMM_DEFAULT = rocblas_gemm_algo_standard;
rocblas_gemm_algo BLAS_GEMM_DEFAULT_TENSOR_OP = rocblas_gemm_algo_standard;
rocblas_gemm_algo BLAS_GEMM_ALGO15_TENSOR_OP = rocblas_gemm_algo_standard;
rocblas_operation BLAS_OP_N = rocblas_operation_none;
rocblas_status BLAS_STATUS_SUCCESS = rocblas_status_success;

rocblas_status blasGemmEx(rocblas_handle handle,
                          rocblas_operation transa,
                          rocblas_operation transb,
                          int m,
                          int n,
                          int k,
                          const void* alpha,
                          const void* A,
                          rocblas_datatype Atype,
                          int lda,
                          const void* B,
                          rocblas_datatype Btype,
                          int ldb,
                          const void* beta,
                          void* C,
                          rocblas_datatype Ctype,
                          int ldc,
                          rocblas_datatype computeType,
                          rocblas_gemm_algo algo)
{
    rocblas_status status = rocblas_gemm_ex(handle,
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            alpha,
                                            A,
                                            Atype,
                                            lda,
                                            B,
                                            Btype,
                                            ldb,
                                            beta,
                                            C,
                                            Ctype,
                                            m,
                                            C,
                                            Ctype,
                                            m,
                                            computeType,
                                            algo,
                                            0,
                                            0);
    return status
}

rocblas_status blasGemmStridedBatchedEx(rocblas_handle handle,
                                        rocblas_operation transa,
                                        rocblas_operation transb,
                                        int m,
                                        int n,
                                        int k,
                                        const void* alpha,
                                        const void* A,
                                        rocblas_datatype Atype,
                                        int lda,
                                        long long int strideA,
                                        const void* B,
                                        rocblas_datatype Btype,
                                        int ldb,
                                        long long int strideB,
                                        const void* beta,
                                        void* C,
                                        rocblas_datatype Ctype,
                                        int ldc,
                                        long long int strideC,
                                        int batchCount,
                                        rocblas_datatype computeType,
                                        rocblas_gemm_algo algo)
{
    rocblas_status status = rocblas_gemm_strided_batched_ex(handle,
                                                            transa,
                                                            transb,
                                                            m,
                                                            n,
                                                            k,
                                                            alpha,
                                                            A,
                                                            Atype,
                                                            lda,
                                                            strideA,
                                                            B,
                                                            Btype,
                                                            ldb,
                                                            strideB,
                                                            beta,
                                                            C,
                                                            Ctype,
                                                            m,
                                                            strideC,
                                                            C,
                                                            Ctype,
                                                            m,
                                                            strideC,
                                                            batchCount,
                                                            computeType,
                                                            algo,
                                                            0,
                                                            0);
    return status;
}

}  // namespace rocm
