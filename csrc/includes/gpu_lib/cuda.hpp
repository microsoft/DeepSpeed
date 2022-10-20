#include <cublas_v2.h>
#include <cuda.h>

namespace cuda {
using blasHandle_t = cublasHandle_t;
using blasOperation_t = cublasOperation_t;
using blasGemmAlgo_t = cublasGemmAlgo_t;
using blasStatus_t = cublasStatus_t;

cudaDataType_t DT_R_32F = CUDA_R_32F;
cudaDataType_t DT_R_16F = CUDA_R_16F;

cublasGemmAlgo_t BLAS_GEMM_DEFAULT = CUBLAS_GEMM_DEFAULT;
cublasGemmAlgo_t BLAS_GEMM_DEFAULT_TENSOR_OP =
    CUBLAS_GEMM_DEFAULT_TENSOR_OP;  // TODO: this is deprecated in cublas
cublasGemmAlgo_t BLAS_GEMM_ALGO15_TENSOR_OP =
    CUBLAS_GEMM_ALGO15_TENSOR_OP;  // TODO: this is deprecated in cublas
cublasOperation_t BLAS_OP_N = CUBLAS_OP_N;
cublasStatus_t BLAS_STATUS_SUCCESS = CUBLAS_STATUS_SUCCESS;

cublasStatus_t blasGemmEx(cublasHandle_t handle,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m,
                          int n,
                          int k,
                          const void* alpha,
                          const void* A,
                          cudaDataType Atype,
                          int lda,
                          const void* B,
                          cudaDataType Btype,
                          int ldb,
                          const void* beta,
                          void* C,
                          cudaDataType Ctype,
                          int ldc,
                          cudaDataType computeType,
                          cublasGemmAlgo_t algo)
{
    cublasStatus_t status = cublasGemmEx(handle,
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
                                         ldc,
                                         computeType,
                                         algo);
    return status;
}
}  // namespace cuda
