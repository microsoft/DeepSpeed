#include <cublas_v2.h>
#include <cuda.h>

namespace cuda {
using blasGemmAlgo_t = cublasGemmAlgo_t;
cublasGemmAlgo_t BLAS_GEMM_DEFAULT = CUBLAS_GEMM_DEFAULT;
cublasGemmAlgo_t BLAS_GEMM_DEFAULT_TENSOR_OP =
    CUBLAS_GEMM_DEFAULT_TENSOR_OP;  // TODO: this is deprecated in cublas
cublasGemmAlgo_t BLAS_GEMM_ALGO15_TENSOR_OP =
    CUBLAS_GEMM_ALGO15_TENSOR_OP;  // TODO: this is deprecated in cublas
}  // namespace cuda
