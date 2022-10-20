namespace rocm {
using blasGemmAlgo_t = rocblas_gemm_algo;
rocblas_gemm_algo BLAS_GEMM_DEFAULT = rocblas_gemm_algo_standard;
rocblas_gemm_algo BLAS_GEMM_DEFAULT_TENSOR_OP = rocblas_gemm_algo_standard;
rocblas_gemm_algo BLAS_GEMM_ALGO15_TENSOR_OP = rocblas_gemm_algo_standard;
}  // namespace rocm
