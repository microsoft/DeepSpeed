#include <ATen/ATen.h>
#include "inference_onemkl_wrappers.hpp"
#include "inference_sycl_layers.hpp"


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
                   T* C)
{
    // TODO: ldb and ldc is right? 
    try {
      int lda = (transa == oneapi::mkl::transpose::nontrans) ? k : m;
      int ldb = (transb == oneapi::mkl::transpose::nontrans) ? n : k;
      int ldc = n;
      oneapi::mkl::blas::row_major::gemm(
          handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
                  << std::endl;
      std::exit(1);
    }

    return 0;
}

// TODO: if stride_A needed
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
                                int batch)
{
    try {
      int lda = (transa == oneapi::mkl::transpose::nontrans) ? k : m;
      int ldb = (transb == oneapi::mkl::transpose::nontrans) ? n : k;
      int ldc = n;

      int stride_A = m * k;
      int stride_B = k * n;
      int stride_C = m * n;

      oneapi::mkl::blas::row_major::gemm_batch(handle,
                                    transa,
                                    transb,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    A,
                                    lda,
                                    stride_A,
                                    B,
                                    ldb,
                                    stride_B,
                                    beta,
                                    C,
                                    ldc,
                                    stride_C,
                                    batch);
    } catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
                << " (batch, m, n, k)" << batch << " " << m << " " << n << " " << k << std::endl;
      std::exit(1);
    }
}

template int onemkl_matmul_ex(sycl::queue handle,
                            oneapi::mkl::transpose transa,
                            oneapi::mkl::transpose transb,
                            int m,
                            int n,
                            int k,
                            const float alpha,
                            const float beta,
                            const sycl::half* A,
                            const sycl::half* B,
                            sycl::half* C);

template int onemkl_matmul_ex(sycl::queue handle,
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

template int onemkl_strided_batched_gemm(sycl::queue handle,
                                         oneapi::mkl::transpose op_A,
                                         oneapi::mkl::transpose op_B,
                                         int m,
                                         int n,
                                         int k,
                                         const float alpha,
                                         const float beta,
                                         const sycl::half* A,
                                         const sycl::half* B,
                                         sycl::half* C,
                                         int batch);

template int onemkl_strided_batched_gemm(sycl::queue handle,
                                         oneapi::mkl::transpose op_A,
                                         oneapi::mkl::transpose op_B,
                                         int m,
                                         int n,
                                         int k,
                                         const float alpha,
                                         const float beta,
                                         const float* A,
                                         const float* B,
                                         float* C,
                                         int batch);
