#pragma once

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include <array>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <memory>
#include "StopWatch.h"
#include "onemkl_wrappers.hpp"

template <typename T>
class GemmTest {
public:
    GemmTest(int m,
             int n,
             int k,
             oneapi::mkl::transpose ta,
             oneapi::mkl::transpose tb,
             sycl::queue* h)
        : M(m), N(n), K(k), transa(ta), transb(tb), handle(h)
    {
        dpct::device_ext& dev_ct1 = dpct::get_current_device();
        sycl::queue& q_ct1 = dev_ct1.default_queue();
        A = (T*)sycl::malloc_device(sizeof(T) * M * K, q_ct1);
        B = (T*)sycl::malloc_device(sizeof(T) * K * N, q_ct1);
        C = (T*)sycl::malloc_device(sizeof(T) * M * N, q_ct1);
    }

    ~GemmTest()
    {
        dpct::device_ext& dev_ct1 = dpct::get_current_device();
        sycl::queue& q_ct1 = dev_ct1.default_queue();
        sycl::free(A, q_ct1);
        sycl::free(B, q_ct1);
        sycl::free(C, q_ct1);
    }

    std::array<int, 3> TestAlgo(int loops)
    {
        float alpha = (T)1.0f;
        float beta = (T)0.0f;

        int algo_fw = Run(loops, [=](int algo) {
            onemkl_gemm_ex(handle,
                           oneapi::mkl::transpose::trans,
                           oneapi::mkl::transpose::nontrans,
                           N,
                           M,
                           K,
                           &alpha,
                           &beta,
                           B,
                           A,
                           C,
                           static_cast<cublasGemmAlgo_t>(algo));
        });

        int algo_bw1 = Run(loops, [=](int algo) {
            onemkl_gemm_ex(handle,
                           oneapi::mkl::transpose::nontrans,
                           oneapi::mkl::transpose::trans,
                           K,
                           N,
                           M,
                           &alpha,
                           &beta,
                           A,
                           C,
                           B,
                           static_cast<cublasGemmAlgo_t>(algo));
        });

        int algo_bw2 = Run(loops, [=](int algo) {
            onemkl_gemm_ex(handle,
                           oneapi::mkl::transpose::nontrans,
                           oneapi::mkl::transpose::nontrans,
                           K,
                           M,
                           N,
                           &alpha,
                           &beta,
                           B,
                           C,
                           A,
                           static_cast<cublasGemmAlgo_t>(algo));
        });

        return std::array<int, 3>({algo_fw, algo_bw1, algo_bw2});
    }

    template <typename Func>
    int Run(int loops, Func f)
    {
        float fast_latency = (std::numeric_limits<float>::max)();
        int fast_algo = 0;

        for (int algo = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
             algo <= (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
             algo++) {
            int warm_up = 5;
            for (int i = 0; i < warm_up; ++i) f(algo);

            cudaDeviceSynchronize();
            Stopwatch timer;
            timer.Restart();

            for (int i = 0; i < loops; ++i) f(algo);

            cudaDeviceSynchronize();
            timer.Stop();

            float avg_latency = (float)timer.GetTimeInSeconds() * 1000 / loops;

            printf("algo-%d: %.3fms\n", algo, avg_latency);

            if (avg_latency < fast_latency) {
                fast_latency = avg_latency;
                fast_algo = algo;
            }
        }

        printf("fast_algo %d: %.3f ms\n", fast_algo, fast_latency);

        return fast_algo;
    }

private:
    int M, N, K;
    sycl::queue* handle;
    oneapi::mkl::transpose transa, transb;
    T *A, *B, *C;
};

template <typename T>
class StridedGemmTest {
public:
    StridedGemmTest(int b,
                    int m,
                    int n,
                    int k,
                    oneapi::mkl::transpose ta,
                    oneapi::mkl::transpose tb,
                    sycl::queue* h)
        : bsz(b), M(m), N(n), K(k), transa(ta), transb(tb), handle(h)
    {
        dpct::device_ext& dev_ct1 = dpct::get_current_device();
        sycl::queue& q_ct1 = dev_ct1.default_queue();
        A = (T*)sycl::malloc_device(sizeof(T) * M * K * bsz, q_ct1);
        B = (T*)sycl::malloc_device(sizeof(T) * K * N * bsz, q_ct1);
        C = (T*)sycl::malloc_device(sizeof(T) * M * N * bsz, q_ct1);
    }

    ~StridedGemmTest()
    {
        dpct::device_ext& dev_ct1 = dpct::get_current_device();
        sycl::queue& q_ct1 = dev_ct1.default_queue();
        sycl::free(A, q_ct1);
        sycl::free(B, q_ct1);
        sycl::free(C, q_ct1);
    }

    std::array<int, 3> TestAlgo(int loops)
    {
        float alpha = (T)1.0f;
        float beta = (T)0.0f;

        int algo_fw = Run(loops, [=](int algo) {
            int stride_a = M * K;
            int stride_b = N * K;
            int stride_c = M * N;

            cublas_strided_batched_gemm(handle,
                                        M,
                                        N,
                                        K,
                                        &alpha,
                                        &beta,
                                        A,
                                        B,
                                        C,
                                        transa,
                                        transb,
                                        stride_a,
                                        stride_b,
                                        stride_c,
                                        bsz,
                                        static_cast<cublasGemmAlgo_t>(algo));
        });

        int algo_bw1 = Run(loops, [=](int algo) {
            int mb = (transa == oneapi::mkl::transpose::trans ? K : M);
            int kb = (transa == oneapi::mkl::transpose::trans ? M : K);

            int stride_a = mb * N;
            int stride_b = N * kb;
            int stride_c = M * K;

            // B need to transpose.
            cublasOperation_t op_b =
                (transb == oneapi::mkl::transpose::trans ? oneapi::mkl::transpose::nontrans
                                                         : oneapi::mkl::transpose::trans);

            // Calculate d_A.
            cublas_strided_batched_gemm(handle,
                                        mb,
                                        kb,
                                        N,
                                        &alpha,
                                        &beta,
                                        (transa == oneapi::mkl::transpose::trans ? B : C),
                                        (transa == oneapi::mkl::transpose::trans ? C : B),
                                        A,
                                        oneapi::mkl::transpose::nontrans,
                                        op_b,
                                        stride_a,
                                        stride_b,
                                        stride_c,
                                        bsz,
                                        static_cast<cublasGemmAlgo_t>(algo));
        });

        int algo_bw2 = Run(loops, [=](int algo) {
            // A need to transpose.
            cublasOperation_t op_a =
                (transa == oneapi::mkl::transpose::trans ? oneapi::mkl::transpose::nontrans
                                                         : oneapi::mkl::transpose::trans);

            int stride_a = M * K;
            int stride_b = M * N;
            int stride_c = N * K;

            // Calculate d_B.
            cublas_strided_batched_gemm(handle,
                                        K,
                                        N,
                                        M,
                                        &alpha,
                                        &beta,
                                        A,
                                        C,
                                        B,
                                        op_a,
                                        oneapi::mkl::transpose::nontrans,
                                        stride_a,
                                        stride_b,
                                        stride_c,
                                        bsz,
                                        static_cast<cublasGemmAlgo_t>(algo));
        });

        return std::array<int, 3>({algo_fw, algo_bw1, algo_bw2});
    }

    template <typename Func>
    int Run(int loops, Func f)
    {
        float fast_latency = (std::numeric_limits<float>::max)();
        int fast_algo = 0;

        for (int algo = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
             algo <= (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
             algo++) {
            int warm_up = 5;
            for (int i = 0; i < warm_up; ++i) f(algo);

            cudaDeviceSynchronize();
            Stopwatch timer;
            timer.Restart();

            for (int i = 0; i < loops; ++i) f(algo);

            cudaDeviceSynchronize();
            timer.Stop();

            float avg_latency = (float)timer.GetTimeInSeconds() * 1000 / loops;

            printf("algo-%d: %.3fms\n", algo, avg_latency);

            if (avg_latency < fast_latency) {
                fast_latency = avg_latency;
                fast_algo = algo;
            }
        }

        printf("fast_algo %d: %.3f ms\n", fast_algo, fast_latency);

        return fast_algo;
    }

private:
    int bsz, M, N, K;
    sycl::queue* handle;
    oneapi::mkl::transpose transa, transb;
    T *A, *B, *C;
};
