// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <stdio.h>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include "context.hpp"
#include "onednn_wrappers.hpp"
#include "onemkl_wrappers.hpp"

template <typename T>
class StridedBatchGemm {
public:
    struct Config {
        int batch_size;
        int m;
        int n;
        int k;
        float alpha;
        float beta;
        oneapi::mkl::transpose op_A;
        oneapi::mkl::transpose op_B;
        std::array<int, 3> gemm_algos;

        Config(int batch,
               int mm,
               int nn,
               int kk,
               float param_alpha,
               float param_beta,
               oneapi::mkl::transpose opA,
               oneapi::mkl::transpose opB,
               const std::array<int, 3>& algos)
            : batch_size(batch),
              m(mm),
              n(nn),
              k(kk),
              alpha(param_alpha),
              beta(param_beta),
              op_A(opA),
              op_B(opB),
              gemm_algos(algos)
        {
        }
        void SetConfig(int mm, int nn, int kk)
        {
            m = mm;
            n = nn;
            k = kk;
        }
    };

    StridedBatchGemm(const Config& config) : _config(config)
    {
        k_buf = NULL;
        q_buf = NULL;
    }

    virtual ~StridedBatchGemm() {}

    void Forward(int bsz, T* output, const T* _buffer_a, const T* _buffer_b, sycl::queue* handle)
    {
        int stride_a = _config.m * _config.k;
        int stride_b = _config.n * _config.k;
        int stride_c = _config.m * _config.n;

        if constexpr (std::is_same_v<T, bf16>) {
            onednn_batchgemm(handle,
                             _config.n,
                             _config.m,
                             _config.k,
                             _config.alpha,
                             _config.beta,
                             _buffer_b,
                             _buffer_a,
                             output,
                             _config.op_B == oneapi::mkl::transpose::trans,
                             _config.op_A == oneapi::mkl::transpose::trans,
                             bsz);
        } else {
            onemkl_strided_batched_gemm(handle,
                                        _config.m,
                                        _config.n,
                                        _config.k,
                                        (T)_config.alpha,
                                        (T)_config.beta,
                                        _buffer_a,
                                        _buffer_b,
                                        output,
                                        _config.op_A,
                                        _config.op_B,
                                        stride_a,
                                        stride_b,
                                        stride_c,
                                        bsz,
                                        int(_config.gemm_algos[0]));
        }
    }

    void ForwardPlusSave(T* output, const T* _buffer_a, const T* _buffer_b, sycl::queue* handle)
    {
        int stride_a = _config.m * _config.k;
        int stride_b = _config.n * _config.k;
        int stride_c = _config.m * _config.n;

        if constexpr (std::is_same_v<T, bf16>) {
            throw std::runtime_error("Unsupport bf16 strided batch gemm");
        } else {
            onemkl_strided_batched_gemm(handle,
                                        _config.m,
                                        _config.n,
                                        _config.k,
                                        (T)_config.alpha,
                                        (T)_config.beta,
                                        _buffer_a,
                                        _buffer_b,
                                        output,
                                        _config.op_A,
                                        _config.op_B,
                                        stride_a,
                                        stride_b,
                                        stride_c,
                                        _config.batch_size,
                                        int(_config.gemm_algos[0]));
        }

        k_buf = _buffer_a;
        q_buf = _buffer_b;
    }

    void Backward(int bsz,
                  const T* d_output,
                  const T* _buffer_a,
                  const T* _buffer_b,
                  sycl::queue* handle,
                  T* inpGradA = nullptr,
                  T* inpGradB = nullptr)
    {
        if constexpr (std::is_same_v<T, bf16>) {
            // calculate d_A
            if (_config.op_A == oneapi::mkl::transpose::trans) {
                onednn_batchgemm(handle,
                                 _config.m,
                                 _config.k,
                                 _config.n,
                                 _config.alpha,
                                 _config.beta,
                                 d_output,
                                 _buffer_b,
                                 inpGradA,
                                 true,
                                 false,
                                 bsz);

                // Calculate d_B.
                onednn_batchgemm(handle,
                                 _config.n,
                                 _config.k,
                                 _config.m,
                                 _config.alpha,
                                 _config.beta,
                                 d_output,
                                 _buffer_a,
                                 inpGradB,
                                 false,
                                 false,
                                 bsz);
            } else {
                onednn_batchgemm(handle,
                                 _config.n,
                                 _config.m,
                                 _config.k,
                                 _config.alpha,
                                 _config.beta,
                                 _buffer_b,
                                 d_output,
                                 inpGradA,
                                 true,
                                 false,
                                 bsz);

                // Calculate d_B.
                onednn_batchgemm(handle,
                                 _config.n,
                                 _config.k,
                                 _config.m,
                                 _config.alpha,
                                 _config.beta,
                                 d_output,
                                 _buffer_a,
                                 inpGradB,
                                 false,
                                 true,
                                 bsz);
            }

        } else {
            int mb = (_config.op_A == oneapi::mkl::transpose::trans ? _config.k : _config.m);
            int kb = (_config.op_A == oneapi::mkl::transpose::trans ? _config.m : _config.k);

            int stride_a = mb * _config.n;
            int stride_b = _config.n * kb;
            int stride_c = _config.m * _config.k;

            // B need to transpose.
            oneapi::mkl::transpose op_b =
                (_config.op_B == oneapi::mkl::transpose::trans ? oneapi::mkl::transpose::nontrans
                                                               : oneapi::mkl::transpose::trans);

            // calculate d_A
            onemkl_strided_batched_gemm(
                handle,
                mb,
                kb,
                _config.n,
                (T)_config.alpha,
                (T)_config.beta,
                (_config.op_A == oneapi::mkl::transpose::trans ? _buffer_b : d_output),
                (_config.op_A == oneapi::mkl::transpose::trans ? d_output : _buffer_b),
                inpGradA,
                oneapi::mkl::transpose::nontrans,
                op_b,
                stride_a,
                stride_b,
                stride_c,
                bsz,
                int(_config.gemm_algos[1]));

            // A need to transpose.
            oneapi::mkl::transpose op_a =
                (_config.op_A == oneapi::mkl::transpose::trans ? oneapi::mkl::transpose::nontrans
                                                               : oneapi::mkl::transpose::trans);

            stride_a = _config.m * _config.k;
            stride_b = _config.m * _config.n;
            stride_c = _config.n * _config.k;

            // Calculate d_B.
            onemkl_strided_batched_gemm(handle,
                                        _config.k,
                                        _config.n,
                                        _config.m,
                                        (T)_config.alpha,
                                        (T)_config.beta,
                                        _buffer_a,
                                        d_output,
                                        inpGradB,
                                        op_a,
                                        oneapi::mkl::transpose::nontrans,
                                        stride_a,
                                        stride_b,
                                        stride_c,
                                        bsz,
                                        int(_config.gemm_algos[2]));
        }
    }

    inline int GetN() const { return _config.k; }

    inline const T* GetBufferA() const { return k_buf; }

    inline const T* GetBufferB() const { return q_buf; }

    inline void SetConfig(int m, int n, int k) { _config.SetConfig(m, n, k); }

private:
    Config _config;
    const T* q_buf;
    const T* k_buf;
};
