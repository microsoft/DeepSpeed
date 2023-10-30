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
#include "custom_sycl_layers.hpp"

template <typename T>
class FeedForward {
public:
    struct Config {
        int batchSize, outputSize;
        int inputSize;
        Config(int batch, int outputs, int inputs)
            : batchSize(batch), outputSize(outputs), inputSize(inputs)
        {
        }
    };

    FeedForward(Config config) : config_(config) {}

    ~FeedForward() {}

    void Forward(int bsz, const T* input_ptr, const T* weights, T* out, sycl::queue* _Q)
    {
        if constexpr (std::is_same_v<bf16, T>) {
            float alpha = 1.0f;
            float beta = 0.0f;
            onednn_matmul_ex(_Q,
                             false,
                             true,
                             bsz,
                             config_.outputSize,
                             config_.inputSize,
                             alpha,
                             beta,
                             input_ptr,
                             weights,
                             out);
        } else {
            T alpha = T(1.);
            T beta = T(0.);
            onemkl_gemm_ex(_Q,
                           oneapi::mkl::transpose::trans,
                           oneapi::mkl::transpose::nontrans,
                           config_.outputSize,
                           bsz,
                           config_.inputSize,
                           alpha,
                           beta,
                           weights,
                           input_ptr,
                           out);
        }
    }
    void Backward(int bsz,
                  const T* out_grad,
                  const T* input_ptr,
                  const T* weights,
                  T* weights_grad,
                  T* bias_grad,
                  sycl::queue* _Q,
                  sycl::queue* stream,
                  T* inp_grad_out = nullptr,
                  T* out_grad_trans_out = nullptr)
    {
        if constexpr (std::is_same_v<bf16, T>) {
            float alpha = 1.0f;
            float beta = 0.0f;
            onednn_matmul_ex(stream,
                             true,
                             false,
                             config_.outputSize,
                             config_.inputSize,
                             bsz,
                             alpha,
                             beta,
                             out_grad,
                             input_ptr,
                             weights_grad);
            onednn_matmul_ex(stream,
                             false,
                             false,
                             bsz,
                             config_.inputSize,
                             config_.outputSize,
                             alpha,
                             beta,
                             out_grad,
                             weights,
                             inp_grad_out);
            launch_fuse_transpose_bias_kernel<T>(
                out_grad, bias_grad, bsz, config_.outputSize, stream);
        } else {
            T alpha = (T)1.0;
            T beta = (T)0.0;
            onemkl_gemm_ex(_Q,
                           oneapi::mkl::transpose::nontrans,
                           oneapi::mkl::transpose::trans,
                           config_.inputSize,
                           config_.outputSize,
                           bsz,
                           alpha,
                           beta,
                           input_ptr,
                           out_grad,
                           weights_grad);
            onemkl_gemm_ex(_Q,
                           oneapi::mkl::transpose::nontrans,
                           oneapi::mkl::transpose::nontrans,
                           config_.inputSize,
                           bsz,
                           config_.outputSize,
                           alpha,
                           beta,
                           weights,
                           out_grad,
                           inp_grad_out);
            launch_fuse_transpose_bias_kernel<T>(
                out_grad, bias_grad, bsz, config_.outputSize, stream);
        }
    }

private:
    Config config_;
};
