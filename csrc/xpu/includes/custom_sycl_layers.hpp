// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <stdio.h>
#include <stdlib.h>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>

#include "context.hpp"
#include "onednn_wrappers.hpp"
#include "onemkl_wrappers.hpp"

#define MAX_THREADS 1024
#define THREADS 256

#define MAX_THREAD_STRIDE 32
#define TILE_DIM 16

// Maximum sequence-length support based on the number of threads (2048) allowed
// in each block and this MAX is 8K For higher sequence length we need to use
// higher Max, like for 64K : 32
#define MAX_THREAD_ITERATIONS 8  // Maximum 8K
#define MAX_WARP_NUM 32

#define MAX_REGISTERS 256

#define MAX_REG 256

template <typename T>
void launch_qunatize_kernel(T* vals,
                            int total_count,
                            int group_num,
                            int num_bits,
                            sycl::queue* stream);
template <typename T>
void launch_sr_qunatize_kernel(T* vals,
                               int total_count,
                               int group_num,
                               int num_bits,
                               sycl::queue* stream);
template <typename T>
void launch_qunatize_kernel_asym(T* vals,
                                 int total_count,
                                 int group_num,
                                 int num_bits,
                                 sycl::queue* stream);
template <typename T>
void launch_sr_qunatize_kernel_asym(T* vals,
                                    int total_count,
                                    int group_num,
                                    int num_bits,
                                    sycl::queue* stream);
// Fused bias add with gelu activation
template <typename T>
void launch_bias_gelu(const T* input,
                      const T* bias,
                      T* output,
                      int intermediate_size,
                      int batch_size,
                      sycl::queue* stream);

template <typename T>
void launch_gelu(const T* input,
                 T* output,
                 int intermediate_size,
                 int batch_size,
                 sycl::queue* stream);

template <typename T>
void launch_d_gelu(T* d_output,
                   const T* input,
                   const T* bias,
                   int intermediate_size,
                   int batch_size,
                   sycl::queue* stream);

// Custom fused bias add with layer normalization
template <typename T>
void launch_bias_residual_layer_norm(T* vals,
                                     const T* residual,
                                     const T* gamma,
                                     const T* beta,
                                     float epsilon,
                                     int batch_size,
                                     int hidden_dim,
                                     sycl::queue* stream,
                                     bool preLayerNorm,
                                     bool training,
                                     T* vars,
                                     T* means);

template <typename T>
void launch_bias_residual_layer_norm(T* vals,
                                     const T* residual,
                                     const T* gamma,
                                     const T* beta,
                                     float epsilon,
                                     int batch_size,
                                     int hidden_dim,
                                     sycl::queue* stream,
                                     bool preLayerNorm,
                                     bool training,
                                     T* vars);

template <typename T>
void launch_layerNorm_backward_fused_add(const T* out_grad1,
                                         const T* out_grad2,
                                         const T* X_data,
                                         const T* vars,
                                         const T* means,
                                         const T* gamma,
                                         T* gamma_grad,
                                         T* betta_grad,
                                         T* inp_grad,
                                         int batch_size,
                                         int hidden_dim,
                                         sycl::queue* stream[2]);
template <typename T>
void launch_layerNorm_backward_fused_add(const T* out_grad1,
                                         const T* out_grad2,
                                         const T* vals_hat,
                                         const T* vars,
                                         const T* gamma,
                                         T* gamma_grad,
                                         T* betta_grad,
                                         T* inp_grad,
                                         int batch_size,
                                         int hidden_dim,
                                         sycl::queue* stream[2],
                                         bool invertible = false,
                                         const T* betta = nullptr);

template <typename T>
void launch_layerNorm_backward(const T* out_grad,
                               const T* X_data,
                               const T* vars,
                               const T* means,
                               const T* gamma,
                               T* gamma_grad,
                               T* betta_grad,
                               T* inp_grad,
                               int batch_size,
                               int hidden_dim,
                               sycl::queue* stream[2]);

template <typename T>
void launch_layerNorm_backward(const T* out_grad,
                               const T* vals_hat,
                               const T* vars,
                               const T* gamma,
                               T* gamma_grad,
                               T* betta_grad,
                               T* inp_grad,
                               int batch_size,
                               int hidden_dim,
                               sycl::queue* stream[2],
                               bool invertible = false,
                               const T* betta = nullptr);

template <typename T>
void launch_layerNorm_backward_nreversible(const T* out_grad,
                                           const T* vals,
                                           const T* out_grad_trans,
                                           const T* vals_trans,
                                           const T* means,
                                           const T* vars,
                                           const T* gamma,
                                           T* gamma_grad,
                                           T* betta_grad,
                                           T* inp_grad,
                                           int batch_size,
                                           int hidden_dim,
                                           sycl::queue* stream[2]);

template <typename T>
void Transpose(const T* inp_mat, T* out_mat, int rows, int cols, sycl::queue* stream);

template <typename T>
void launch_attn_softmax_backward(T* out_grad,
                                  const T* soft_inp,
                                  int batch_size,
                                  int heads,
                                  int seq_length,
                                  sycl::queue* stream);

template <typename T>
void launch_attn_softmax_backward_v2(T* out_grad,
                                     const T* soft_inp,
                                     int batch_size,
                                     int heads,
                                     int seq_length,
                                     sycl::queue* stream);

// Custom softmax with scaling and attention mask addition
template <typename T>
void launch_attn_softmax(T* vals,
                         const T* attn_mask,
                         int batch_size,
                         int heads,
                         int sequence_length,
                         sycl::queue* stream);

template <typename T>
void launch_transform_0213(T* output,
                           const T* vals,
                           int batch_size,
                           int seq_length,
                           int hidden_dim,
                           int heads,
                           sycl::queue* stream);

// Custom bias add
template <typename T>
void launch_bias_add_transform_0213(T* outputs,
                                    const T* vals,
                                    const T* bias,
                                    int batch_size,
                                    int seq_length,
                                    int hidden_dim,
                                    int heads,
                                    sycl::queue* stream,
                                    int trans_count);

// 4D transform [0, 1, 2, 3] -> [0, 2, 1, 3]
template <typename T>
void launch_transform4d_0213(T* out,
                             const T* in,
                             int batch_size,
                             int heads,
                             int seq_length,
                             int hidden_dim,
                             sycl::queue* stream,
                             int trans_count);

template <typename T>
void launch_dropout(T* vals,
                    const T* bias,
                    uint8_t* mask,
                    int batch,
                    int dim,
                    float ratio,
                    sycl::queue* stream);

template <typename T>
void launch_dropout(T* vals_out,
                    const T* vals,
                    uint8_t* mask,
                    int total_count,
                    int dim,
                    float ratio,
                    sycl::queue* stream,
                    bool bwd = false);

template <typename T>
void launch_dropout(T* out,
                    const T* vals,
                    const T* residual,
                    const T* bias,
                    uint8_t* mask,
                    int batch,
                    int dim,
                    float ratio,
                    sycl::queue* stream);

template <typename T>
void launch_dropout_grad(T* vals, uint8_t* mask, int total_count, float ratio, sycl::queue* stream);

template <typename T>
void launch_dropout_grad(T* vals_out,
                         const T* vals,
                         uint8_t* mask,
                         int total_count,
                         float ratio,
                         sycl::queue* stream);

template <typename T>
void launch_fuse_transpose_bias_kernel(const T* inp,
                                       T* out,
                                       int rows,
                                       int cols,
                                       sycl::queue* stream);

void launch_param_update(const float* input, sycl::half* output, int size, sycl::queue* stream);
