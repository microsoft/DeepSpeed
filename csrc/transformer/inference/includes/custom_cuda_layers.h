#pragma once

#ifdef __HIP_PLATFORM_HCC__
#define HALF_PRECISION_AVAILABLE = 1
#include <hip/hip_cooperative_groups.h>
#else
#if __CUDA_ARCH__ >= 700
#define HALF_PRECISION_AVAILABLE = 1
#endif
#include <cooperative_groups.h>
#endif

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <iostream>

#define MAX_THREADS 1024
#define MAX_WARP_NUM 32
#define WARP_SIZE 32
#define SMs 80

#define MAX_REGISTERS 256
template <typename T>
void launch_attn_softmax_v2(T* vals,
                            T* mask,
                            bool triangular,
                            bool recompute,
                            bool local_attention,
                            int window_size,
                            int batch_size,
                            int heads,
                            int num_seq,
                            int sequence_length,
                            float scale,
                            cudaStream_t stream);

// Fused bias add with gelu activation
template <typename T>
void launch_bias_gelu(T* input,
                      const T* bias,
                      int intermediate_size,
                      int batch_size,
                      cudaStream_t stream);
template <typename T>
void launch_bias_add(T* input, const T* bias, int hidden_size, int batch_size, cudaStream_t stream);

template <typename T>
void launch_bias_residual(T* input,
                          const T* residual,
                          const T* bias,
                          int size,
                          int intermediate_size,
                          bool add_bias,
                          cudaStream_t stream);

template <typename T>
void launch_layer_norm(T* out,
                       T* vals,
                       const T* gamma,
                       const T* beta,
                       float epsilon,
                       int batch_size,
                       int hidden_dim,
                       cudaStream_t stream);

template <typename T>
void launch_residual_layer_norm(T* norm,
                                T* res_add,
                                T* vals,
                                T* residual,
                                const T* bias,
                                const T* gamma,
                                const T* beta,
                                float epsilon,
                                int batch_size,
                                int hidden_dim,
                                bool preLN,
                                cudaStream_t stream);
template <typename T>
void launch_dequantize(T* output,
                       const int8_t* input,
                       const float* qscale,
                       unsigned output_size,
                       unsigned hidden_dim,
                       unsigned groups,
                       unsigned merge_count,
                       cudaStream_t stream);

template <typename T>
void launch_gptj_residual_add(T* input,
                              T* output,
                              T* attn,
                              T* bias,
                              int batch,
                              int head_size,
                              cudaStream_t stream);
template <typename T>
void launch_apply_rotary_pos_emb(T* mixed_query,
                                 T* key_layer,
                                 unsigned head_size,
                                 unsigned seq_len,
                                 unsigned rotary_dim,
                                 unsigned offset,
                                 unsigned num_heads,
                                 unsigned batch,
                                 cudaStream_t stream);
template <typename T>
void launch_apply_rotary_pos_emb1(T* query_key_value,
                                 unsigned head_size,
                                 unsigned seq_len,
                                 unsigned rotary_dim,
                                 unsigned offset,
                                 unsigned num_heads,
                                 unsigned batch,
                                 cudaStream_t stream);

template <typename T>
void launch_moe_res_matmul(T* residual,
                           T* coef,
                           T* mlp_out,
                           int seq_len,
                           int hidden_dim,
                           cudaStream_t stream);
template<typename T>
void launch_transform_scale(T* vals,
                                   T* query,
                                   T* kv_cache,
                                   int batch_size,
                                   int seq_length,
                                   unsigned cur_tokens,
                                   size_t value_offset,
                                   unsigned hidden_dim,
                                   int heads,
                                   cudaStream_t stream,
                                   int trans_count,
                                   float norm_factor);
template <typename T>
void launch_transform4d_0213(T* out,
                             const T* in,
                             unsigned batch_size,
                             int heads,
                             int seq_length,
                             int hidden_dim,
                             cudaStream_t stream,
                             int trans_count);