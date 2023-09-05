#pragma once
#include "compatible.hpp"

template <typename T>
void launch_attn_softmax_v2(T* vals,
                            T* mask,
                            T* alibi,
                            float layer_scale,
                            bool triangular,
                            bool recompute,
                            bool local_attention,
                            int window_size,
                            int batch_size,
                            int heads,
                            int num_seq,
                            int sequence_length,
                            int head_offset,
                            int mask_stride,
                            int mp_size,
                            sycl::queue stream);

// Fused bias add with gelu activation
template <typename T>
void launch_bias_gelu(T* input,
                      const T* bias,
                      int intermediate_size,
                      int batch_size,
                      sycl::queue stream);

template <typename T>
void launch_bias_add(T* input,
                     const T* bias,
                     int intermediate_size,
                     int batch_size,
                     sycl::queue stream);

template <typename T>
void launch_bias_residual(T* residual,
                          T* hidden_state,
                          T* attn,
                          T* bias,
                          T* attn_bias,
                          int batch,
                          int hidden_dim,
                          int mp_size,
                          bool preln,
                          sycl::queue stream);

template <typename T>
void launch_fused_ln(T* output,
                     const T* vals,
                     const T* gamma,
                     const T* beta,
                     float epsilon,
                     int rows,
                     int elems_per_row,
                     sycl::queue stream);

template <typename T>
void launch_fused_residual_ln(T* output,
                              const T* vals,
                              const T* residual,
                              const T* bias,
                              const T* gamma,
                              const T* beta,
                              float epsilon,
                              int rows,
                              int elems_per_row,
                              sycl::queue stream);

template <typename T>
void launch_vector_add(T* out,
                       const T* a,
                       const T* b,
                       float gamma,
                       int num_elems,
                       sycl::queue stream);
