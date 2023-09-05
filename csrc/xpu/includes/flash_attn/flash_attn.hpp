#pragma once

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <stddef.h>
#include <ipex.h>
#include <torch/extension.h>

#define DPCPP_Q_SUBMIT(q, cgf, ...)                                          \
  {                                                                          \
    auto e = (q).submit((cgf), ##__VA_ARGS__);                               \
    (q).throw_asynchronous();                                                \
    xpu::profiler_record("dpcpp_kernel", e);                                 \                                            
  }

namespace xpu {
namespace xetla {

bool flash_scaled_attn_bf16_inf(
    sycl::queue& queue,
    // void* output, // pointer to output buffer, [Bs, Hn, Sl, Hs] ==> [Bs, Sl,
    // Hn, Hs] ==> [Sl, Bs, Hn, Hs]
    void* output, // pointer to output buffer, [Bs, Hn, Sl, Hs]
    void* out_buffer, // [Bs, Hn, Sl, Hs]
    const uint32_t Bs, // Batch sizes,
    const uint32_t Hn, // number of heads
    const uint32_t Sl, // sequence length, current assume Sq == Sk, fixed Sl
                       // for current Batches
    const uint32_t Hs, // head sizes
    const float hs_rsqrt_scale, // hs_rsqrt_scale = 1 / sqrt(hs)
    const void* q_ptr, // pointer to Q data buffer, [Bs, Hn, Sl, Hs]
    const void* k_ptr, // pointer to K data buffer, [Bs, Hn, Sl, Hs]
    const void* v_ptr, // pointer to V data buffer, [Bs, Hn, Sl, Hs]
    const bool is_casual =
        true); // Indicate whether do mask_fill before softmax

bool flash_scaled_attn_bf16_fwd(
    sycl::queue& queue,
    void* output, // pointer to output buffer, [Bs, Hn, Sl, Hs], will consider
                  // permute later
    void* out_buffer, // [Bs, Hn, Sl, Hs]
    void* softmax_workspace, // if store_sofmax_out is true,
                             //   it's pointer to softmax output buffer, sizes
                             //   are [Bs, Hn, Sl, Sl]
                             // if store_sofmax_out is false,
                             //   it's pointer to softmax row_max and row_sum
                             //   buffer, sizes are [Bs*Hn, 2, Sl], row_max is
                             //   stored at [Bs*Hn, 0, Sl], row_sum is stored at
                             //   [Bs*Hn, 1, Sl]
    const uint32_t Bs, // Batch sizes,
    const uint32_t Hn, // number of heads
    const uint32_t Sl, // sequence length, current assume Sq == Sk, fixed Sl
                       // for current Batches
    const uint32_t Hs, // head sizes
    const float hs_rsqrt_scale, // hs_rsqrt_scale = 1 / sqrt(hs)
    const void* q_ptr, // pointer to Q data buffer, [Bs, Hn, Sl, Hs]
    const void* k_ptr, // pointer to K data buffer, [Bs, Hn, Sl, Hs]
    const void* v_ptr, // pointer to V data buffer, [Bs, Hn, Sl, Hs]
    const void* drop_mask =
        nullptr, // for dtopout mask if has, use uint8_t as data type
    const float dropout_prob = 0.0,
    const float dropout_scale = 1.0, // dropout_scale = 1 / (1 - drop_p)
    const uint64_t dropout_rand_seed = 0, // dropout random generator seed
    const bool is_casual = true, // Indicate whether do mask_fill before softmax
    const bool store_softmax_out =
        false); // Indicate whether output softmax result

bool flash_scaled_attn_bf16_bwd(
    sycl::queue& queue,
    void* dq, // gradient of Q, [Bs, Hn, Sl, Hs]
    void* dk, // gradient of K, [Bs, Hn, Sl, Hs]
    void* dv, // gradient of V, [Bs, Hn, Sl, Hs]
    void* grad_softmax, // temp buffer for grad_softmax output, [Bs, Hn, Sl, Sl]
    const void* out, // output, [Bs, Hn, Sl, Hs]
    const void*
        gradout, // gradient of output, has been permuted as [Bs, Hn, Sl, Hs]
    const uint32_t Bs, // saved Bs from forward
    const uint32_t Hn, // saved Hn from forward
    const uint32_t Sl, // saved Sl from forward
    const uint32_t Hs, // saved Hs from forward
    const float hs_rsqrt_scale, // saved hs_rsqrt_scale from forward
    const void* q_ptr, // saved Q input from forward
    const void* k_ptr, // saved K input from forward
    const void* v_ptr, // saved V input from forward
    const void* softmax_workspace_ptr, // saved softmax output or
                                       // row_max/row_sum from forward
    const void* drop_mask_ptr =
        nullptr, // may be saved drop_mask from forward or regenrated drop mask
                 // use uint8_t as data type
    const float dropout_prob =
        0.0, // dropout probility 0-1, if 0, there woukd be no dropout
    const float dropout_scale = 1.0,
    const uint64_t rand_seed = 0, // regenrated drop mask by same random seed
    const bool is_casual = true, // Indicate whether do mask_fill before softmax
    const bool softmax_out_saved =
        false); // Indicate whether softmax result has been saved and not need
                // to be re-computed

} // namespace xetla
} // namespace xpu

class FlashAttention {
public:
    virtual ~FlashAttention() {}
    
    bool Forward(sycl::queue &stream,
                 void* output,
                 void* out_buffer,
                 void* softmax_storespace,
                 const uint32_t Bs,
                 const uint32_t Hn,
                 const uint32_t Sl,
                 const uint32_t Hs,
                 const float hs_rsqrt_scale,
                 const void* q_ptr,
                 const void* k_ptr,
                 const void* v_ptr,
                 const void* drop_mask = nullptr,
                 const float dropout_prob = 0.0,
                 const float dropout_scale = 1.0,
                 const uint64_t dropout_rand_seed = 0,
                 const bool is_causal = true,
                 const bool store_softmax_out = false) {
        RECORD_FUNCTION("flash_scaled_attn_bf16_fwd", c10::ArrayRef<c10::IValue>({}));
        return xpu::xetla::flash_scaled_attn_bf16_fwd(
            stream,
            output,
            out_buffer,
            softmax_storespace,
            Bs,
            Hn,
            Sl,
            Hs,
            hs_rsqrt_scale,
            q_ptr,
            k_ptr,
            v_ptr,
            drop_mask,
            dropout_prob,
            dropout_scale,
            dropout_rand_seed,
            is_causal,
            store_softmax_out
        );
    }

    bool Backward(sycl::queue &stream,
                  void* dq,
                  void* dk,
                  void* dv,
                  void* grad_softmax,
                  const void* out,
                  const void* gradout,
                  const uint32_t Bs,
                  const uint32_t Hn,
                  const uint32_t Sl,
                  const uint32_t Hs,
                  const float hs_rsqrt_scale,
                  const void* q_ptr,
                  const void* k_ptr,
                  const void* v_ptr,
                  const void* softmax_workspace_ptr,
                  const void* drop_mask = nullptr,
                  const float dropout_prob = 0.0,
                  const float dropout_scale = 1.0,
                  const uint64_t dropout_rand_seed = 0,
                  const bool is_causal = true,
                  const bool store_softmax_out = false) {
        RECORD_FUNCTION("flash_scaled_attn_bf16_bwd", c10::ArrayRef<c10::IValue>({}));
        return xpu::xetla::flash_scaled_attn_bf16_bwd(
            stream,
            dq,
            dk,
            dv,
            grad_softmax,
            out,
            gradout,
            Bs,
            Hn,
            Sl,
            Hs,
            hs_rsqrt_scale,
            q_ptr,
            k_ptr,
            v_ptr,
            softmax_workspace_ptr,
            drop_mask,
            dropout_prob,
            dropout_scale,
            dropout_rand_seed,
            is_causal,
            store_softmax_out
        );
    }
};
