/*******************************************************************************
 * Copyright (c) 2022-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "flash_attn.hpp"
#include "flash_attn_fwd.hpp"

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
    const bool is_casual) {
  return false;
}
template <typename P>
static int flash_scaled_attn_bf16_fwd_run(
    sycl::queue& queue,
    typename P::arguments_t& args) {
  P kernel(args);
  sycl::nd_range<3> nd_range = P::utils::get_nd_range(kernel);

  auto cgf = ([&](sycl::handler& cgh) {
    cgh.parallel_for(nd_range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
      using namespace gpu::xetla;
      using namespace gpu::xetla::group;
      using namespace gpu::xetla::kernel;
      using namespace gpu::xetla::subgroup;

      xetla_exec_item<3> ei(item);
      kernel.run(ei);
    });
  });
  DPCPP_Q_SUBMIT(queue, cgf);
  // evt.wait();

  return 0;
}

#define FLASH_ATTN_FWD_INVOKE(                                               \
    hidden_size,                                                             \
    is_casual,                                                               \
    enable_mask,                                                             \
    dropout_enable,                                                          \
    dump_dropout_mask,                                                       \
    prefetch_q)                                                              \
  {                                                                          \
    using R = xpu::xetla::FLASH_ATTENTION_FWD_PARAM;                         \
    using param_dtype = R::param_dtype_default;                              \
    using param_debug = R::param_debug_t<float, dump_dropout_mask>;          \
    using param_impl = R::param_impl_t<                                      \
        is_casual,                                                           \
        enable_mask,                                                         \
        dropout_enable,                                                      \
        prefetch_q,                                                          \
        R::mat_buffer_type::local,                                           \
        R::mat_buffer_type::reg,                                             \
        32,                                                                  \
        param_debug>;                                                        \
    using flash_attn_fwd_h##hidden_size = R::tuning_parameter_t<             \
        param_dtype,                                                         \
        param_impl,                                                          \
        hidden_size,                                                         \
        128,                                                                 \
        128,                                                                 \
        32,                                                                  \
        32,                                                                  \
        128,                                                                 \
        32>;                                                                 \
    using P =                                                                \
        xpu::xetla::FLASH_ATTENTION_FWD_IMPL<flash_attn_fwd_h##hidden_size>; \
    using arguments_t = P::arguments_t;                                      \
    P::dtype_q* ptr_q =                                                      \
        reinterpret_cast<P::dtype_q*>(const_cast<void*>(q_ptr));             \
    P::dtype_k* ptr_k =                                                      \
        reinterpret_cast<P::dtype_k*>(const_cast<void*>(k_ptr));             \
    P::dtype_v* ptr_v =                                                      \
        reinterpret_cast<P::dtype_v*>(const_cast<void*>(v_ptr));             \
    P::dtype_o* ptr_o =                                                      \
        reinterpret_cast<P::dtype_o*>(const_cast<void*>(output));            \
    P::dtype_m* ptr_m =                                                      \
        reinterpret_cast<P::dtype_m*>(const_cast<void*>(softmax_workspace)); \
    P::dtype_b* ptr_b =                                                      \
        reinterpret_cast<P::dtype_b*>(const_cast<void*>(out_buffer));        \
    P::dtype_d* ptr_d =                                                      \
        reinterpret_cast<P::dtype_d*>(const_cast<void*>(drop_mask));         \
    arguments_t args(                                                        \
        Bs,                                                                  \
        Hn,                                                                  \
        Sl,                                                                  \
        Hs,                                                                  \
        hs_rsqrt_scale,                                                      \
        dropout_prob,                                                        \
        dropout_scale,                                                       \
        dropout_rand_seed,                                                   \
        ptr_q,                                                               \
        ptr_k,                                                               \
        ptr_v,                                                               \
        ptr_o,                                                               \
        ptr_m,                                                               \
        ptr_b,                                                               \
        ptr_d);                                                              \
    flash_scaled_attn_bf16_fwd_run<P>(queue, args);                          \
  }

#define FLASH_ATTN_FWD_IF(cond, cond_name, code) \
  {                                              \
    if (cond) {                                  \
      static constexpr bool cond_name = true;    \
      code                                       \
    } else {                                     \
      static constexpr bool cond_name = false;   \
      code                                       \
    }                                            \
  }

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
    const void* drop_mask, // for dtopout mask if has, use uint8_t as data type
    const float dropout_prob,
    const float dropout_scale, // dropout_scale = 1 / (1 - drop_p)
    const uint64_t dropout_rand_seed, // dropout random generator seed
    const bool is_casual, // Indicate whether do mask_fill before softmax
    const bool store_softmax_out) {
  bool ret = false;
  if (Hs == 128) {
    FLASH_ATTN_FWD_IF(
        is_casual,
        is_casual_flag,
        FLASH_ATTN_FWD_IF(
            is_casual,
            enable_mask_flag,
            FLASH_ATTN_FWD_IF(
                dropout_prob > 0.0f,
                dropout_enable_flag,
                FLASH_ATTN_FWD_IF(
                    drop_mask,
                    dump_dropout_mask_flag,
                    FLASH_ATTN_FWD_INVOKE(
                        128,
                        is_casual_flag,
                        enable_mask_flag,
                        dropout_enable_flag,
                        dump_dropout_mask_flag,
                        false)))));
    ret = true;
  } else if (Hs == 96) {
    FLASH_ATTN_FWD_IF(
        is_casual,
        is_casual_flag,
        FLASH_ATTN_FWD_IF(
            is_casual,
            enable_mask_flag,
            FLASH_ATTN_FWD_IF(
                dropout_prob > 0.0f,
                dropout_enable_flag,
                FLASH_ATTN_FWD_IF(
                    drop_mask,
                    dump_dropout_mask_flag,
                    FLASH_ATTN_FWD_INVOKE(
                        96,
                        is_casual_flag,
                        enable_mask_flag,
                        dropout_enable_flag,
                        dump_dropout_mask_flag,
                        false)))));
    ret = true;
  }
  return ret;
}

#undef FLASH_ATTN_FWD_IF
#undef FLASH_ATTN_FWD_INVOKE

} // namespace xetla
} // namespace xpu