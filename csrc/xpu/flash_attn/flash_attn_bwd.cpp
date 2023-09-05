
#include "flash_attn.hpp"
#include "flash_attn_bwd.hpp"

namespace xpu {
namespace xetla {

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
    const void* drop_mask_ptr, // may be saved drop_mask from forward or
                               // regenrated drop mask use uint8_t as data type
    const float
        dropout_prob, // dropout probility 0-1, if 0, there woukd be no dropout
    const float dropout_scale,
    const uint64_t rand_seed, // regenrated drop mask by same random seed
    const bool is_casual, // Indicate whether do mask_fill before softmax
    const bool softmax_out_saved) // Indicate whether softmax result has been
                                  // saved and not need to be re-computed
{
  if (Hs == 128 && is_casual && !softmax_out_saved) {
    return flash_attn_bwd<kernel_traits<bf16, bf16, float, 128, 128>>(
        queue,
        dq,
        dk,
        dv,
        grad_softmax,
        out,
        gradout,
        Bs,
        Hn,
        Sl,
        hs_rsqrt_scale,
        q_ptr,
        k_ptr,
        v_ptr,
        softmax_workspace_ptr,
        drop_mask_ptr,
        dropout_prob,
        dropout_scale,
        rand_seed);
  } else if (Hs == 96 && is_casual && !softmax_out_saved) {
    return flash_attn_bwd<kernel_traits<bf16, bf16, float, 96, 128>>(
        queue,
        dq,
        dk,
        dv,
        grad_softmax,
        out,
        gradout,
        Bs,
        Hn,
        Sl,
        hs_rsqrt_scale,
        q_ptr,
        k_ptr,
        v_ptr,
        softmax_workspace_ptr,
        drop_mask_ptr,
        dropout_prob,
        dropout_scale,
        rand_seed);
  } else {
    return false;
  }
}

} // namespace xetla
} // namespace xpu