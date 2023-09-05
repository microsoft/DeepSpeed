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

#pragma once
#include <limits>
#include "flash_attn_fwd_kernel.hpp"
#include "flash_attn_fwd_utils.hpp"

namespace xpu {
namespace xetla {

template <typename tuning_parameter_>
struct FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::program {
  static __XETLA_API KERNEL_FUNC void initialization(utils::work_item3& ei);

  static __XETLA_API KERNEL_FUNC void prefetch_Q_i(
      utils::work_item3& ei,
      uint32_t seq_len,
      dtype_q* ptr_q);

  static __XETLA_API KERNEL_FUNC void calculate_S_ij(
      utils::work_item3& ei,
      uint32_t seq_len,
      uint32_t T_r,
      param_S::mat_out_t& matS,
      dtype_q* ptr_q,
      dtype_k* ptr_k,
      float hs_rsqrt_scale);

  static __XETLA_API KERNEL_FUNC void calculate_mask(
      utils::work_item3& ei,
      uint32_t seq_len,
      param_S::mat_out_t& matM);

  static __XETLA_API KERNEL_FUNC void apply_mask(
      utils::work_item3& ei,
      uint32_t seq_len,
      param_S::mat_out_t& matS);

  static __XETLA_API KERNEL_FUNC void calculate_P_ij(
      utils::work_item3& ei,
      param_S::mat_out_t& matS,
      param_P::rowmax_t& m_tilde_vec,
      param_P::rowsum_t& l_tilde_vec);

  static __XETLA_API KERNEL_FUNC void apply_dropout(
      utils::work_item3& ei,
      uint32_t seq_len,
      uint32_t threshold,
      uint32_t T_r,
      uint32_t T_c,
      float dropout_scale,
      uint64_t rand_seed,
      param_P::mat_in_t& matP,
      dtype_d* ptr_d);

  static __XETLA_API KERNEL_FUNC void init_ml(
      utils::work_item3& ei,
      param_P::rowmax_t& m_vec,
      param_P::rowsum_t& l_vec);

  static __XETLA_API KERNEL_FUNC void load_ml(
      utils::work_item3& ei,
      dtype_m* ptr_m,
      param_P::rowmax_t& m_vec,
      param_P::rowsum_t& l_vec);

  static __XETLA_API KERNEL_FUNC void calculate_new_ml(
      utils::work_item3& ei,
      param_P::rowmax_t& m_tilde_diff_exp_vec,
      param_P::rowmax_t& m_new_vec,
      param_P::rowmax_t& m_tilde_vec,
      param_P::rowmax_t& m_vec,
      param_P::rowsum_t& l_new_vec,
      param_P::rowsum_t& l_tilde_vec,
      param_P::rowsum_t& l_vec);

  static __XETLA_API KERNEL_FUNC void calculate_PV(
      utils::work_item3& ei,
      uint32_t batch_size,
      uint32_t seq_len,
      uint32_t T_r,
      uint32_t T_c,
      dtype_b* ptr_b,
      param_O::mat_out_t& matO_new,
      param_P::mat_out_t& matP,
      dtype_v* ptr_v,
      param_P::rowmax_t& m_tilde_diff_exp_vec);

  static __XETLA_API KERNEL_FUNC void load_O(
      utils::work_item3& ei,
      param_O::mat_out_t& matO,
      param_P::rowmax_t& m_new_vec,
      param_P::rowmax_t& m_vec);

  static __XETLA_API KERNEL_FUNC void update_O(
      utils::work_item3& ei,
      param_O::mat_out_t& matO,
      param_O::mat_out_t& matO_new,
      param_P::rowsum_t& l_new_vec,
      param_P::rowsum_t& l_vec);

  static __XETLA_API KERNEL_FUNC void store_O(
      utils::work_item3& ei,
      param_O::mat_out_t& matO);

  static __XETLA_API KERNEL_FUNC void store_ml(
      utils::work_item3& ei,
      param_P::rowmax_t& m_vec,
      param_P::rowmax_t& m_new_vec,
      param_P::rowsum_t& l_vec,
      param_P::rowsum_t& l_new_vec);

  static __XETLA_API KERNEL_FUNC void store_P_ij(
      utils::work_item3& ei,
      uint32_t batch_size,
      uint32_t seq_len,
      uint32_t T_r,
      uint32_t T_c,
      dtype_b* ptr_b,
      param_P::mat_out_t& matP);

  static __XETLA_API KERNEL_FUNC void store_global_O(
      utils::work_item3& ei,
      uint32_t batch_dim,
      uint32_t head_num,
      uint32_t seq_len,
      dtype_o* ptr_o,
      param_O::mat_out_t& matO,
      param_P::rowsum_t& l_vec);

  static __XETLA_API KERNEL_FUNC void store_global_ml(
      utils::work_item3& ei,
      dtype_m* ptr_m,
      param_P::factor_tile_t& m_store,
      param_P::factor_payload_t& m_payload,
      param_P::factor_tile_t& l_store,
      param_P::factor_payload_t& l_payload);

  static __XETLA_API KERNEL_FUNC void setup_ml_store(
      utils::work_item3& ei,
      uint32_t batch_size,
      uint32_t seq_len,
      uint32_t T_r,
      uint32_t T_c,
      uint32_t B_r,
      uint32_t B_c,
      dtype_m* ptr_m,
      param_P::factor_payload_t& m_payload,
      param_P::factor_payload_t& l_payload);

  static __XETLA_API KERNEL_FUNC void causal_skip(
      utils::work_item3& ei,
      uint32_t batch_size,
      uint32_t seq_len,
      uint32_t T_r,
      uint32_t T_c,
      dtype_b* ptr_b,
      param_S::mat_out_t& matS,
      param_P::rowmax_t& m_tilde_vec,
      param_P::rowsum_t& l_tilde_vec);
};

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<
    tuning_parameter_>::program::initialization(utils::work_item3& ei) {
  static constexpr uint32_t barrier_count =
      param_S::barrier_count + param_P::barrier_count + param_O::barrier_count;
  if constexpr (param_S::fine_tuning::periodic_sync_interval != 0) {
    static_assert(
        param_S::barrier_count == param_S::n_x + param_S::n_y,
        "param_S::barrier_count mismatch");
  }
  if constexpr (param_O::fine_tuning::periodic_sync_interval != 0) {
    if constexpr (param_S::fine_tuning::periodic_sync_interval != 0) {
      static_assert(
          param_S::n_x == param_O::n_x,
          "when periodic_sync_interval is enabled, expect n_x to "
          "match");
      static_assert(
          param_S::n_y == param_O::n_y,
          "when periodic_sync_interval is enabled, expect n_x to "
          "match");
    }
  }
  static constexpr uint32_t slm_size =
      std::max({param_S::slm_size, param_P::slm_size, param_O::slm_size}) + 1;
  gpu::xetla::xetla_nbarrier_init<barrier_count>();
  gpu::xetla::xetla_local_init<slm_size>();
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::prefetch_Q_i(
        utils::work_item3& ei,
        uint32_t seq_len,
        dtype_q* ptr_q) {
  int batch_idx = ei.get_group(0);
  int start_m = batch_idx * seq_len + ei.get_group(1) * param_S::m_w;
  int start_n = ei.get_group(2) * param_S::n_w;
  int start_k = 0;
  uint32_t boundary_q_m =
      utils::cap_val(start_m + param_S::m_w, (batch_idx + 1) * seq_len);
  uint32_t boundary_q_n = utils::cap_val(start_n + param_S::n_w, H);

  using mem_desc_src_q = typename param_S::mem_desc_input_q;
  using mem_desc_dst_q = typename param_S::mem_desc_prefetch_q;
  mem_desc_src_q md_src_q(
      {ptr_q}, {boundary_q_n, boundary_q_m, H}, {start_k, start_m});
  mem_desc_dst_q md_dst_q(
      {param_S::prefetch_q_slm_base_addr}, {H, param_S::m_w, H}, {0, 0});

  using transfer_op_t = utils::template transfer_op_t<
      mem_desc_src_q,
      mem_desc_dst_q,
      typename param_S::brgemm_t>;

  transfer_op_t prefetch_op;
  typename transfer_op_t::arguments_t args(
      md_src_q, md_dst_q, param_S::split_k_cnt);
  typename transfer_op_t::work_group_t g(ei.get_local_linear_id());
  prefetch_op(g, args, param_S::brgemm_slm_base_addr, param_S::barrier_offset);

  gpu::xetla::xetla_fence<gpu::xetla::memory_kind::shared_local>();
  typename param_P::store_nbarrier_t nbarrier;
  nbarrier.init_nbarrier(
      param_P::local_store_barrier_offset,
      gpu::xetla::nbarrier_role::producer_consumer);
  nbarrier.arrive_wait();
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::calculate_S_ij(
        utils::work_item3& ei,
        uint32_t seq_len,
        uint32_t T_r,
        param_S::mat_out_t& matS,
        dtype_q* ptr_q,
        dtype_k* ptr_k,
        float hs_rsqrt_scale) {
  // S_ij = Q_i @ K_j.T
  // where:
  //   - shape(Q_i)   = [B_r, H]
  //   - shape(K_j.T) = [H, B_c]
  //   - shape(S_ij)  = [B_r, B_c]
  int batch_idx = ei.get_group(0);
  int start_q_m =
      batch_idx * T_r * param_S::m_w + ei.get_group(1) * param_S::m_w;
  int start_q_k = 0;
  int start_k_k = start_q_k;
  int start_k_n = batch_idx * seq_len + ei.get_group(2) * param_S::n_w;
  uint32_t boundary_q_m = utils::cap_val(
      start_q_m + param_S::m_w, (batch_idx + 1) * T_r * param_S::m_w);
  uint32_t boundary_q_n = utils::cap_val(start_q_k + param_S::n_w, H);
  uint32_t boundary_k_m = utils::cap_val(start_k_k + H, H);
  uint32_t boundary_k_n =
      utils::cap_val(start_k_n + param_S::n_w, (batch_idx + 1) * seq_len);

  using mem_desc_q = std::conditional_t<
      prefetch_q,
      typename param_S::mem_desc_prefetch_q,
      typename param_S::mem_desc_input_q>;
  mem_desc_q md_q;
  if constexpr (prefetch_q) {
    md_q.init(
        {param_S::prefetch_q_slm_base_addr}, {H, param_S::m_w, H}, {0, 0});
  } else {
    md_q.init({ptr_q}, {boundary_q_n, boundary_q_m, H}, {start_q_k, start_q_m});
  }

  typename param_S::mem_desc_input_k md_k(
      {ptr_k}, {boundary_k_n, boundary_k_m, H}, {start_k_n, start_k_k});
  typename param_S::brgemm_t matmul_op;
  typename param_S::brgemm_t::arguments_t args(
      md_q, md_k, param_S::split_k_cnt);
  typename param_S::brgemm_t::work_group_t g(ei.get_local_linear_id());
  matS.init(0);
  matmul_op(
      g, matS, args, param_S::brgemm_slm_base_addr, param_S::barrier_offset);
  matS.reg *= hs_rsqrt_scale;
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::calculate_mask(
        utils::work_item3& ei,
        uint32_t seq_len,
        param_S::mat_out_t& matM) {
  matM.init(0.0f);
  // 1. attention mask
  // TODO: not implemented at the moment
  // 2. causal mask:
  // S[i_g, j_g] = -inf
  // where:
  //   - j_g > i_g
  //   - i_g, j_g ∈ {0..seq_len-1}
  // apply causal mask only on block S_ij := S<i_w, j_w> which contains S[k_g,
  // k_g] where:
  //   - i_w ∈ {0..T_r-1}
  //   - j_w ∈ {0..T_c-1}
  //   - k_g ∈ {0..seq_len-1}
  utils::template causal_mask<typename param_S::tile_shape>(ei, seq_len, matM);
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::apply_mask(
        utils::work_item3& ei,
        uint32_t seq_len,
        param_S::mat_out_t& matS) {
  typename param_S::mat_out_t matM;
  // TODO: currently supports causual mask only
  program::calculate_mask(ei, seq_len, matM);
  // apply mask
  matS.reg += matM.reg;
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::calculate_P_ij(
        utils::work_item3& ei,
        param_S::mat_out_t& matS,
        param_P::rowmax_t& m_tilde_vec,
        param_P::rowsum_t& l_tilde_vec) {
  using mat_t = param_S::mat_out_t;
  using mat_dtype = mat_t::dtype;
  using vec_dtype = param_P::rowmax_t::element_type;
  static constexpr int vec_length = param_P::rowmax_t::length;
  static_assert(
      std::is_same_v<vec_dtype, typename param_P::rowsum_t::element_type>,
      "dtype mismatch");
  static_assert(vec_length == param_P::rowsum_t::length, "vec_length mismatch");
  using mat_tile_shape = param_S::mat_tile_shape;
  using mat_worker_scope_t = mat_tile_shape::work_group_t;
  mat_worker_scope_t g(ei.get_local_linear_id());
  int32_t j_s = g.get_id() % mat_tile_shape::wg_size_x;
  int32_t i_s = g.get_id() / mat_tile_shape::wg_size_x;
  uint32_t nbarrier_id = param_P::reduce_barrier_offset + i_s;
  uint32_t slm_base_addr = param_P::reduce_slm_base_addr +
      i_s * param_P::reduce_list_xlength * param_P::rowmax_t::length *
          sizeof(vec_dtype);

  gpu::xetla::xetla_mask<param_P::rowmax_t::length> vec_mask;

  // 1. m_tilde_ij = rowmax(S_ij)
  // where:
  //   - shape(m_tilde_ij) = [B_r,]
  {
    using rowmax_t = param_P::rowmax_t;
    static constexpr auto reduce_op_t = gpu::xetla::reduce_op::max;
    rowmax_t m_tilde_vec_thread =
        gpu::xetla::subgroup::tile_reduce<reduce_op_t, mat_t, mat_dtype, 1>(
            matS);
    typename param_P::wg_reduce_max_t wg_reduce_max(
        j_s, nbarrier_id, slm_base_addr);
    m_tilde_vec = wg_reduce_max(m_tilde_vec_thread);
  }
  // 2. P_ij = exp(S_ij - m_tilde_ij)
  // where:
  //   - shape(P_ij) = [B_r, B_c]
  {
    using broadcast_op_t = gpu::xetla::subgroup::tile_minus;
    if constexpr (enable_mask) {
      // special handling for masking the whole row
      vec_mask = m_tilde_vec <= -utils::inf;
      m_tilde_vec.merge(utils::inf, vec_mask);
    }
    gpu::xetla::subgroup::tile_broadcast_op<broadcast_op_t, mat_t>(
        matS, m_tilde_vec);
    if constexpr (enable_mask) {
      // special handling for masking the whole row
      m_tilde_vec.merge(-utils::inf, vec_mask);
    }
    matS.reg = gpu::xetla::xetla_exp<mat_dtype>(matS.reg);
  }
  auto& matP = matS;
  // 3. l_tilde_ij = rowsum(P_ij)
  // where:
  //   - shape(l_tilde_ij) = [B_r,]
  {
    using rowsum_t = param_P::rowsum_t;
    static constexpr auto reduce_op_t = gpu::xetla::reduce_op::sum;
    rowsum_t l_tilde_vec_thread =
        gpu::xetla::subgroup::tile_reduce<reduce_op_t, mat_t, mat_dtype, 1>(
            matP);
    typename param_P::wg_reduce_sum_t wg_reduce_sum(
        j_s, nbarrier_id, slm_base_addr);
    l_tilde_vec = wg_reduce_sum(l_tilde_vec_thread);
  }
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::apply_dropout(
        utils::work_item3& ei,
        uint32_t seq_len,
        uint32_t T_r,
        uint32_t T_c,
        uint32_t threshold,
        float dropout_scale,
        uint64_t rand_seed,
        param_P::mat_in_t& matP,
        dtype_d* ptr_d) {
  int batch_idx = ei.get_group(0);
  int i_w = ei.get_group(1);
  int j_w = ei.get_group(2);

  static constexpr uint32_t rand_sz = param_S::m_s * param_S::n_s;
  gpu::xetla::dropout_fwd_t<
      rand_sz,
      uint8_t,
      tuning_parameter::fine_tune::dropout_random_simd>
      dropout_op;

  uint64_t offset = batch_idx;

  using mat_tile_shape = param_S::mat_tile_shape;
  using mat_worker_scope_t = mat_tile_shape::work_group_t;
  mat_worker_scope_t g(ei.get_local_linear_id());
  int32_t j_s = g.get_id() % mat_tile_shape::wg_size_x;
  int32_t i_s = g.get_id() / mat_tile_shape::wg_size_x;
  int coord_x = j_w * param_S::n_w + j_s * param_S::n_s;
  int coord_y = i_w * param_S::m_w + i_s * param_S::m_s;
  const uint64_t subseq = uint64_t(coord_y) << 32 | uint64_t(coord_x);
  dropout_op.init(rand_seed, subseq, offset, threshold, dropout_scale);

  matP.reg = dropout_op.template process<dtype_p>(matP.reg);

  if constexpr (dump_dropout_mask) {
    using matM_t = gpu::xetla::subgroup::
        tile_t<uint8_t, typename param_P::mat_in_t::tile_desc>;
    matM_t matM;
    matM.reg = dropout_op.get_mask();

    int start_m = batch_idx * seq_len + i_w * param_S::m_w;
    int start_n = j_w * param_S::n_w;

    using mem_desc_output_t = gpu::xetla::mem_desc_t<
        dtype_d,
        gpu::xetla::mem_layout::row_major,
        gpu::xetla::mem_space::global>;
    uint32_t boundary_m =
        utils::cap_val(start_m + param_S::m_w, (batch_idx + 1) * seq_len);
    uint32_t boundary_n = utils::cap_val(start_n + param_S::n_w, seq_len);
    mem_desc_output_t md_dst(
        {ptr_d}, {boundary_n, boundary_m, seq_len}, {start_n, start_m});
    using epilogue_t = gpu::xetla::group::epilogue_t<
        gpu::xetla::group::epilogue_policy_default<gpu::xetla::gpu_arch::Xe>,
        typename param_S::tile_shape,
        mem_desc_output_t>;
    epilogue_t epilogue;
    typename param_S::brgemm_t::work_group_t g(ei.get_local_linear_id());
    epilogue(g, matM, md_dst);
  }
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::init_ml(
        utils::work_item3& ei,
        param_P::rowmax_t& m_vec,
        param_P::rowsum_t& l_vec) {
  m_vec = -utils::inf;
  l_vec = 0.0f;
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::load_ml(
        utils::work_item3& ei,
        dtype_m* ptr_m,
        param_P::rowmax_t& m_vec,
        param_P::rowsum_t& l_vec) {
  // m_i, l_i is created and persists in register
  // storing the result to ptr_m is skipped to reduce memory read/write
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::calculate_new_ml(
        utils::work_item3& ei,
        param_P::rowmax_t& m_tilde_diff_exp_vec,
        param_P::rowmax_t& m_new_vec,
        param_P::rowmax_t& m_tilde_vec,
        param_P::rowmax_t& m_vec,
        param_P::rowsum_t& l_new_vec,
        param_P::rowsum_t& l_tilde_vec,
        param_P::rowsum_t& l_vec) {
  // 1. m_new_i = max(m_i, m_tilde_ij)
  // 2. l_new_i = sum(exp(m_i - m_new_i) * l_i, exp(m_tilde_ij - m_new_i) *
  // l_tilde_ij)
  // 3. m_tilde_diff_exp_vec = exp(m_tilde_ij - m_new_i)
  // where:
  //   - shape(m) = [B_r,]
  //   - shape(l) = [B_r,]
  static constexpr uint32_t block_elems = std::min({32, param_P::vec_length});
  using calculate_new_ml_op_t =
      utils::template calculate_new_ml_op_t<block_elems>;
  calculate_new_ml_op_t max_sum_op;
  max_sum_op(
      m_tilde_diff_exp_vec,
      m_new_vec,
      m_tilde_vec,
      m_vec,
      l_new_vec,
      l_tilde_vec,
      l_vec);
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::calculate_PV(
        utils::work_item3& ei,
        uint32_t batch_size,
        uint32_t seq_len,
        uint32_t T_r,
        uint32_t T_c,
        dtype_b* ptr_b,
        param_O::mat_out_t& matO_new,
        param_P::mat_out_t& matP,
        dtype_v* ptr_v,
        param_P::rowmax_t& m_tilde_diff_exp_vec) {
  // 1. depending on pv_buffer flag, store P_ij to either:
  //   - global memory
  //   - shared local memory (slm)
  // where:
  //   - shape(P_ij) = [B_r, B_c]
  store_P_ij(ei, batch_size, seq_len, T_r, T_c, ptr_b, matP);
  // 2. O_tilde_i = P_ij @ V_j
  // where:
  //   - shape(V_j) = [B_c, H]
  //   - mem_space(P_ij) = global / slm
  //   - mem_space(V_j)  = global
  {
    int batch_idx = ei.get_group(0);
    int i_w = ei.get_group(1);
    int j_w = ei.get_group(2);
    int start_v_m = batch_idx * seq_len + j_w * param_S::m_w;
    int start_v_n = 0;
    uint32_t boundary_v_m =
        utils::cap_val(start_v_m + param_O::m_w, (batch_idx + 1) * seq_len);
    uint32_t boundary_v_n = utils::cap_val(start_v_n + param_O::n_w, H);
    typename param_O::mem_desc_input_p md_p;
    if constexpr (pv_buffer == mat_buffer_type::global) {
      // use global memory for storing P_ij
      int start_m = (batch_idx * T_r + i_w) * param_S::m_w;
      int start_n = 0;
      uint32_t boundary_p_m = utils::cap_val(
          start_m + param_S::m_w, (batch_idx + 1) * T_r * param_S::m_w);
      uint32_t boundary_p_n =
          utils::cap_val(start_n + param_S::n_w, param_S::n_w);
      md_p.init(
          {ptr_b},
          {boundary_p_n, boundary_p_m, param_S::n_w},
          {start_n, start_m});
    } else if constexpr (pv_buffer == mat_buffer_type::local) {
      // use shared local memory for storing P_ij
      using addr_t = std::conditional_t<
          pv_buffer == mat_buffer_type::global,
          dtype_p*,
          uint32_t>;
      addr_t slm_base_addr =
          reinterpret_cast<addr_t>(param_P::local_store_slm_base_addr);
      md_p.init(
          {slm_base_addr}, {param_S::n_w, param_S::m_w, param_S::n_w}, {0, 0});
    } else {
      // P_ij remains in register
      // TODO: to be implemented
      static_assert(
          pv_buffer != mat_buffer_type::reg, "pv_buffer == reg unimplemented");
    }
    typename param_O::mem_desc_input_v md_v(
        {ptr_v}, {boundary_v_n, boundary_v_m, H}, {start_v_n, start_v_m});
    typename param_O::brgemm_t matmul_op;
    typename param_O::brgemm_t::arguments_t args(
        md_p, md_v, param_O::split_k_cnt);
    typename param_O::brgemm_t::work_group_t g(ei.get_local_linear_id());
    matO_new.init(0);
    matmul_op(
        g,
        matO_new,
        args,
        param_O::brgemm_slm_base_addr,
        param_O::barrier_offset);
  }
  // 3. O_tilde_i_weighted = diag(exp(m_tilde_ij - m_new_i)) @ O_tilde_i
  {
    using row_mul_op_t = utils::row_mul_op_t;
    row_mul_op_t row_mul_op;
    row_mul_op(m_tilde_diff_exp_vec, matO_new);
  }
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::load_O(
        utils::work_item3& ei,
        param_O::mat_out_t& matO,
        param_P::rowmax_t& m_new_vec,
        param_P::rowmax_t& m_vec) {
  // O_i_weighted = diag(exp(m_i - m_new_i)) @ O_i
  using row_exp_mul_op_t = utils::row_exp_mul_op_t;
  row_exp_mul_op_t row_exp_mul_op;
  row_exp_mul_op(m_new_vec, m_vec, matO);
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::update_O(
        utils::work_item3& ei,
        param_O::mat_out_t& matO,
        param_O::mat_out_t& matO_new,
        param_P::rowsum_t& l_new_vec,
        param_P::rowsum_t& l_vec) {
  // 1. O_i_weighted_unnorm = diag(l_i) @ O_i_weighted
  if constexpr (disable_shortcut) {
    using row_mul_op_t = utils::row_mul_op_t;
    row_mul_op_t row_mul_op;
    row_mul_op(l_vec, matO);
  }
  // 2. O_i_new_weighted_unnorm = O_i_weighted_unnorm + O_tilde_i_weighted
  { matO.reg = matO.reg + matO_new.reg; }
  // 3. O_i_new = inv(diag(l_new_i)) @ O_i_new_weighted_unnorm
  if constexpr (disable_shortcut) {
    using row_div_op_t = utils::row_div_op_t;
    row_div_op_t row_div_op;
    row_div_op(l_new_vec, matO);
  }
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::store_O(utils::work_item3& ei, param_O::mat_out_t& matO) {
  // O_i is created and persists in register
  // storing the result to shared local memory is skipped to reduce memory
  // read/write
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::store_ml(
        utils::work_item3& ei,
        param_P::rowmax_t& m_vec,
        param_P::rowmax_t& m_new_vec,
        param_P::rowsum_t& l_vec,
        param_P::rowsum_t& l_new_vec) {
  // m_i, l_i are stored in register instead of shared local memory
  // 1. m_i = m_new_i
  m_vec = m_new_vec;
  // 2. l_i = l_new_i
  l_vec = l_new_vec;
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::store_P_ij(
        utils::work_item3& ei,
        uint32_t batch_size,
        uint32_t seq_len,
        uint32_t T_r,
        uint32_t T_c,
        dtype_b* ptr_b,
        param_P::mat_out_t& matP) {
  using mat_tile_shape = param_P::mat_tile_shape;
  using mat_worker_scope_t = mat_tile_shape::work_group_t;
  mat_worker_scope_t g(ei.get_local_linear_id());
  typename param_P::mem_desc_output_p md_p;

  if constexpr (pv_buffer == mat_buffer_type::global) {
    // store P_ij from register to global memory
    int batch_idx = ei.get_group(0);
    int i_w = ei.get_group(1);
    int j_w = ei.get_group(2);
    int start_m = (batch_idx * T_r + i_w) * param_S::m_w;
    int start_n = 0;
    uint32_t boundary_p_m = utils::cap_val(
        start_m + param_S::m_w, (batch_idx + 1) * T_r * param_S::m_w);
    uint32_t boundary_p_n =
        utils::cap_val(start_n + param_S::n_w, param_S::n_w);
    md_p.init(
        {ptr_b},
        {boundary_p_n, boundary_p_m, param_S::n_w},
        {start_n, start_m});
  } else if constexpr (pv_buffer == mat_buffer_type::local) {
    // store P_ij from register to shared local memory
    using addr_t = std::
        conditional_t<pv_buffer == mat_buffer_type::global, dtype_p*, uint32_t>;
    addr_t slm_base_addr =
        reinterpret_cast<addr_t>(param_P::local_store_slm_base_addr);
    md_p.init(
        {slm_base_addr}, {param_S::n_w, param_S::m_w, param_S::n_w}, {0, 0});
  } else {
    // P_ij remains in register
    return;
  }
  typename param_P::epilogue_t epilogue;
  epilogue(g, matP, md_p);

  if constexpr (pv_buffer == mat_buffer_type::global) {
    gpu::xetla::xetla_fence<gpu::xetla::memory_kind::untyped_global>();
  } else if constexpr (pv_buffer == mat_buffer_type::local) {
    gpu::xetla::xetla_fence<gpu::xetla::memory_kind::shared_local>();
  }
  typename param_P::store_nbarrier_t nbarrier;
  nbarrier.init_nbarrier(
      param_P::local_store_barrier_offset,
      gpu::xetla::nbarrier_role::producer_consumer);
  nbarrier.arrive_wait();
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::store_global_O(
        utils::work_item3& ei,
        uint32_t batch_dim,
        uint32_t head_num,
        uint32_t seq_len,
        dtype_o* ptr_o,
        param_O::mat_out_t& matO,
        param_P::rowsum_t& l_vec) {
  // const uint32_t B = batch_dim;
  // const uint32_t N = head_num;
  // const uint32_t F = seq_len;
  // static constexpr uint32_t tile_size_y = param_O::m_s;
  // static constexpr uint32_t tile_size_x = param_O::n_s;

  // gpu::xetla::xetla_tdescriptor transpose_store_tdecs;
  // // Define a temprary vector as output buffer
  // gpu::xetla::xetla_vector<dtype_o, tile_size_x> out_reg;

  // param_O::brgemm_t::work_group_t g(ei.get_local_linear_id());
  // // Calculate new coordination of each element
  // uint32_t b = ei.get_group(2) / N; // dim0 coord
  // uint32_t n = ei.get_group(2) % N; // dim1 coord
  // uint32_t start_m = ei.get_group(0) * param_O::m_w;
  // uint32_t start_n = ei.get_group(1) * param_O::n_w;
  // uint32_t f
  //         = start_m + param_O::brgemm_t::get_matC_offset_y(g); // dim2 coord
  // uint32_t h
  //         = start_n + param_O::brgemm_t::get_matC_offset_x(g); // dim3 coord

  // // transpose 8 * 16 tile and store to global

  // for (uint32_t j = 0; j < tile_size_y; ++j, ++f) {
  //     // TODO: vairiable align with FLASH_ATTENTION_FWD_IMPL
  //     uint32_t dst_offset = b * N * F * H + n * F * H + f * H + h;
  //     out_reg = matO.reg.xetla_select<tile_size_x, 1>(j * tile_size_x);
  //     gpu::xetla::xetla_fill_tdesc<dtype_o, tile_size_x /*, 1, 1*/>(
  //             transpose_store_tdecs.xetla_format<uint32_t>(),
  //             ptr_o + dst_offset, H, 1, H, h, 0);
  //     gpu::xetla::xetla_tstore_global<dtype_o, tile_size_x,
  //             gpu::xetla::cache_hint::write_back,
  //             gpu::xetla::cache_hint::write_back>(
  //             transpose_store_tdecs, out_reg);
  // }

  // O_i = inv(diag(l_i)) @ O_i_unnorm
  if constexpr (!disable_shortcut) {
    using row_div_op_t = utils::row_div_op_t;
    row_div_op_t row_div_op;
    row_div_op(l_vec, matO);
  }
  // store O_i untransposed
  int batch_idx = ei.get_group(0);
  int start_m = batch_idx * seq_len + ei.get_group(1) * param_O::m_w;
  int start_n = 0;
  using mem_desc_output_t = gpu::xetla::mem_desc_t<
      dtype_o,
      gpu::xetla::mem_layout::row_major,
      gpu::xetla::mem_space::global>;
  uint32_t boundary_m =
      utils::cap_val(start_m + param_O::m_w, (batch_idx + 1) * seq_len);
  uint32_t boundary_n = utils::cap_val(start_n + param_O::n_w, H);
  mem_desc_output_t md_dst(
      {ptr_o}, {boundary_n, boundary_m, H}, {start_n, start_m});
  using epilogue_t = gpu::xetla::group::epilogue_t<
      gpu::xetla::group::epilogue_policy_default<gpu::xetla::gpu_arch::Xe>,
      typename param_O::tile_shape,
      mem_desc_output_t>;
  epilogue_t epilogue;
  typename param_O::brgemm_t::work_group_t g(ei.get_local_linear_id());
  epilogue(g, matO, md_dst);
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::store_global_ml(
        utils::work_item3& ei,
        dtype_m* ptr_m,
        param_P::factor_tile_t& m_store,
        param_P::factor_payload_t& m_payload,
        param_P::factor_tile_t& l_store,
        param_P::factor_payload_t& l_payload) {
  using mat_tile_shape = param_S::mat_tile_shape;
  using mat_worker_scope_t = mat_tile_shape::work_group_t;
  mat_worker_scope_t g(ei.get_local_linear_id());
  int32_t j_s = g.get_id() % mat_tile_shape::wg_size_x;
  int32_t i_s = g.get_id() / mat_tile_shape::wg_size_x;
  // store m, l once, threads with coordinate j_s != 0 can skip
  if (j_s != 0) {
    return;
  }
  // store m_i, l_i to global memory
  gpu::xetla::subgroup::tile_store<gpu::xetla::cache_hint::uncached>(
      m_store, m_payload);
  gpu::xetla::subgroup::tile_store<gpu::xetla::cache_hint::uncached>(
      l_store, l_payload);
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::setup_ml_store(
        utils::work_item3& ei,
        uint32_t batch_size,
        uint32_t seq_len,
        uint32_t T_r,
        uint32_t T_c,
        uint32_t B_r,
        uint32_t B_c,
        dtype_m* ptr_m,
        param_P::factor_payload_t& m_payload,
        param_P::factor_payload_t& l_payload) {
  // M[i] = m_i
  // L[i] = l_i
  // where:
  //   - shape(M) = [T_r, B_r]
  //   - mem_space(M) = global
  //   - mem_location(M) = {base_address: ptr_m, offset: batch_idx * factor_size
  //   - shape(L) = [T_r, B_r]
  //   - mem_space(L) = global
  //   - mem_location(L) = {base_address: ptr_m, offset: batch_idx * factor_size
  //   + T_r * B_r}

  int batch_idx = ei.get_group(0);
  int i_w = ei.get_group(1);

  using mat_t = param_S::mat_out_t;
  using mat_dtype = mat_t::dtype;
  using mat_tile_shape = param_S::mat_tile_shape;
  using mat_worker_scope_t = mat_tile_shape::work_group_t;
  mat_worker_scope_t g(ei.get_local_linear_id());
  int32_t i_s = g.get_id() / mat_tile_shape::wg_size_x;
  static constexpr uint32_t vec_length = param_P::vec_length;

  {
    typename param_P::mem_desc_output_m md_m;
    int start_m = batch_idx * 2;
    int start_n = i_w * B_r + i_s * vec_length;
    uint32_t boundary_m = utils::cap_val(start_m, batch_idx * 2 + 1);
    uint32_t boundary_n = utils::cap_val(start_n + B_r, seq_len);
    md_m.init({ptr_m}, {boundary_n, boundary_m, seq_len}, {start_n, start_m});
    m_payload.init(md_m);
  }
  {
    typename param_P::mem_desc_output_m md_l;
    int start_m = batch_idx * 2 + 1;
    int start_n = i_w * B_r + i_s * vec_length;
    uint32_t boundary_m = utils::cap_val(start_m, (batch_idx + 1) * 2);
    uint32_t boundary_n = utils::cap_val(start_n + B_r, seq_len);
    md_l.init({ptr_m}, {boundary_n, boundary_m, seq_len}, {start_n, start_m});
    l_payload.init(md_l);
  }
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    program::causal_skip(
        utils::work_item3& ei,
        uint32_t batch_size,
        uint32_t seq_len,
        uint32_t T_r,
        uint32_t T_c,
        dtype_b* ptr_b,
        param_S::mat_out_t& matS,
        param_P::rowmax_t& m_tilde_vec,
        param_P::rowsum_t& l_tilde_vec) {
  // skip calculating S_ij and skip m_i, l_i, O_i update
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::run(
    gpu::xetla::xetla_exec_item<3>& ei) const {
  auto uei = utils::make_work_item(ei);

  program::initialization(uei);
  typename param_P::factor_payload_t m_payload;
  typename param_P::factor_payload_t l_payload;
  program::setup_ml_store(
      uei,
      batch_size,
      seq_len,
      T_r,
      T_c,
      B_r,
      B_c,
      ptr_m,
      m_payload,
      l_payload);

  typename param_O::mat_out_t matO;
  matO.init(0.0f);

  typename param_P::factor_tile_t m_store;
  typename param_P::factor_tile_t l_store;
  typename param_P::rowmax_t& m_vec = m_store.reg;
  typename param_P::rowsum_t& l_vec = l_store.reg;
  program::init_ml(uei, m_vec, l_vec);

  if constexpr (prefetch_q) {
    program::prefetch_Q_i(uei, seq_len, ptr_q);
  }

  for (int j_w = 0; j_w < T_c; ++j_w) {
    decltype(matO) matO_new;

    typename param_P::rowmax_t m_new_vec;
    typename param_P::rowsum_t l_new_vec;

    uei.update_group_coord(2, j_w);
    {
      typename param_S::mat_out_t matS;
      typename param_P::rowmax_t m_tilde_vec;
      typename param_P::rowsum_t l_tilde_vec;
      typename param_P::rowmax_t m_tilde_diff_exp_vec;
      if constexpr (is_causal) {
        // TODO: currently supports causal mask only
        int t = utils::template check_diag_intersection<
            typename param_S::tile_shape>(uei);
        if (t > 0) {
          // causal mask applied, but S_ij is in upper triangle
          program::causal_skip(
              uei,
              batch_size,
              seq_len,
              T_r,
              T_c,
              ptr_b,
              matS,
              m_tilde_vec,
              l_tilde_vec);
          continue;
        }
      }

      program::calculate_S_ij(
          uei, seq_len, T_r, matS, ptr_q, ptr_k, hs_rsqrt_scale);

      if constexpr (enable_mask) {
        program::apply_mask(uei, seq_len, matS);
      }

      program::calculate_P_ij(uei, matS, m_tilde_vec, l_tilde_vec);
      auto& matP = matS;
      if constexpr (enable_dropout) {
        program::apply_dropout(
            uei,
            seq_len,
            T_r,
            T_c,
            dropout_threshold,
            dropout_scale,
            dropout_rand_seed,
            matP,
            ptr_d);
      }

      program::load_ml(uei, ptr_m, m_vec, l_vec);

      program::calculate_new_ml(
          uei,
          m_tilde_diff_exp_vec,
          m_new_vec,
          m_tilde_vec,
          m_vec,
          l_new_vec,
          l_tilde_vec,
          l_vec);

      program::calculate_PV(
          uei,
          batch_size,
          seq_len,
          T_r,
          T_c,
          ptr_b,
          matO_new,
          matP,
          ptr_v,
          m_tilde_diff_exp_vec);
    }

    program::load_O(uei, matO, m_new_vec, m_vec);

    program::update_O(uei, matO, matO_new, l_new_vec, l_vec);
    program::store_O(uei, matO);

    program::store_ml(uei, m_vec, m_new_vec, l_vec, l_new_vec);
  }
  program::store_global_O(
      uei, batch_dim, head_num, seq_len, ptr_o, matO, l_vec);
  program::store_global_ml(uei, ptr_m, m_store, m_payload, l_store, l_payload);
}

} // namespace xetla
} // namespace xpu