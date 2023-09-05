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
#include "flash_attn_bwd_inner_loop.hpp"

template <
    typename T,
    typename out_T,
    typename acc_T,
    // typename mem_desc_brxbc_t_, typename mem_desc_brxd_t_, typename
    // mem_desc_bcxd_t_,
    typename gemm_brxd_block_tile_t,
    typename gemm_bcxd_block_tile_t,
    typename gemm_brxbc_block_tile_t,
    uint32_t accum_stride = 16,
    uint32_t prefetch_distance = 3,
    uint32_t periodic_sync_interval = 0>
struct flash_attention_bwd {
  using mem_desc_brxd_t =
      mem_desc_t<bf16, mem_layout::row_major, mem_space::global>;
  using mem_desc_bcxd_t =
      mem_desc_t<bf16, mem_layout::row_major, mem_space::global>;
  using mem_desc_brxbc_t =
      mem_desc_t<bf16, mem_layout::row_major, mem_space::local>;
  using mem_desc_l_m_t =
      mem_desc_t<acc_T, mem_layout::row_major, mem_space::global>;

  using fmha_bwd_inner_loop_t = fmha_block_t<
      T,
      out_T,
      float,
      mem_desc_brxbc_t,
      mem_desc_brxd_t,
      mem_desc_bcxd_t,
      mem_desc_l_m_t,
      gemm_brxbc_block_tile_t,
      gemm_brxd_block_tile_t,
      gemm_bcxd_block_tile_t>;
  using inner_loop_arguments = typename fmha_bwd_inner_loop_t::arguments_t;
  // using worker_scope_t = typename fmha_bwd_inner_loop_t::worker_scope_t;

  // transposed [brxbc] while store into slm
  // using mem_desc_brxbc_trans_t = mem_desc_t<bf16, mem_layout::row_major,
  // mem_space::local>;
  struct arguments_t : public inner_loop_arguments {
    arguments_t(
        T* ptr_q,
        T* ptr_k,
        T* ptr_v,
        T* ptr_o,
        acc_T* ptr_l,
        acc_T* ptr_m,
        T* ptr_dO,
        T* ptr_dQ,
        T* ptr_dK,
        T* ptr_dV,
        uint32_t seq_q,
        uint32_t seq_k,
        float scale,
        const float dropout_prob = 0,
        const float dropout_scale = 0,
        const uint64_t rand_seed = 67280421310721,
        uint32_t matP_base = 0,
        T* drop_mask_ptr = nullptr)
        : inner_loop_arguments(
              ptr_q,
              ptr_k,
              ptr_v,
              ptr_o,
              ptr_l,
              ptr_m,
              ptr_dO,
              ptr_dQ,
              ptr_dK,
              ptr_dV,
              seq_q,
              seq_k,
              scale,
              dropout_prob,
              dropout_scale,
              rand_seed,
              matP_base,
              drop_mask_ptr){};
  };

  __XETLA_API KERNEL_FUNC void operator()(
      xetla_exec_item<3> ei,
      arguments_t& args) {
    fmha_bwd_inner_loop_t fmha_inner_loop;
    int bc = gemm_brxbc_block_tile_t::blocked_N;
    xetla_nbarrier_init<
        gemm_brxbc_block_tile_t::tile_shape_t::wg_size_x +
        gemm_brxd_block_tile_t::tile_shape_t::wg_size_y +
        gemm_brxbc_block_tile_t::tile_shape_t::wg_size_y>();
    int max_loop_steps = (args.seq_k + bc - 1) / bc;
    for (int loop_idx = 0; loop_idx < max_loop_steps; loop_idx++) {
      fmha_inner_loop(ei, args, loop_idx);
    }
  }
};