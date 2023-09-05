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
#include <cmath>
#include "xetla.hpp"

namespace gpu::xetla::group {

template <
    typename epilogue_policy,
    typename tile_shape_,
    typename mem_desc_c_t_>
class epilogue_transp_t {};
template <
    typename tile_op_t_,
    typename update_method_,
    typename tile_shape_,
    typename mem_desc_c_t_>
class epilogue_transp_t<
    epilogue_policy_tile_op<tile_op_t_, update_method_, gpu_arch::Xe>,
    tile_shape_,
    mem_desc_c_t_> {
 public:
  using update_method = result_overwrite;
  using tile_shape = tile_shape_;
  using mem_desc_c_t = mem_desc_c_t_;
  static constexpr gpu_arch arch_tag = gpu_arch::Xe;
  static constexpr uint32_t barrier_count = 0;
  static constexpr uint32_t slm_size = 0;
  /// @brief Epilogue arguments.
  struct arguments_t {};

 private:
  using work_group_t = typename tile_shape::work_group_t;
  static constexpr uint32_t wg_tile_m = tile_shape::wg_tile_size_y;
  static constexpr uint32_t wg_tile_n = tile_shape::wg_tile_size_x;
  static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
  static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
  static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
  static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
  using dtype_c = typename mem_desc_c_t::dtype;
  static constexpr mem_layout mem_layout_c = mem_desc_c_t::layout;
  static constexpr mem_space mem_space_c = mem_desc_c_t::space;
  static constexpr msg_type msg_type_c =
      (mem_space_c == mem_space::global ? msg_type::block_2d
                                        : msg_type::scatter);
  /// @brief Updates tile base descriptor based on the tid.
  __XETLA_API static void update_sg_tile_tdesc(
      work_group_t& g,
      mem_desc_c_t& mem_desc_c) {
    int32_t sg_idy = g.get_id() % wg_size_x;
    int32_t sg_idx = g.get_id() / wg_size_x;
    int32_t tile_offset_n = sg_idx * sg_tile_m;
    int32_t tile_offset_m = sg_idy * sg_tile_n;
    mem_desc_c.update_coord(tile_offset_n, tile_offset_m);
  }

 public:
  /// @brief Default epilogue.
  /// 1) Convert dtype_acc to dtype_c 2) Overwrite to memory.
  /// @tparam matAcc_t Is the type of the input tile.
  /// @param g Is the workgroup of the current tile.
  /// @param matAcc Is the input tile.
  /// @param mem_desc_c Is the memory description of matC, including base, shape
  /// and coordinate.
  /// @param args Is the additional arguments for epilogue.
  /// @param slm_base Is the slm base address.
  /// @param nbarrier_base Is the named barrier base.
  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      work_group_t& g,
      matAcc_t& matAcc,
      mem_desc_c_t mem_desc_c,
      arguments_t args = {},
      uint32_t slm_base = 0,
      uint32_t nbarrier_base = 0) {
    static_assert(
        mem_layout_c == mem_layout::row_major &&
            mem_space_c == mem_space::local,
        "layout should be row_major and space should be local");
    using matC_tile_desc_t = subgroup::tile_desc_t<
        matAcc_t::tile_size_x,
        matAcc_t::tile_size_y,
        matAcc_t::block_size_x,
        matAcc_t::block_size_y,
        reg_layout::vnni_tiled_col_major>;

    using matC_t = subgroup::tile_t<dtype_c, matC_tile_desc_t>;
    using matC_payload_t = subgroup::mem_payload_t<
        dtype_c,
        matC_tile_desc_t,
        msg_type_c,
        mem_layout_c,
        mem_space_c,
        arch_tag>;

    update_sg_tile_tdesc(g, mem_desc_c);
    matC_payload_t matC_payload(mem_desc_c);
    matC_t matC;
    subgroup::vnni_transform(matC, matAcc);
    subgroup::tile_store(matC, matC_payload);
  }
};

template <
    // The number of rows in the gemm block.
    uint32_t M_,
    // The number of cols in the gemm block.
    uint32_t N_,
    // The number of elements in the the K dimension of the block gemm loop.
    uint32_t K_,
    // The number of rows of workgroup.
    uint32_t wg_tile_m_,
    // The number of cols of workgroup.
    uint32_t wg_tile_n_,
    // The number of rows of subgroup.
    uint32_t sg_tile_m_,
    // The number of cols of subgroup.
    uint32_t sg_tile_n_,
    // The number elements of subgroup in the K dimension of the block gemm
    // loop.
    uint32_t sg_tile_k_>
struct gemm_block_tile_t {
  static constexpr uint32_t blocked_M = M_, blocked_N = N_, blocked_K = K_;
  static constexpr uint32_t sg_tile_k = sg_tile_k_;
  static constexpr uint32_t inner_loop_count =
      (blocked_K + sg_tile_k - 1) / sg_tile_k;
  using tile_shape_t =
      tile_shape_t<wg_tile_n_, wg_tile_m_, sg_tile_n_, sg_tile_m_>;
};

template <
    typename tile_shape_t_,
    int br,
    int bc,
    typename matAcc_t,
    typename worker_scope_t,
    bool is_causal = true>
struct casual_mask {
  using tile_shape_t = tile_shape_t_;
  static constexpr float Inf = INFINITY;
  static constexpr float negative_Inf = Inf * -1;
  static constexpr uint32_t tg_size_x = tile_shape_t::wg_size_x;
  static constexpr uint32_t sg_tile_m = tile_shape_t::sg_tile_size_y;
  static constexpr uint32_t sg_tile_n = tile_shape_t::sg_tile_size_x;
  static constexpr uint32_t block_size_x = matAcc_t::tile_desc_t::block_size_x;
  static constexpr uint32_t block_size_y = matAcc_t::tile_desc_t::block_size_y;
  static constexpr uint32_t num_block_x = matAcc_t::tile_desc_t::num_block_x;
  static constexpr uint32_t num_block_y =
      matAcc_t::tile_desc_t::num_block / num_block_x;
  static constexpr uint32_t block_elems = matAcc_t::tile_desc_t::block_elems;
  static inline KERNEL_FUNC void apply_mask(
      worker_scope_t& g,
      matAcc_t& matAcc,
      const int m_idx,
      const int n_idx) {
    int32_t sg_idx = g.get_id() % tg_size_x;
    int32_t sg_idy = g.get_id() / tg_size_x;

    const int start_col = n_idx * bc + sg_idx * sg_tile_n;
    const int start_row = m_idx * br + sg_idy * sg_tile_m;

    if (start_row > start_col + sg_tile_n - 1) {
      return;
    }
#pragma unroll
    for (int ii = 0; ii < sg_tile_m; ii++) {
      int cur_row = start_row + ii;
#pragma unroll
      for (int jj = 0; jj < sg_tile_n; jj++) {
        int cur_col = start_col + jj;
        int cur_block_linear_id =
            jj / block_size_x + ii / block_size_y * num_block_x;
        int bj = jj % block_size_x, bi = ii % block_size_y;
        if (cur_row < cur_col) {
          // TODO: which kind of -inf should be adapt?
          // matAcc.reg[ii * sg_tile_n + jj] = negative_Inf; //-inf;
          matAcc
              .reg[bi * block_size_x + bj + cur_block_linear_id * block_elems] =
              negative_Inf;
        }
      }
    }
  }
};

}; // namespace gpu::xetla::group