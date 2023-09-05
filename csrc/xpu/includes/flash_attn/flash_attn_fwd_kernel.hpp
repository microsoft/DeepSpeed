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

#include <xetla.hpp>

namespace xpu {
namespace xetla {

class FLASH_ATTENTION_FWD_PARAM {
 public:
  enum class mat_buffer_type { global, local, reg };

  template <
      typename dtype_q_ = gpu::xetla::bf16,
      typename dtype_k_ = gpu::xetla::bf16,
      typename dtype_v_ = gpu::xetla::bf16,
      typename dtype_o_ = gpu::xetla::bf16,
      typename dtype_m_ = float,
      typename dtype_b_ = float,
      typename dtype_acc_ = float,
      typename dtype_s_ = float,
      typename dtype_p_ = float>
  struct param_dtype_t {
    using dtype_q = dtype_q_;
    using dtype_k = dtype_k_;
    using dtype_v = dtype_v_;
    using dtype_o = dtype_o_;
    using dtype_m = dtype_m_;
    using dtype_b = dtype_b_;

    using dtype_acc = dtype_acc_;
    using dtype_s = dtype_s_;
    using dtype_p = dtype_p_;
  };
  using param_dtype_default = param_dtype_t<>;

  template <
      typename dtype_d_ = float,
      bool dump_dropout_mask_ = false,
      bool disable_shortcut_ = false>
  struct param_debug_t {
    using dtype_d = dtype_d_;
    static constexpr bool dump_dropout_mask = dump_dropout_mask_;
    static constexpr bool disable_shortcut = disable_shortcut_;
  };
  using param_debug_default = param_debug_t<>;

  template <
      bool is_causal_ = true,
      bool enable_mask_ = true,
      bool enable_dropout_ = true,
      bool prefetch_q_ = true,
      mat_buffer_type pv_buffer_ = mat_buffer_type::local,
      mat_buffer_type o_buffer_ = mat_buffer_type::reg,
      int thread_num_ = 32,
      typename debugging_impl_ = param_debug_default>
  struct param_impl_t {
    static constexpr bool is_causal = is_causal_;
    static constexpr bool enable_dropout = enable_dropout_;

    static constexpr bool prefetch_q = prefetch_q_;

    static constexpr mat_buffer_type pv_buffer = pv_buffer_;
    static constexpr mat_buffer_type o_buffer = o_buffer_;

    static constexpr int thread_num = thread_num_;

    using debug_param = debugging_impl_;
  };
  using param_impl_default = param_impl_t<>;

  template <
      uint32_t matS_periodic_sync_interval_ = 8,
      uint32_t matS_prefetch_distance_ = 3,
      uint32_t matO_periodic_sync_interval_ = 8,
      uint32_t matO_prefetch_distance_ = 3,
      uint32_t dropout_random_simd_ = 16>
  struct param_fine_tune_t {
    static constexpr uint32_t matS_periodic_sync_interval =
        matS_periodic_sync_interval_;
    static constexpr uint32_t matS_prefetch_distance = matS_prefetch_distance_;
    static constexpr uint32_t matO_periodic_sync_interval =
        matO_periodic_sync_interval_;
    static constexpr uint32_t matO_prefetch_distance = matO_prefetch_distance_;
    static constexpr uint32_t dropout_random_simd = dropout_random_simd_;
  };
  using param_fine_tune_default = param_fine_tune_t<>;

  template <
      typename param_dtype_ = param_dtype_default,
      typename param_impl_ = param_impl_default,
      uint32_t H_ = 128,
      int B_r_ = 128,
      int B_c_ = 128,
      int matS_n_s_ = 64,
      int matS_k_s_ = 32,
      int matO_n_w_ = 128,
      int matO_k_s_ = 16,
      typename param_fine_tune_ = param_fine_tune_default>
  struct tuning_parameter_t {
    using dtype_q = param_dtype_::dtype_q;
    using dtype_k = param_dtype_::dtype_k;
    using dtype_v = param_dtype_::dtype_v;
    using dtype_o = param_dtype_::dtype_o;
    using dtype_m = param_dtype_::dtype_m;
    using dtype_b = param_dtype_::dtype_b;

    using dtype_acc = param_dtype_::dtype_acc;
    using dtype_s = param_dtype_::dtype_s;
    using dtype_p = param_dtype_::dtype_p;

    static constexpr bool is_causal = param_impl_::is_causal;
    static constexpr bool enable_mask = is_causal;
    static constexpr bool enable_dropout = param_impl_::enable_dropout;
    // use global memory ptr_b for storing P_ij
    // slm version ignores ptr_b
    // TODO: register version
    static constexpr mat_buffer_type pv_buffer = param_impl_::pv_buffer;
    static constexpr mat_buffer_type o_buffer = param_impl_::o_buffer;

    static constexpr bool prefetch_q = param_impl_::prefetch_q;

    static constexpr int thread_num = param_impl_::thread_num;

    // hidden size per head
    static constexpr uint32_t H = H_;
    static constexpr int B_r = B_r_;
    static constexpr int B_c = B_c_;

    static constexpr int matS_n_s = matS_n_s_;
    static constexpr int matS_k_s = matS_k_s_;

    static constexpr int matO_n_w = matO_n_w_;
    static constexpr int matO_k_s = matO_k_s_;

    using fine_tune = param_fine_tune_;
    using debug_param = param_impl_::debug_param;
  };
};

template <typename tuning_parameter_>
class FLASH_ATTENTION_FWD_IMPL {
 public:
  using tuning_parameter = tuning_parameter_;
  using debug_parameter = tuning_parameter::debug_param;

  using dtype_q = tuning_parameter::dtype_q;
  using dtype_k = tuning_parameter::dtype_k;
  using dtype_v = tuning_parameter::dtype_v;
  using dtype_o = tuning_parameter::dtype_o;
  using dtype_m = tuning_parameter::dtype_m;
  using dtype_b = tuning_parameter::dtype_b;
  using dtype_d = debug_parameter::dtype_d;

  using dtype_acc = tuning_parameter::dtype_acc;
  using dtype_s = tuning_parameter::dtype_s;
  using dtype_p = tuning_parameter::dtype_p;

  using mat_buffer_type = FLASH_ATTENTION_FWD_PARAM::mat_buffer_type;

 public:
  struct arguments_t {
    const uint32_t batch_dim;
    const uint32_t head_num;
    const uint32_t batch_size;
    const uint32_t sequence_length;
    const uint32_t head_dim;
    const float hs_rsqrt_scale;
    const float dropout_prob;
    const float dropout_scale;
    const uint64_t dropout_rand_seed;
    dtype_q* const ptr_q;
    dtype_k* const ptr_k;
    dtype_v* const ptr_v;
    dtype_o* const ptr_o;
    dtype_m* const ptr_m;
    dtype_b* const ptr_b;
    dtype_d* const ptr_d;

    arguments_t(
        uint32_t batch_dim_,
        const uint32_t head_num_,
        uint32_t sequence_length_,
        uint32_t head_dim_,
        float hs_rsqrt_scale_,
        float dropout_prob_,
        float dropout_scale_,
        uint64_t dropout_rand_seed_,
        dtype_q* ptr_q_,
        dtype_k* ptr_k_,
        dtype_v* ptr_v_,
        dtype_o* ptr_o_,
        dtype_m* ptr_m_,
        dtype_b* ptr_b_,
        dtype_d* ptr_d_)
        : batch_dim(batch_dim_),
          head_num(head_num_),
          batch_size(batch_dim_ * head_num_),
          sequence_length(sequence_length_),
          head_dim(head_dim_),
          hs_rsqrt_scale(hs_rsqrt_scale_),
          dropout_prob(dropout_prob_),
          dropout_scale(dropout_scale_),
          dropout_rand_seed(dropout_rand_seed_),
          ptr_q(ptr_q_),
          ptr_k(ptr_k_),
          ptr_v(ptr_v_),
          ptr_o(ptr_o_),
          ptr_m(ptr_m_),
          ptr_b(ptr_b_),
          ptr_d(ptr_d_){};
  };

  // max head dim
  static constexpr uint32_t H = tuning_parameter::H;
  static constexpr int thread_num = tuning_parameter::thread_num;
  static constexpr bool is_causal = tuning_parameter::is_causal;
  static constexpr bool enable_mask = tuning_parameter::enable_mask;
  static constexpr bool enable_dropout = tuning_parameter::enable_dropout;
  static constexpr bool dump_dropout_mask = debug_parameter::dump_dropout_mask;
  static constexpr bool disable_shortcut = debug_parameter::disable_shortcut;
  static constexpr mat_buffer_type pv_buffer = tuning_parameter::pv_buffer;
  static constexpr mat_buffer_type o_buffer = tuning_parameter::o_buffer;
  static constexpr bool prefetch_q = tuning_parameter::prefetch_q;
  static constexpr int B_r = tuning_parameter::B_r;
  static constexpr int B_c = tuning_parameter::B_c;
  const uint32_t batch_dim;
  const uint32_t head_num;
  const uint32_t batch_size;
  const uint32_t seq_len;
  const uint32_t head_dim;
  const float hs_rsqrt_scale;
  const float dropout_prob;
  const float dropout_scale;
  const uint32_t dropout_threshold;
  const uint64_t dropout_rand_seed;
  dtype_q* const ptr_q;
  dtype_k* const ptr_k;
  dtype_v* const ptr_v;
  dtype_o* const ptr_o;
  dtype_m* const ptr_m;
  dtype_b* const ptr_b;
  dtype_d* const ptr_d;
  const uint32_t T_r;
  const uint32_t T_c;
  const uint32_t t_y;
  const uint32_t t_x;

  explicit FLASH_ATTENTION_FWD_IMPL(arguments_t& args)
      : batch_dim(args.batch_dim),
        head_num(args.head_num),
        batch_size(args.batch_size),
        seq_len(args.sequence_length),
        head_dim(args.head_dim),
        hs_rsqrt_scale(args.hs_rsqrt_scale),
        dropout_prob(args.dropout_prob),
        dropout_scale(args.dropout_scale),
        dropout_threshold(uint32_t(dropout_prob * float(4294967296))),
        dropout_rand_seed(args.dropout_rand_seed),
        ptr_q(args.ptr_q),
        ptr_k(args.ptr_k),
        ptr_v(args.ptr_v),
        ptr_o(args.ptr_o),
        ptr_m(args.ptr_m),
        ptr_b(args.ptr_b),
        ptr_d(args.ptr_d),
        T_r((args.sequence_length + B_r - 1) / B_r),
        T_c((args.sequence_length + B_c - 1) / B_c),
        t_y(param_S::m_w / param_S::m_s),
        t_x(param_S::n_w / param_S::n_s) {}

  static constexpr int slm_base_addr = 0;

  template <
      uint32_t m_w_,
      uint32_t n_w_,
      uint32_t k_w_,
      uint32_t n_s_,
      uint32_t k_s_>
  struct param_S_t {
    using compute_attr =
        gpu::xetla::group::compute_attr_t<dtype_q, dtype_k, dtype_acc>;
    struct fine_tuning {
      static constexpr uint32_t periodic_sync_interval =
          tuning_parameter::fine_tune::matS_periodic_sync_interval;
      static constexpr uint32_t prefetch_distance =
          tuning_parameter::fine_tune::matS_prefetch_distance;
    };
    // should larger than 8
    static constexpr uint32_t k_iter_num = k_s_;
    using perf_tuning_knob = gpu::xetla::group::perf_tuning_knob_t<
        k_iter_num,
        fine_tuning::prefetch_distance,
        fine_tuning::periodic_sync_interval>;
    using compute_policy = gpu::xetla::group::compute_policy_default_xmx<
        compute_attr,
        perf_tuning_knob,
        gpu::xetla::gpu_arch::Xe>;
    using mem_desc_input_q = gpu::xetla::mem_desc_t<
        dtype_q,
        gpu::xetla::mem_layout::row_major,
        gpu::xetla::mem_space::global>;
    using mem_desc_prefetch_q = gpu::xetla::mem_desc_t<
        dtype_q,
        gpu::xetla::mem_layout::row_major,
        gpu::xetla::mem_space::local>;
    using mem_desc_input_k = gpu::xetla::mem_desc_t<
        dtype_k,
        gpu::xetla::mem_layout::col_major,
        gpu::xetla::mem_space::global>;
    static constexpr int m_w = m_w_;
    static constexpr int n_w = n_w_;
    static constexpr int n_s = n_s_;
    static_assert(n_w % n_s == 0, "invalid configuration for n_w, n_s");
    static_assert(
        m_w * n_w % (n_s * thread_num) == 0,
        "invalid configuration for m_w");
    static constexpr int m_s = m_w * n_w / n_s / thread_num;
    static constexpr int n_x = n_w / n_s;
    static constexpr int n_y = m_w / m_s;
    using tile_shape = gpu::xetla::group::tile_shape_t<n_w, m_w, n_s, m_s>;
    using brgemm_t = std::conditional_t<
        prefetch_q,
        gpu::xetla::group::brgemm_t<
            compute_policy,
            tile_shape,
            mem_desc_prefetch_q,
            mem_desc_input_k>,
        gpu::xetla::group::brgemm_t<
            compute_policy,
            tile_shape,
            mem_desc_input_q,
            mem_desc_input_k>>;
    using mat_tile_shape = brgemm_t::tile_shape;
    using mat_out_t = brgemm_t::matAcc_t;
    static constexpr int split_k_cnt = k_w_ / k_iter_num;
    static constexpr int barrier_count = brgemm_t::barrier_count;
    static constexpr int barrier_offset = 0;
    static constexpr int prefetch_q_slm_base_addr = slm_base_addr;
    static_assert(
        prefetch_q_slm_base_addr % 4 == 0,
        "prefetch_q_slm_base_addr alignment failed");
    static constexpr int prefetch_q_slm_size =
        prefetch_q ? m_w * H * sizeof(dtype_q) : 0;
    static constexpr int brgemm_slm_base_addr =
        prefetch_q_slm_base_addr + prefetch_q_slm_size;
    static_assert(
        brgemm_slm_base_addr % 4 == 0,
        "brgemm_slm_base_addr alignment failed");
    static constexpr int slm_size = brgemm_slm_base_addr + brgemm_t::slm_size;
    static_assert(slm_size <= 128 * 1024, "slm size exceeds 128k!");
  };

  using param_S = param_S_t<
      B_r,
      B_c,
      H,
      tuning_parameter::matS_n_s,
      tuning_parameter::matS_k_s>;

  struct param_P {
    using mat_type = param_S::mat_out_t;
    using mat_in_t = mat_type;
    using mat_out_t = mat_type;
    using mat_tile_shape = param_S::mat_tile_shape;

    using rowmax_t = gpu::xetla::
        xetla_vector<typename mat_out_t::dtype, mat_in_t::tile_size_y>;
    using rowsum_t = rowmax_t;
    static constexpr int vec_length = rowmax_t::length;
    using vec_dtype = rowmax_t::element_type;

    using mem_desc_output_m = gpu::xetla::mem_desc_t<
        dtype_m,
        gpu::xetla::mem_layout::row_major,
        gpu::xetla::mem_space::global>;
    using factor_tile_desc_t = gpu::xetla::subgroup::tile_desc_t<
        vec_length,
        1,
        vec_length,
        1,
        gpu::xetla::reg_layout::tiled>;
    using factor_tile_t =
        gpu::xetla::subgroup::tile_t<dtype_m, factor_tile_desc_t>;
    using factor_payload_t = gpu::xetla::subgroup::mem_payload_t<
        dtype_m,
        factor_tile_desc_t,
        gpu::xetla::msg_type::block_1d,
        mem_desc_output_m::layout,
        mem_desc_output_m::space,
        param_S::brgemm_t::arch_tag>;

    static constexpr int reduce_list_ylength = mat_tile_shape::wg_size_y;
    static constexpr int reduce_list_xlength = mat_tile_shape::wg_size_x;
    static constexpr int reduce_elem_count = vec_length;

    using wg_reduce_max_t = gpu::xetla::group::group_reduce_t<
        typename mat_out_t::dtype,
        1,
        vec_length,
        gpu::xetla::reduce_op::max,
        reduce_list_xlength,
        true,
        factor_payload_t::arch_tag>;
    using wg_reduce_sum_t = gpu::xetla::group::group_reduce_t<
        typename mat_out_t::dtype,
        1,
        vec_length,
        gpu::xetla::reduce_op::sum,
        reduce_list_xlength,
        true,
        factor_payload_t::arch_tag>;

    using reduce_nbarrier_t =
        gpu::xetla::xetla_nbarrier_t<reduce_list_xlength, reduce_list_xlength>;
    static constexpr int reduce_barrier_count =
        ((param_S::barrier_count == 0) && (reduce_list_xlength > 0))
        ? reduce_list_ylength
        : 0;
    static constexpr int reduce_barrier_offset = 0;
    static_assert(
        (param_S::barrier_count == 0) ||
            ((param_S::barrier_count > 0) &&
             (param_S::n_x == reduce_list_xlength)),
        "param_S::n_x expect to match reduce_list_xlength");
    static constexpr int reduce_slm_size = (reduce_list_xlength > 0)
        ? (reduce_list_ylength * reduce_list_xlength * reduce_elem_count *
           sizeof(vec_dtype))
        : 0;

    static constexpr uint32_t reduce_slm_base_addr =
        param_S::prefetch_q_slm_base_addr + param_S::prefetch_q_slm_size;
    static_assert(
        reduce_slm_base_addr % 4 == 0,
        "reduce_slm_base_addr alignment failed");
    using mem_desc_output_p = std::conditional_t<
        pv_buffer == mat_buffer_type::global,
        gpu::xetla::mem_desc_t<
            dtype_b,
            gpu::xetla::mem_layout::row_major,
            gpu::xetla::mem_space::global>,
        gpu::xetla::mem_desc_t<
            dtype_p,
            gpu::xetla::mem_layout::row_major,
            gpu::xetla::mem_space::local>>;
    using epilogue_t = gpu::xetla::group::epilogue_t<
        gpu::xetla::group::epilogue_policy_default<gpu::xetla::gpu_arch::Xe>,
        mat_tile_shape,
        mem_desc_output_p>;
    using store_nbarrier_t =
        gpu::xetla::xetla_nbarrier_t<thread_num, thread_num>;
    static constexpr int local_store_barrier_count = 1;
    static constexpr int local_store_barrier_offset =
        param_S::barrier_count + reduce_barrier_count;
    static constexpr uint32_t local_store_slm_base_addr =
        reduce_slm_base_addr + reduce_slm_size;
    static_assert(
        local_store_slm_base_addr % 4 == 0,
        "local_store_slm_base_addr alignment failed");
    static constexpr int local_store_slm_size =
        param_S::m_w * param_S::n_w * sizeof(typename mat_type::dtype);

    static constexpr int barrier_count =
        reduce_barrier_count + local_store_barrier_count;
    static constexpr int slm_size =
        reduce_slm_base_addr + reduce_slm_size + local_store_slm_size;
    static_assert(slm_size <= 128 * 1024, "slm size exceeds 128k!");
  };

  template <uint32_t m_w_, uint32_t n_w_, uint32_t k_w_, uint32_t k_s_>
  struct param_O_t {
    using compute_attr =
        gpu::xetla::group::compute_attr_t<dtype_p, dtype_acc, dtype_acc>;
    struct fine_tuning {
      static constexpr uint32_t periodic_sync_interval =
          tuning_parameter::fine_tune::matO_periodic_sync_interval;
      static constexpr uint32_t prefetch_distance =
          tuning_parameter::fine_tune::matO_prefetch_distance;
    };
    // should larger than 8
    static constexpr uint32_t k_iter_num = k_s_;
    using perf_tuning_knob = gpu::xetla::group::perf_tuning_knob_t<
        k_iter_num,
        fine_tuning::prefetch_distance,
        fine_tuning::periodic_sync_interval>;
    using compute_policy = gpu::xetla::group::compute_policy_default_xmx<
        compute_attr,
        perf_tuning_knob,
        gpu::xetla::gpu_arch::Xe>;
    using mem_desc_input_p = param_P::mem_desc_output_p;
    using mem_desc_input_v = gpu::xetla::mem_desc_t<
        dtype_v,
        gpu::xetla::mem_layout::row_major,
        gpu::xetla::mem_space::global>;
    static constexpr int m_w = m_w_;
    static constexpr int n_w = n_w_;
    static constexpr int v_s = n_w * param_S::n_s / param_S::n_w;
    static constexpr int n_s = v_s;
    static_assert(n_w % n_s == 0, "invalid configuration for n_w, n_s");
    static_assert(
        m_w * n_w % (n_s * thread_num) == 0,
        "invalid configuration for m_w");
    static constexpr int m_s = m_w * n_w / n_s / thread_num;
    static constexpr int n_x = n_w / n_s;
    static constexpr int n_y = m_w / m_s;
    using tile_shape = gpu::xetla::group::tile_shape_t<n_w, m_w, n_s, m_s>;
    using brgemm_t = gpu::xetla::group::brgemm_t<
        compute_policy,
        tile_shape,
        mem_desc_input_p,
        mem_desc_input_v>;
    using mat_out_t = brgemm_t::matAcc_t;
    static constexpr int split_k_cnt = k_w_ / k_iter_num;
    static constexpr int barrier_count =
        (param_S::barrier_count == 0) ? brgemm_t::barrier_count : 0;
    static_assert(
        (param_S::barrier_count == 0) ||
            ((param_S::barrier_count > 0) &&
             (param_S::n_x == n_x && param_S::n_y == n_y)),
        "when periodic_sync_interval is enabled, expect (n_x, n_y) to "
        "match");
    static constexpr int barrier_offset =
        (barrier_count == 0) ? 0 : param_P::barrier_count;
    static constexpr int brgemm_slm_base_addr =
        param_P::local_store_slm_base_addr + param_P::local_store_slm_size;
    static_assert(
        brgemm_slm_base_addr % 4 == 0,
        "brgemm_slm_base_addr alignment failed");

    static constexpr int o_buffer_slm_size =
        (o_buffer == mat_buffer_type::local) ? m_w * n_w * sizeof(dtype_acc)
                                             : 0;
    static constexpr int o_buffer_slm_base_addr = std::max(
        {param_S::slm_size,
         param_P::slm_size,
         static_cast<int>(brgemm_slm_base_addr + brgemm_t::slm_size)});
    static_assert(
        o_buffer_slm_base_addr % 4 == 0,
        "o_buffer_slm_base_addr alignment failed");

    static constexpr int slm_size = o_buffer_slm_base_addr + o_buffer_slm_size;
    static_assert(slm_size <= 128 * 1024, "slm size exceeds 128k!");
  };

  using param_O = param_O_t<
      param_S::m_w,
      tuning_parameter::matO_n_w,
      param_S::n_w,
      tuning_parameter::matO_k_s>;

  struct utils;
  struct program;

 public:
  __XETLA_API KERNEL_FUNC void run(gpu::xetla::xetla_exec_item<3>& ei) const;
};

} // namespace xetla
} // namespace xpu