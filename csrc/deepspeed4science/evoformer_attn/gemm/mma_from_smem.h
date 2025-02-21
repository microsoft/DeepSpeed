/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/functional.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/platform/platform.h"
#include "cutlass/transform/threadblock/vector_iterator.h"

#include "../epilogue/epilogue_thread_apply_logsumexp.h"
#include "../gemm/mma_accum_lambda_iterator.h"
#include "../gemm_kernel_utils.h"
#include "../iterators/make_residual_last.h"
#include "../iterators/transpose_warp_iterator.h"
#include "../iterators/warp_iterator_from_smem.h"
#include "cutlass/epilogue/threadblock/epilogue_smem_accumulator.h"
#include "cutlass/gemm/threadblock/mma_base.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_access_iterator.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

/// Shared storage object needed by accumulator
/// From 13_two_tensor_op_fusion/threadblock/b2b_mma_base_smem_accumulator.h
template <typename Shape_, typename Element_, typename Layout_, typename Padding_>
class AccumulatorSharedStorage {
public:
    //
    // Type definitions
    //
    using Shape = Shape_;
    using Element = Element_;
    using Layout = Layout_;
    using Padding = Padding_;

    /// Tensor reference to the accumulator
    using TensorRefAccum = cutlass::TensorRef<Element, Layout>;

    /// Shape of the accumulator matrix in shared memory
    using ShapeAccum =
        cutlass::MatrixShape<Shape::kM + Padding::kRow, Shape::kN + Padding::kColumn>;

public:
    //
    // Data members
    //

    /// Buffer for accumulator
    cutlass::AlignedBuffer<Element, ShapeAccum::kCount> accum;

public:
    //
    // Methods
    //

    /// Returns a layout object for the Accum matrix
    CUTLASS_DEVICE
    static Layout LayoutAccum() { return Layout::packed({ShapeAccum::kRow, ShapeAccum::kColumn}); }

    /// Returns a TensorRef to the Accumulator
    CUTLASS_HOST_DEVICE
    TensorRefAccum accum_ref() { return TensorRefAccum{accum.data(), LayoutAccum()}; }
};

////////////////////////////////////////////////////////////////////////////////
// Taken from
// https://github.com/NVIDIA/cutlass/blob/master/examples/13_two_tensor_op_fusion/threadblock/b2b_mma_base_smem_accumulator.h
////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    // Maximum value for K
    int kMaxK,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Used for partial specialization
    typename Enable = bool>
class MmaBaseFromSharedMemory {
public:
    ///< Size of the Gemm problem - concept: gemm::GemmShape<>
    using Shape = Shape_;

    ///< Policy describing tuning details
    using Policy = Policy_;

    //
    // Dependent types
    //

    /// Warp-level Mma
    using Operator = typename Policy::Operator;

    /// Shape describing the overall GEMM computed from shared memory
    /// by each warp.
    using WarpGemm = typename Policy::Operator::Shape;

    /// Shape describing the number of warps filling the CTA
    using WarpCount =
        GemmShape<Shape::kM / WarpGemm::kM, Shape::kN / WarpGemm::kN, Shape::kK / WarpGemm::kK>;
    using WarpCount1 = WarpCount;

    /// Number of warp-level GEMM oeprations
    static int const kWarpGemmIterations = (WarpGemm::kK / Operator::Policy::MmaShape::kK);
    static int const kWarpGemmIterations1 = kWarpGemmIterations;

    /// Number of stages
    static int const kStages = Stages;

    /// If this is true, we fill the entire shmem buffer at start
    /// and don't need to iterate through it in a circular fashion
    static bool const kSmemContainsEntireB = kMaxK <= Shape::kK * kStages;

    /// Tensor reference to the A operand
    using TensorRefA = TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;

    /// Tensor reference to the B operand
    using TensorRefB = TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;

    //
    // Nested structs
    //

    /// Shared storage object needed by threadblock-scoped GEMM
    class SharedStorage {
    public:
        //
        // Type definitions
        //

        /// Shape of the B matrix operand in shared memory
        using ShapeB = MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow,
                                   Shape::kN + Policy::SmemPaddingB::kColumn>;

    public:
        //
        // Data members
        //

        /// Buffer for B operand
        AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;

    public:
        //
        // Methods
        //

        /// Returns a layout object for the B matrix
        CUTLASS_HOST_DEVICE
        static typename Operator::LayoutB LayoutB()
        {
            return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
        }

        /// Returns a TensorRef to the B operand
        CUTLASS_HOST_DEVICE
        TensorRefB operand_B_ref() { return TensorRefB{operand_B.data(), LayoutB()}; }
    };

protected:
    //
    // Data members
    //

    // /// Iterator to load a warp-scoped tile of A operand from shared memory
    // typename Operator::IteratorA warp_tile_iterator_A_;

    /// Iterator to load a warp-scoped tile of B operand from shared memory
    typename Operator::IteratorB warp_tile_iterator_B_;

public:
    /// Construct from tensor references
    CUTLASS_DEVICE
    MmaBaseFromSharedMemory(
        ///< Shared storage needed for internal use by threadblock-scoped GEMM
        SharedStorage& shared_storage,
        ///< ID within the threadblock
        int thread_idx,
        ///< ID of warp
        int warp_idx,
        ///< ID of each thread within a warp
        int lane_idx)
        : warp_tile_iterator_B_(shared_storage.operand_B_ref(), lane_idx)
    {
    }
};

namespace {

// has necessary trait compliance with WarpIteratorFromSmem but doesn't do
// anything, can be default initialized, and uses fragment that takes up
// (almost) no space. this warp iterator is selected at compile time when
// elementwise on-the-fly scaling for operand A is disabled, in which case
// operations related to loading scale factors for operand A get wiped out by
// the compiler.
template <typename TensorRef>
class NoOpWarpIteratorScale {
public:
    // in pipelined+multistage MMA implementations we keep an array of fragments.
    // if we aren't using scaling we don't want to waste registers on fragments
    // of scale elements, so ideally this would be sized 0.
    // Since arrays of zero-sized objects are not allowed, using size as 1.
    // The compiler will most likely wipe it out anyways.
    using Fragment = cutlass::Array<char, 1>;

    CUTLASS_HOST_DEVICE
    NoOpWarpIteratorScale() {}

    CUTLASS_HOST_DEVICE
    NoOpWarpIteratorScale(TensorRef const&, int) {}

    CUTLASS_HOST_DEVICE
    NoOpWarpIteratorScale& add_tile_offset(typename TensorRef::TensorCoord const&) { return *this; }

    CUTLASS_HOST_DEVICE
    NoOpWarpIteratorScale& operator++() { return *this; }

    CUTLASS_DEVICE
    void load(Fragment&) const {}
};

// if scaling is enabled, performs fragment elementwise multiplication between
// fragment and its scaling factor.
template <typename Fragment, typename FragmentScale, bool ScalingEnabled>
class FragmentElementwiseScaler;

// specialization for scaling being enabled.
template <typename Fragment, typename FragmentScale>
class FragmentElementwiseScaler<Fragment, FragmentScale, true> {
public:
    // cast scale_frag to correct type then apply elementwise to fragment
    CUTLASS_DEVICE
    static Fragment apply(Fragment frag, FragmentScale const& scale_frag)
    {
        Fragment converted_scale_frag =
            cutlass::NumericArrayConverter<typename Fragment::Element,
                                           typename FragmentScale::Element,
                                           FragmentScale::kElements>()(scale_frag);
        return cutlass::multiplies<Fragment>()(frag, converted_scale_frag);
    }
};

// specialization for scaling being disabled. doesn't do anything and should
// just get wiped out by the compiler.
template <typename Fragment, typename FragmentScale>
class FragmentElementwiseScaler<Fragment, FragmentScale, false> {
public:
    CUTLASS_DEVICE
    static Fragment apply(Fragment frag, FragmentScale const&) { return frag; }
};
}  // namespace

////////////////////////////////////////////////////////////////////////////////
// Taken from
// https://github.com/NVIDIA/cutlass/blob/master/examples/13_two_tensor_op_fusion/threadblock/b2b_mma_pipelined_smem_accumulator.h
////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    // BEGIN smem
    /// Iterates over the intermediate accumulator tile in shared memory
    typename WarpIteratorA,
    /// whether or not to perform elementwise multiplication of A
    //  by another matrix (A_scale) that is also kept in shared memory prior
    //  to matmul A @ B
    bool ScaleOperandA_,
    // Accumulator type
    typename AccumulatorSharedStorage,
    // END smem
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Transformation applied to B operand
    typename TransformB_ = NumericArrayConverter<typename SmemIteratorB_::Element,
                                                 typename IteratorB_::Element,
                                                 IteratorB_::Fragment::kElements>,
    /// Used for partial specialization
    typename Enable = bool>
class MmaPipelinedFromSharedMemory
    : public MmaBaseFromSharedMemory<Shape_, AccumulatorSharedStorage::Shape::kN, Policy_, 2> {
public:
    ///< Base class
    using Base = MmaBaseFromSharedMemory<Shape_, AccumulatorSharedStorage::Shape::kN, Policy_, 2>;

    using Shape = Shape_;  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
    static constexpr bool ScaleOperandA = ScaleOperandA_;

    ///< loads fragments of A_scale from shared memory if operand A scaling is
    ///< enabled. otherwise no-op.
    using WarpIteratorAScale = typename cutlass::platform::conditional<
        ScaleOperandA,
        WarpIteratorA,
        NoOpWarpIteratorScale<typename WarpIteratorA::TensorRef>>::type;

    using IteratorB = IteratorB_;  ///< Iterates over tiles of B operand in global memory
    using ElementC = ElementC_;    ///< Data type of accumulator matrix
    using LayoutC = LayoutC_;      ///< Layout of accumulator matrix
    using Policy = Policy_;        ///< Policy describing tuning details

    using SmemIteratorB = SmemIteratorB_;

    using TransformB = TransformB_;

    //
    // Dependent types
    //

    /// Fragment of operand B loaded from global memory
    using FragmentB = typename IteratorB::Fragment;

    /// Fragment of accumulator tile
    using FragmentC = typename Policy::Operator::FragmentC;

    /// Warp-level Mma
    using Operator = typename Policy::Operator;

    /// Obtain the arch tag from the warp-level operator
    using ArchTag = typename Policy::Operator::ArchTag;

    /// Complex transform on B operand
    static ComplexTransform const kTransformB = Operator::kTransformB;

    // staticaly assert kStages for MmaPipelined is two (Double-buffered pipeline)
    static_assert((Base::kStages == 2), "MmaPipelined requires kStages set to value 2");

private:
    using WarpFragmentA = typename Operator::FragmentA;

    /// fragment type of OperandA elementwise scaling matrix. (almost) empty
    /// if operand A scaling is disabled.
    using WarpFragmentAScale = typename WarpIteratorAScale::Fragment;

    using WarpFragmentB = typename Operator::FragmentB;

    /// applies scaling factor to operand A fragment if operand A scaling is
    /// enabled. otherwise no-op.
    using FragmentAScaler =
        FragmentElementwiseScaler<WarpFragmentA, WarpFragmentAScale, ScaleOperandA>;

protected:
    // /// Iterator to write threadblock-scoped tile of A operand to shared memory
    // SmemIteratorA smem_iterator_A_;

    /// Iterator to write threadblock-scoped tile of B operand to shared memory
    SmemIteratorB smem_iterator_B_;

    /// Iterator to load a warp-scoped tile of A operand from intermediate
    /// accumulator tile
    WarpIteratorA warp_tile_iterator_A_;

    /// Iterator to load a warp-scoped tile of A_scale from intermediate
    /// accumulator tile (only used if ScaleOperandA_ is true)
    WarpIteratorAScale warp_tile_iterator_A_scale_;

public:
    /// constructor for MMA with operand A scaling enabled.
    CUTLASS_DEVICE
    MmaPipelinedFromSharedMemory(
        // shared storage needed for internal use by threadblock-scoped GEMM
        typename Base::SharedStorage& shared_storage,
        // warp iterator over A tile held in shared memory
        WarpIteratorA warp_iter_a,
        // warp iterator over A_scale tile held in shared memory
        WarpIteratorAScale warp_iter_a_scale,
        int thread_idx,
        int warp_idx,
        int lane_idx)
        : Base(shared_storage, thread_idx, warp_idx, lane_idx),
          warp_tile_iterator_A_(warp_iter_a),
          warp_tile_iterator_A_scale_(warp_iter_a_scale),
          smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx)
    {
        // Compute warp location within threadblock tile by mapping the warp_id to
        // three coordinates:
        //   _m: the warp's position within the threadblock along the M dimension
        //   _n: the warp's position within the threadblock along the N dimension
        //   _k: the warp's position within the threadblock along the K dimension
        int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
        int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);
        int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
        int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

        // Add per-warp offsets in units of warp-level tiles
        this->warp_tile_iterator_A_.add_tile_offset(
            {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
        this->warp_tile_iterator_A_scale_.add_tile_offset(
            {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
        this->warp_tile_iterator_B_.add_tile_offset(
            {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    }

    /// Construct from tensor references
    CUTLASS_DEVICE
    MmaPipelinedFromSharedMemory(
        typename Base::SharedStorage& shared_storage,  ///< Shared storage needed for internal use
                                                       ///< by threadblock-scoped GEMM
        AccumulatorSharedStorage& accumulator_shared_storage,
        int thread_idx,  ///< ID within the threadblock
        int warp_idx,    ///< ID of warp
        int lane_idx,    ///< ID of each thread within a warp
        int problem_size_0_n)
        : Base(shared_storage, thread_idx, warp_idx, lane_idx),
          warp_tile_iterator_A_(accumulator_shared_storage.accum_ref(), lane_idx),
          smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx)
    {
        // Compute warp location within threadblock tile by mapping the warp_id to
        // three coordinates:
        //   _m: the warp's position within the threadblock along the M dimension
        //   _n: the warp's position within the threadblock along the N dimension
        //   _k: the warp's position within the threadblock along the K dimension

        int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
        int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

        int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
        int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

        // Add per-warp offsets in units of warp-level tiles
        this->warp_tile_iterator_A_.add_tile_offset(
            {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
        this->warp_tile_iterator_B_.add_tile_offset(
            {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    }

    // For API compatibility with MmaMultistageFromSharedMemory
    // but not supported as it worsens perf: older gpus < sm80 don't
    // support async transfers and have to waste registers
    CUTLASS_DEVICE
    void set_prologue_done(bool value) {}
    CUTLASS_DEVICE
    static void prologue(typename Base::SharedStorage& shared_storage,
                         IteratorB iterator_B1,
                         int thread_idx,
                         int problem_size_0_n)
    {
    }

    CUTLASS_DEVICE
    static void drain_cp_asyncs() {}

    /// Perform a threadblock-scoped matrix multiply-accumulate
    CUTLASS_DEVICE
    void operator()(int gemm_k_iterations,  ///< number of iterations of the mainloop
                    FragmentC& accum,       ///< destination accumulator tile
                    // IteratorA iterator_A,                             ///< iterator over A
                    // operand in global memory
                    IteratorB iterator_B,        ///< iterator over B operand in global memory
                    FragmentC const& src_accum,  ///< source accumulator tile
                    // TransformA transform_A = TransformA(),            ///< transformation
                    // applied to A fragment
                    TransformB transform_B = TransformB())
    {  ///< transformation applied to B fragment

        //
        // Prologue
        //

        // Perform accumulation in the 'd' output operand
        accum = src_accum;

        FragmentB tb_frag_B;

        tb_frag_B.clear();

        // The last kblock is loaded in the prolog
        iterator_B.set_residual_tile(gemm_k_iterations == 1);
        iterator_B.load(tb_frag_B);

        ++iterator_B;

        this->smem_iterator_B_.store(transform_B(tb_frag_B));

        ++this->smem_iterator_B_;

        __syncthreads();

        // remember that WarpFragmentAScale and WarpIteratorAScale are empty/no-op
        // if scaling is disabled.

        // Pair of fragments used to overlap shared memory loads and math
        // instructions
        WarpFragmentA warp_frag_A[2];
        WarpFragmentAScale warp_frag_A_scale[2];
        WarpFragmentB warp_frag_B[2];
        warp_frag_A[0].clear();
        warp_frag_A_scale[0].clear();
        warp_frag_B[0].clear();

        this->warp_tile_iterator_B_.set_kgroup_index(0);

        this->warp_tile_iterator_A_.load(warp_frag_A[0]);
        this->warp_tile_iterator_A_scale_.load(warp_frag_A_scale[0]);
        this->warp_tile_iterator_B_.load(warp_frag_B[0]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_A_scale_;
        ++this->warp_tile_iterator_B_;

        Operator warp_mma;

        int smem_write_stage_idx = 1;

        // Avoid reading out of bounds
        iterator_B.set_residual_tile(gemm_k_iterations == 2);
        iterator_B.clear_mask(gemm_k_iterations <= 1);

        // Issue loads during the first warp-level matrix multiply-add *AFTER*
        // issuing shared memory loads (which have the tightest latency requirement).

        //
        // Mainloop
        //

        // Note: The main loop does not support Base::kWarpGemmIterations == 2.
        CUTLASS_GEMM_LOOP
        for (; gemm_k_iterations > 0; --gemm_k_iterations) {
            //
            // Loop over GEMM K dimension
            //

            CUTLASS_PRAGMA_UNROLL
            for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
                // Load warp-level tiles from shared memory, wrapping to k offset if
                // this is the last group as the case may be.
                bool hasNext = true;

                if (warp_mma_k == Base::kWarpGemmIterations - 1) {
                    // Write fragments to shared memory
                    this->smem_iterator_B_.store(transform_B(tb_frag_B));

                    __syncthreads();

                    ++this->smem_iterator_B_;

                    // Add negative offsets to return iterators to the 'start' of the
                    // circular buffer in shared memory SMEM: Don't reset iterator A, as
                    // we are continuing our iteration at this point
                    if (smem_write_stage_idx == 1) {
                        this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
                    } else {
                        this->warp_tile_iterator_B_.add_tile_offset(
                            {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
                    }

                    smem_write_stage_idx ^= 1;
                    hasNext = gemm_k_iterations > 1;
                }

                // Only read the next if we need to
                if (hasNext) {
                    this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) %
                                                                 Base::kWarpGemmIterations);

                    this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                    this->warp_tile_iterator_A_scale_.load(warp_frag_A_scale[(warp_mma_k + 1) % 2]);
                    this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

                    ++this->warp_tile_iterator_A_;
                    ++this->warp_tile_iterator_A_scale_;
                    ++this->warp_tile_iterator_B_;

                    if (warp_mma_k == 0) {
                        iterator_B.load(tb_frag_B);

                        ++iterator_B;

                        // Avoid reading out of bounds if this was the last loop iteration
                        iterator_B.set_residual_tile(gemm_k_iterations == 3);
                        iterator_B.clear_mask(gemm_k_iterations <= 2);
                    }
                }

                warp_mma(accum,
                         FragmentAScaler::apply(warp_frag_A[warp_mma_k % 2],
                                                warp_frag_A_scale[warp_mma_k % 2]),
                         warp_frag_B[warp_mma_k % 2],
                         accum);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////
// Taken from
// https://github.com/NVIDIA/cutlass/blob/master/examples/13_two_tensor_op_fusion/threadblock/b2b_mma_multistage_smem_accumulator.h
////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape1_,
    /// Iterates over the intermediate accumulator tile in shared memory
    typename WarpIteratorA1_,
    /// whether or not to perform elementwise multiplication of A
    //  by another matrix (A_scale) that is also kept in shared memory prior
    //  to matmul A @ B
    bool ScaleOperandA_,
    // Accumulator type
    typename AccumulatorSharedStorage,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB1_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB1_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB1,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy1_,
    /// Number of stages,
    int Stages_,
    int kMaxK_,
    /// Used for partial specialization
    typename Enable = bool>
class MmaMultistageFromSharedMemory
    : public MmaBaseFromSharedMemory<Shape1_, kMaxK_, Policy1_, Stages_> {
public:
    ///< Base class
    using Base = MmaBaseFromSharedMemory<Shape1_, kMaxK_, Policy1_, Stages_>;

    ///< Size of the Gemm problem - concept: gemm::GemmShape<>
    using Shape1 = Shape1_;
    ///< Iterates over tiles of B operand in global memory
    using IteratorB1 = IteratorB1_;
    using IteratorB = IteratorB1;
    ///< Policy describing tuning details
    using Policy1 = Policy1_;

    using SmemIteratorB1 = SmemIteratorB1_;
    using WarpIteratorA1 = WarpIteratorA1_;  ///< Iterates over the intermediate
                                             ///< accumulator tile in shared memory
    static constexpr bool ScaleOperandA = ScaleOperandA_;

    ///< warp level iterator over A_scale matrix tile kept in shared memory.
    ///< if elementwise A scaling is disabled then everything this does is no-op.
    using WarpIteratorAScale = typename cutlass::platform::conditional<
        ScaleOperandA,
        WarpIteratorA1,
        NoOpWarpIteratorScale<typename WarpIteratorA1::TensorRef>>::type;
    ///< Data type of accumulator matrix
    using ElementC = ElementC_;
    ///< Layout of accumulator matrix
    using LayoutC = LayoutC_;

    static cutlass::arch::CacheOperation::Kind const kCacheOpB1 = CacheOpB1;
    static constexpr bool kSmemContainsEntireB = Base::kSmemContainsEntireB;

    //
    // Dependent types
    //

    /// Fragment of accumulator tile
    using FragmentC1 = typename Policy1::Operator::FragmentC;
    using FragmentC = FragmentC1;

    /// Warp-level Mma
    using Operator1 = typename Policy1::Operator;

    /// Minimum architecture is Sm80 to support cp.async
    using ArchTag = arch::Sm80;

    /// Complex transform on B operand
    static ComplexTransform const kTransformB1 = Operator1::kTransformB;

    /// Internal structure exposed for introspection.
    struct Detail {
        static_assert(Base::kWarpGemmIterations1 > 1,
                      "The pipelined structure requires at least two warp-level "
                      "GEMM operations.");

        /// Number of cp.async instructions to load one stage of operand B
        static int const TBLoadIterationsB1 = IteratorB1::ThreadMap::Iterations::kCount;

        /// Number of cp.async instructions to load on group of operand B
        static int const kAccessesPerGroupB1 =
            (TBLoadIterationsB1 + Base::kWarpGemmIterations1 - 1) / Base::kWarpGemmIterations1;
    };

    static constexpr int kNumStagesConcurrentLoad = kSmemContainsEntireB ? Base::kStages
                                                                         : Base::kStages - 1;

private:
    using WarpLoadedFragmentA1 = typename Operator1::FragmentA;
    /// fragment of OperandA scale matrix. if operand A scaling is disabled this
    /// is (almost) empty.
    using WarpLoadedFragmentA1Scale = typename WarpIteratorAScale::Fragment;
    using WarpLoadedFragmentB1 = typename Operator1::FragmentB;
    using WarpTransformedFragmentA1 = typename Operator1::TransformedFragmentA;
    using WarpTransformedFragmentB1 = typename Operator1::TransformedFragmentB;

    /// applies elementwise scaling to fragment of A. if operand A scaling is
    /// disabled this is a no-op.
    using FragmentAScaler =
        FragmentElementwiseScaler<WarpLoadedFragmentA1, WarpLoadedFragmentA1Scale, ScaleOperandA>;

private:
    //
    // Data members
    //

    /// Iterator to load a warp-scoped tile of A1 operand from intermediate
    /// accumulator tile
    WarpIteratorA1 warp_tile_iterator_A1_;

    /// Iterator to load a warp-scoped tile of A1_scale operand from shared memory
    /// if operand A scaling is disabled everything this does is a no-op.
    WarpIteratorAScale warp_tile_iterator_A1_scale_;

    /// Iterator to write threadblock-scoped tile of B operand to shared memory
    SmemIteratorB1 smem_iterator_B1_;

    bool prologue_done_;

public:
    /// constructor for MMA with operand A scaling enabled.
    CUTLASS_DEVICE
    MmaMultistageFromSharedMemory(
        // shared storage needed for internal use by threadblock-scoped GEMM
        typename Base::SharedStorage& shared_storage,
        // warp level iterator over operand A tile kept in shared memory
        WarpIteratorA1 warp_tile_iterator_A1,
        // warp level iterator over operand A elementwise scale tile kept in
        // shared memory.
        WarpIteratorAScale warp_tile_iterator_A1_scale,
        int thread_idx,
        int warp_idx,
        int lane_idx)
        : Base(shared_storage, thread_idx, warp_idx, lane_idx),
          warp_tile_iterator_A1_(warp_tile_iterator_A1),
          warp_tile_iterator_A1_scale_(warp_tile_iterator_A1_scale),
          smem_iterator_B1_(shared_storage.operand_B_ref(), thread_idx),
          prologue_done_(false)
    {
        // Compute warp location within threadblock tile by mapping the warp_id to
        // three coordinates:
        //   _m: the warp's position within the threadblock along the M dimension
        //   _n: the warp's position within the threadblock along the N dimension
        //   _k: the warp's position within the threadblock along the K dimension
        int warp_idx_mn_1 = warp_idx % (Base::WarpCount1::kM * Base::WarpCount1::kN);
        int warp_idx_k_1 = warp_idx / (Base::WarpCount1::kM * Base::WarpCount1::kN);
        int warp_idx_m_1 = warp_idx_mn_1 % Base::WarpCount1::kM;
        int warp_idx_n_1 = warp_idx_mn_1 / Base::WarpCount1::kM;

        // Add per-warp offsets in units of warp-level tiles
        warp_tile_iterator_A1_.add_tile_offset(
            {warp_idx_m_1, Base::kWarpGemmIterations1 * warp_idx_k_1});
        warp_tile_iterator_A1_scale_.add_tile_offset(
            {warp_idx_m_1, Base::kWarpGemmIterations1 * warp_idx_k_1});
        this->warp_tile_iterator_B_.add_tile_offset(
            {Base::kWarpGemmIterations1 * warp_idx_k_1, warp_idx_n_1});
    }

    /// Construct from tensor references
    CUTLASS_DEVICE
    MmaMultistageFromSharedMemory(
        typename Base::SharedStorage& shared_storage,  ///< Shared storage needed for internal use
                                                       ///< by threadblock-scoped GEMM
        AccumulatorSharedStorage& accumulator_shared_storage,
        ///< ID within the threadblock
        int thread_idx,
        ///< ID of warp
        int warp_idx,
        ///< ID of each thread within a warp
        int lane_idx,
        ///< GEMM0 N is used for accumulator extent
        int problem_size_0_n)
        : Base(shared_storage, thread_idx, warp_idx, lane_idx),
          warp_tile_iterator_A1_(accumulator_shared_storage.accum_ref(), lane_idx),
          smem_iterator_B1_(shared_storage.operand_B_ref(), thread_idx),
          prologue_done_(false)
    {
        // Compute warp location within threadblock tile by mapping the warp_id to
        // three coordinates:
        //   _m: the warp's position within the threadblock along the M dimension
        //   _n: the warp's position within the threadblock along the N dimension
        //   _k: the warp's position within the threadblock along the K dimension

        int warp_idx_mn_1 = warp_idx % (Base::WarpCount1::kM * Base::WarpCount1::kN);
        int warp_idx_k_1 = warp_idx / (Base::WarpCount1::kM * Base::WarpCount1::kN);

        int warp_idx_m_1 = warp_idx_mn_1 % Base::WarpCount1::kM;
        int warp_idx_n_1 = warp_idx_mn_1 / Base::WarpCount1::kM;

        // Add per-warp offsets in units of warp-level tiles
        warp_tile_iterator_A1_.add_tile_offset(
            {warp_idx_m_1, Base::kWarpGemmIterations1 * warp_idx_k_1});
        this->warp_tile_iterator_B_.add_tile_offset(
            {Base::kWarpGemmIterations1 * warp_idx_k_1, warp_idx_n_1});
    }

    CUTLASS_DEVICE
    void set_prologue_done(bool value) { prologue_done_ = value; }

    CUTLASS_DEVICE
    static void prologue(typename Base::SharedStorage& shared_storage,
                         IteratorB iterator_B1,
                         int thread_idx,
                         int problem_size_0_n)
    {
        SmemIteratorB1 smem_iterator_B1(shared_storage.operand_B_ref(), thread_idx);
        _prologue(iterator_B1,
                  (problem_size_0_n + Base::Shape::kK - 1) / Base::Shape::kK,
                  smem_iterator_B1);
    }

    CUTLASS_DEVICE
    static void drain_cp_asyncs()
    {
        // commit and drain all pending and predicated cp.async pnz from the GEMM
        // mainloop
        cutlass::arch::cp_async_fence();
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();
    }

    CUTLASS_DEVICE
    void copy_tiles_and_advance_1(IteratorB1& iterator_B1, int group_start_B1 = 0)
    {
        iterator_B1.set_iteration_index(group_start_B1 * IteratorB1::kAccessesPerVector);
        this->smem_iterator_B1_.set_iteration_index(group_start_B1);

        // Load for operand B
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < Detail::kAccessesPerGroupB1; ++j) {
            if (group_start_B1 + j < Detail::TBLoadIterationsB1) {
                typename IteratorB1::AccessType* dst_ptr =
                    reinterpret_cast<typename IteratorB1::AccessType*>(
                        this->smem_iterator_B1_.get());

                int const kSrcBytes = sizeof_bits<typename IteratorB1::Element>::value *
                                      IteratorB1::ThreadMap::kElementsPerAccess /
                                      IteratorB1::kAccessesPerVector / 8;

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorB1::kAccessesPerVector; ++v) {
                    auto gmem_ptr = iterator_B1.get();

                    cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB1>(
                        dst_ptr + v, gmem_ptr, iterator_B1.valid());

                    ++iterator_B1;
                }
                ++this->smem_iterator_B1_;
            }
        }
    }

    CUTLASS_DEVICE
    static void _prologue(IteratorB& iterator_B1,
                          int32_t gemm_k_iterations_1,
                          SmemIteratorB1& smem_iterator_B1_)
    {
        // Issue several complete stages
        CUTLASS_PRAGMA_UNROLL
        for (int stage = 0; stage < kNumStagesConcurrentLoad; ++stage, --gemm_k_iterations_1) {
            iterator_B1.set_residual_tile(gemm_k_iterations_1 == 1);
            iterator_B1.clear_mask(gemm_k_iterations_1 == 0);

            iterator_B1.set_iteration_index(0);
            smem_iterator_B1_.set_iteration_index(0);

            // Load for operand B
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < Detail::TBLoadIterationsB1; ++j) {
                typename IteratorB1::AccessType* dst_ptr =
                    reinterpret_cast<typename IteratorB1::AccessType*>(smem_iterator_B1_.get());

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorB1::kAccessesPerVector; ++v) {
                    int const kSrcBytes = sizeof_bits<typename IteratorB1::Element>::value *
                                          IteratorB1::ThreadMap::kElementsPerAccess /
                                          IteratorB1::kAccessesPerVector / 8;

                    cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB1>(
                        dst_ptr + v, iterator_B1.get(), iterator_B1.valid());

                    ++iterator_B1;
                }

                ++smem_iterator_B1_;
            }

            // Move to the next stage
            iterator_B1.add_tile_offset({1, 0});

            smem_iterator_B1_.add_tile_offset({1, 0});

            // Defines the boundary of a stage of cp.async.
            cutlass::arch::cp_async_fence();
        }
        iterator_B1.set_residual_tile(gemm_k_iterations_1 == 1);
        iterator_B1.clear_mask(gemm_k_iterations_1 == 0);
    }

    /// Perform a threadblock-scoped matrix multiply-accumulate
    CUTLASS_DEVICE
    void operator()(
        ///< problem size of GEMM
        int gemm_k_iterations_1_,
        ///< destination accumulator tile
        FragmentC1& accum,
        ///< iterator over B1 operand in global memory
        IteratorB1 iterator_B1,
        ///< initial value of accumulator
        FragmentC1 const& src_accum)
    {
        // 2nd Gemm

        //
        // Prologue
        //
        // Perform accumulation in the 'd' output operand
        accum = src_accum;

        if (!prologue_done_) {
            _prologue(iterator_B1, gemm_k_iterations_1_, smem_iterator_B1_);
        } else if (!kSmemContainsEntireB) {
            // Restore the iterators increments

            int gemm_k_iterations_1 = gemm_k_iterations_1_;
            // Issue several complete stages
            CUTLASS_PRAGMA_UNROLL
            for (int stage = 0; stage < kNumStagesConcurrentLoad; ++stage, --gemm_k_iterations_1) {
                iterator_B1.set_iteration_index(0);
                this->smem_iterator_B1_.set_iteration_index(0);

                // Load for operand B
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < Detail::TBLoadIterationsB1; ++j) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int v = 0; v < IteratorB1::kAccessesPerVector; ++v) { ++iterator_B1; }
                    ++this->smem_iterator_B1_;
                }
                iterator_B1.add_tile_offset({1, 0});
                this->smem_iterator_B1_.add_tile_offset({1, 0});
            }
            iterator_B1.set_residual_tile(gemm_k_iterations_1 <= 1);
            iterator_B1.clear_mask(gemm_k_iterations_1 <= 0);
        }

        // DEPBAR+SYNC
        cutlass::arch::cp_async_wait<kNumStagesConcurrentLoad - 1>();
        __syncthreads();

        // remember that WarpFragmentAScale and WarpIteratorAScale are no-op/empty
        // if scaling is disabled.

        // Pair of fragments used to overlap shared memory loads and math
        // instructions
        WarpLoadedFragmentA1 warp_loaded_frag_A1[2];
        WarpLoadedFragmentA1Scale warp_loaded_frag_A1_scale[2];
        WarpLoadedFragmentB1 warp_loaded_frag_B1[2];
        WarpTransformedFragmentA1 warp_transformed_frag_A1[2];
        WarpTransformedFragmentB1 warp_transformed_frag_B1[2];

        Operator1 warp_mma1;

        warp_tile_iterator_A1_.load(warp_loaded_frag_A1[0]);
        ++warp_tile_iterator_A1_;

        warp_tile_iterator_A1_scale_.load(warp_loaded_frag_A1_scale[0]);
        ++warp_tile_iterator_A1_scale_;

        this->warp_tile_iterator_B_.set_kgroup_index(0);
        this->warp_tile_iterator_B_.load(warp_loaded_frag_B1[0]);
        ++this->warp_tile_iterator_B_;

        int smem_write_stage_idx = Base::kStages - 1;
        int smem_read_stage_idx = 0;

        warp_mma1.transform(
            warp_transformed_frag_A1[0],
            warp_transformed_frag_B1[0],
            FragmentAScaler::apply(warp_loaded_frag_A1[0], warp_loaded_frag_A1_scale[0]),
            warp_loaded_frag_B1[0]);

        // tf32x3 kernels use staging accumulation. warp_mma uses a temporary
        // accumulator and this temporary accumulator is added to the final
        // accumulator once in every mainloop iteration.
        plus<FragmentC1> plus_accum;

        FragmentC1 tmp_accum;

        if (platform::is_same<typename Operator1::MathOperator,
                              arch::OpMultiplyAddFastF32>::value ||
            platform::is_same<typename Operator1::MathOperator,
                              arch::OpMultiplyAddComplexFastF32>::value) {
            tmp_accum.clear();
        }

        //
        // Mainloop
        //

        CUTLASS_PRAGMA_UNROLL
        for (int gemm_k_iterations_1 = gemm_k_iterations_1_ - (Base::kStages - 1);
             gemm_k_iterations_1 > (-Base::kStages + 1);
             gemm_k_iterations_1--) {
            //
            // Loop over GEMM K dimension
            //

            // Computes a warp-level GEMM on data held in shared memory
            // Each "warp_mma_k" refers to a warp-level matrix multiply-accumulate
            CUTLASS_PRAGMA_UNROLL
            for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations1; ++warp_mma_k) {
                // Load warp-level tile from accumulator fragment (A)
                // or shared memory (operand B)
                this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) %
                                                             Base::kWarpGemmIterations1);
                // skip warp tile loading for the last kgroup (we are out of the buf)
                if (gemm_k_iterations_1 > (-Base::kStages + 2) ||
                    warp_mma_k < Base::kWarpGemmIterations1 - 1) {
                    warp_tile_iterator_A1_.load(warp_loaded_frag_A1[(warp_mma_k + 1) % 2]);
                    warp_tile_iterator_A1_scale_.load(
                        warp_loaded_frag_A1_scale[(warp_mma_k + 1) % 2]);
                    this->warp_tile_iterator_B_.load(warp_loaded_frag_B1[(warp_mma_k + 1) % 2]);
                }
                ++warp_tile_iterator_A1_;
                ++warp_tile_iterator_A1_scale_;
                ++this->warp_tile_iterator_B_;

                if (warp_mma_k > 0)
                    warp_mma1.transform(
                        warp_transformed_frag_A1[warp_mma_k % 2],
                        warp_transformed_frag_B1[warp_mma_k % 2],
                        FragmentAScaler::apply(warp_loaded_frag_A1[warp_mma_k % 2],
                                               warp_loaded_frag_A1_scale[warp_mma_k % 2]),
                        warp_loaded_frag_B1[warp_mma_k % 2]);

                if (platform::is_same<typename Operator1::MathOperator,
                                      arch::OpMultiplyAddFastF32>::value ||
                    platform::is_same<typename Operator1::MathOperator,
                                      arch::OpMultiplyAddComplexFastF32>::value) {
                    warp_mma1(tmp_accum,
                              warp_transformed_frag_A1[warp_mma_k % 2],
                              warp_transformed_frag_B1[warp_mma_k % 2],
                              tmp_accum);

                    if (warp_mma_k == 0) {
                        accum = plus_accum(accum, tmp_accum);
                        tmp_accum.clear();
                    }
                } else {
                    warp_mma1(accum,
                              warp_transformed_frag_A1[warp_mma_k % 2],
                              warp_transformed_frag_B1[warp_mma_k % 2],
                              accum);
                }

                // Issue global->shared copies for the this stage
                if (warp_mma_k < Base::kWarpGemmIterations1 - 1) {
                    int group_start_iteration_B1;

                    group_start_iteration_B1 = warp_mma_k * Detail::kAccessesPerGroupB1;

                    if (!kSmemContainsEntireB) {
                        copy_tiles_and_advance_1(iterator_B1, group_start_iteration_B1);
                    }
                }

                if (warp_mma_k + 2 == Base::kWarpGemmIterations1) {
                    int group_start_iteration_B1;
                    group_start_iteration_B1 = (warp_mma_k + 1) * Detail::kAccessesPerGroupB1;

                    if (!kSmemContainsEntireB) {
                        copy_tiles_and_advance_1(iterator_B1, group_start_iteration_B1);
                    }

                    // Inserts a memory fence between stages of cp.async instructions.
                    cutlass::arch::cp_async_fence();

                    // Waits until kStages-2 stages have committed.
                    arch::cp_async_wait<kNumStagesConcurrentLoad - 1>();
                    __syncthreads();

                    // Move to the next stage
                    iterator_B1.add_tile_offset({1, 0});

                    this->smem_iterator_B1_.add_tile_offset({1, 0});

                    // Add negative offsets to return iterators to the 'start' of the
                    // circular buffer in shared memory
                    if (!kSmemContainsEntireB) {
                        if (smem_write_stage_idx == (Base::kStages - 1)) {
                            this->smem_iterator_B1_.add_tile_offset({-Base::kStages, 0});
                            smem_write_stage_idx = 0;
                        } else {
                            ++smem_write_stage_idx;
                        }

                        if (smem_read_stage_idx == (Base::kStages - 1)) {
                            this->warp_tile_iterator_B_.add_tile_offset(
                                {-Base::kStages * Policy1::kPartitionsK *
                                     Base::kWarpGemmIterations1,
                                 0});
                            smem_read_stage_idx = 0;
                        } else {
                            ++smem_read_stage_idx;
                        }
                    }

                    iterator_B1.set_residual_tile(gemm_k_iterations_1 == 2);
                    iterator_B1.clear_mask(gemm_k_iterations_1 == 1);
                }

                // Do any conversions feeding the first stage at the end of the loop so
                // we can start right away on mma instructions
                if (warp_mma_k + 1 == Base::kWarpGemmIterations1)
                    warp_mma1.transform(
                        warp_transformed_frag_A1[(warp_mma_k + 1) % 2],
                        warp_transformed_frag_B1[(warp_mma_k + 1) % 2],
                        FragmentAScaler::apply(warp_loaded_frag_A1[(warp_mma_k + 1) % 2],
                                               warp_loaded_frag_A1_scale[(warp_mma_k + 1) % 2]),
                        warp_loaded_frag_B1[(warp_mma_k + 1) % 2]);
            }
        }

        if (platform::is_same<typename Operator1::MathOperator,
                              arch::OpMultiplyAddFastF32>::value ||
            platform::is_same<typename Operator1::MathOperator,
                              arch::OpMultiplyAddComplexFastF32>::value) {
            accum = plus_accum(accum, tmp_accum);
        }
    }
};

template <typename WarpShape,
          typename InstructionShape,
          typename RegularWarpIterator,
          typename Policy,
          typename Enable = void>
struct DefaultWarpIteratorAFromSharedMemory {
};

// TensorOp - Ampere half
template <typename RegularWarpIterator, typename Policy>
struct DefaultWarpIteratorAFromSharedMemory<
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    RegularWarpIterator,
    Policy,
    typename platform::enable_if<(sizeof_bits<typename RegularWarpIterator::Element>::value == 16 &&
                                  Policy::Operator::Policy::OpDelta::kRow == 1)>::type> {
    static constexpr auto kWarpSize = 32;
    using OpDelta = typename Policy::Operator::Policy::OpDelta;
    using WarpShape = cutlass::MatrixShape<32, 32>;

    using WarpIterator =
        cutlass::gemm::warp::WarpIteratorFromSmem<cutlass::gemm::Operand::kA,
                                                  typename RegularWarpIterator::Element>;
};

// TensorOp - Ampere f32
template <typename WarpShape, typename RegularWarpIterator, typename Policy>
struct DefaultWarpIteratorAFromSharedMemory<
    WarpShape,
    cutlass::gemm::GemmShape<16, 8, 8>,
    RegularWarpIterator,
    Policy,
    typename platform::enable_if<(sizeof_bits<typename RegularWarpIterator::Element>::value != 16 ||
                                  Policy::Operator::Policy::OpDelta::kRow != 1)>::type> {
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
    static constexpr auto kWarpSize = 32;
    using OpDelta = typename Policy::Operator::Policy::OpDelta;

    using WarpIterator = cutlass::gemm::warp::MmaTensorOpMultiplicandTileAccessIterator<
        cutlass::MatrixShape<WarpShape::kM, WarpShape::kK>,
        cutlass::gemm::Operand::kA,
        typename RegularWarpIterator::Element,
        cutlass::layout::RowMajor,
        cutlass::MatrixShape<InstructionShape::kM, InstructionShape::kK>,
        OpDelta::kRow,
        kWarpSize>;
};

// TensorOp - Volta
template <typename WarpShape, typename RegularWarpIterator, typename Policy>
struct DefaultWarpIteratorAFromSharedMemory<WarpShape,
                                            cutlass::gemm::GemmShape<16, 16, 4>,
                                            RegularWarpIterator,
                                            Policy> {
    using InstructionShape = cutlass::gemm::GemmShape<16, 16, 4>;
    static constexpr auto kWarpSize = 32;
    using OpDelta = typename Policy::Operator::Policy::OpDelta;

    using WarpIterator = cutlass::gemm::warp::MmaVoltaTensorOpMultiplicandTileIterator<
        cutlass::MatrixShape<32, 32>,  // MatrixShape<WarpShape::kM,
                                       // WarpShape::kK>,
        cutlass::gemm::Operand::kA,
        typename RegularWarpIterator::Element,
        cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<16, 32>,
        cutlass::MatrixShape<16, 4>,
        OpDelta::kRow,
        kWarpSize>;
};

// Simt
template <typename WarpShape, typename RegularWarpIterator, typename Policy>
struct DefaultWarpIteratorAFromSharedMemory<WarpShape,
                                            cutlass::gemm::GemmShape<1, 1, 1>,
                                            RegularWarpIterator,
                                            Policy> {
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    static constexpr auto kWarpSize = 32;

    // We just use the same iterator, as we reproduced the same shared-memory
    // schema. Just modify it to handle non-complete tiles.
    using WarpIterator = RegularWarpIterator;
};

// Converts a "regular" Mma into their counterpart from shared memory
template <typename Mma_,
          typename AccumulatorSharedStorage,
          /// whether or not to apply elementwise multiplication of operand A by
          /// another matrix in shared memory before usage in A @ B
          bool kScaleOperandA,
          bool kTransposeA = false>
struct DefaultMmaFromSharedMemory;

// Mma pipelined
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Transformation applied to A operand
    typename TransformA_,
    /// Transformation applied to B operand
    typename TransformB_,
    typename AccumulatorSharedStorage_,
    /// whether or not to apply elementwise multiplication of operand A by
    /// another matrix in shared memory before usage in A @ B
    bool kScaleOperandA,
    bool kTransposeA>
struct DefaultMmaFromSharedMemory<MmaPipelined<Shape_,
                                               IteratorA_,
                                               SmemIteratorA_,
                                               IteratorB_,
                                               SmemIteratorB_,
                                               ElementC_,
                                               LayoutC_,
                                               Policy_,
                                               TransformA_,
                                               TransformB_>,
                                  AccumulatorSharedStorage_,
                                  kScaleOperandA,
                                  kTransposeA> {
    static constexpr int kWarpSize = 32;
    using SmemAccumulatorLayout = cutlass::layout::RowMajor;

    using RegularMma = MmaPipelined<Shape_,
                                    IteratorA_,
                                    SmemIteratorA_,
                                    IteratorB_,
                                    SmemIteratorB_,
                                    ElementC_,
                                    LayoutC_,
                                    Policy_,
                                    TransformA_,
                                    TransformB_>;

    using WarpShape = typename Policy_::Operator::Shape;
    using InstructionShape = typename Policy_::Operator::InstructionShape;
    using ArchMmaOperator = typename Policy_::Operator;

    static constexpr bool kIsTransposedA = false;
    using WarpIteratorA =
        typename DefaultWarpIteratorAFromSharedMemory<WarpShape,
                                                      InstructionShape,
                                                      typename RegularMma::Operator::IteratorA,
                                                      Policy_>::WarpIterator;
    using IteratorB =
        typename cutlass::transform::threadblock::MakeIteratorResidualLast<IteratorB_>::Iterator;

    using Mma =
        typename cutlass::gemm::threadblock::MmaPipelinedFromSharedMemory<Shape_,
                                                                          WarpIteratorA,
                                                                          kScaleOperandA,
                                                                          AccumulatorSharedStorage_,
                                                                          IteratorB,
                                                                          SmemIteratorB_,
                                                                          ElementC_,
                                                                          LayoutC_,
                                                                          Policy_>;
};

template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    typename AccumulatorSharedStorage_,
    /// whether or not to apply elementwise multiplication of operand A by
    /// another matrix in shared memory before usage in A @ B
    bool kScaleOperandA,
    bool kTransposeA>
struct DefaultMmaFromSharedMemory<MmaMultistage<Shape_,
                                                IteratorA_,
                                                SmemIteratorA_,
                                                CacheOpA,
                                                IteratorB_,
                                                SmemIteratorB_,
                                                CacheOpB,
                                                ElementC_,
                                                LayoutC_,
                                                Policy_,
                                                Stages,
                                                SharedMemoryClear>,
                                  AccumulatorSharedStorage_,
                                  kScaleOperandA,
                                  kTransposeA> {
    static constexpr int kWarpSize = 32;

    using RegularMma = MmaMultistage<Shape_,
                                     IteratorA_,
                                     SmemIteratorA_,
                                     CacheOpA,
                                     IteratorB_,
                                     SmemIteratorB_,
                                     CacheOpB,
                                     ElementC_,
                                     LayoutC_,
                                     Policy_,
                                     Stages,
                                     SharedMemoryClear>;

    using WarpShape = typename Policy_::Operator::Shape;
    using InstructionShape = typename Policy_::Operator::InstructionShape;
    using WarpIteratorA_ =
        typename DefaultWarpIteratorAFromSharedMemory<WarpShape,
                                                      InstructionShape,
                                                      typename RegularMma::Operator::IteratorA,
                                                      Policy_>::WarpIterator;
    using WarpIteratorTranspose = TransposeWarpIterator<WarpIteratorA_>;
    static constexpr bool kIsTransposedA = WarpIteratorTranspose::kSupportsTranspose && kTransposeA;
    using WarpIteratorA = typename platform::
        conditional<kIsTransposedA, typename WarpIteratorTranspose::Iterator, WarpIteratorA_>::type;

    static int constexpr kMaxK = kIsTransposedA ? AccumulatorSharedStorage_::Shape::kM
                                                : AccumulatorSharedStorage_::Shape::kN;
    // Reduce the number of stages if we don't need that many
    static int constexpr kStagesMax = (kMaxK + int(Shape_::kK) - 1) / int(Shape_::kK);
    static int constexpr kStages = cutlass::const_min(Stages, kStagesMax);

    using IteratorB =
        typename cutlass::transform::threadblock::MakeIteratorResidualLast<IteratorB_>::Iterator;
    using Mma = typename cutlass::gemm::threadblock::MmaMultistageFromSharedMemory<
        Shape_,
        WarpIteratorA,
        kScaleOperandA,
        AccumulatorSharedStorage_,
        IteratorB,
        SmemIteratorB_,
        RegularMma::kCacheOpB,
        ElementC_,
        LayoutC_,
        Policy_,
        kStages,
        kMaxK>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename IteratorC,
          typename Operator,
          typename scalar_t,
          typename WarpShape_,
          typename ThreadblockShape_>
struct B2bGemm;

// Tensor Cores >= Sm75 specialization (Ampere ...)
template <  /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element type
    typename Element_,
    /// Layout of operand in memory
    typename Layout_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_,
    typename Operator,
    typename scalar_t,
    typename WarpShape_,
    typename ThreadblockShape_>
struct B2bGemm<
    cutlass::gemm::warp::
        MmaTensorOpAccumulatorTileIterator<Shape_, Element_, Layout_, InstructionShape_, OpDelta_>,
    Operator,
    scalar_t,
    WarpShape_,
    ThreadblockShape_> {
    using IteratorC = typename cutlass::gemm::warp::
        MmaTensorOpAccumulatorTileIterator<Shape_, Element_, Layout_, InstructionShape_, OpDelta_>;
    using FragmentC = typename IteratorC::Fragment;
    using InstructionShape = InstructionShape_;
    using WarpShape = WarpShape_;
    using ThreadblockShape = ThreadblockShape_;
    using accum_t = Element_;
    using lse_scalar_t = float;

    using SmemAccumulatorLayout = cutlass::layout::RowMajor;

    // Iterator to load accumulators (results of matmul in registers)
    using FragmentIteratorAccumulator = cutlass::epilogue::warp::FragmentIteratorTensorOp<
        WarpShape,
        InstructionShape,
        accum_t,
        typename Operator::Policy::Operator::FragmentC,
        cutlass::layout::RowMajor>;

    // Iterator to store to shared-memory
    using SmemIteratorD0 =
        typename cutlass::epilogue::warp::TileIteratorTensorOp<WarpShape,
                                                               InstructionShape,
                                                               scalar_t,  // accum_t,
                                                               SmemAccumulatorLayout>;
    using AccumulatorSharedStorage =
        cutlass::gemm::threadblock::AccumulatorSharedStorage<ThreadblockShape,
                                                             typename SmemIteratorD0::Element,
                                                             typename SmemIteratorD0::TensorLayout,
                                                             typename SmemIteratorD0::Padding>;
    // We need to provide an operation for the epilogue. Let's create an
    // operation that does nothing (ScaleType::Nothing), just converts
    // from accum_t (float) -> scalar_t (can be half)
    using OutputOpNoOp = cutlass::epilogue::thread::LinearCombination<
        typename SmemIteratorD0::Element,  // ElementOutput
        FragmentIteratorAccumulator::Fragment::kElements,
        accum_t,                           // ElementAccumulator
        typename SmemIteratorD0::Element,  // ElementCompute
        cutlass::epilogue::thread::ScaleType::Nothing>;
    using Epilogue = cutlass::epilogue::threadblock::EpilogueSmemAccumulator<
        SmemIteratorD0,
        FragmentIteratorAccumulator,
        SmemIteratorD0,  // ScaleBiasIterator
                         // - not used
        OutputOpNoOp>;

    // Epilogue 2: with LSE (for backwards pass)
    static int const kElementsPerAccess = 2;  // TODO: Why 2?
    using IteratorAccumulatorLSE = cutlass::transform::threadblock::VectorIterator<
        cutlass::transform::threadblock::PredicatedVectorAccessIterator<
            // Shape
            cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kN>,
            // WarpShape
            cutlass::MatrixShape<WarpShape::kM, WarpShape::kN>,
            lse_scalar_t,
            cutlass::layout::RowMajor,
            kElementsPerAccess>>;
    using EpilogueOpApplyLSE = cutlass::epilogue::thread::ApplyLogSumExp<
        scalar_t,      // ElementOutput_
        lse_scalar_t,  // ElementLSE_
        accum_t,       // ElementAccumulator_
        accum_t,       // ElementCompute_
        128 / cutlass::sizeof_bits<scalar_t>::value
        // FragmentIteratorAccumulator::Fragment::kElements
        // InstructionShape::kM * InstructionShape::kN / 32
        >;
    using EpilogueWithLSE =
        cutlass::epilogue::threadblock::EpilogueSmemAccumulator<SmemIteratorD0,
                                                                FragmentIteratorAccumulator,
                                                                IteratorAccumulatorLSE,
                                                                EpilogueOpApplyLSE>;

    static void CUTLASS_DEVICE accumToSmem(AccumulatorSharedStorage& shared_storage,
                                           FragmentC const& accum,
                                           int lane_id,
                                           cutlass::MatrixCoord const& tile_coords)
    {
        SmemIteratorD0 smem_iterator_attn(shared_storage.accum_ref(), lane_id);
        smem_iterator_attn.add_tile_offset(
            tile_coords * cutlass::MatrixCoord{SmemIteratorD0::TileIterations::kRow,
                                               SmemIteratorD0::TileIterations::kColumn});
        Epilogue epilogue;
        epilogue(OutputOpNoOp({}), smem_iterator_attn, accum);
    }

    static void CUTLASS_DEVICE accumApplyLSEToSmem(AccumulatorSharedStorage& shared_storage,
                                                   FragmentC& accum,
                                                   lse_scalar_t const* lse,
                                                   int32_t lse_extents,
                                                   int thread_id,
                                                   int warp_id,
                                                   int lane_id,
                                                   cutlass::MatrixCoord const& tile_coords)
    {
        constexpr int32_t kAlignLSE = 32;
        IteratorAccumulatorLSE iterator_lse(
            lse,
            {(int32_t)0, (int32_t)ceil_div(lse_extents, kAlignLSE) * kAlignLSE},
            thread_id,
            warp_id,
            cutlass::MatrixCoord{0, 0}  // offset
        );

        SmemIteratorD0 smem_iterator_attn(shared_storage.accum_ref(), lane_id);
        smem_iterator_attn.add_tile_offset(
            tile_coords * cutlass::MatrixCoord{SmemIteratorD0::TileIterations::kRow,
                                               SmemIteratorD0::TileIterations::kColumn});
        EpilogueWithLSE epilogue;
        EpilogueOpApplyLSE minus_lse_exp({});
        epilogue(minus_lse_exp,
                 smem_iterator_attn,
                 accum,
                 // scale - unused
                 iterator_lse,
                 // bias
                 iterator_lse);
    }
};

// Volta Specialization
// only supported for f16
template <typename Operator, typename WarpShape_, typename ThreadblockShape_>
struct B2bGemm<cutlass::gemm::warp::MmaVoltaTensorOpAccumulatorTileIterator<
                   cutlass::MatrixShape<32, 32>,
                   float,
                   cutlass::layout::RowMajor,
                   cutlass::gemm::GemmShape<16, 16, 4>,
                   cutlass::MatrixShape<1, 1>>,
               Operator,
               cutlass::half_t,
               WarpShape_,
               ThreadblockShape_> {
    using IteratorC = cutlass::gemm::warp::MmaVoltaTensorOpAccumulatorTileIterator<
        cutlass::MatrixShape<32, 32>,
        float,
        cutlass::layout::RowMajor,
        cutlass::gemm::GemmShape<16, 16, 4>,
        cutlass::MatrixShape<1, 1>>;
    using scalar_t = cutlass::half_t;
    using accum_t = IteratorC::Element;
    using WarpShape = WarpShape_;
    using ThreadblockShape = ThreadblockShape_;
    using FragmentC = IteratorC::Fragment;
    using lse_scalar_t = float;

    using SmemAccumulatorLayout = cutlass::layout::RowMajor;
    using SmemIteratorD0 =
        cutlass::epilogue::warp::TileIteratorVoltaTensorOp<WarpShape,
                                                           cutlass::gemm::GemmShape<32, 32, 4>,
                                                           scalar_t,
                                                           SmemAccumulatorLayout>;

    // // Storage in shared-memory for Q.Kt
    using AccumulatorSharedStorage = cutlass::gemm::threadblock::AccumulatorSharedStorage<
        ThreadblockShape,
        scalar_t,
        cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<
            16,
            32>,                    // typename SmemIteratorD0::TensorLayout,
        cutlass::MatrixShape<0, 0>  // Padding
        >;

    using OutputLayout = cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<16, 32>;
    using TensorRef = cutlass::TensorRef<scalar_t, OutputLayout>;
    using Policy = typename IteratorC::Policy;
    using Element = accum_t;
    // Those are MmaVoltaTensorOpAccumulatorTileIterator private fields
    // Let's copy their values
    static int const kElementsPerPartial = 4;
    using EleShapePerPatial =
        typename cutlass::platform::conditional<cutlass::platform::is_same<Element, float>::value,
                                                cutlass::MatrixShape<2, 2>,
                                                cutlass::MatrixShape<1, 4>>::type;
    static int const kElementsPerMma = 8;
    static int const kAccumulatorPatials = 2;
    using QuadShapePerPatialMma = cutlass::MatrixShape<4, 4>;

    static void CUTLASS_DEVICE accumToSmem(AccumulatorSharedStorage& shared_storage,
                                           FragmentC const& accum,
                                           int lane_id,
                                           cutlass::MatrixCoord const& tile_coords)
    {
        // ctor - from MmaVoltaTensorOpAccumulatorTileIterator
        TensorRef ref_(shared_storage.accum_ref());
        int quad = (lane_id >> 2);
        int lane_in_quad = (lane_id & 3);
        int accum_m, accum_n;

        if (cutlass::platform::is_same<Element, float>::value) {
            // (quad[2],quad[0])+lane_in_quad[0]
            accum_m = (((quad & 0x4) >> 1) + (quad & 0x1)) * 8 + (lane_in_quad & 1);
            // (quad[1])+lane_in_quad[1]
            accum_n = ((quad >> 1) & 0x1) * kElementsPerPartial * kAccumulatorPatials +
                      (lane_in_quad & 2);
        } else {
            accum_m = (((quad & 0x4) >> 1) + (quad & 0x1)) * 8 + lane_in_quad;  // (quad[2],quad[0])
            accum_n = ((quad >> 1) & 0x1) * kElementsPerPartial * kAccumulatorPatials;
        }
        cutlass::MatrixCoord lane_offset(accum_m, accum_n);

        // Tile offset
        ref_.add_coord_offset(tile_coords * cutlass::MatrixCoord({IteratorC::Shape::kRow,
                                                                  IteratorC::Shape::kColumn}));

        using AccessType = cutlass::Array<scalar_t, EleShapePerPatial::kColumn>;

        // store - from MmaVoltaTensorOpAccumulatorTileIterator
        CUTLASS_PRAGMA_UNROLL
        for (int tile_n = 0; tile_n < Policy::TileIterations::kColumn; ++tile_n) {
            CUTLASS_PRAGMA_UNROLL
            for (int tile_m = 0; tile_m < Policy::TileIterations::kRow; ++tile_m) {
                CUTLASS_PRAGMA_UNROLL
                for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
                        int mma_accum_start = (((tile_n * Policy::TileIterations::kRow + tile_m) *
                                                    Policy::MmaIterations::kColumn +
                                                mma_n) *
                                                   Policy::MmaIterations::kRow +
                                               mma_m) *
                                              kElementsPerMma;

                        CUTLASS_PRAGMA_UNROLL
                        for (int p = 0; p < kAccumulatorPatials; ++p) {
                            CUTLASS_PRAGMA_UNROLL
                            for (int m = 0; m < EleShapePerPatial::kRow; ++m) {
                                int accum_m = tile_m * Policy::InterleavedTile::kRow +
                                              mma_m * QuadShapePerPatialMma::kRow + m * 2;
                                int accum_n = tile_n * Policy::InterleavedTile::kColumn +
                                              mma_n * QuadShapePerPatialMma::kColumn +
                                              p * Policy::InterleavedTile::kColumn / 2;
                                int r = (accum_m + lane_offset.row());
                                AccessType to_store;
                                CUTLASS_PRAGMA_UNROLL
                                for (int n = 0; n < EleShapePerPatial::kColumn; ++n) {
                                    int idx = mma_accum_start + p * kElementsPerPartial +
                                              m * EleShapePerPatial::kColumn + n;
                                    int c = (accum_n + n + lane_offset.column());
                                    to_store[n] = scalar_t(accum[idx]);
                                }
                                int c = (accum_n + lane_offset.column());
                                assert(r < 32);
                                assert(c < 32);
                                *reinterpret_cast<AccessType*>(ref_.data() + ref_.offset({r, c})) =
                                    to_store;
                            }
                        }
                    }
                }
            }
        }
    }

    static void CUTLASS_DEVICE accumApplyLSEToSmem(AccumulatorSharedStorage& shared_storage,
                                                   typename IteratorC::Fragment& accum,
                                                   lse_scalar_t const* lse,
                                                   int lse_extent,
                                                   int thread_id,
                                                   int warp_id,
                                                   int lane_id,
                                                   cutlass::MatrixCoord const& tile_coords)
    {
        // Non-optimized way to apply LSE to registers
        // NOTE: accum is attn.T
        // TODO: Optimize for each architecture
        static constexpr int WarpSize = 32;
        using AccumLambdaIterator =
            typename DefaultMmaAccumLambdaIterator<IteratorC, accum_t, WarpSize>::Iterator;
        auto lane_offset = AccumLambdaIterator::get_lane_offset(lane_id, warp_id, tile_coords);

        cutlass::Array<lse_scalar_t, IteratorC::Fragment::kElements> lse_prefetched;
        lse_prefetched.clear();
        int rowIdx = 0;
        int colIdx = 0;
        AccumLambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) {
                ++rowIdx;
                colIdx = 0;
            },
            [&](int accum_m, int accum_n, int idx) {
                if (rowIdx == 1) {
                    lse_prefetched[colIdx] = accum_n < lse_extent
                                                 ? lse[accum_n]
                                                 : platform::numeric_limits<accum_t>::infinity();
                }
                accum[idx] = expf(accum[idx] - lse_prefetched[colIdx]);
                ++colIdx;
            },
            [&](int accum_m) {});
        accumToSmem(shared_storage, accum, lane_id, tile_coords);
    }
};

// Simt Specialization
// for f32 on Sm70-Sm75 and f16/f32 below

template <typename Operator,
          typename OperatorPolicy,
          typename scalar_t,
          typename WarpShape_,
          typename ThreadblockShape_>
struct B2bGemm<cutlass::gemm::warp::MmaSimtTileIterator<cutlass::MatrixShape<32, 32>,
                                                        cutlass::gemm::Operand::kC,
                                                        float,
                                                        cutlass::layout::RowMajor,
                                                        OperatorPolicy,
                                                        1,
                                                        1>,
               Operator,
               scalar_t,
               WarpShape_,
               ThreadblockShape_> {
    using IteratorC = cutlass::gemm::warp::MmaSimtTileIterator<cutlass::MatrixShape<32, 32>,
                                                               cutlass::gemm::Operand::kC,
                                                               float,
                                                               cutlass::layout::RowMajor,
                                                               OperatorPolicy,
                                                               1,
                                                               1>;
    using accum_t = typename IteratorC::Element;
    using WarpShape = WarpShape_;
    using ThreadblockShape = ThreadblockShape_;
    using FragmentC = typename IteratorC::Fragment;
    using lse_scalar_t = float;

    // Storage in shared-memory for Q.Kt
    using AccumulatorSharedStorage =
        cutlass::gemm::threadblock::AccumulatorSharedStorage<ThreadblockShape,
                                                             scalar_t,
                                                             cutlass::layout::ColumnMajor,
                                                             cutlass::MatrixShape<0, 0>  // Padding
                                                             >;

    static void CUTLASS_DEVICE accumToSmem(AccumulatorSharedStorage& shared_storage,
                                           FragmentC const& accum,
                                           int lane_id,
                                           cutlass::MatrixCoord const& tile_coords)
    {
        using Policy = typename IteratorC::Policy;
        using Element = typename IteratorC::Element;
        using Iterations = typename IteratorC::Iterations;
        using Delta = typename IteratorC::Delta;

        auto ref_ = shared_storage.accum_ref();
        // ctor - MmaSimtTileIterator
        // compute offset based on thread ID and lane layout
        typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

        MatrixCoord lane_offset = lane_layout.inverse(lane_id) *
                                  MatrixCoord(Policy::LaneMmaShape::kM, Policy::LaneMmaShape::kN);

        ref_.add_coord_offset(lane_offset);

        // Tile offset
        ref_.add_coord_offset(tile_coords * cutlass::MatrixCoord({IteratorC::Shape::kRow,
                                                                  IteratorC::Shape::kColumn}));

        // store - MmaSimtTileIterator
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Iterations::kColumn; ++mma_n) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < Policy::LaneMmaShape::kN; ++n) {
                CUTLASS_PRAGMA_UNROLL
                for (int mma_m = 0; mma_m < Iterations::kRow; ++mma_m) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int m = 0; m < Policy::LaneMmaShape::kM; ++m) {
                        int r = Policy::LaneMmaShape::kM * (mma_m * Policy::WarpShape::kRow) + m;
                        int c = mma_n * Delta::kColumn + n;
                        int idx = n + Policy::LaneMmaShape::kN *
                                          (mma_n + Iterations::kColumn *
                                                       (m + mma_m * Policy::LaneMmaShape::kM));
                        ref_.at({r, c}) = scalar_t(accum[idx]);
                    }
                }
            }
        }
    }

    static void CUTLASS_DEVICE accumApplyLSEToSmem(AccumulatorSharedStorage& shared_storage,
                                                   typename IteratorC::Fragment& accum,
                                                   lse_scalar_t const* lse,
                                                   int lse_extent,
                                                   int thread_id,
                                                   int warp_id,
                                                   int lane_id,
                                                   cutlass::MatrixCoord const& tile_coords)
    {
        // Non-optimized way to apply LSE to registers
        // NOTE: accum is attn.T
        // TODO: Optimize for each architecture
        static constexpr int WarpSize = 32;
        using AccumLambdaIterator =
            typename DefaultMmaAccumLambdaIterator<IteratorC, accum_t, WarpSize>::Iterator;
        auto lane_offset = AccumLambdaIterator::get_lane_offset(lane_id, warp_id, tile_coords);

        cutlass::Array<lse_scalar_t, IteratorC::Fragment::kElements> lse_prefetched;
        lse_prefetched.clear();
        int rowIdx = 0;
        int colIdx = 0;
        AccumLambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) {
                ++rowIdx;
                colIdx = 0;
            },
            [&](int accum_m, int accum_n, int idx) {
                if (rowIdx == 1) {
                    lse_prefetched[colIdx] = accum_n < lse_extent
                                                 ? lse[accum_n]
                                                 : platform::numeric_limits<accum_t>::infinity();
                }
                accum[idx] = expf(accum[idx] - lse_prefetched[colIdx]);
                ++colIdx;
            },
            [&](int accum_m) {});
        accumToSmem(shared_storage, accum, lane_id, tile_coords);
    }
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
