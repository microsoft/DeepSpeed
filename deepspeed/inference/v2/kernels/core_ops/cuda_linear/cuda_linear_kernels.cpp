// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <ATen/cuda/CUDAContext.h>

#include "cuda_linear_kernels.h"

namespace {

// For bit-level debugging.
template <typename T>
void printBits(T num)
{
    char bits[sizeof(T) * 8 + 1] = {'\0'};
    for (int bit = 0; bit < (sizeof(T) * 8); bit++) {
        bits[sizeof(T) * 8 - 1 - bit] = '0' + (num & 0x01);
        num = num >> 1;
    }
    printf("%s\n", bits);
}

void printBits(half num)
{
    char bits[sizeof(half) * 8 + 1] = {'\0'};
    auto int_num = *reinterpret_cast<uint16_t*>(&num);
    for (int bit = 0; bit < (sizeof(half) * 8); bit++) {
        bits[sizeof(half) * 8 - 1 - bit] = '0' + (int_num & 0x01);
        int_num = int_num >> 1;
    }
    printf("%s\n", bits);
}

// Utils to prepack FP16 weights into continuous FP6 values.

// TODO: the following cast seems not the same with that in qtorch:
// https://github.com/Tiiiger/QPyTorch/blob/f58bba72113e696099ef3e15e06cf421a06ff289/qtorch/quant/quant_cuda/float_kernel.cu#L41
void Cast_FP16_FP6(uint16_t* FP16x4, uint8_t* FP6x4)
{
    // Constants for FP6
    constexpr int exponent_nbits_fp6 = 3;
    constexpr int mantissa_nbits_fp6 = 2;
    constexpr int exp_bias_fp6 = (1 << (exponent_nbits_fp6 - 1)) - 1;
    // constexpr int max_exponent_fp6 = (1 << exponent_nbits_fp6) - 2;
    // Constants for FP16
    constexpr int exponent_nbits_fp16 = 5;
    constexpr int mantissa_nbits_fp16 = 10;
    constexpr int exp_bias_fp16 = (1 << (exponent_nbits_fp16 - 1)) - 1;

    int fp6_temp[4];

    for (int i = 0; i < 4; ++i) {
        uint16_t source = FP16x4[i];
        // It is not safe to do shift operation on uint16_t. So we promote it to int.
        int source_promote = int(source);

        int sign = (source_promote >> 15);
        // Extracting exponent represented in FP16. The sign mask 0x7FFF is '0111 1111 1111 1111'
        int exp = (source_promote & 0x7FFF) >> mantissa_nbits_fp16;
        // Extracting mantissa represented in FP16
        int mant = source_promote & ((1 << mantissa_nbits_fp16) - 1);

        int new_exp;
        int new_mant;
        if (exp == 0 || exp == ((1 << exponent_nbits_fp16) - 1)) {
            // When all bits of exponent are zero, the calculation of the float value will not
            // include the spacific value of the exponent. Thus we can just copy the value to the
            // new variable after bit cutting.
            // TODO: a problem here is that the mantissa actually affects the results when all bits
            // of exponent are one. We need to consider this case in the next version if it matters
            // in practice.
            new_exp = exp >> (exponent_nbits_fp16 - exponent_nbits_fp6);
            new_mant = mant >> (mantissa_nbits_fp16 - mantissa_nbits_fp6);
        } else {
            int target_exp = int(exp) - int(exp_bias_fp16);
            constexpr int min_exp_fp6 = -((1 << (exponent_nbits_fp6 - 1)) - 2);
            constexpr int max_exp_fp6 = (1 << (exponent_nbits_fp6 - 1)) - 1;
            if (target_exp < min_exp_fp6) {
                // The exponent is too small to be represented in FP6. We need to round it to zero.
                // Keep the sign.
                // TODO: do we round it to zero or the smallest FP6 value?
                new_exp = 0;
                new_mant = 0;
            } else if (target_exp > max_exp_fp6) {
                // The exponent is too large to be represented in FP6. We need to round it to the
                // largest value. Keep the sign.
                new_exp = max_exp_fp6 + exp_bias_fp6;
                new_mant = (1 << mantissa_nbits_fp6) - 1;
            } else {
                new_exp = target_exp + exp_bias_fp6;
                new_mant = mant >> (mantissa_nbits_fp16 - mantissa_nbits_fp6);
            }

#if 0
            if (target_exp < min_exp_fp6) {
                uint16_t casted = (sign << (exponent_nbits_fp6 + mantissa_nbits_fp6)) |
                                (new_exp << mantissa_nbits_fp6) | new_mant;
                printf("%f exp too small, new value is: %f\n",
                    __half2float(*((half*)(&source))),
                    __half2float(*((half*)(&casted))));
                printBits(source);
                printBits(casted);
            } else if (target_exp > max_exp_fp6) {
                uint16_t casted = (sign << (exponent_nbits_fp6 + mantissa_nbits_fp6)) |
                                (new_exp << mantissa_nbits_fp6) | new_mant;
                printf("%f exp too large, new value is: %f\n",
                    __half2float(*((half*)(&source))),
                    __half2float(*((half*)(&casted))));
                printBits(source);
                printBits(casted);
            }
#endif
        }

        fp6_temp[i] = (sign << (exponent_nbits_fp6 + mantissa_nbits_fp6)) |
                      (new_exp << mantissa_nbits_fp6) | new_mant;
    }
    // Pack the values
    FP6x4[0] = fp6_temp[0] << 2 | (fp6_temp[1] >> 4);
    FP6x4[1] = (fp6_temp[1] & 0x0F) << 4 | (fp6_temp[2] >> 2);
    FP6x4[2] = (fp6_temp[2] & 0x03) << 6 | fp6_temp[3];
}

/*
 * Inputs:
 * (1) uint16_t Weight_16bit[M*K]
 * Outputs:
 * (1) unsigned char Weight_6bit[M*K*6/8]
 */
void PackMatrix_Weight_FP6(uint16_t* Weight_16bit, uint8_t* Weight_6bit, size_t M, size_t K)
{
#pragma omp parallel for
    for (auto m = 0; m < M; m++) {
        uint8_t* ptr_6bit = Weight_6bit + m * K * 6 / 8;
        uint16_t* ptr_16bit = Weight_16bit + m * K;
        for (auto k = 0; k < K; k += 4) {
            Cast_FP16_FP6(ptr_16bit, ptr_6bit);
            ptr_16bit += 4;
            ptr_6bit += 3;
        }
    }
}

}  // namespace

cudaError_t QuantGEMM_API(
    cudaStream_t stream,
    const uint4* Weight1,
    const uint4* Weight2,
    const half* Scales,
    const half* B,
    half* C,
    const size_t M_Global,
    const size_t N_Global,
    const size_t K_Global,
    float* Reduction_Workspace,  // Identical workspace for all QuantGEMM kernel launches
    int Split_K);

void cuda_wf6af16_linear(torch::Tensor& output,
                         torch::Tensor& hidden_states,
                         torch::Tensor& weights_2bit,
                         torch::Tensor& weights_4bit,
                         torch::Tensor& scales,
                         torch::Tensor& workspace,
                         int M,
                         int N,
                         int K,
                         int split_k)
{
    TORCH_CHECK(weights_2bit.device().type() == torch::kCUDA, "weight_2bit must be on CUDA");
    TORCH_CHECK(weights_4bit.device().type() == torch::kCUDA, "weight_4bit must be on CUDA");
    TORCH_CHECK(hidden_states.device().type() == torch::kCUDA, "X must be on CUDA");
    TORCH_CHECK(scales.device().type() == torch::kCUDA, "scales must be on CUDA");

    auto status = QuantGEMM_API(at::cuda::getCurrentCUDAStream(),
                                (uint4*)(weights_2bit.data_ptr<uint8_t>()),
                                (uint4*)(weights_4bit.data_ptr<uint8_t>()),
                                (half*)(scales.data_ptr<at::Half>()),
                                (half*)(hidden_states.data_ptr<at::Half>()),
                                (half*)(output.data_ptr<at::Half>()),
                                M,
                                N,
                                K,
                                workspace.data_ptr<float>(),
                                split_k);
    if (status != cudaSuccess) {
        AT_ERROR("QuantGEMM_API failed with error: ", cudaGetErrorString(status));
    }
}

void GenMatrix_Weight_FP6(unsigned char* Weight_6bit,
                          unsigned char* Weight_2bit,
                          unsigned char* Weight_4bit,
                          size_t M,
                          size_t K);

/*
 * Inputs:
 * (1) torch::Tensor weight[M, K] in FP16
 * Outputs:
 * (1) torch::Tensor weight_2bit and weight_4bit
 */
std::vector<torch::Tensor> preprocess_weight(torch::Tensor& Weight)
{
    TORCH_CHECK(Weight.dim() == 2, "weight must be 2-dimensional");
    TORCH_CHECK(Weight.scalar_type() == torch::kFloat16, "weight must be FP16");
    TORCH_CHECK(Weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(Weight.device().type() == torch::kCPU, "weight must be on CPU");
    auto M = Weight.size(0);
    auto K = Weight.size(1);
    TORCH_CHECK(K % 4 == 0, "K must be multiple of 4");

    // Pack Weight
    auto Weight_ptr = Weight.data_ptr<at::Half>();
    std::vector<uint8_t> Weight_6bit_Packed(M * K * 6 / 8);
    PackMatrix_Weight_FP6((uint16_t*)Weight_ptr, Weight_6bit_Packed.data(), M, K);

    // Split Weight
    auto Weight_2bit = torch::empty({M * K * 2 / 8}, torch::kUInt8);
    auto Weight_4bit = torch::empty({M * K * 4 / 8}, torch::kUInt8);
    GenMatrix_Weight_FP6(Weight_6bit_Packed.data(),
                         Weight_2bit.data_ptr<uint8_t>(),
                         Weight_4bit.data_ptr<uint8_t>(),
                         M,
                         K);

    return {Weight_2bit, Weight_4bit};
}