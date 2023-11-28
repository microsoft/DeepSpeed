#include "Kernel_QuantGEMM.cuh"
#include "Kernel_Reduction.cuh"
#include "GenMatrix_QuantLLM.h"

#include <stdio.h>
#include <assert.h>

template<typename TilingConfig, typename OutputDataType>
static void Kernel_QuantGEMM_Ex(cudaStream_t    stream,
                                const uint4     *Weight1,
                                const uint4     *Weight2,
                                const half      *Scales,
                                const int       QUANT_GROUP_SIZE_DIVIDED_BY_64,
                                const half      *B,
                                OutputDataType  *C,
                                const size_t    M_Global,
                                const size_t    N_Global,
                                const size_t    K_Global, 
                                int             Split_K) 
{
    #ifdef DEBUG_MODE
        printf("\n");
        printf("Launcher.cu->Kernel_QuantGEMM_Ex():\n");
        printf("M: %d, N: %d, K: %d, SplitK: %d, QUANT_GROUP_SIZE_DIVIDED_BY_64: %d\n", M_Global, N_Global, K_Global, Split_K, QUANT_GROUP_SIZE_DIVIDED_BY_64);
        printf("TILE_M: %d, TILE_K: %d, TILE_N: %d\n", TilingConfig::TILE_M, TilingConfig::TILE_K, TilingConfig::TILE_N);
        assert(N_Global % TilingConfig::TILE_N          == 0);
        assert(M_Global*Split_K % TilingConfig::TILE_M  == 0);
        assert(K_Global % TilingConfig::TILE_K          == 0);
    #endif
    static int SHMEM_SZ = max(TilingConfig::SMEM_SIZE_B_TILE+SMEM_SIZE_A1_TILE+SMEM_SIZE_A2_TILE, TilingConfig::SMEM_SIZE_C_TILE);
    cudaFuncSetAttribute(QUANT_GEMM_Kernel<TilingConfig, OutputDataType>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    int  dimN = (N_Global-1) / TilingConfig::TILE_N + 1;
    int  dimM = M_Global * Split_K / TilingConfig::TILE_M;
    dim3 GridDim(dimN, dimM, 1);
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);
    //
    #ifdef DEBUG_MODE
        printf("GridDim.x: %d, GridDim.y: %d, GridDim.z: %d, BlockDim.x: %d, BlockDim.y: %d, BlockDim.z: %d SHMEM_SZ: %d\n",
                GridDim.x, GridDim.y, GridDim.z, BlockDim.x, BlockDim.y, BlockDim.z, SHMEM_SZ);
        printf("\n");
    #endif
    QUANT_GEMM_Kernel<TilingConfig, OutputDataType><<<GridDim, BlockDim, SHMEM_SZ, stream>>>
                    (Weight1, Weight2, Scales, QUANT_GROUP_SIZE_DIVIDED_BY_64, B, C, M_Global, N_Global, K_Global, Split_K);
}

/*
 *half* Reduction_Workspace:  1. Requiring an extra memory space in device memory for un-reducted intermediate output tensors
 *                            2. Reduction_Workspace_Size = Split_K * M_Global * N_Global * sizeof(fp32)
 */
cudaError_t QuantGEMM_API(cudaStream_t    stream,
                          const uint4     *Weight1,
                          const uint4     *Weight2,
                          const half      *Scales,
                          const int       QUANT_GROUP_SIZE_DIVIDED_BY_64,
                          const half      *B,
                          half            *C,
                          const size_t    M_Global,
                          const size_t    N_Global,
                          const size_t    K_Global, 
                          float           *Reduction_Workspace,  // Identical workspace for all QuantGEMM kernel launches
                          int             Split_K)
{
    if(N_Global<=0) {printf("QuantLLM_API Error: Unsupported N dimension %d!\n", N_Global); return cudaErrorUnknown;}

    // Work around to support more N shapes: Pretending that the input is 2^n
    int N_PowerOf2;
    if(N_Global>0 &&  N_Global<=8)      N_PowerOf2 = 8;
    if(N_Global>8 &&  N_Global<=16)     N_PowerOf2 = 16;
    if(N_Global>16 && N_Global<=32)     N_PowerOf2 = 32;
    if(N_Global>32 && N_Global<=64)     N_PowerOf2 = 64;
    if(N_Global>64 && N_Global<=128)    N_PowerOf2 = 128;
    if(N_Global>128)                    N_PowerOf2 = ((N_Global-1)/128+1) * 128;
    //printf("N_Global:%d N_PowerOf2:%d\n", N_Global, N_PowerOf2);

    if (Split_K == 1) {
        switch (N_PowerOf2) {
            case 8:     Kernel_QuantGEMM_Ex<TilingConfig<4, 1, 1>, half>(stream, Weight1, Weight2, Scales, QUANT_GROUP_SIZE_DIVIDED_BY_64, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 16:    Kernel_QuantGEMM_Ex<TilingConfig<4, 1, 2>, half>(stream, Weight1, Weight2, Scales, QUANT_GROUP_SIZE_DIVIDED_BY_64, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 32:    Kernel_QuantGEMM_Ex<TilingConfig<4, 1, 4>, half>(stream, Weight1, Weight2, Scales, QUANT_GROUP_SIZE_DIVIDED_BY_64, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 64:    Kernel_QuantGEMM_Ex<TilingConfig<4, 1, 8>, half>(stream, Weight1, Weight2, Scales, QUANT_GROUP_SIZE_DIVIDED_BY_64, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 128:   Kernel_QuantGEMM_Ex<TilingConfig<4, 1, 8>, half>(stream, Weight1, Weight2, Scales, QUANT_GROUP_SIZE_DIVIDED_BY_64, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            default:    if (N_PowerOf2 % 128 != 0) {
                            printf("QuantLLM_API Error: Unsupported N dimension %d!\n", N_PowerOf2);
                            return cudaErrorUnknown;
                        }
                        Kernel_QuantGEMM_Ex<TilingConfig<4, 1, 8>, half>(stream, Weight1, Weight2, Scales, QUANT_GROUP_SIZE_DIVIDED_BY_64, B, C, M_Global, N_Global, K_Global, Split_K);  break;
        }
    }
    else {
        switch (N_PowerOf2) {
            case 8:     Kernel_QuantGEMM_Ex<TilingConfig<4, 1, 1>, float>(stream, Weight1, Weight2, Scales, QUANT_GROUP_SIZE_DIVIDED_BY_64, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 16:    Kernel_QuantGEMM_Ex<TilingConfig<4, 1, 2>, float>(stream, Weight1, Weight2, Scales, QUANT_GROUP_SIZE_DIVIDED_BY_64, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 32:    Kernel_QuantGEMM_Ex<TilingConfig<4, 1, 4>, float>(stream, Weight1, Weight2, Scales, QUANT_GROUP_SIZE_DIVIDED_BY_64, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 64:    Kernel_QuantGEMM_Ex<TilingConfig<4, 1, 8>, float>(stream, Weight1, Weight2, Scales, QUANT_GROUP_SIZE_DIVIDED_BY_64, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 128:   Kernel_QuantGEMM_Ex<TilingConfig<4, 1, 8>, float>(stream, Weight1, Weight2, Scales, QUANT_GROUP_SIZE_DIVIDED_BY_64, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            default:    if (N_PowerOf2 % 128 != 0) {
                            printf("QuantLLM_API Error: Unsupported N dimension %d!\n", N_PowerOf2);
                            return cudaErrorUnknown;
                        }
                        Kernel_QuantGEMM_Ex<TilingConfig<4, 1, 8>, float>(stream, Weight1, Weight2, Scales, QUANT_GROUP_SIZE_DIVIDED_BY_64, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
        }
        // Reduction for SplitK
        dim3 GridDim((M_Global * N_Global) / REDUCTION_ELEMENT_PER_THREADBLOCK, 1, 1);
        dim3 BlockDim(WARP_SIZE, 1, 1);
        SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    }
    return cudaGetLastError();
}
