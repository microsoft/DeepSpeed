#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <iostream>
#include <vector>
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"
#include "gemm_test.h"
#include <stack>

#define WARP_SIZE 32

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                                                          \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x) \
        for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); j += blockDim.y * gridDim.y)

#define DS_CUDA_NUM_THREADS 512
#define DS_MAXIMUM_NUM_BLOCKS 262144

inline int DS_GET_BLOCKS(const int N)
{
    return std::max(
        std::min((N + DS_CUDA_NUM_THREADS - 1) / DS_CUDA_NUM_THREADS, DS_MAXIMUM_NUM_BLOCKS),
        // Use at least 1 block, since CUDA does not allow empty block
        1);
}

class Context {
public:
    Context() : _workspace(nullptr), _seed(42), _curr_offset(0), _prev_offset(0), _offset_stored(false), _grad_enable(true)
    {
        curandCreateGenerator(&_gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(_gen, 123);
        if (cublasCreate(&_cublasHandle) != CUBLAS_STATUS_SUCCESS) {
            auto message = std::string("Fail to create cublas handle.");
            std::cerr << message << std::endl;
            throw std::runtime_error(message);
        }
    }

    virtual ~Context()
    {
        cublasDestroy(_cublasHandle);
        cudaFree(_workspace);
    }

    static Context& Instance()
    {
        static Context _ctx;
        return _ctx;
    }

    void GenWorkSpace(size_t size)
    {
        if (!_workspace) {
            assert(_workspace == nullptr);
            cudaMalloc(&_workspace, size);
        } else if (_workSpaceSize != size) {
            cudaFree(_workspace);
            cudaMalloc(&_workspace, size);
        }

        _workSpaceSize = size;
    }

    void* GetWorkSpace() { return _workspace; }

    curandGenerator_t& GetRandGenerator() { return _gen; }

    cudaStream_t GetCurrentStream()
    {
        // get current pytorch stream.
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        return stream;
    }

    cublasHandle_t GetCublasHandle() { return _cublasHandle; }

    std::pair<uint64_t, uint64_t> IncrementOffset(uint64_t offset_inc)
    {
        //if(_local_rank == 0)printf("current offset is %lu\n", _curr_offset);
        uint64_t offset = _curr_offset;
        if(_grad_enable)
            _backward_offsets.push(_curr_offset);
        _curr_offset += offset_inc;
        return std::pair<uint64_t, uint64_t>(_seed, offset);
    }

    std::pair<uint64_t, uint64_t> RestoreBackwardRandOffset() { 
        if(_backward_offsets.empty())        
            throw std::runtime_error("Can't restore random offset!");

        auto offset = _backward_offsets.top();
        _backward_offsets.pop(); 
        return std::pair<uint64_t, uint64_t>(_seed, offset);
    }
    
    inline void Enable_Grad(bool grad_enable) { _grad_enable = grad_enable;}

    inline void StoreRandOffset() { 
        //if(_local_rank == 0)printf("storing offset %lu\n", _curr_offset);
        _prev_offset = _curr_offset; }

    inline void RestoreRandOffset(bool grad_enable) { 
        if(grad_enable)
        {
            if(!_offsets.empty()){
                _curr_offset = _offsets.top();
                _offsets.pop();
                _offset_stored = _offsets.empty();
            }
        }
        else
        {
            if(_offset_stored){
               // if(_local_rank == 0)printf("offset restored from %lu to %lu\n", _curr_offset, _prev_offset);
                _curr_offset = _prev_offset; 
                _offset_stored = false; 
            }
            _offsets.push(_curr_offset);
        }
    }

    void SetSeed(uint64_t new_seed) { _seed = new_seed; }

    void TestGemmFP16(bool test_gemm, int batch_size, int seq_len, int head_num, int size_per_head)
    {
        // avoid rerun.
        if (_gemm_algos.size() > 0) return;

        if (test_gemm) {
            cublasHandle_t handle = GetCublasHandle();

            std::unique_ptr<GemmTest<__half>> test_qkv_fw(
                new GemmTest<__half>(batch_size * seq_len,      // M
                                     head_num * size_per_head,  // N
                                     head_num * size_per_head,  // K
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     handle));

            std::unique_ptr<GemmTest<__half>> test_inter(
                new GemmTest<__half>(batch_size * seq_len,          // M
                                     4 * head_num * size_per_head,  // N
                                     head_num * size_per_head,      // K
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     handle));

            std::unique_ptr<GemmTest<__half>> test_output(
                new GemmTest<__half>(batch_size * seq_len,          // M
                                     head_num * size_per_head,      // N
                                     4 * head_num * size_per_head,  // K
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     handle));

            std::unique_ptr<StridedGemmTest<__half>> test_attn_scores(
                new StridedGemmTest<__half>(batch_size * head_num,  // batch
                                            seq_len,                // M
                                            seq_len,                // N
                                            size_per_head,          // K
                                            CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            handle));

            std::unique_ptr<StridedGemmTest<__half>> test_attn_context(
                new StridedGemmTest<__half>(batch_size * head_num,  // batch
                                            size_per_head,          // M
                                            seq_len,                // N
                                            seq_len,                // K
                                            CUBLAS_OP_N,
                                            CUBLAS_OP_N,
                                            handle));

            _gemm_algos.push_back(test_qkv_fw->TestAlgo(100));
            _gemm_algos.push_back(test_inter->TestAlgo(100));
            _gemm_algos.push_back(test_output->TestAlgo(100));
            _gemm_algos.push_back(test_attn_scores->TestAlgo(100));
            _gemm_algos.push_back(test_attn_context->TestAlgo(100));
        } else {
            // Use default algo.
            _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
            _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
            _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
            _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
            _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
        }
    }
    inline int Get_local_rank() const { return _local_rank; }
    inline void Set_local_rank(int local_rank) { _local_rank = local_rank; }
    const std::vector<std::array<int, 3>>& GetGemmAlgos() const { return _gemm_algos; }

private:
    curandGenerator_t _gen;
    cublasHandle_t _cublasHandle;
    void* _workspace;
    uint64_t _seed;
    uint64_t _curr_offset;
    size_t _workSpaceSize;
    uint64_t _prev_offset;
    std::stack<uint64_t> _offsets;
    std::stack<uint64_t> _backward_offsets;
    bool _offset_stored;
    bool _grad_enable;
    int _local_rank;
    std::vector<std::array<int, 3>> _gemm_algos;
};
