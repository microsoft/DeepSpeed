#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <iostream>
#include <vector>
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"

#include </usr/local/mpi/include/mpi.h>
#include <THC/THC.h>
#include <cuda.h>
#include <nccl.h>
#include <stdlib.h>
#include <sys/time.h>
#include <map>
#include <memory>
#include <stack>
#include <string>

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
    Context()
        : _workspace(nullptr),
          _seed(42),
          _curr_offset(0),
          _comm_stream(0),
          _comp_stream(0),
          _comm_created(false)
    {
        curandCreateGenerator(&_gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(_gen, 123);
        if (cublasCreate(&_cublasHandle) != CUBLAS_STATUS_SUCCESS) {
            auto message = std::string("Fail to create cublas handle.");
            std::cerr << message << std::endl;
            throw std::runtime_error(message);
        }
        cublasSetMathMode(_cublasHandle, CUBLAS_TENSOR_OP_MATH);
        cudaEventCreate(&_comp_event, (cudaEventDisableTiming | cudaEventBlockingSync));
        cudaEventCreate(&_comm_event, (cudaEventDisableTiming | cudaEventBlockingSync));
    }

    virtual ~Context()
    {
        cublasDestroy(_cublasHandle);
        cudaFree(_workspace);
        ncclCommDestroy(_nccl_comm);
        MPI_Group_free(&_group);
        // MPI_Comm_free(&_comm);
        cudaEventDestroy(_comp_event);
        cudaEventDestroy(_comm_event);
    }

    static Context& Instance()
    {
        static Context _ctx;
        return _ctx;
    }

    void create_comm_group(std::vector<int> comm_ranks, int rank)
    {
        if (_comm_created) return;
        int world_rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        _comm = MPI_COMM_WORLD;
        MPI_Comm_group(_comm, &_group);

        unsigned num_ranks = comm_ranks.size();

        if (num_ranks < world_size) {
            auto total_group = _group;
            MPI_Group_incl(total_group, num_ranks, comm_ranks.data(), &_group);
            MPI_Group_free(&total_group);
        } else if (num_ranks > world_size) {
            auto message = std::string(
                "Fail to create comm group (number of ranks is higher than world_size).");
            std::cerr << message << std::endl;
            throw std::runtime_error(message);
        }

        ncclUniqueId _nccl_uid;
        ncclGetUniqueId(&_nccl_uid);

        MPI_Bcast((void*)&_nccl_uid, sizeof(ncclUniqueId), MPI_BYTE, 0, _comm);

        ncclCommInitRank(&_nccl_comm, num_ranks, _nccl_uid, rank);

        _comm_created = true;
    }
    inline ncclComm_t GetNCCLComm() { return _nccl_comm; }

    inline void barrier() { MPI_Barrier(_comm); }

    inline void SynchComp()
    {
        cudaEventRecord(_comp_event, _comp_stream);
        cudaStreamWaitEvent(_comm_stream, _comp_event, 0);
    }
    inline void SynchComm()
    {
        cudaEventRecord(_comm_event, _comm_stream);
        cudaStreamWaitEvent(_comp_stream, _comm_event, 0);
    }
    void GenWorkSpace(size_t size)
    {
        if (!_workspace) {
            assert(_workspace == nullptr);
            cudaMalloc(&_workspace, size);
        } else if (_workSpaceSize < size) {
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

    cudaStream_t GetCommStream(bool async_op = false)
    {
        if (!_comm_stream)
            _comm_stream = async_op ? at::cuda::getStreamFromPool(true)
                                    : at::cuda::getCurrentCUDAStream();
        return _comm_stream;
    }

    cublasHandle_t GetCublasHandle() { return _cublasHandle; }

    std::pair<uint64_t, uint64_t> IncrementOffset(uint64_t offset_inc)
    {
        uint64_t offset = _curr_offset;
        _curr_offset += offset_inc;
        return std::pair<uint64_t, uint64_t>(_seed, offset);
    }

    void SetSeed(uint64_t new_seed) { _seed = new_seed; }

    const std::vector<std::array<int, 3>>& GetGemmAlgos() const { return _gemm_algos; }

private:
    curandGenerator_t _gen;
    cublasHandle_t _cublasHandle;
    cudaEvent_t _comp_event;
    cudaEvent_t _comm_event;

    void* _workspace;
    uint64_t _seed;
    uint64_t _curr_offset;
    size_t _workSpaceSize;
    std::vector<std::array<int, 3>> _gemm_algos;
    cudaStream_t _comp_stream;
    cudaStream_t _comm_stream;

    MPI_Group _group;
    MPI_Comm _comm;
    ncclComm_t _nccl_comm;
    bool _comm_created;
};
