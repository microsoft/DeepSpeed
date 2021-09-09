#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"

template<typename stream_type, typename randgen_type, typename handle_type>    
Context<stream_type, randgen_type, handle_type>::Context() : _workspace(nullptr), _seed(42), _curr_offset(0)
    {
        curandCreateGenerator(&_gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(_gen, 123);
        if (cublasCreate(&_cublasHandle) != CUBLAS_STATUS_SUCCESS) {
            auto message = std::string("Fail to create cublas handle.");
            std::cerr << message << std::endl;
            throw std::runtime_error(message);
        }
    }
template<typename stream_type, typename randgen_type, typename handle_type>
virtual Context<stream_type, randgen_type, handle_type>::~Context()
    {
        cublasDestroy(_cublasHandle);
        cudaFree(_workspace);
    }
template<typename stream_type, typename randgen_type, typename handle_type>
stream_type Context<stream_type, randgen_type, handle_type>::GetCurrentStream()
    {
        // get current pytorch stream.
        stream_type stream = at::cuda::getCurrentCUDAStream();
        return stream;
    }

template<typename stream_type, typename randgen_type, typename handle_type>
stream_type Context<stream_type, randgen_type, handle_type>::GetNewStream() { return at::cuda::getStreamFromPool(); }


template<typename stream_type, typename randgen_type, typename handle_type>
void Context<stream_type, randgen_type, handle_type>::TestGemmFP16(bool test_gemm, int batch_size, int seq_len, int head_num, int size_per_head)
    {
    }
