#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "custom_cuda_layers.h"

#include <fstream>

using namespace std;

template <typename T>
class Softmax {
public:
    struct Config {
        size_t batchSize;
        size_t heads;
        size_t seq_length;
        size_t prob_depth;
        float temprature;
        bool mem_alloc;
        Config(size_t batch, size_t h, size_t seq, int prob_size = 0, bool mem_alloc = false)
            : batchSize(batch),
              heads(h),
              seq_length(seq),
              prob_depth(prob_size),
              temprature(1.0),
              mem_alloc(mem_alloc)
        {
        }
    };

    Softmax(Config config) : config_(config) {
        cudaMalloc(&_masked_softmax, 
                (size_t)ceil((float)(config.batchSize * config.heads * config.seq_length * config.seq_length * 15) / 100.0) * sizeof(T));
    }

    ~Softmax() {
        cudaFree(_masked_softmax);
    }

    void Forward_fused_dropout(int bsz, T * vals, const T * attn_mask, uint8_t* mask, float ratio, cudaStream_t &stream, int threads)
    {
        launch_attn_softmax_dropout<T>(vals, attn_mask, mask, _masked_softmax,
                                ratio, bsz, config_.heads, config_.seq_length, stream, threads);
    }

    void Backward_fused_dropout(int bsz, T *out_grad, const T* soft_out, uint8_t* mask, float ratio, cudaStream_t stream, int threads)
    {
        launch_attn_softmax_dropout_grad<T>(out_grad, soft_out, mask, _masked_softmax, ratio, bsz, config_.heads, config_.seq_length, stream, threads);
    }

    void Forward(int bsz, T* vals, const T* attn_mask, cudaStream_t& stream)
    {
        launch_attn_softmax<T>(vals, attn_mask, bsz, config_.heads, config_.seq_length, stream);
    }

    void Forward1(int bsz, T* vals, const T* attn_mask, cudaStream_t& stream, int threads = 512, int blocks = 512, int reduce_threads = 32)
    {
        launch_attn_softmax_v2<T>(vals, attn_mask, bsz, config_.heads, config_.seq_length, stream, threads, blocks, reduce_threads);
    }

    void Backward(int bsz, T* out_grad, const T* soft_out, cudaStream_t stream)
    {
        launch_attn_softmax_backward_v2<T>(
            out_grad, soft_out, bsz, config_.heads, config_.seq_length, stream);
    }

    inline int GetProbDepth() const { return config_.prob_depth; }

    inline int GetBatchSize() const { return config_.batchSize; }

    inline int GetNumHeads() const { return config_.heads; }

    inline int GetSeqLength() const { return config_.seq_length; }

private:
    T *_masked_softmax;
    Config config_;
};
