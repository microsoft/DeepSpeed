

#pragma once

namespace DeepSpeedAPI {

void bias_gelu(void* inout,
               void* bias,
               unsigned bsz,
               unsigned intermediate_size,
               bool is_float,
               cudaStream_t stream);

void bias_residual(void* inout,
                   void* residual,
                   void* bias,
                   unsigned bsz,
                   unsigned hidden_dim,
                   bool is_float);

}  // namespace DeepSpeedAPI
