# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import triton
import triton.language as tl
from deepspeed.accelerator import get_accelerator


@triton.autotune(configs=[
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 1024}, num_stages=4, num_warps=1),
  ],
  key=['M','N'] # the two above configs will be evaluated anytime
                 # the value of x_size changes
)

@triton.jit
def _fused_bias_silu(Mataddr,  # *Pointer* to first input vector.
               biasaddr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               stride_mat,
               M, N,    # Size of the vector.
               BLOCK_SIZE_M: tl.constexpr,  # Number of elements each program should process.
               BLOCK_SIZE_N: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    Mataddr += pid//num_pid_n * stride_mat *BLOCK_SIZE_M + ((pid%num_pid_n) * BLOCK_SIZE_N)
    output_ptr += pid//num_pid_n * stride_mat *BLOCK_SIZE_M + ((pid%num_pid_n) * BLOCK_SIZE_N)
    biasaddr += pid % num_pid_n * BLOCK_SIZE_N
    
    for off in range(0, BLOCK_SIZE_M): 
        cols = tl.arange(0, BLOCK_SIZE_N)

        mask = cols < M*N 
        x = tl.load(Mataddr + off*stride_mat + cols, mask= mask)
        y = tl.load(biasaddr + cols)
        val = x + y
        numerator = val
        denominator = 1.0 + tl.exp(-val)
        output = numerator / denominator
        # Write x + y back to DRAM.
        tl.store(output_ptr +off*stride_mat + cols, output, mask= mask)

@triton.autotune(configs=[
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 1024}, num_stages=4, num_warps=1),
  ],
  key=['M','N'] # the two above configs will be evaluated anytime
                 # the value of x_size changes
)

@triton.jit
def _fused_bias_relu(Mataddr,  # *Pointer* to first input vector.
               biasaddr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               stride_mat,
               M, N,    # Size of the vector.
               BLOCK_SIZE_M: tl.constexpr,  # Number of elements each program should process.
               BLOCK_SIZE_N: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    Mataddr += pid//num_pid_n * stride_mat *BLOCK_SIZE_M + ((pid%num_pid_n) * BLOCK_SIZE_N)
    output_ptr += pid//num_pid_n * stride_mat *BLOCK_SIZE_M + ((pid%num_pid_n) * BLOCK_SIZE_N)
    biasaddr += pid % num_pid_n * BLOCK_SIZE_N
    
    for off in range(0, BLOCK_SIZE_M): 
        cols = tl.arange(0, BLOCK_SIZE_N)

        mask = cols < M*N 
        x = tl.load(Mataddr + off*stride_mat + cols, mask= mask)
        y = tl.load(biasaddr + cols)
        val = x + y
        zr = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
        val = tl.where(val < zr, zr, val)
        output = val
        # Write x + y back to DRAM.
        tl.store(output_ptr +off*stride_mat + cols, output, mask= mask)


@triton.autotune(configs=[
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 1024}, num_stages=4, num_warps=2),
  ],
  key=['M','N'] # the two above configs will be evaluated anytime
                 # the value of x_size changes
)

@triton.jit
def _fused_bias_gelu(Mataddr,  # *Pointer* to first input vector.
               biasaddr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               stride_mat,
               M, N,    # Size of the vector.
               sqrt_addr,
               mul_addr,
               BLOCK_SIZE_M: tl.constexpr,  # Number of elements each program should process.
               BLOCK_SIZE_N: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    Mataddr += pid//num_pid_n * stride_mat *BLOCK_SIZE_M + ((pid%num_pid_n) * BLOCK_SIZE_N)
    output_ptr += pid//num_pid_n * stride_mat *BLOCK_SIZE_M + ((pid%num_pid_n) * BLOCK_SIZE_N)
    biasaddr += pid % num_pid_n * BLOCK_SIZE_N
    
    for off in range(0, BLOCK_SIZE_M): 
        cols = tl.arange(0, BLOCK_SIZE_N)

        mask = cols < M*N 
        x = tl.load(Mataddr + off*stride_mat + cols, mask= mask)
        y = tl.load(biasaddr + cols)
        val = x + y
        #sqrt_param = tl.load(sqrt_addr)
        #mul_param = tl.load(mul_addr)
        #square = val * val
        #cube = square * val
        #mul_cubed = mul_param * cube
        #valaddmul = val + mul_cubed
        #tanh_param = 1/sqrt_param * valaddmul
        erf = tl.math.erf(val/1.41421356237)
        add2 = 1.0 + erf
        mul2 = add2 * val
        output = 0.5 * mul2        
        # Write x + y back to DRAM.
        tl.store(output_ptr +off*stride_mat + cols, output, mask= mask)


def fused_bias_act(x: torch.Tensor, y: torch.Tensor, act: str)-> torch.Tensor:
    assert x.is_contiguous(), "Matrix x must be contiguous"
    assert y.is_contiguous(), "Matrix y must be contiguous"
    output = torch.empty_like(x)
    x = x.view(-1, x.shape[-1])

    M, N = x.shape
    assert x.shape[1] == y.shape[0], "Incompatible dimensions"
    # We need to preallocate the output.
    #print("x.shape: ", x.shape, "y.shape", y.shape, M, N)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    if act == 'gelu':
        sqrt = torch.sqrt(torch.tensor(2.0*7/22, device='cuda'))
        mulparam = torch.tensor(0.044715, device='cuda')
        _fused_bias_gelu[(grid)](x, y, output, 
                               x.stride(0), M, N,
                               sqrt, mulparam)
    elif act == 'relu':
        _fused_bias_relu[(grid)](x, y, output,
                               x.stride(0), M, N)
    elif act == 'silu':
        _fused_bias_silu[(grid)](x, y, output, 
                               x.stride(0), M, N)
    
    return output
