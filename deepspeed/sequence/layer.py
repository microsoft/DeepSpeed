# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from typing import Any, Tuple
from torch import Tensor
from torch.nn import Module

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator


def wait_stream(stream):
    get_accelerator().wait_stream(stream)
    
def print0(msg):
    if dist.get_rank()==0:
        print(msg)
def single_all_to_all(input, scatter_idx, gather_idx, group, async_op=False):
    seq_world_size = dist.get_world_size(group)
    inp_shape = list(input.shape)
    inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size
    if scatter_idx < 2:
        input_t = input.reshape(
            [seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).contiguous()
    else:
        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        input_t = input.reshape(
            [-1, seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)
    work= dist.all_to_all_single(output, input_t, group=group, async_op=async_op)
   
    # if scattering the seq-dim, transpose the heads back to the original dimension
    if scatter_idx < 2:
        output = output.transpose(0, 1).contiguous()

    return output.reshape(
        inp_shape[: gather_idx] + \
        [inp_shape[gather_idx] * seq_world_size,] + \
        inp_shape[gather_idx + 1:]).contiguous() ,work


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int, stream=None, fwd_async=False,bwd_async=False) -> Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        
        ctx.gather_idx = gather_idx
        ctx.stream=stream
        ctx.bwd_async=bwd_async
        # if stream != None:
        #     with get_accelerator().stream(stream):
        #         res, work=single_all_to_all(input, scatter_idx, gather_idx, group,fwd_async)
        # else:
        #     res , work=single_all_to_all(input, scatter_idx, gather_idx, group,fwd_async)
        res , work=single_all_to_all(input, scatter_idx, gather_idx, group,False)
        # def single_all_to_all(input, scatter_idx, gather_idx, group, async_op=False):

        if fwd_async:
            get_accelerator().current_stream().wait_stream(ctx.stream)

        return  res 
    

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        # print0("all2all o before")
        # import pydevd  
        # pydevd.settrace()
        
        #def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int, stream=None, fwd_async=False,bwd_async=False) -> Tensor:
        q= (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.stream,ctx.bwd_async,False), None, None,None,None,None)
        # print0("all2all o after")

        return q


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        local_attention: Module,
        sequence_process_group: dist.ProcessGroup,
        scatter_idx: int = 2,
        gather_idx: int = 0,
        sp_stream=None
    ) -> None:

        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        # self.q_stream=get_accelerator().Stream()
        # self.k_stream=get_accelerator().Stream()
        # self.v_stream=get_accelerator().Stream()
        self.sp_stream=sp_stream
        b=0

    # query = slef.linearq(hidden)
    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        #in shape : e.g.,  [s/p:h:]
        
        
        #step1   get q ,k ,v outside out this function
        # def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int, stream=None, fwd_async=False,bwd_async=False) -> Tensor:

        query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx) #[1,512,32,32]
        key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx) #[1,512,32,32]
        value_layer= _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx) #[1,512,32,32]

        #out shape : e.g., [s:h/p:]
        # print(query_layer) #2,8, 2,4  sp=2 2gpus
        #                    #
        # print(key_layer)
        # print(value_layer)  #seq_len 16 , sp 2 , head_dim = 4, num_heads=4, hidding=16
        
        context_layer = self.local_attn(query_layer, key_layer, value_layer, *args)       #[8,512,4,32]
        bwd_o_async=False
        if self.sp_stream is not None:
            bwd_o_async=True
        output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx,self.sp_stream,False,bwd_o_async)

        # dO=wdY        

        #out e.g., [s/p::h]
        return output

        #o= self.dense(output)