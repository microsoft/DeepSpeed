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
def single_all_to_all(input, scatter_idx, gather_idx, group, async_op=False,handle=None,type=None):
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
    if async_op:
        shape=( inp_shape[: gather_idx] + \
            [inp_shape[gather_idx] * seq_world_size,] + \
            inp_shape[gather_idx + 1:])
        c=output.reshape(shape).contiguous()
        if type=='dq' or type=='dk':
            handle[type+'_grad']=output
            handle[type+'_grad_shape']=shape
        return c, work
    #!! need to delete
    c= output.reshape(
        inp_shape[: gather_idx] + \
        [inp_shape[gather_idx] * seq_world_size,] + \
        inp_shape[gather_idx + 1:]).contiguous()
    return c,work


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int, stream=None, fwd_async=False,bwd_async=False, handle=None,type=None,is_fwd=True) -> Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        
        ctx.gather_idx = gather_idx
        ctx.stream=stream
        ctx.bwd_async=bwd_async
        ctx.handle=handle
        ctx.type=type
        
        # if fwd_async and stream!=None:
        if not is_fwd and type=='o':
            assert stream!=None
            # print0('')
            res , work=single_all_to_all(input, scatter_idx, gather_idx, group,False)

            get_accelerator().current_stream().wait_stream(ctx.stream)
        # elif fwd_async and handle!=None:
        elif not is_fwd and (type=='q' or type=='k'):
            assert fwd_async==True
            type='d'+type
            res , work=single_all_to_all(input, scatter_idx, gather_idx, group,True,handle,type)
          
            handle[type]=work
        elif is_fwd and (type=='q' or type=='k'):
            type='fwd_'+type

            res , work=single_all_to_all(input, scatter_idx, gather_idx, group,True,handle,type)
            handle[type]=work
        else:
            res , work=single_all_to_all(input, scatter_idx, gather_idx, group,False)

        return  res 
    

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        # print0("all2all o before")
        # import pydevd  
        # pydevd.settrace()
        
        #def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int, stream=None, fwd_async=False,bwd_async=False) -> Tensor:
        q= (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.stream,ctx.bwd_async,False,ctx.handle,ctx.type,False), None, None,None,None,None,None,None,None)
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
        sp_stream=None,
    ) -> None:

        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

        
        self.sp_stream=sp_stream
        self.bwd_all2all_handels={}
        self.bwd_all2all_handels['dq']=None
        self.bwd_all2all_handels['dq_grad']=None
        self.bwd_all2all_handels['dk']=None
        self.bwd_all2all_handels['dk_grad']=None
        self.dafult_stream=get_accelerator().default_stream()
  
        self.hook_register=False

        

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
        
        
        def q_hook(*notneeded):

            self.bwd_all2all_handels['dq'].wait()
            self.sp_stream.wait_stream(torch.cuda.default_stream())

            tmp=self.bwd_all2all_handels['dq_grad']
            notneeded=list(notneeded)
            notneeded[0]=list(notneeded[0])
            notneeded[0][0]=tmp.reshape(self.bwd_all2all_handels['dq_grad_shape']).contiguous()
            notneeded[0]=tuple(notneeded[0])
            notneeded=tuple(notneeded)
            
         
        def k_hook(*notneeded):            
            self.bwd_all2all_handels['dk'].wait()
            self.sp_stream.wait_stream(torch.cuda.default_stream())
            tmp=self.bwd_all2all_handels['dk_grad']
            notneeded=list(notneeded)
            notneeded[0]=list(notneeded[0])
            notneeded[0][0]=tmp.reshape(self.bwd_all2all_handels['dk_grad_shape']).contiguous()
            notneeded[0]=tuple(notneeded[0])
            notneeded=tuple(notneeded)
            
            

        
        async_bwd_comm_q=True
        async_bwd_comm_k=True


            
        self.dafult_stream.wait_event(query.done_event)
        query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx,None,False,async_bwd_comm_q,self.bwd_all2all_handels,'q') #[1,512,32,32]
        self.dafult_stream.wait_event(key.done_event)
        key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx,None,False,async_bwd_comm_k, self.bwd_all2all_handels,'k') #[1,512,32,32]
        self.dafult_stream.wait_stream(self.sp_stream)
        value_layer= _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx,None,False,False,  self.bwd_all2all_handels,'v') #[1,512,32,32]
        
        if True:
            async_bwd_comm_q=True
            async_bwd_comm_k=True
            #eval interval
            fn_q = query.grad_fn.next_functions[0][0]
            fn_q.register_prehook(q_hook)
            fn_k = key.grad_fn.next_functions[0][0]
            fn_k.register_prehook(k_hook)
        #do dq qk   k v
        # def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int, stream=None, fwd_async=False,bwd_async=False, handle=None,type=None) -> Tensor:

        self.bwd_all2all_handels['fwd_q'].wait()
        self.bwd_all2all_handels['fwd_k'].wait()
        # self.bwd_all2all_handels['fwd_q'].wait()
        #all2all ayns to k_dense_bwd wait
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


        #out e.g., [s/p::h]
        return output

        #o= self.dense(output)