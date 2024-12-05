# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed import comm as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.tp_shard import get_shard_size, get_shard_size_list
from abc import ABC, abstractmethod
from typing import Iterable
from deepspeed.utils import groups
from .fusedqkv_utils import shard_value_with_share_qk, shard_chunk_mlp, prepare_tp_fused_qkvw

def move(tensor, device):
    #TODO: the data parallelism (DP) is greater than 2,
    # we need to consider when to delete the CPU data.
    if tensor.is_meta:
        return torch.empty_like(tensor, device=device)
    else:
        # Using new tensors help in freeing memory (after split for example) was done before by calling clone().
        # Using copy=True instead of clone() will help in case of cpu --> cpu.
        # Otherwise to() will not create a new copy for the view of the full tensor, and it will not be de-referenced.
        return tensor.to(device, copy=True)
class RowParallel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, group: dist.ProcessGroup, input_):
        ctx.group = group
        if group == None:
            return input_
        # for debug ,will apply dist.inference_all_reduce
        dist.all_reduce(input_, group=group)
        return input_

    @staticmethod
    def backward(ctx, grad_output):

        return None, grad_output


class ColumnParallel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, group, input_):
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):

        if ctx.group == None:
            return None, grad_output
        # for debug ,will apply dist.inference_all_reduce
        dist.all_reduce(grad_output, group=ctx.group)
        return None, grad_output


#Parent class handling common logic
class Replaced_Layer(nn.Module, ABC):
    mode = "INFERENCE" 
    def __init__(self, mp_group, name=None):
        super().__init__()
        self.support_training = False
        if mp_group is not None:
            self.mp_group = mp_group
            self.tp_world_sz = dist.get_world_size(self.mp_group)
            self.tp_index = dist.get_rank(mp_group)
        if name is not None:
            self.name=name
    @abstractmethod
    def forward(self, input):
        """
        Forward pass method. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def gather_params(self, params_list):
        pass

    def partition(self, params_list, move_to_device=False):
        for idx, param in enumerate(params_list):
            params_list[idx].data = param.data_partition
            del param.data_partition

        # for param in params_list:
        #     param.data=torch.empty(0, dtype=param.dtype, device=param.device)
    def config_tp_training(self, weight):
        assert self.support_training, "No implementation of backward."
        if weight is not None:
            weight.requires_grad = True
            setattr(weight, 'tensor_model_parallel', True)
            weight.ds_is_preleace_module = True
            weight.gather_params = self.gather_params
            weight.partition = self.partition


class GatherReplacedLayerParams:

    def __init__(self, params, module, enabled=True):
        self.enabled = enabled
        self.module = module
        if not enabled:
            return
        if isinstance(params, Iterable) and not isinstance(params, torch.Tensor):
            # deal with generators like model.parameters()
            # must convert to list to be able to iterate more than once if we get a generator
            params = list(params)
        else:
            # single param
            params = [params]

        self.params = params

        if not any(self._is_replaced_module_weight(p) for p in params):
            self.enabled = False
            return

    def _is_replaced_module_weight(self, param):
        return getattr(param, 'ds_is_preleace_module', False)

    def __enter__(self):

        if self.enabled:
            self.params[0].gather_params(self.params)

    def __exit__(self, exc_type, exc_value, traceback):
        #TODO : Check whether there are any missing attributes.
        if self.enabled:
            self.params[0].partition(self.params)


class LinearAllreduce(Replaced_Layer):

    def __init__(self, module, mp_group, name=None):
        super(LinearAllreduce, self).__init__(mp_group, name)
        self.weight = module.weight
        self.bias = module.bias
        
        self.partition([self.weight, self.bias], move_to_device=True)
        self.support_training = True
        self.config_tp_training(self.weight)
        if self.bias is not None:
            self.config_tp_training(self.bias)

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        output = RowParallel.apply(self.mp_group, output)
        if self.bias is not None:
            output += self.bias
        return output

    def gather_params(self, params_list):

        for idx, param in enumerate(params_list):
            params_list[idx].data_partition = param.data
            param = param.transpose(0, 1).contiguous()
            output_param = torch.empty(self.tp_world_sz * param.shape[0],
                                       param.shape[1],
                                       dtype=param.dtype,
                                       device=param.device)
            dist.all_gather_into_tensor(output_param, param, group=self.mp_group)
            params_list[idx].data = output_param.transpose(0, 1).contiguous()
        return
    def partition(self, params_list, move_to_device=False):
        for idx, param in enumerate(params_list):
            if param is None:
                return 
            _partition=torch.chunk(param, self.tp_world_sz, dim=1)[self.tp_index]

            if move_to_device:
                partition=move(_partition, get_accelerator().current_device()).detach()
                del _partition
                _partition=partition
            
            params_list[idx].data = _partition

class LinearLayer(Replaced_Layer):

    def __init__(self, module, mp_group, name=None, skip_partition=False):
        super(LinearLayer, self).__init__(mp_group, name)
        self.weight = module.weight
        self.bias = module.bias
        if not skip_partition:
            self.partition([self.weight, self.bias], move_to_device=True)
        self.support_training = True
        self.config_tp_training(self.weight)
        if self.bias is not None:
            self.config_tp_training(self.bias)
        self.config_tp_training(self.weight)
        self.config_tp_training(self.bias)

    def forward(self, input):
        input = ColumnParallel.apply(self.mp_group, input)
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.bias is not None:
            output += self.bias
        return output

    def gather_params(self, params_list):

        for idx, param in enumerate(params_list):
            # TODO: uneven support
            # shape_tensor=torch.tensor(param.shape[0],dtype=param.dtype,device=param.device)
            # dist.all_reduce(shape_tensor, group=self.mp_group)
            params_list[idx].data_partition = param.data
            output_param = torch.empty(self.tp_world_sz * param.shape[0],
                                       param.shape[1],
                                       dtype=param.dtype,
                                       device=param.device)
            dist.all_gather_into_tensor(output_param, param, group=self.mp_group)
            params_list[idx].data = output_param.contiguous()
    def partition(self, params_list, move_to_device=False):
        
        for idx, param in enumerate(params_list):
            if param is None:
                return 
            _partition=torch.chunk(param, self.tp_world_sz, dim=0)[self.tp_index]

            if move_to_device:
                partition=move(_partition, get_accelerator().current_device()).detach()
                del _partition
                _partition=partition
            
            params_list[idx].data = _partition
    # for bwc
    @classmethod
    def from_weights(cls, weight_shape=None, dtype=torch.half, weight=None, bias=None):
        if weight is not None:
            in_features = weight.shape[1] 
            out_features = weight.shape[0]  
            linear = nn.Linear(in_features, out_features, bias=(bias is not None))
            linear.weight.data = weight
            if bias is not None:
                linear.bias.data = bias
        else:
            in_features = weight_shape[1] 
            out_features = weight_shape[0]  
            linear = nn.Linear(in_features, out_features, bias=(bias is not None))
        return cls(linear, skip_partition=True)
        

class fused_LinearLayer(LinearLayer):
    def partition(self, params_list, move_to_device=False): 
        def prepare_tp_fused_qkvw(module, src, mp_size, gpu_index):
            
            for idx, param in params_list:
                if param is None:
                    return 
                _partition=prepare_tp_fused_qkvw(self.name, param, self.tp_world_sz, self.tp_index )
            if move_to_device:
                partition=move(_partition, get_accelerator().current_device()).detach()
                del _partition
                _partition=partition
            params_list[idx].data = _partition

class conv_LinearLayer(LinearLayer):
    def partition(self, params_list, move_to_device=False):
        weight = None
        bias = None
        if len(params_list)==1:
            weight=params_list[0]
        elif len(params_list)==2:
            weight, bias=params_list[0], params_list[1]
        _partition = weight.data.split(get_shard_size_list(weight.shape[0],  self.tp_world_sz, self.name), dim=1)
        partition=move(_partition, get_accelerator().current_device()).detach()
        del _partition
        weight.data=partition
        
        if bias is not None:
            _partition = bias.data.split(get_shard_size_list(
                        weight.shape[1] ,self.tp_world_sz, self.name),
                                                      dim=0)
            partition=move(_partition, get_accelerator().current_device()).detach()
            del _partition
            bias.data=partition

            
        
    
            
class bwc_LinearLayer(nn.Module):

    def __init__(self, weight_shape=None, dtype=torch.half, weight=None, bias=None):
        super(LinearLayer, self).__init__()
        if weight is not None:
            self.weight = weight
            self.bias = bias
        else:
            self.weight = Parameter(
                torch.empty(weight_shape, dtype=dtype, device=get_accelerator().current_device_name()))

            self.bias = Parameter(
                torch.empty(weight_shape[0],
                            dtype=dtype,
                            device=get_accelerator().current_device_name())) \
                if bias is not None else None

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.bias is not None:
            output += self.bias
        return output




#override the subclasses related to weight splitting.
def Yuan_LinearALlreduce(LinearAllreduce):
    def partition(self, params_list, move_to_device=False):
        params_list[0], params_list[1]=shard_value_with_share_qk(params_list[0],params_list[1],self.tp_world_size, self.tp_index, False)

def Yuan_LinearLayer(LinearLayer):
    def partition(self, params_list, move_to_device=False):
        params_list[0], params_list[1]=shard_value_with_share_qk(params_list[0],params_list[1],self.tp_world_size, self.tp_index, False)

def GLM_LinearLayer(LinearLayer):
    def partition(self, params_list, move_to_device=False):
        params_list[0], params_list[1]=shard_chunk_mlp(params_list[0],params_list[1],self.tp_world_size, self.tp_index, False)

def Conv_LinearALlreduce(LinearALlreduce):
    def partition(self, params_list, move_to_device=False):            
        for idx, param in enumerate(params_list):
            if param is None:
                return 
            param.data= param.data.transpose(-1, -2).contiguous()
            
            _partition=param.split(get_shard_size_list(
                param.shape[0] , self.tp_world_size, self.name),
                                           dim=1)

            if move_to_device:
                partition=move(_partition, get_accelerator().current_device())
                del _partition
                _partition=partition
            
            params_list[idx].data = _partition
        
        
    

#override the subclasses related to reward.
class LmHeadLinearAllreduce(LinearAllreduce):

    def forward(self, input):
        input_shard_size = get_shard_size(input.shape[-1], self.tp_world_sz, "lm_head")
        input_shard_offset = sum(get_shard_size_list(input.shape[-1], self.world_size, "lm_head")[0:self.tp_index])
        output = torch.matmul(input[:, :, input_shard_offset:input_shard_offset + input_shard_size],
                              self.weight.transpose(-1, -2))
        if self.mp_group is not None:
            dist.inference_all_reduce(output, group=self.mp_group)
        if self.bias is not None:
            output += self.bias
        return output
        
        
        
        
class TensorParallelConv2d(nn.Module):

    def __init__(self, conv, rank, world_size, shard_by_oc):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.shard_by_oc = shard_by_oc
        self.shard_weights(conv)

    # Split along the input/output channel depending on whether it is the last conv layer.
    def shard_weights(self, conv):
        if self.shard_by_oc:
            total_size = conv.weight.shape[0]
        else:
            total_size = conv.weight.shape[1]
        bias_data = None
        cols_per_rank = [0]
        for i in range(self.world_size - 1, -1, -1):
            cols = total_size // self.world_size
            if i < total_size % self.world_size:
                cols += 1
            cols_per_rank.append(cols_per_rank[-1] + cols)
        weight_data = conv.weight.data
        if self.shard_by_oc:
            # not last conv layer, split output channel
            weight_data = weight_data[cols_per_rank[self.rank]:cols_per_rank[self.rank + 1]]
            if conv.bias is not None:
                bias_data = conv.bias.data[cols_per_rank[self.rank]:cols_per_rank[self.rank + 1]]
        else:
            # last conv layer, split input channel
            weight_data = weight_data[:, cols_per_rank[self.rank]:cols_per_rank[self.rank + 1]]
            if conv.bias is not None:
                bias_data = conv.bias.data / float(self.world_size)
        self.conv = nn.Conv2d(weight_data.shape[1], weight_data.shape[0], conv.kernel_size, conv.stride, conv.padding,
                              conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode)
        self.conv.weight = torch.nn.Parameter(weight_data)
        if conv.bias is not None:
            self.conv.bias = torch.nn.Parameter(bias_data)
        del conv

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class TensorParallelOcShardConv2d(TensorParallelConv2d):

    def __init__(self, conv, rank, world_size):
        super().__init__(conv, rank, world_size, True)


class TensorParallelIcShardConv2d(TensorParallelConv2d):

    def __init__(self, conv, rank, world_size):
        super().__init__(conv, rank, world_size, False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.conv(input)
        if self.world_size > 1:
            dist.inference_all_reduce(out)
        return out


class Normalize(nn.Module):

    def __init__(self, dim=None, dtype=torch.float, eps=1e-5, weight=None, bias=None):
        super(Normalize, self).__init__()
        if weight is not None:
            self.weight = weight
            self.bias = bias
        else:
            self.norm = nn.LayerNorm(dim, eps=eps).to(dtype).to(get_accelerator().current_device_name())
            self.weight = self.norm.weight
            self.bias = self.norm.bias

        self.eps = eps

    def forward(self, input):
        return nn.functional.layer_norm(input, input.shape[-1:], self.weight, self.bias, eps=self.eps)


class EmbeddingLayer(nn.Module):

    def __init__(self, weight_shape=None, dtype=torch.half, weight=None, bias=None):
        super(EmbeddingLayer, self).__init__()
        if weight is None:
            self.weight = Parameter(
                torch.empty(weight_shape[0],
                            weight_shape[1],
                            dtype=dtype,
                            device=get_accelerator().current_device_name()))
        else:
            self.weight = weight

    def forward(self, input):
        return F.embedding(input, self.weight)


class OPTEmbedding(EmbeddingLayer):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, weight_shape=None, weight=None, bias=None):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(weight_shape, weight=weight)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


class RMSNormalize(nn.Module):

    def __init__(self, dim=None, dtype=torch.float, eps=1e-5, weight=None):
        super(RMSNormalize, self).__init__()
        if weight is not None:
            self.weight = weight
        else:
            self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=get_accelerator().current_device_name()))

        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return hidden_states * self.weight
