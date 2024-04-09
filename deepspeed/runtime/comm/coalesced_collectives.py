# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
batched collective operations for overhead amortization and better
bandwidth utilization
"""

import math
from typing import List
import torch
from torch import Tensor
from deepspeed import comm as dist
from deepspeed.comm import ProcessGroup, all_to_all_single
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import instrument_w_nvtx
from deepspeed.ops import op_builder
from deepspeed.utils import logger


def _torch_reduce_scatter_fn(input_tensor: Tensor, output_tensor: Tensor, group=None, async_op=False, prof=False):
    return instrument_w_nvtx(dist.reduce_scatter_fn)(output_tensor, input_tensor, group=group, async_op=False)


quantizer_module = None


@instrument_w_nvtx
@torch.no_grad()
def all_to_all_quant_reduce(tensors: List[Tensor], groups: {}) -> List[Tensor]:
    global quantizer_module
    if quantizer_module is None:
        quantizer_module = op_builder.QuantizerBuilder().load()
    local_world_size = get_accelerator().device_count()
    global_world_size = dist.get_world_size()
    num_nodes = global_world_size // local_world_size
    this_rank = dist.get_rank()
    intra_idx = int(this_rank / local_world_size)
    inter_idx = this_rank % local_world_size
    output_lst: List[Tensor] = [None] * len(tensors)
    for idx, tensor in enumerate(tensors):
        if tensor.dim() == 1:
            output_lst[idx] = reduce_scatter_coalesced([tensor])[0]
        elif tensor.numel() % (2 * global_world_size) != 0:
            # Due to the constraint of 2-stage all-to-all, the input tensor must be divisible by 2 * global_world_size
            # Otherwise, all-to-all cannot be performed because of shape mismatch.
            # See more at https://github.com/microsoft/DeepSpeed/pull/5056
            logger.warning(
                f"qgZ falls back to reduce_scatter because tensor size = {tensor.numel()} is not divisible by (2 * global_world_size) = {2 * global_world_size}. Please consider allocating a new world to enable qgZ"
            )
            output_lst[idx] = reduce_scatter_coalesced([tensor])[0]
        else:
            intra_quant_group = max(tensor.shape[0], tensor.shape[1], global_world_size)

            inter_quant_group = intra_quant_group // local_world_size
            intra_quant_int4, intra_q_scales = quantizer_module.swizzle_quant(tensor, intra_quant_group, 4,
                                                                              quantizer_module.Symmetric, 1, num_nodes,
                                                                              local_world_size)
            local_output = torch.empty_like(intra_quant_int4)
            scale_output = torch.empty_like(intra_q_scales)
            all_to_all_single(local_output, intra_quant_int4, group=groups[f'local_{intra_idx}'])
            all_to_all_single(scale_output, intra_q_scales, group=groups[f'local_{intra_idx}'])
            global_input_tensor, global_scales = quantizer_module.quantized_reduction(
                local_output, scale_output, intra_quant_group, inter_quant_group, 4, quantizer_module.Symmetric,
                local_world_size)
            global_output = torch.empty_like(global_input_tensor)
            global_scale_output = torch.empty_like(global_scales)
            all_to_all_single(global_output, global_input_tensor, group=groups[f'global_{inter_idx}'])
            all_to_all_single(global_scale_output, global_scales, group=groups[f'global_{inter_idx}'])
            final_output = quantizer_module.dequantize(global_output, global_scale_output, global_scale_output.numel(),
                                                       4, quantizer_module.Symmetric)
            assert final_output.numel(
            ) % num_nodes == 0, f"final_output.numel()={final_output.numel()} is not divisible by num_nodes={num_nodes}"
            output_lst[idx] = (sum(list(final_output.chunk(num_nodes))) / num_nodes).view(-1)
    return output_lst


@instrument_w_nvtx
@torch.no_grad()
def reduce_scatter_coalesced(
    tensors: List[Tensor],
    group: ProcessGroup = None,
) -> List[Tensor]:

    this_rank = dist.get_rank(group)
    world_sz = dist.get_world_size(group)
    if dist.has_reduce_scatter_coalesced():
        padding_size_list = tuple(math.ceil(t.numel() / world_sz) * world_sz - t.numel() for t in tensors)

        flattened_tensor_list = [tensor.view(-1).div_(world_sz) for tensor in tensors]

        chunked_tensors = []
        for tensor_idx, flattened_tensor in enumerate(flattened_tensor_list):
            if padding_size_list[tensor_idx] > 0:
                flattened_tensor = torch.cat([
                    flattened_tensor,
                    torch.empty(padding_size_list[tensor_idx],
                                dtype=flattened_tensor.dtype,
                                device=flattened_tensor.device)
                ])
            chunks = torch.chunk(flattened_tensor, world_sz)
            chunked_tensors.append(chunks)

        output_lst = [chunked_tensor[this_rank] for chunked_tensor in chunked_tensors]
        dist.reduce_scatter_coalesced(output_tensors=output_lst,
                                      input_tensors=flattened_tensor_list,
                                      group=group,
                                      async_op=False)
        for tensor_idx in range(len(output_lst)):
            if padding_size_list[tensor_idx] > 0:
                tensor_size = output_lst[tensor_idx].numel()
                last_real_data_rank = world_sz - (padding_size_list[tensor_idx] // tensor_size + 1)
                if this_rank > last_real_data_rank:
                    # drop full pad tensor
                    output_lst[tensor_idx] = output_lst[tensor_idx][:0]
                elif this_rank == last_real_data_rank:
                    # drop pad tensor from index
                    output_lst[tensor_idx] = output_lst[tensor_idx][:tensor_size -
                                                                    padding_size_list[tensor_idx] % tensor_size]
        return output_lst

    # simultaneously reduce-scatter a list of tensors
    partition_lst_for_each_tensor = [None] * len(tensors)
    for tensor_idx, tensor in enumerate(tensors):
        flattened_tensor = tensor.view(-1)
        chunk_sz = math.ceil(tensor.numel() / world_sz)
        partition_lst_for_each_tensor[tensor_idx] = [
            flattened_tensor[rank * chunk_sz:rank * chunk_sz + chunk_sz] for rank in range(0, world_sz)
        ]

    padded_partition_sz_for_each_tensor = tuple(math.ceil(t.numel() / world_sz) for t in tensors)

    if len(tensors) == 1 and tensors[0].numel() % world_sz == 0:
        # if there's only one tensor being reduced and we don't need to pad
        # we have an opportunity to avoid a memory allocation
        tensor_partition_flat_buffer = tensors[0].view(-1)
    else:
        # interleave tensor partitions such that the correct reduced partitions of each tensor
        # end up at each rank
        tensor_partitions_lst_with_padding = []
        for rank in range(world_sz):
            for tensor_idx in range(len(tensors)):
                # add tensor content
                tensor_chunk = partition_lst_for_each_tensor[tensor_idx][rank]
                tensor_partitions_lst_with_padding.append(tensor_chunk)

                # add padding if necessary
                padding_sz = padded_partition_sz_for_each_tensor[tensor_idx] - tensor_chunk.numel()
                if padding_sz > 0:
                    tensor_partitions_lst_with_padding.append(
                        torch.empty(padding_sz, dtype=tensor_chunk.dtype, device=tensor_chunk.device))

        tensor_partition_flat_buffer = instrument_w_nvtx(torch.cat)(tensor_partitions_lst_with_padding)

    tensor_partition_flat_buffer.div_(world_sz)  # pre-divide
    tensor_partition_buffer_for_each_rank: List[Tensor] = torch.chunk(tensor_partition_flat_buffer, world_sz)

    # batched reduce-scatter call
    _torch_reduce_scatter_fn(tensor_partition_flat_buffer,
                             tensor_partition_buffer_for_each_rank[this_rank],
                             group=group)

    # reverse procedure of the interleaving done previously, done on the
    # result of the batched reduce-scatter
    output_lst: List[Tensor] = [None] * len(tensors)
    offset = 0
    for tensor_idx in range(len(tensors)):
        output_lst[tensor_idx] = tensor_partition_buffer_for_each_rank[this_rank].narrow(
            0, offset, partition_lst_for_each_tensor[tensor_idx][this_rank].numel())

        offset += padded_partition_sz_for_each_tensor[tensor_idx]
    return output_lst
