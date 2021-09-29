"""batched collective operations for overhead amortization and better
bandwidth utilization"""

import math
from typing import List

import torch
from torch import Tensor
import torch.distributed
from torch.distributed import ProcessGroup
import torch.nn.functional

from deepspeed.utils import instrument_w_nvtx


@instrument_w_nvtx
@torch.no_grad()
def reduce_scatter_coalesced(
        tensors: List[Tensor],
        group: ProcessGroup = None,
) -> List[Tensor]:
    """simultaneously reduce-scatter a list of tensors - this can be done more
    efficiently than individual reduce scatter calls

    TODO. see if PyTorch team wants a c++ verson of this for ProcessGroupNCCL
    """
    this_rank = torch.distributed.get_rank(group)
    world_sz = torch.distributed.get_world_size(group)

    partition_lst_for_each_tensor = tuple(
        torch.chunk(tensor.view(-1),
                    world_sz) for tensor in tensors)
    padded_partition_sz_for_each_tensor = tuple(
        math.ceil(t.numel() / world_sz) for t in tensors)

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
                padding_sz = padded_partition_sz_for_each_tensor[
                    tensor_idx] - tensor_chunk.numel()
                if padding_sz > 0:
                    tensor_partitions_lst_with_padding.append(
                        torch.empty(padding_sz,
                                    dtype=tensor_chunk.dtype,
                                    device=tensor_chunk.device))

        tensor_partition_flat_buffer = instrument_w_nvtx(
            torch.cat)(tensor_partitions_lst_with_padding)

    tensor_partition_buffer_for_each_rank: List[Tensor] = torch.chunk(
        tensor_partition_flat_buffer,
        world_sz)

    # batched reduce-scatter call
    instrument_w_nvtx(torch.distributed._reduce_scatter_base)(
        tensor_partition_buffer_for_each_rank[this_rank],
        tensor_partition_flat_buffer,
        group=group,
    )

    # post-divide
    tensor_partition_buffer_for_each_rank[this_rank].div_(world_sz)

    # reverse procedure of the interleaving done previously, done on the
    # result of the batched reduce-scatter
    output_lst: List[Tensor] = [None] * len(tensors)
    offset = 0
    for tensor_idx in range(len(tensors)):
        output_lst[tensor_idx] = tensor_partition_buffer_for_each_rank[this_rank].narrow(
            0,
            offset,
            partition_lst_for_each_tensor[tensor_idx][this_rank].numel())

        offset += padded_partition_sz_for_each_tensor[tensor_idx]

    return output_lst
