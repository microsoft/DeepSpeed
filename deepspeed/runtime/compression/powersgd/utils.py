from typing import List, Tuple
import torch
from types import SimpleNamespace


def pack(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Size]]:
    """Packs a list of tensors into one buffer for sending to other workers"""
    buffer = torch.cat([t.view(-1) for t in tensors])  # copies
    shapes = [tensor.shape for tensor in tensors]
    return buffer, shapes


def unpack(buffer: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
    """Provides pointers to tensors of original `shapes` in a flat-packed buffer."""
    idx = 0
    entries = []
    for tensor_shape in shapes:
        end = idx + tensor_shape.numel()
        entries.append(buffer[idx:end].view(size=tensor_shape))
        idx = end

    return entries


def params_in_optimizer(optimizer: torch.optim.Optimizer) -> List[torch.Tensor]:
    params = []
    for group in optimizer.param_groups:
        params.extend(group["params"])
    # if torch.distributed.get_rank() == 0:
    #     print(">> In params_in_optimizer:")
    #     for p in params:
    #         print(p.shape) 
    return params


def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()  # type: ignore


def flatten(tensors: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    out = []
    for list in tensors:
        out.extend(list)
    return out


def allreduce_average(data, *args, **kwargs):
    """All-reduce average if torch.distributed is available, otherwise do nothing"""
    if is_distributed():
        data.div_(torch.distributed.get_world_size())  # type: ignore
        return torch.distributed.all_reduce(data, *args, **kwargs)  # type: ignore
    else:
        return SimpleNamespace(wait=lambda: None)
