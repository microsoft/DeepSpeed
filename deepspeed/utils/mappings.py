import torch
import deepspeed


def _gather_tokens(input_, dim=0):
    """Gather tensors and concatinate them along a dimension"""
    mpu = deepspeed.utils.groups.mpu
    if mpu.get_tensor_model_parallel_world_size() == 1:
        return input_

    input_ = input_.contiguous()
    # Size and dimension.
    rank = mpu.get_tensor_model_parallel_rank()

    tensor_list = [
        torch.empty_like(input_)
        for _ in range(mpu.get_tensor_model_parallel_world_size())
    ]
    tensor_list[rank] = input_
    deepspeed.comm.all_gather(tensor_list,
                              input_,
                              group=mpu.get_tensor_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


def _drop_tokens(input_, dim=0):
    mpu = deepspeed.utils.groups.mpu
    if mpu.get_tensor_model_parallel_world_size() == 1:
        return input_
    total_chunks = mpu.get_tensor_model_parallel_world_size()
    this_chunk = mpu.get_tensor_model_parallel_rank()
    assert input_.shape[dim] % total_chunks == 0, f"input dimension {dim} ({input_.shape[dim]}) is not divisible by tensor parallel world size ({total_chunks})"
    chunk_size = input_.shape[dim] // total_chunks

    return torch.narrow(input_, dim, this_chunk * chunk_size, chunk_size)


class _GatherTokens(torch.autograd.Function):
    """Reduce scatter output of self attention for MoE"""
    @staticmethod
    def symbolic(graph, input_, dim):
        return _gather_tokens(input_, dim)

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        return _gather_tokens(input_, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _drop_tokens(grad_output, ctx.dim), None


class _DropTokens(torch.autograd.Function):
    "Drop tokens (this is a hacky approach until we can do reduce scatter)"

    @staticmethod
    def symbolic(graph, input_, dim):
        return _drop_tokens(input_, dim)

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        return _drop_tokens(input_, dim)

    @staticmethod
    def backward(ctx, input_):
        return _gather_tokens(input_, ctx.dim), None
