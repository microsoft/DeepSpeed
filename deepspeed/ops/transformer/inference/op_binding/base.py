import torch
import deepspeed


class BaseOp(torch.nn.Module):
    def __init__(self, builder="InferenceBuilder"):
        super(BaseOp, self).__init__()
        builder = getattr(deepspeed.ops.op_builder, builder)()
        self.inference_cuda_module = builder.load()
