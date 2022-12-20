import torch
import deepspeed
from ..config import DeepSpeedInferenceConfig


class BaseOp(torch.nn.Module):
    def __init__(self, config: DeepSpeedInferenceConfig):
        super(BaseOp, self).__init__()
        self.config = config
        builder = deepspeed.ops.op_builder.InferenceBuilder()
        self.inference_cuda_module = builder.load()
