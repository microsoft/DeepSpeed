import torch
from ..config import DeepSpeedInferenceConfig

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder.builder_names import InferenceBuilder


class BaseOp(torch.nn.Module):
    inference_cuda_module = None

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(BaseOp, self).__init__()
        self.config = config
        if BaseOp.inference_cuda_module is None:
            builder = get_accelerator().create_op_builder(InferenceBuilder)
            BaseOp.inference_cuda_module = builder.load()
