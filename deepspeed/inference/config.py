import torch
from pydantic import validator
from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from deepspeed.runtime.zero.config import DeepSpeedZeroConfig
from pydantic import Field
from typing import Dict
from enum import Enum


class DtypeEnum(str, Enum):
    fp16 = "torch.float16"
    #fp16 = "half"
    #fp16 = "torch.float16"
    #fp16 = "torch.half"
    fp32 = "fp32"
    #fp32 = "torch.float32"
    #fp32 = "torch.float"
    int8 = "int8"


class MoETypeEnum(str, Enum):
    residual = "residual"
    standard = "standard"


class DeepSpeedTPConfig(DeepSpeedConfigModel):
    enabled: bool = True
    tp_size: int = 1
    mpu: object = None
    tp_group: object = None


class DeepSpeedMoEConfig(DeepSpeedConfigModel):
    enabled: bool = True
    ep_size: int = 1
    moe_experts = Field(1, alias="num_experts")
    moe_type: MoETypeEnum = MoETypeEnum.standard
    ep_mp_group: object = None
    ep_group: object = Field(None, alias="expert_group")


class QuantTypeEnum(str, Enum):
    asym = "asymmetric"
    sym = "symmetric"


class BaseQuantConfig(DeepSpeedConfigModel):
    enabled = True
    num_bits = 8
    q_type: QuantTypeEnum = QuantTypeEnum.sym
    q_groups: int = 1


class WeightQuantConfig(BaseQuantConfig):
    enabled = True


class ActivationQuantConfig(BaseQuantConfig):
    enabled = True


class QKVQuantConfig(DeepSpeedConfigModel):
    enabled = True


class QuantizationConfig(DeepSpeedConfigModel):
    enabled: bool = True
    activation: ActivationQuantConfig = ActivationQuantConfig()
    weight: WeightQuantConfig = WeightQuantConfig()
    qkv: QKVQuantConfig = QKVQuantConfig()


# todo: brainstorm on how to do ckpt loading for DS inference
class InferenceCheckpointConfig(DeepSpeedConfigModel):
    checkpoint_dir: str = None
    save_mp_checkpoint_path: str = None
    base_dir: str = None


''' Public DS Inference config is defined in this class.
    If you plan to extend the config, please create a new subclass 
    e.g. NewQuantConfig and add as a field to this class

Arguments:

        triangular_masking: Required: this shows the type of masking for attention scores in transformer layer
            note that the masking is application specific.

        mp_size: Optional: Desired model parallel size, default is 1 meaning no
            model parallelism.

        training_mp_size: Optional: if loading a checkpoint this is the mp size that it was trained with,
            it may be different than what the mp size that you want to use during inference.

        mpu: Optional: A model parallelism unit object that implements
            get_{model,data}_parallel_{rank,group,world_size}()

        checkpoint: Optional: Path to deepspeed compatible checkpoint or path to
            JSON with load policy.

        dtype: Optional: Desired model data type, will convert model to this type.
            Supported target types: torch.half, torch.int8, torch.float

        injection_policy: Optional: Dictionary mapping a client nn.Module to its corresponding
            injection policy. e.g., {BertLayer : deepspeed.inference.HFBertLayerPolicy}

        replace_method: Optional: If 'auto' DeepSpeed will automatically try and replace
            model modules with its optimized versions. If an injection_policy is set this will
            override the automatic replacement behavior.

        quantization_setting: Optional: Quantization settings used for quantizing your model using the MoQ.
            The setting can be one element or a tuple. If one value is passed in, we consider it as the number
            of groups used in quantization. A tuple is passed in if we want to mention that there is extra-grouping
            for the MLP part of a Transformer layer (e.g. (True, 8) shows we quantize the model using 8 groups for
            all the network except the MLP part that we use 8 extra grouping).
        replace_with_kernel_inject: this flag need to be set to true to inject inference kernels for models such as, Bert, GPT2, GPT-Neo and GPT-J. Otherwise,
            the injection_dict provides the names of two linear layers as a tuple: (attention_output projection, transformer output projection)
        return_tuple: Specify whether or not the transformer layers need to return a tuple or a Tensor. It is set to True by default (returning a tuple).
        ep_size: The expert-parallelism size which is used for partitioning the experts across the GPUs in the expert-parallel group.
        moe: Specify if the type of Transformer is MoE. It is set to False by default.
        moe_experts: The global number of experts used in an MoE layer.
        moe_type: Specify the type of MoE layer. We have two types of MoE layer: 'Standard' and 'Residual'. It is set to 'Standard' type by default.
        args: All the arguments used for launching the inference api that can be useful at the inference-engine for injecting the optimizations.
        enable_cuda_graph: use this flag for capturing the CUDA-Graph of the inference ops, so that it can run faster using the graph replay method,
            this is set to False by default
        save_mp_checkpoint_path: The path for which we want to save the loaded model with a checkpoint. This feature is used for adjusting the
            parallelism degree to help alleviate the model loading overhead. It does not save any new checkpoint if no path is passed.
        base_dir: This shows the root directory under which all the checkpoint files exists. This can be passed through the json config too.
'''
class DeepSpeedInferenceConfig(DeepSpeedConfigModel):
    kernel_inject: bool = Field(False,
                                description="Injects the kernel into the model",
                                alias="replace_with_kernel_inject")
    dtype: object = torch.float # todo: fix later. objects cannot be serialized to json
    tensor_parallel: DeepSpeedTPConfig = Field(DeepSpeedTPConfig(), alias="tp")
    enable_cuda_graph: bool = False
    zero: DeepSpeedZeroConfig = DeepSpeedZeroConfig()
    triangular_masking = Field(True, alias="tm")
    moe = DeepSpeedMoEConfig()
    quant: QuantizationConfig = QuantizationConfig()
    checkpoint: Dict = None
    checkpoint_config: InferenceCheckpointConfig = Field(InferenceCheckpointConfig(),
                                                         alias="ckpt_config")
    return_tuple: bool = True
    training_mp_size: int = 1
    replace_method: str = 'auto'
    injection_policy: Dict = None
    config: Dict = None  # todo: really no need for this field if we can refactor
    mp_size: int = 1

    @validator('mp_size')
    def tp_size_set(cls, value, values):
        if values['tensor_parallel'].tp_size is None:
            values['tensor_parallel'].tp_size = value
        return value
