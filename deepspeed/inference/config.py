import torch
from pydantic import validator
from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from deepspeed.runtime.zero.config import DeepSpeedZeroConfig
from pydantic import Field
from typing import Dict
from enum import Enum


class DtypeEnum(Enum):
    # The torch dtype must always be the first value (so we return torch.dtype)
    fp16 = torch.float16, "torch.float16", "fp16", "float16", "half"
    fp32 = torch.float32, "torch.float32", "fp32", "float32", "float"
    int8 = torch.int8, "torch.int8", "int8"

    # Copied from https://stackoverflow.com/a/43210118
    # Allows us to use multiple values for each Enum index and returns first
    # listed value when Enum is called
    def __new__(cls, *values):
        obj = object.__new__(cls)
        # first value is canonical value
        obj._value_ = values[0]
        for other_value in values[1:]:
            cls._value2member_map_[other_value] = obj
        obj._all_values = values
        return obj

    def __repr__(self):
        return "<%s.%s: %s>" % (
            self.__class__.__name__,
            self._name_,
            ", ".join([repr(v) for v in self._all_values]),
        )


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
    moe_experts: list = Field([1], alias="num_experts")
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


""" Public DS Inference config is defined in this class.
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
        max_out_tokens: This argument shows the maximum number of tokens inference-engine can work with, including the input and output tokens.
            Please consider increasing it to the required token-length required for your use-case.
"""


class DeepSpeedInferenceConfig(DeepSpeedConfigModel):
    replace_with_kernel_inject: bool = Field(
        False,
        description="Injects the kernel into the model",
        alias="kernel_inject")
    dtype: DtypeEnum = DtypeEnum.fp16
    tensor_parallel: DeepSpeedTPConfig = Field(DeepSpeedTPConfig(), alias="tp")
    enable_cuda_graph: bool = False
    zero: DeepSpeedZeroConfig = DeepSpeedZeroConfig()
    triangular_masking = Field(True, alias="tm")
    moe = DeepSpeedMoEConfig()
    quant: QuantizationConfig = QuantizationConfig()
    #todo: refactor the following 3 into the new checkpoint_config
    checkpoint: str = None
    base_dir: str = None
    save_mp_checkpoint_path: str = None
    checkpoint_config: InferenceCheckpointConfig = Field(InferenceCheckpointConfig(),
                                                         alias="ckpt_config")
    return_tuple: bool = True
    training_mp_size: int = 1
    replace_method: str = "auto"
    injection_policy: Dict = Field(None, alias="injection_dict")
    injection_policy_tuple: tuple = Field(None)

    config: Dict = None  # todo: really no need for this field if we can refactor
    max_out_tokens: int = 1024
    mp_size: int = 1

    @validator("mp_size")
    def tp_size_set(cls, field_value, values):
        print(values["tensor_parallel"].__fields_set__)
        if "tp_size" in values["tensor_parallel"].__fields_set__:
            assert (
                values["tensor_parallel"].tp_size == field_value
            ), f"Cannot provide different values for mp_size ({field_value}) and tensor_parallel.tp_size ({values['tensor_parallel'].tp_size})"
        else:
            values["tensor_parallel"].tp_size = field_value
        return field_value

    class Config:
        # Get the str representation of the datatype for serialization
        json_encoders = {torch.dtype: lambda x: str(x)}
