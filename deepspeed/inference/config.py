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
    If you plan to extend the config, please create a new subclass e.g. NewQuantConfig
    and add as a field to this class'''


class DeepSpeedInferenceConfig(DeepSpeedConfigModel):
    kernel_inject: bool = Field(False,
                                description="Injects the kernel into the model",
                                alias="replace_with_kernel_inject")
    dtype: DtypeEnum = DtypeEnum.fp16
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
