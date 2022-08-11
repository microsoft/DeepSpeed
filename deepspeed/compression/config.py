from pydantic import Field
from typing import Dict, List, Any
from enum import Enum
from deepspeed.runtime.config_utils import DeepSpeedConfigModel

COMPRESSION_TRAINING = "compression_training"


def get_compression_config(param_dict):
    compression_config_dict = param_dict.get(COMPRESSION_TRAINING, {})
    return DeepSpeedCompressionConfig(**compression_config_dict)


class QuantizationTypeEnum(str, Enum):
    symmetric = "symmetric"
    asymmetric = "asymmetric"


class QuantizationRoundingEnum(str, Enum):
    nearest = "nearest"
    stochastic = "stochastic"


class QuantizationRangeEnum(str, Enum):
    dynamic = "dynamic"
    static = "static"


class PruningMethodEnum(str, Enum):
    l1 = "l1"
    topk = "topk"


""" Weights """


class FP16MixedQuantizeConfig(DeepSpeedConfigModel):
    enabled: bool = False
    quantize_change_ratio: float = Field(0.001, ge=0)


class WeightQuantizationSharedParamsConfig(DeepSpeedConfigModel):
    enabled: bool = False
    quantizer_kernel: bool = False
    schedule_offset: int = Field(0, ge=0)
    quantize_groups: int = Field(1, gt=0)
    quantize_verbose: bool = False
    quantization_type: QuantizationTypeEnum = QuantizationTypeEnum.symmetric
    quantize_weight_in_forward: bool = False
    rounding: QuantizationRoundingEnum = QuantizationRoundingEnum.nearest
    fp16_mixed_quantize: FP16MixedQuantizeConfig = {}


class WeightQuantizationDifferentGroupsParamsConfig(DeepSpeedConfigModel):
    start_bits: int
    target_bits: int
    quantization_period: int = Field(1, ge=0)


class WeightQuantizationDifferentGroupsConfig(DeepSpeedConfigModel):
    params: WeightQuantizationDifferentGroupsParamsConfig
    modules: List[str] = ["*"]
    related_modules: List[str] = None


class WeightQuantizationConfig(DeepSpeedConfigModel):
    shared_parameters: WeightQuantizationSharedParamsConfig = {}
    different_groups: Dict[str, WeightQuantizationDifferentGroupsConfig] = {}


""" Activation """


class ActivationQuantizationSharedParamsConfig(DeepSpeedConfigModel):
    enabled: bool = False
    quantization_type: QuantizationTypeEnum = QuantizationTypeEnum.symmetric
    range_calibration: QuantizationRangeEnum = QuantizationRangeEnum.dynamic
    schedule_offset: int = Field(1000, ge=0)


class ActivationQuantizationDifferentGroupsParamConfig(DeepSpeedConfigModel):
    bits: int


class ActivationQuantizationDifferentGroupsConfig(DeepSpeedConfigModel):
    params: ActivationQuantizationDifferentGroupsParamConfig
    modules: List[str] = ["*"]
    related_modules: Any = None


class ActivationQuantizationConfig(DeepSpeedConfigModel):
    shared_parameters: ActivationQuantizationSharedParamsConfig = {}
    different_groups: Dict[str, ActivationQuantizationDifferentGroupsConfig] = {}


""" Pruning """


class PruningSharedParamsConfig(DeepSpeedConfigModel):
    enabled: bool = False
    method: PruningMethodEnum = PruningMethodEnum.l1
    schedule_offset: int = Field(1000, ge=0)


class PruningDifferentGroupsParamConfig(DeepSpeedConfigModel):
    dense_ratio: float


class PruningDifferentGroupsConfig(DeepSpeedConfigModel):
    params: PruningDifferentGroupsParamConfig
    modules: List[str] = ["*"]
    related_modules: Any = None


class PruningConfig(DeepSpeedConfigModel):
    shared_parameters: PruningSharedParamsConfig = {}
    different_groups: Dict[str, PruningDifferentGroupsConfig] = {}


# Head pruning is slightly different:


class HeadPruningSharedParamsConfig(DeepSpeedConfigModel):
    enabled: bool = False
    method: PruningMethodEnum = PruningMethodEnum.l1
    schedule_offset: int = Field(1000, ge=0)
    num_heads: int = Field(None, ge=0)


class HeadPruningConfig(DeepSpeedConfigModel):
    shared_parameters: HeadPruningSharedParamsConfig = {}
    different_groups: Dict[str, PruningDifferentGroupsConfig] = {}


""" Layer Reduction """


class LayerReductionConfig(DeepSpeedConfigModel):
    enabled: bool = False
    keep_number_layer: int = Field(None, ge=0)
    module_name_prefix: str = ""
    teacher_layer: List[int] = []
    other_module_name: List[str] = []


""" Compression Config """


class DeepSpeedCompressionConfig(DeepSpeedConfigModel):
    weight_quantization: WeightQuantizationConfig = {}
    activation_quantization: ActivationQuantizationConfig = {}
    sparse_pruning: PruningConfig = {}
    row_pruning: PruningConfig = {}
    head_pruning: HeadPruningConfig = {}
    channel_pruning: PruningConfig = {}
    layer_reduction: LayerReductionConfig = {}
