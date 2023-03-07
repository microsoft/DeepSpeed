'''Copyright The Microsoft DeepSpeed Team'''
from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from enum import Enum
from pydantic import root_validator, validator, Field
from typing import Dict, List

COMPRESSION_TRAINING = "compression_training"


def get_compression_config(param_dict):
    if COMPRESSION_TRAINING not in param_dict:
        param_dict[COMPRESSION_TRAINING] = {}
    return DeepSpeedCompressionConfig(**param_dict[COMPRESSION_TRAINING])


# Enum classes for pydantic models
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


class DifferentGroupsParamsConfig(DeepSpeedConfigModel):
    start_bits: int
    target_bits: int
    quantization_period: int = Field(1, ge=0)


class DifferentGroupsConfig(DeepSpeedConfigModel):
    params: DifferentGroupsParamsConfig = {}
    modules: List[str] = ["*"]
    related_modules: List[str] = None


class ActivationDifferentGroupsParamsConfig(DeepSpeedConfigModel):
    bits: int


class ActivationDifferentGroupsConfig(DeepSpeedConfigModel):
    params: ActivationDifferentGroupsParamsConfig = {}
    modules: List[str] = ["*"]
    related_modules: List[str] = None


class PruningDifferentGroupsParamsConfig(DeepSpeedConfigModel):
    dense_ratio: float


class PruningDifferentGroupsConfig(DeepSpeedConfigModel):
    params: PruningDifferentGroupsParamsConfig = {}
    modules: List[str] = ["*"]
    related_modules: List[List[str]] = None


class FP16MixedQuantizeConfig(DeepSpeedConfigModel):
    enabled: bool = False
    quantize_change_ratio: float = Field(0.001, ge=0)


class WeightQuantizationSharedParamsConfig(DeepSpeedConfigModel):
    enabled: bool = False
    quantizer_kernel: bool = False
    schedule_offset: int = Field(0, ge=0)
    quantize_groups: int = Field(1, ge=1)
    quantize_verbose: bool = False
    quantize_weight_in_forward: bool = False
    quantization_type: QuantizationTypeEnum = QuantizationTypeEnum.symmetric
    rounding: QuantizationRoundingEnum = QuantizationRoundingEnum.nearest
    fp16_mixed_quantize: FP16MixedQuantizeConfig = {}


class ActivationQuantizationSharedParamsConfig(DeepSpeedConfigModel):
    enabled: bool = False
    quantization_type: QuantizationTypeEnum = QuantizationTypeEnum.symmetric
    range_calibration: QuantizationRangeEnum = QuantizationRangeEnum.dynamic
    schedule_offset: int = Field(1000, ge=0)


class PruningSharedParamsConfig(DeepSpeedConfigModel):
    enabled: bool = False
    method: PruningMethodEnum = PruningMethodEnum.l1
    schedule_offset: int = Field(1000, ge=0)


class HeadPruningSharedParamsConfig(DeepSpeedConfigModel):
    enabled: bool = False
    method: PruningMethodEnum = PruningMethodEnum.l1
    schedule_offset: int = Field(1000, ge=0)
    num_heads: int = Field(None, ge=0)

    @root_validator
    def assert_num_heads(cls, values):
        if values.get("enabled"):
            assert values.get("num_heads") != None, "'num_heads' must be specified for head pruning"
        return values


class WeightQuantizationConfig(DeepSpeedConfigModel):
    different_groups: Dict[str, DifferentGroupsConfig] = {}
    shared_parameters: WeightQuantizationSharedParamsConfig = {}

    @validator("shared_parameters")
    def set_enabled(cls, field_value, values):
        values["enabled"] = field_value.enabled
        return field_value

    @root_validator
    def assert_different_groups(cls, values):
        if values.get("enabled"):
            assert values.get("different_groups"), "Weight Quantization is enabled, 'different_groups' must be specified"
        return values


class ActivationQuantizationConfig(DeepSpeedConfigModel):
    different_groups: Dict[str, ActivationDifferentGroupsConfig] = {}
    shared_parameters: ActivationQuantizationSharedParamsConfig = {}

    @validator("shared_parameters")
    def set_enabled(cls, field_value, values):
        values["enabled"] = field_value.enabled
        return field_value

    @root_validator
    def assert_different_groups(cls, values):
        if values.get("enabled"):
            assert values.get("different_groups"), "Activation Quantization is enabled, 'different_groups' must be specified"
        return values


class SparsePruningConfig(DeepSpeedConfigModel):
    different_groups: Dict[str, PruningDifferentGroupsConfig] = {}
    shared_parameters: PruningSharedParamsConfig = {}

    @validator("shared_parameters")
    def set_enabled(cls, field_value, values):
        values["enabled"] = field_value.enabled
        return field_value

    @root_validator
    def assert_different_groups(cls, values):
        if values.get("enabled"):
            assert values.get("different_groups"), "Sparse Pruning is enabled, 'different_groups' must be specified"
        return values


class RowPruningConfig(DeepSpeedConfigModel):
    different_groups: Dict[str, PruningDifferentGroupsConfig] = {}
    shared_parameters: PruningSharedParamsConfig = {}

    @validator("shared_parameters")
    def set_enabled(cls, field_value, values):
        values["enabled"] = field_value.enabled
        return field_value

    @root_validator
    def assert_different_groups(cls, values):
        if values.get("enabled"):
            assert values.get("different_groups"), "Row Pruning is enabled, 'different_groups' must be specified"
        return values


class HeadPruningConfig(DeepSpeedConfigModel):
    different_groups: Dict[str, PruningDifferentGroupsConfig] = {}
    shared_parameters: HeadPruningSharedParamsConfig = {}


class ChannelPruningConfig(DeepSpeedConfigModel):
    different_groups: Dict[str, PruningDifferentGroupsConfig] = {}
    shared_parameters: PruningSharedParamsConfig = {}

    @validator("shared_parameters")
    def set_enabled(cls, field_value, values):
        values["enabled"] = field_value.enabled
        return field_value

    @root_validator
    def assert_different_groups(cls, values):
        if values.get("enabled"):
            assert values.get("different_groups"), "Channel Pruning is enabled, 'different_groups' must be specified"
        return values


class LayerReductionConfig(DeepSpeedConfigModel):
    enabled: bool = False
    keep_number_layer: int = Field(None, ge=0)
    module_name_prefix: str = ""
    teacher_layer: List[int] = []
    other_module_name: List[str] = []


class DeepSpeedCompressionConfig(DeepSpeedConfigModel):
    weight_quantization: WeightQuantizationConfig = {}
    activation_quantization: ActivationQuantizationConfig = {}
    sparse_pruning: SparsePruningConfig = {}
    row_pruning: RowPruningConfig = {}
    head_pruning: HeadPruningConfig = {}
    channel_pruning: ChannelPruningConfig = {}
    layer_reduction: LayerReductionConfig = {}
