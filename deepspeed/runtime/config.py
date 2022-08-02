import torch
from pydantic import Field, validator
from typing import Union, Literal, Type, Optional
from enum import Enum
from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from deepspeed.runtime.zero.config import DeepSpeedZeroConfig
from deepspeed.runtime.activation_checkpointing.config import DeepSpeedActivationCheckpointingConfig
from deepspeed.monitor.config import TensorBoardConfig, WandbConfig, CSVConfig
from deepspeed.comm.config import DeepSpeedCommsConfig


class TorchDTypeEnum(str, Enum):
    torch.float32 = 'fp32'
    torch.float16 = 'fp16'

TORCH_TYPE_LOOKUP = {"float":torch.float32, "float32": torch.float32, "fp32":torch.float32,
                     "half":torch.float16, "float16": torch.float16, "fp16":torch.float16,
                     "bf16":torch.bfloat16}

class DeepSpeedFP16Config(DeepSpeedConfigModel):
    enabled: bool = False
    auto_cast: bool = False
    loss_scale: float = Field(0., ge=0)
    initial_scale_power: int = Field(32, ge=0)
    loss_scale_window: int = Field(1000, gt=0)
    hysteresis: int = Field(2, ge=0)
    min_loss_scale: float = Field(1., ge=0)
    fp16_master_weights_and_grads: bool = False

class DeepSpeedBF16Config(DeepSpeedConfigModel):
    enabled: bool = False

class DeepSpeedAMPConfig(DeepSpeedConfigModel):
    enabled: bool = False

    # NOTE: Defining this for subclasses will keep all configs from parent
    # class, except the ones that are redefined below
    class Config:
        extra = "allow" # So we can pass arbitrary kwargs to AMP

class DeepSpeedConfig(DeepSpeedConfigModel):
    train_batch_size: int = Field(None, gt=0)
    train_micro_batch_size_per_gpu: int = Field(None, gt=0)
    gradient_accumulation_steps: int = Field(None, gt=0)
    steps_per_print: int = Field(10, gt=0)
    dump_state: bool = False

    disable_allgather: bool = False
    communication_data_type: Union[Literal[None], Literal[tuple(TORCH_TYPE_LOOKUP.keys())]] = None
    prescale_gradients: bool = False
    gradient_predivide_factor: float = 1.0
    sparse_gradients_enabled: bool = False

    zero_config: DeepSpeedZeroConfig = {}

    activation_checkpointing_config: DeepSpeedActivationCheckpointingConfig = Field({}, alias='activation_checkpointing')

    comms_config: DeepSpeedCommsConfig = Field({}, alias="comms_logger")
    # These replace the previous DeepSpeedMonitorConfig
    tensorboard: TensorBoardConfig = {}
    wandb: WandbConfig = {}
    csv_monitor: CSVConfig = {}

    gradient_clipping: float = 0.
    fp16: DeepSpeedFP16Config = {}
    bf16: DeepSpeedBF16Config = {}
    amp: DeepSpeedAMPConfig = {}

    @validator('communication_data_type')
    def convert_to_torch_dtype(cls, field_value, values):
        if field_value is None:
            return field_value
        return TORCH_TYPE_LOOKUP[field_value]

    @validator('bf16')
    def assert_fp16_or_bf16(cls, field_value, values):
        assert "fp16" in values, "FP16 field must be defined prior to BF16 in DeepSpeedConfig class"
        if field_value.enabled:
            assert not values["fp16"].enabled, "FP16 and BF16 cannot both be enabled"
        return field_value

if __name__ == '__main__':
    ds_dict = {'fp16':{'enabled':True}, 'amp':{'enabled':True, 'opt_level':'O1'}}
    conf = DeepSpeedConfig(**ds_dict)
    #ds_dict = {'communication_data_type': None}
    #print(DeepSpeedConfig(**ds_dict))
