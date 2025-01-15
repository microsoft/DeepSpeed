from enum import Enum
from deepspeed.runtime.config_utils import DeepSpeedConfigModel
import torch
from pydantic import Field, field_validator
from typing import Dict, Union, Optional

class AUTOTP_MODE(Enum):
    TRAINING = "TRAINING"
    INFERENCE = "INFERENCE"
    
class DeepSpeedTPConfig(DeepSpeedConfigModel):
    """ Configure tensor parallelism settings """

    enabled: bool = True
    """ Turn tensor parallelism on/off. """

    tp_size: int = 1
    """ Number of devices to split the model across using tensor parallelism. """

    tp_grain_size: int = 64
    "Desired MLP/lm_head tp size granularity. DNN library favors tensor size in granularity of power of 2, we pick 64 as a default size."

    mpu: object = None
    """
    A model parallelism unit object that implements
    ``get_{model,data}_parallel_{rank,group,world_size}()``.
    """

    tp_group: object = None
    
class DeepSpeedTPTrainingConfig(DeepSpeedConfigModel):

    dtype: torch.dtype = torch.float16
    """
    Desired model data type, will convert model to this type.
    Supported target types: `torch.half`, `torch.int8`, `torch.float`
    """

    tensor_parallel: DeepSpeedTPConfig = Field({}, alias="tp")
    """
    Configuration for tensor parallelism used to split the model across several
    GPUs. Expects a dictionary containing values for :any:`DeepSpeedTPConfig`.
    """
    
    injection_policy_tuple: Optional[tuple] = None
