'''Copyright The Microsoft DeepSpeed Team'''
"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from pydantic import Field, validator
from deepspeed.runtime.config_utils import DeepSpeedConfigModel

# TODO: remove once main deepspeed config uses pydantic
NEBULA = "nebula"


def get_nebula_config(param_dict):
    nebula_config_dict = param_dict.get(NEBULA, {})
    return DeepSpeedNebulaConfig(**nebula_config_dict)


class DeepSpeedNebulaConfig(DeepSpeedConfigModel):
    """ Sets parameters for Nebula checkpoint engine. """

    enabled: bool = False
    """ Enable or disable Nebula checkpoint engine. """

    load_path: str = None
    """
    When you want to resume the previous checkpoint saved by nebula, you can
    set `load_path` as the parent folder of checkpoint.  If `load_path` is
    None, the `persistent_storage_path` will be the default path to load.
    """

    persistent_storage_path: str = None
    """ Nebula will save the checkpoint under `load_path` in the asynchronous way. """

    persistent_time_interval: int = Field(None, gt=0)
    """ Time interval to trigger the nebula persistence. """

    num_of_version_in_retention: int = Field(2, gt=0)
    """
    Checkpoint number which will be kept in memory. Let us say, if the value is
    `2`. Then we have checkpoints `1` and `2` are ready now. When it comes to
    checkpoint `3`, the `1` will be removed if `1` has been persisted to disk.
    """

    enable_nebula_load: bool = True
    """
    There is a case where customer want to load the checkpoint saved by raw
    torch. Because nebula cannot load torch checkpoint directly as they have
    different folder structures to bring the gap for loading(the data are
    totaly same in bytes for torch and nebula saving).  In this case, we must
    disable nebula load to use raw torch load.  Customer can just set
    `enable_nebula_load` to False. Then use original way of deepspeed to load,
    i.e. set the value of "--load".
    """
    @validator("persistent_storage_path")
    def load_path_check(cls, field_value, values):
        if values["load_path"] is None:
            values["load_path"] = field_value
        return field_value
