# Copyright ______
# Maintained by ______

from deepspeed.runtime.config_utils import DeepSpeedConfigObject
class DeepSpeedDataStatesConfig(DeepSpeedConfigObject):
    def __init__(self, param_dict):
        super(DeepSpeedDataStatesConfig, self).__init__()

        self.enabled = None
        self.config = {}

        if "datastates_ckpt" in param_dict.keys():
            self.enabled = True
            self.config = param_dict["datastates_ckpt"]