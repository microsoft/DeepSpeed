# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Dict

_WEIGNT_QUANTIZATION_ = 'weight_quantization'
_QUANTIZED_INITIALIZATION_ = 'quantized_initialization'
_POST_INIT_QUIANT_ = 'post_init_quantization'


class WeightQuantizationConfig:

    def __init__(self, param_dict: Dict) -> None:
        super(WeightQuantizationConfig, self).__init__()
        self.quantized_initialization = None
        self.post_init_quant = None

        if _WEIGNT_QUANTIZATION_ in param_dict:
            weight_quantization_config = param_dict[_WEIGNT_QUANTIZATION_]

            assert not (_QUANTIZED_INITIALIZATION_ in weight_quantization_config and _POST_INIT_QUIANT_
                        in weight_quantization_config), 'Must choose only one quantization flavor.'

            if _QUANTIZED_INITIALIZATION_ in weight_quantization_config:
                self.quantized_initialization = weight_quantization_config[_QUANTIZED_INITIALIZATION_]
            if _POST_INIT_QUIANT_ in weight_quantization_config:
                self.post_init_quant = weight_quantization_config[_POST_INIT_QUIANT_]
