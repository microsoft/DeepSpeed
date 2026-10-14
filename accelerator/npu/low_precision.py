# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""TorchAO public HiFloat8 interface for the NPU accelerator."""

import importlib
from importlib.metadata import PackageNotFoundError, version

import torch
from packaging.version import Version

from ..low_precision import LowPrecisionProvider


class TorchAOHiFloat8Provider(LowPrecisionProvider):

    def __init__(self):
        self._api = None

    def _get_api(self):
        if self._api is not None:
            return self._api
        hint = ("Install torchao>=0.18,<0.19 and a compatible torchao-npu build "
                "providing torchao.hifloat8; keep its automatic activation enabled.")
        try:
            installed_version = Version(version("torchao"))
        except PackageNotFoundError as error:
            raise ImportError(f"TorchAO low-precision provider is unavailable. {hint}") from error
        if not Version("0.18") <= installed_version < Version("0.19"):
            raise RuntimeError(f"Unsupported torchao version {installed_version}. {hint}")
        try:
            api = importlib.import_module("torchao.hifloat8")
        except ImportError as error:
            raise ImportError(f"TorchAO HiFloat8 API is unavailable. {hint}") from error
        required = ("HiFloat8Config", "assert_hifloat8_training_available", "convert_to_hifloat8_training",
                    "hifloat8_grouped_mm")
        missing = [name for name in required if not hasattr(api, name)]
        if missing:
            raise ImportError(f"TorchAO HiFloat8 API is missing {missing}. {hint}")
        self._api = api
        return api

    def config_from_dict(self, values):
        return self._get_api().HiFloat8Config.from_dict(values)

    def assert_training_available(self, *, device, config, model_dtype, linear, grouped):
        if model_dtype != torch.bfloat16:
            raise RuntimeError("TorchAO HiFloat8 training requires BF16 model parameters")
        self._get_api().assert_hifloat8_training_available(device=device,
                                                           config=config,
                                                           linear=linear,
                                                           grouped=grouped)

    def convert_to_training(self, model, config, *, filter_fn):
        return self._get_api().convert_to_hifloat8_training(model, config, filter_fn=filter_fn)

    def grouped_mm(self, x, weight, offsets, *, config, trans_b=False):
        output = self._get_api().hifloat8_grouped_mm(x, weight, offsets, config=config, trans_b=trans_b)
        # Working precision is backend policy; residual paths retain model dtype.
        return output.to(x.dtype)
