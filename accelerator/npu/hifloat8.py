# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Resolve the optional public TorchAO HiFloat8 APIs on NPU."""

import importlib
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version


def get_hifloat8_provider():
    hint = ("Install torchao>=0.18,<0.19 and a compatible torchao-npu build "
            "providing torchao.hifloat8; keep automatic activation enabled.")
    try:
        installed_version = Version(version("torchao"))
    except PackageNotFoundError as error:
        raise ImportError(f"TorchAO HiFloat8 is unavailable. {hint}") from error
    if not Version("0.18") <= installed_version < Version("0.19"):
        raise RuntimeError(f"Unsupported torchao version {installed_version}. {hint}")
    try:
        api = importlib.import_module("torchao.hifloat8")
    except ImportError as error:
        raise ImportError(f"TorchAO HiFloat8 API is unavailable. {hint}") from error
    required = ("HiFloat8Config", "assert_hifloat8_training_available")
    missing = [name for name in required if not hasattr(api, name)]
    if missing:
        raise ImportError(f"TorchAO HiFloat8 API is missing {missing}. {hint}")
    return api
