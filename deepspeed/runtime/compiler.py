# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Union, Callable, Dict, Any
import importlib
import torch
from ..pydantic_v1 import validator
from .config_utils import DeepSpeedConfigModel

COMPILE_CONFIG = "compile"


def is_compile_supported():
    return hasattr(torch, "compiler")


def disable(func):
    if is_compile_supported():
        return torch.compiler.disable(func)
    return func


def get_compile_config(param_dict):
    if COMPILE_CONFIG in param_dict:
        compile_config_dict = param_dict[COMPILE_CONFIG]
    else:
        compile_config_dict = {}
    return CompileConfig(**compile_config_dict)


def get_backend_fn(backend: Union[str, Callable]) -> Union[str, Callable]:
    if isinstance(backend, Callable):
        return backend

    elif isinstance(backend, str):
        if backend in torch._dynamo.list_backends(exclude_tags=()):
            return backend

        # Get module name from backend name
        module_name = '.'.join(backend.split('.')[:-1])
        fn_name = backend.split('.')[-1]

        try:
            module = importlib.import_module(module_name)
            backend_fn = getattr(module, fn_name)
        except ImportError:
            raise ValueError(
                f"The backend {backend} is not in the list of available backends and could not be imported.")
        return backend_fn

    raise ValueError(f"backend for torch.compile must be a string or Callable: {backend}")


class CompileConfig(DeepSpeedConfigModel):
    """
    [EXPERIMENTAL] This configuration enables users to activate `torch.compile` within DeepSpeed and customize its settings.
    Please be aware that these features and API designs are experimental and subject to change.
    """

    enabled: bool = False
    """
    Enable torch.compile when True.
    """

    backend: str = "inductor"
    """
    Passed to `backend` argument of torch.compile.
    If the given value is not in torch._dynamo.list_backends(),
    DeepSpeed attempts to import and instantiate the module with the given name.
    """

    kwargs: Dict[str, Any] = {}
    """
    Passed to `kwargs` argument of torch.compile.
    """

    @validator("enabled")
    def validate_enabled(cls, field_value, values):
        if field_value and not is_compile_supported():
            raise ValueError("torch.compile is not supported on this version of PyTorch.")
        return field_value
