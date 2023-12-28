# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional, Union, Callable, Dict
import importlib
import torch
from ..pydantic_v1 import validator
from .config_utils import DeepSpeedConfigModel

COMPILE_CONFIG = "compile"


def is_compile_supported():
    return hasattr(torch, "compile")


def disable(func):

    if is_compile_supported():
        return torch.compiler.disable(func)
    return func


def get_compile_config(param_dict):
    if COMPILE_CONFIG in param_dict:
        compile_config_dict = param_dict[COMPILE_CONFIG]
    else:
        compile_config_dict = {"disable": True}
    return CompileConfig(**compile_config_dict)


def get_backend_fn(backend_name: str):
    # Get module name from backend name
    module_name = '.'.join(backend_name.split('.')[:-1])
    fn_name = backend_name.split('.')[-1]

    try:
        module = importlib.import_module(module_name)
        backend_fn = getattr(module, fn_name)
    except ImportError:
        raise ValueError(f"Could not import module {module_name} for torch.compile backend.")
    return backend_fn


class CompileConfig(DeepSpeedConfigModel):

    backend: Union[str, Callable] = "inductor"
    """
    Passed to `backend` argument of torch.compile.
    If the given value is a string and not in torch._dynamo.list_backends(),
    DeepSpeed attempts to import and instantiate the module with the given name.
    """

    fullgraph: bool = False
    """
    Passed to `fullgraph` argument of torch.compile.
    """

    dynamic: Optional[bool] = None
    """
    Passed to `dynamic` argument of torch.compile.
    """

    mode: Union[str, None] = None
    """
    Passed to `mode` argument of torch.compile.
    """

    options: Optional[Dict[str, Union[str, int, bool]]] = None
    """
    Passed to `options` argument of torch.compile.
    """

    disable: bool = False
    """
    Passed to `disable` argument of torch.compile.
    """

    @validator("backend")
    def validate_backend(cls, field_value, values):
        if isinstance(field_value, str):
            if field_value not in torch._dynamo.list_backends():
                return get_backend_fn(field_value)
        elif not callable(field_value):
            raise ValueError(f"backend for torch.compile must be a string or Callable: {field_value}")

        return field_value


class CompiledModuleWrapper(torch.nn.Module):

    def __init__(self, module, compile_config: Union[CompileConfig, None] = None):
        super().__init__()

        assert is_compile_supported(), "torch.compile is not supported on this version of PyTorch."

        modules = self.__dict__.get('_modules')
        modules['wrapped'] = module
        self.__dict__['wrapped'] = module
        self.is_compiled = False
        self.custom_backend = None
        self.compile_config = compile_config

    def __getattr__(self, name):
        return getattr(self.__dict__['wrapped'], name)

    def set_backend(self, backend: Union[str, Callable]):
        if isinstance(backend, str):
            self.custom_backend = get_backend_fn(backend)
        elif callable(backend):
            self.custom_backend = backend
        else:
            raise ValueError(f"backend for torch.compile must be a string or Callable: {backend}")

    def forward(self, *args, **kwargs):
        if not self.is_compiled:
            if self.custom_backend is not None:
                self.compile_config.backend = self.custom_backend

            self.__dict__['wrapped'] = torch.compile(self.wrapped, **self.compile_config.dict())
            self.is_compiled = True

        return self.__dict__['wrapped'](*args, **kwargs)
