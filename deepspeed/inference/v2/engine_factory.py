# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import logging
from typing import Any
from abc import ABC, abstractmethod

import deepspeed.inference.v2.model_implementations as model_impl
from .engine_v2 import InferenceEngineV2
from .config_v2 import RaggedInferenceEngineConfig
from .checkpoint import HuggingFaceCheckpointEngine
from .logging import inference_logger


class EngineFactory(ABC):
    """
    Abstract factory for creating inference engines.
    """

    @property
    @abstractmethod
    def supported_policies():
        """
        Returns a dictionary of supported policies.
        """
        pass

    @abstractmethod
    def build_engine(self,
                     path: str,
                     engine_config: RaggedInferenceEngineConfig,
                     debug_level: int = logging.INFO,
                     random_weights_config: Any = None,
                     fill_random: bool = False) -> InferenceEngineV2:
        """
        Build an InferenceV2 engine.
        """
        pass


class HFEngineFactory(EngineFactory):

    @property
    def supported_policies():
        return {
            "opt": model_impl.opt.policy.OPTPolicy,
            "llama": model_impl.llama_v2.llama_v2_policy.Llama2Policy,
            "mistral": model_impl.mistral.policy.MistralPolicy
        }

    def build_engine(self,
                     model_name_or_path: str,
                     engine_config: RaggedInferenceEngineConfig,
                     debug_level: int = logging.INFO,
                     random_weights_config: Any = None,
                     fill_random: bool = False) -> InferenceEngineV2:
        """
        Build an InferenceV2 engine for HuggingFace models.
        """
        # Set up logging
        inference_logger(level=debug_level)

        # get HF checkpoint engine
        checkpoint_engine = HuggingFaceCheckpointEngine(model_name_or_path)

        # get model config from HF AutoConfig
        model_config = checkpoint_engine.model_config

        if model_config.model_type not in self.supported_policies:
            raise ValueError(f"Unsupported model type {model_config.model_type}")

        policy = self.supported_policies[model_config.model_type](checkpoint_engine, model_config)
        return InferenceEngineV2(policy, engine_config)


def build_hf_engine(model_name_or_path: str,
                    engine_config: RaggedInferenceEngineConfig,
                    debug_level: int = logging.INFO,
                    random_weights_config: Any = None,
                    fill_random: bool = False) -> InferenceEngineV2:
    """
    Build an InferenceV2 engine for HuggingFace models.
    """
    return HFEngineFactory().build_engine(model_name_or_path, engine_config, debug_level, random_weights_config,
                                          fill_random)
