# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import json
import logging
import os
import pickle

from .engine_v2 import InferenceEngineV2
from .config_v2 import RaggedInferenceEngineConfig
from .checkpoint import HuggingFaceCheckpointEngine
from .logging import inference_logger
from .model_implementations import (
    OPTPolicy,
    Llama2Policy,
    MistralPolicy,
)
from .model_implementations.inference_policy_base import POLICIES, InferenceV2Policy
from .model_implementations.flat_model_helpers import make_metadata_filename, ModelMetadata

def buid_engine_from_ds_checkpoint(path:str, engine_config: RaggedInferenceEngineConfig,
                    debug_level: int = logging.INFO) -> InferenceEngineV2::
    
    inference_logger(level=debug_level)
    # Load metadata, for grabbing the policy name we'll have all ranks just check for
    # rank 0.
    metadata_filename = make_metadata_filename(path, 0, engine_config.tensor_parallel.tp_size)
    metadata = json.load(open(metadata_filename, "r"))
    metadata = ModelMetadata.parse_raw(metadata)

    # Get the policy
    try:
        policy_cls: InferenceV2Policy = POLICIES[metadata.policy]
    except KeyError:
        raise ValueError(f"Unknown policy {metadata.policy} for model {path}")

    # Load the model config
    model_config = pickle.load(open(os.path.join(path, "ds_model_config.pkl"), "rb"))
    policy = policy_cls(model_config, inf_checkpoint_path=path)
    
    return InferenceEngineV2(policy, engine_config)

def build_hf_engine(path: str,
                    engine_config: RaggedInferenceEngineConfig,
                    debug_level: int = logging.INFO) -> InferenceEngineV2:
    """
    Build an InferenceV2 engine for HuggingFace models.
    """

    if os.path.exists(os.path.join(path, "ds_model_config.pkl")):
        return buid_engine_from_ds_checkpoint(path, engine_config)
    else:
        # Set up logging
        inference_logger(level=debug_level)
        # get HF checkpoint engine
        checkpoint_engine = HuggingFaceCheckpointEngine(path)

        # get model config from HF AutoConfig
        model_config = checkpoint_engine.model_config

        # get the policy
        # TODO: generalize this to other models
        if model_config.model_type == "opt":
            policy = OPTPolicy(model_config, checkpoint_engine=checkpoint_engine)
        elif model_config.model_type == "llama":
            policy = Llama2Policy(model_config, checkpoint_engine=checkpoint_engine)
        elif model_config.model_type == "mistral":
            policy = MistralPolicy(model_config, checkpoint_engine=checkpoint_engine)
        else:
            raise ValueError(f"Unsupported model type {model_config.model_type}")

        return InferenceEngineV2(policy, engine_config)
