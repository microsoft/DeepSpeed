# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import json
import torch
from .base_engine import CheckpointEngineBase
from typing import Iterable, Tuple

from ..logging import inference_logger


class HuggingFaceCheckpointEngine(CheckpointEngineBase):

    def __init__(self, model_name_or_path: str, auth_token: str = None) -> None:
        super().__init__()
        from transformers import AutoConfig, GenerationConfig

        self.model_name_or_path = model_name_or_path
        self.auth_token = auth_token
        self.model_config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.generation_config = GenerationConfig.from_pretrained(self.model_name_or_path)
        # Define this property here so we can use it in the model implementation
        if not hasattr(self.model_config, "max_seq_length"):
            self.model_config.max_seq_length = self.model_config.max_position_embeddings
        else:
            self.model_config.max_seq_length = self.generation_config.max_length

        self._all_ckpt_paths = self._fetch_checkpoint_files()

    def _fetch_checkpoint_files(self):
        """
        Fetch the checkpoint files from the HuggingFace Hub.
        """
        # TODO(jeff): for models like llama-2 the user will have to provide an auth `token`,
        # currently coming from the ckpt engine init but maybe a catch all kwargs for other
        # snapshot download parameters would be more flexible.

        # NOTE(jeff): allow_patterns here are explicitly not using safetensors or other
        # checkpoint files that may be present. Example of all files in the llama-2-7b
        # repo here: https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main

        # UPDATE: allow safetensors now. Will auto fetch `*.safetensors` by default;
        # fallback to fetch the `.bin` files if there are no `*.safetensors`.
        from huggingface_hub import snapshot_download

        def snapshot_download_bin():
            return snapshot_download(self.model_name_or_path,
                                                            allow_patterns=[
                                                                "*.bin",
                                                                "*.json",
                                                                "*.pt",
                                                            ],
                                                            revision=None,
                                                            token=self.auth_token)

        def snapshot_download_safetensors():
            return snapshot_download(self.model_name_or_path,
                                                            allow_patterns=[
                                                                "*.json",
                                                                "*.pt",
                                                                "*.safetensors",
                                                            ],
                                                            revision=None,
                                                            token=self.auth_token)

        if os.path.isdir(self.model_name_or_path):
            self._local_checkpoint_dir = self.model_name_or_path
        else:
            self._local_checkpoint_dir = snapshot_download_safetensors()

        assert os.path.isdir(
            self._local_checkpoint_dir
        ), f"Checkpoint dir {self._local_checkpoint_dir} is not a directory, cannot load checkpoint."

        safe_model_param_json = os.path.join(self._local_checkpoint_dir, "model.safetensors.index.json")
        safe_model_file = os.path.join(self._local_checkpoint_dir, 'model.safetensors')
        model_param_json = os.path.join(self._local_checkpoint_dir, "pytorch_model.bin.index.json")
        model_file = os.path.join(self._local_checkpoint_dir, 'pytorch_model.bin')

        # Prioritize `*.safetensors` over `*.bin`
        if os.path.isfile(safe_model_param_json):
            param_map = json.load(open(safe_model_param_json, "r"))
        else:
            if os.path.isfile(safe_model_file):
                all_checkpoint_files = [safe_model_file]

                return all_checkpoint_files
            else:
                self._local_checkpoint_dir = snapshot_download_bin()

                if os.path.isfile(model_param_json):
                    param_map = json.load(open(model_param_json, "r"))
                else:
                    all_checkpoint_files = [model_file]

                    return all_checkpoint_files

        # weight_map -> { "lm_head.weight": "pytorch_model-00002-of-00002.bin", ... }
        weight_map = param_map["weight_map"]

        # unique set of all checkpoint files
        all_checkpoint_files = set(weight_map.values())

        # get absolute path of all unique checkpoint files
        all_checkpoint_files = [os.path.join(self._local_checkpoint_dir, f) for f in all_checkpoint_files]

        return all_checkpoint_files

    def parameters(self) -> Iterable[Tuple[str, torch.Tensor]]:
        """
        Generator of model parameters (satisfies the CheckpointEngineBase interface).
        """
        from safetensors.torch import load_file

        for checkpoint in self._all_ckpt_paths:
            inference_logger().info(f"Loading checkpoint: {checkpoint}")
            if checkpoint.endswith(".safetensors"):
                checkpoint_sd = load_file(checkpoint, device="cpu")
            else:
                checkpoint_sd = torch.load(checkpoint, map_location='cpu')
            param_keys = list(checkpoint_sd.keys())
            for param_name in param_keys:
                param = checkpoint_sd[param_name]
                yield param_name, param

            del checkpoint_sd


if __name__ == "__main__":
    # To test, add your auth_token here and run `python huggingface_engine.py`
    engine = HuggingFaceCheckpointEngine(model_name_or_path="meta-llama/Llama-2-7b-hf",
                                         auth_token="hf_xxxxxxxxxxxxxxxxx")
    for name, param in engine.parameters():
        print(name, param.shape)
