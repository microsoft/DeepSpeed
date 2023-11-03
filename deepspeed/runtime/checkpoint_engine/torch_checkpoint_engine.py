# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

import torch
from deepspeed.utils import logger, log_dist
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine


class TorchCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params=None, rank=0):
        super().__init__(config_params)
        self.rank = rank

    def create(self, tag):
        log_dist(f"[Torch] Checkpoint {tag} is about to be saved!", ranks=[0])

    def save(self, state_dict, path: str):
        logger.info(f"[Torch] Saving {path}...")
        torch.save(state_dict, path)
        logger.info(f"[Torch] Saved {path}.")
        return None

    def commit(self, tag):
        # record tag of most recent checkpoint in 'latest' file
        if self.rank == 0:
            with open(os.path.join(save_dir, 'latest'), 'w') as fd:
                fd.write(tag)

        logger.info(f"[Torch] Checkpoint {tag} is ready now!")
        return True

    def open(self, load_dir=None, tag=None):
        # read tag from latest file if not given an actual name
        if tag in ['latest', 'latest_universal']:
            latest_path = os.path.join(load_dir, tag)
            if os.path.isfile(latest_path):
                with open(latest_path, "r") as fd:
                    tag = fd.read().strip()
                    return tag
            else:
                return None
        return tag

    def load(self, path: str, map_location=None):
        logger.info(f"[Torch] Loading checkpoint from {path}...")
        partition = torch.load(path, map_location=map_location)
        logger.info(f"[Torch] Loaded checkpoint from {path}.")
        return partition
