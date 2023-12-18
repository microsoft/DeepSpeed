'''Copyright The Microsoft DeepSpeed Team'''

import torch
from deepspeed.utils import logger, log_dist
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine

import scr


class SCRCheckpointEngine(CheckpointEngine):
    def __init__(self, config_params=None):
        super().__init__(config_params)

    def create(self, save_dir, tag):
        # TODO: check that save_dir is within SCR_PREFIX

        log_dist(f"[SCR] Checkpoint {tag} is starting!", ranks=[0])
        scr.start_output(tag, scr.FLAG_CHECKPOINT)

    def makedirs(self, path, exist_ok=False):
        # SCR delays creating directories until it flushes the checkpoint.
        # Based on how the user has configured their run,
        # SCR may discard some checkpoints without ever flushing them.
        pass

    def save(self, state_dict, path: str):
        path = scr.route_file(path)
        torch.save(state_dict, path)

    def commit(self, tag):
        scr.complete_output(True)
        log_dist(f"[SCR] Checkpoint {tag} is complete!", ranks=[0])
        return True

    def open(self, load_dir=None, tag=None):
        # TODO: ensure load_dir is within SCR_PREFIX

        # If caller provided a name, ask SCR to load that checkpoint
        load_latest = (tag in (None, 'latest', 'latest_universal'))
        if not load_latest:
            scr.current(tag)

        # Get name of checkpoint that SCR loaded
        name = scr.have_restart()

        # Return None if SCR failed to find a checkpoint
        if name is None:
            return None

        # Raise an error if the checkpoint that was loaded does not match the requested name
        if not load_latest and name != tag:
            raise RuntimeError(f"[SCR] Loaded checkpoint '{name}' does not match requested '{tag}'")

        # Open checkpoint for reading and return tag that we loaded
        log_dist(f"[SCR] Opened checkpoint '{name}'", ranks=[0])
        scr.start_restart()
        return name

    def load(self, path: str, map_location=None):
        path = scr.route_file(path)
        partition = torch.load(path, map_location=map_location)
        return partition

    def close(self, tag):
        scr.complete_restart(True)
        log_dist(f"[SCR] Closed checkpoint '{tag}'", ranks=[0])
