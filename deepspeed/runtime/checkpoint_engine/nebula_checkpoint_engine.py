import os
import torch_nebula

from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine
from deepspeed.utils import logger
from deepspeed.nebula.constants import *


class NebulaCheckpointEngine(CheckpointEngine):
    def __init__(self, config_params=None):
        nebula_config_params = {
            NEBULA_PERSISTENT_STORAGE_PATH: config_params.persistent_storage_path,
            NEBULA_PERSISTENT_TIME_INTERVAL: config_params.persistent_storage_path,
            NEBULA_NUM_OF_VERSION_IN_RETENTION:
            config_params.num_of_version_in_retention,
        }
        torch_nebula.init(**nebula_config_params)

    def save(self, state_dict, path: str, tag: str):
        logger.info(f"[Nebula] Saving {path} under tag{tag}...")
        partititon_name = os.path.basename(path)

        # -2 means: customer needs to  explicitly tell nebula
        # current checkpoint is complete by commit methond.
        checkpoint = torch_nebula.Checkpoint(tag, -2)
        checkpoint.save(partititon_name, state_dict)
        logger.info(f"[Nebula] Saved {path} under tag{tag}.")
        return None

    def load(self,
             path: str,
             tag: str = None,
             persist_path: str = None,
             map_location=None):
        logger.info(f"[Nebula] Loading {path} under tag{tag} from {persist_path}...")
        partititon_name = os.path.basename(path)
        checkpoint = None
        if tag is None:
            checkpoint = torch_nebula.get_latest_checkpoint(persist_path=persist_path)
            if checkpoint is None or (checkpoint is not None and checkpoint.tag == ''):
                logger.warning(f"Unable to find latest valid checkpoint from Nebula!")
                return None
        else:
            checkpoint = torch_nebula.get_checkpoint(tag=tag, persist_path=persist_path)
        partition = checkpoint.load(partititon_name)
        logger.info(f"[Nebula] Loaded {path} under tag{tag} from {persist_path}.")
        return partition

    def commit(self, tag):
        checkpoint = torch_nebula.Checkpoint(tag, -2)
        commit_rls = checkpoint.commit()
        return commit_rls
