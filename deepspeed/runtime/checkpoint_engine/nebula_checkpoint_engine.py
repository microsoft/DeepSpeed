'''Copyright The Microsoft DeepSpeed Team'''

import os
import torch
import torch_nebula

from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine
from deepspeed.utils import logger, log_dist


def _get_tag_from_path(path):
    return os.path.basename(os.path.dirname(path))


class NebulaCheckpointEngine(CheckpointEngine):
    def __init__(self, config_params=None):
        super().__init__(config_params)
        self.checkpoint = None
        self.tag_flag = None
        self.enable_nebula_load = config_params.enable_nebula_load
        self.nebula_load_path = config_params.load_path

        nebula_config_params = {
            key: getattr(config_params,
                         key)
            for key in ("persistent_storage_path",
                        "persisten_time_interval",
                        "num_of_version_in_retention")
        }
        torch_nebula.init(**nebula_config_params)

    def create(self, tag):
        log_dist(f"[Nebula] Start Checkpoint for tag:{tag}", ranks=[0])
        # -2 means: customer needs to  explicitly tell nebula
        # current checkpoint is complete by commit methond.
        self.checkpoint = torch_nebula.Checkpoint(tag, -2)

    def save(self, state_dict, path: str):
        log_dist(f"[Nebula] Create dummy files for loading.")
        torch.save("", path)

        tag = _get_tag_from_path(path)
        partititon_name = os.path.basename(path)
        logger.info(f"[Nebula] Saving {partititon_name} under tag {tag}...")
        self.checkpoint.save(partititon_name, state_dict)
        logger.info(f"[Nebula] Saved {partititon_name} under tag {tag}.")
        return None

    def load(self, path: str, map_location=None):
        tag = _get_tag_from_path(path)
        first_load_flag = self.tag_flag is None or self.tag_flag == tag
        if not self.enable_nebula_load and first_load_flag:
            self.tag_flag = tag
            logger.info(
                f"[Nebula] Disable nebula load. Loading checkpoint from {path} ...")
            partition = torch.load(path, map_location=map_location)
            logger.info(f"[Nebula] Disable nebula load. Loaded checkpoint from {path} .")
            return partition

        partititon_name = os.path.basename(path)
        logger.info(
            f"[Nebula] Loading {path} under tag {tag} from nebula path {self.nebula_load_path}..."
        )

        checkpoint = None
        if tag in (None, 'latest', 'latest_universal'):
            # In some cases, there is the inconsistent tag between deepspeed metadata (latest file)
            # and nebula metadata, will lead to the failure on loading with deepspeed tag. Then we
            # will try to load the valid latest checkpoint from nebula(tier3 > tier1). So, in summary
            # when met failure loading for given tag, the loading priority would be like:
            #               nebula tier3 latest > nebula tier1 latest.
            checkpoint = torch_nebula.get_latest_checkpoint(
                persist_path=self.nebula_load_path)
        else:
            checkpoint = torch_nebula.get_checkpoint(tag=tag,
                                                     persist_path=self.nebula_load_path)

        if checkpoint is None or (checkpoint is not None and checkpoint.tag == ''):
            logger.info(
                f"Unable to find valid checkpoint tag:{tag} from Nebula, try to get latest checkpoint again from nebula {self.nebula_load_path} path!"
            )
            # nebula tier3 latest
            checkpoint = torch_nebula.get_latest_checkpoint(
                persist_path=self.nebula_load_path)
            if checkpoint is None or (checkpoint is not None and checkpoint.tag == ''):
                logger.info(
                    f"Unable to find latest checkpoint from Nebula tier3, try to get latest checkpoint again from nebula tier1 path!"
                )
                # nebula tier1 latest
                checkpoint = torch_nebula.get_latest_checkpoint()
                logger.warning(
                    f"Unable to find valid checkpoint from Nebula under tag:{tag}.")
                return None

        tag = checkpoint.tag
        self.tag_flag = -1
        partition = checkpoint.load(partititon_name, map_location=map_location)
        logger.info(
            f"[Nebula] Loaded {path} under tag {tag} from {self.nebula_load_path}.")
        return partition

    def commit(self, tag):
        # nebula commit will be call when all files under give tag are ready to be persisted in the async way.
        logger.info(
            f"[Nebula] all files for {tag} are saved in tier1. It is ready to start persisting"
        )
        commit_rls = self.checkpoint.commit()
        if not commit_rls:
            logger.error(
                f"[Nebula] failed to commit the checkpoint, please check the log.")
            return False
        return commit_rls
