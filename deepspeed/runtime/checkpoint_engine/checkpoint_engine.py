import torch
from deepspeed.utils import logger


class CheckpointEngine(object):
    def __init__(self):
        pass

    def save(self, state_dict, path: str):
        logger.info(f"Saving {path}...")
        torch.save(state_dict, path)
        logger.info(f"Saved {path}.")
        return None

    def load(self, path: str, map_location=None):
        logger.info(f"Loading checkpoint from {path}...")
        partition = torch.load(path, map_location=map_location)
        logger.info(f"Loaded checkpoint from {path}.")
        return partition

    def commit(self, tag):
        print(f"Checkpoint {tag} is ready now!")
        return True
