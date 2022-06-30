import torch
from deepspeed.utils import logger


class CheckpointEngine(object):

    def __init__(self):
        return

    def save(self, state_dict, path: str, tag: str):
        logger.info(f"Saving {path} under tag{tag}...")
        torch.save(state_dict, path)
        logger.info(f"Saved {path} under tag{tag}.")
        return None

    def load(self, path: str, tag: str = None, persist_path: str = None, map_location=None):
        logger.info(f"Loading {path} under tag{tag} from {persist_path}...")
        partition = torch.load(path, map_location=map_location)
        logger.info(f"Loaded {path} under tag{tag} from {persist_path}.")
        return partition

    def commit(self, tag):
        print(f"Checkpoint {tag} is ready now!")
        return True
