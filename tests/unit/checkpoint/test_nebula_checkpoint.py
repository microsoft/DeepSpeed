from unit.common import DistributedTest
from unit.checkpoint.common import checkpoint_correctness_verification
from unit.simple_model import *
import pytest
class TestNebulaCheckpoint(DistributedTest):
    world_size = 4

    def test_checkpoint_nebula_engine(self, tmpdir):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "nebula": {
                "enabled": "true",
                "persistent_storage_path": "/tmp/data_storage",
                "persistent_time_interval": 100,
                "num_of_version_in_retention": 2,
                "enable_nebula_load": "true"
            }
        }
        hidden_dim = 10
        models = [SimpleModel(hidden_dim=hidden_dim) for _ in range(2)]
        checkpoint_correctness_verification(config_dict=config_dict,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=True,
                                            load_lr_scheduler_states=False,
                                            fp16=False,
                                            empty_tag=True)

