from unit.common import DistributedTest
from unit.checkpoint.common import checkpoint_correctness_verification
from unit.simple_model import *

class TestNebulaCheckpoint(DistributedTest):
    world_size = 4

    @pytest.mark.parametrize("zero_stage", [0, 1])
    def test_checkpoint_nebula_engine(self, zero_stage, tmpdir):
        config_dict = {
            "train_batch_size": 2,
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-5
                }
            },
            "zero_optimization": {
                "stage": zero_stage
            },
            "fp16": {
                "enabled": zero_stage > 0
            },
            "scheduler": {
                "type": "OneCycle",
                "params": {
                    "cycle_first_step_size": 1000,
                    "cycle_first_stair_count": 500,
                    "cycle_second_step_size": 1000,
                    "cycle_second_stair_count": 500,
                    "decay_step_size": 1000,
                    "cycle_min_lr": 0.0001,
                    "cycle_max_lr": 0.0010,
                    "decay_lr_rate": 0.001,
                    "cycle_min_mom": 0.85,
                    "cycle_max_mom": 0.99,
                    "decay_mom_rate": 0.0
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
        models = [LinearStackPipe(num_stages=2) for _ in range(2)]
        checkpoint_correctness_verification(config_dict=config_dict,
                                            models=models,
                                            hidden_dim=models[0].hidden_dim,
                                            tmpdir=tmpdir,
                                            fp16=config_dict['fp16']['enabled'],
                                            load_optimizer_states=True,
                                            load_lr_scheduler_states=True,
                                            train_batch=True)