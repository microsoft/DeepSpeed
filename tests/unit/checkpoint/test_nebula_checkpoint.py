from unit.common import DistributedTest
from unit.checkpoint.common import *
from unit.simple_model import *
import pytest


class TestNebulaCheckpoint(DistributedTest):
    world_size = 2

    # @pytest.mark.parametrize('zero_stage', [3])
    # def test_save_16bit_model(self, tmpdir, zero_stage):
    #     config_dict = {
    #         "optimizer": {
    #             "type": 'Adam'
    #         },
    #         "fp16": {
    #             "enabled": True,
    #             "initial_scale_power": 8
    #         },
    #         "zero_optimization": {
    #             "stage": zero_stage,
    #             "stage3_gather_fp16_weights_on_model_save": True,
    #         },
    #         "gradient_accumulation_steps": 1,
    #         "train_micro_batch_size_per_gpu": 1,
    #         "train_batch_size": 4,
    #         "nebula": {
    #             "enabled": True,
    #             "persistent_storage_path": "/tmp/nebula_checkpoint/",
    #             "persistent_time_interval": 10,
    #             "num_of_version_in_retention": 2,
    #             "enable_nebula_load": True
    #         }
    #     }
    #     hidden_dim = 10
    #     models = [SimpleModel(hidden_dim=hidden_dim) for _ in range(2)]

    #     ds_model = create_deepspeed_model(config_dict=config_dict,
    #                                       model=models[0],
    #                                       base_optimizer=None)

    #     data_loader = random_dataloader(model=ds_model,
    #                                     total_samples=2,
    #                                     hidden_dim=hidden_dim,
    #                                     device=ds_model.device,
    #                                     dtype=torch.half)

    #     batch = next(iter(data_loader))
    #     loss = ds_model(batch[0], batch[1])
    #     ds_model.backward(loss)
    #     ds_model.step()

    #     save_folder = os.path.join(tmpdir, 'save_16bit_model')
    #     save_tag = None
    #     ds_model.save_16bit_model(save_folder)
        
    #     loaded_model = create_deepspeed_model(config_dict=config_dict,
    #                                         model=models[1],
    #                                         base_optimizer=None)
        
    #     assert list(ds_model.parameters())[0].dtype == list(loaded_model.parameters())[0].dtype

    #     loaded_model.load_checkpoint(tmpdir,
    #                                  tag=save_tag)

    #     compare_model_states(ds_model,
    #                          loaded_model)
        
    def test_save_checkpoint(self, tmpdir):
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
                "enabled": True,
                "persistent_storage_path": "/tmp/nebula_checkpoint/",
                "persistent_time_interval": 10,
                "num_of_version_in_retention": 2,
                "enable_nebula_load": True
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
