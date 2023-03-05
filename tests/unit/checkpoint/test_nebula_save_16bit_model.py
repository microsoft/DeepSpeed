from unit.common import DistributedTest
from unit.checkpoint.common import *
from unit.simple_model import *
import pytest
import json
class TestNebulaCheckpoint2(DistributedTest):
    world_size = 2

    def test_save_16bit_model(self, tmpdir):
        config_dict = {
            "optimizer": {
                "type": 'Adam'
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "zero_optimization": {
                "stage": 3,
                "stage3_gather_fp16_weights_on_model_save": True,
            },
            "gradient_accumulation_steps": 2,
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": 4,
            "nebula": {
                "enabled": True,
                "persistent_storage_path": "/tmp/nebula_checkpoint_16bit/",
                "persistent_time_interval": 10,
                "num_of_version_in_retention": 2,
                "enable_nebula_load": True
            }
        }
        hidden_dim = 10
        models = [SimpleModel(hidden_dim=hidden_dim) for _ in range(2)]

        ds_model = create_deepspeed_model(config_dict=config_dict,
                                          model=models[0],
                                          base_optimizer=None)

        data_loader = random_dataloader(model=ds_model,
                                        total_samples=2,
                                        hidden_dim=hidden_dim,
                                        device=ds_model.device,
                                        dtype=torch.half)

        batch = next(iter(data_loader))
        loss = ds_model(batch[0], batch[1])
        ds_model.backward(loss)
        ds_model.step()

        save_tag = "global_step0"
        save_folder = os.path.join(tmpdir, 'save_16bit_model', save_tag)
        
        ds_model.save_16bit_model(save_folder)
        dist.barrier()
        loaded_model = create_deepspeed_model(config_dict=config_dict,
                                            model=models[1],
                                            base_optimizer=None)
        
        assert list(ds_model.parameters())[0].dtype == list(loaded_model.parameters())[0].dtype

        import torch_nebula as tn
        
        tn.flush_persistence(save_tag)

        global_tag_ckpt = tn.list_checkpoints()
        js = json.dumps(global_tag_ckpt, sort_keys=True, indent=4, separators=(",", ":"))
        print(js)

        # get latest checkpoints by name
        latest_ckpt = tn.get_latest_checkpoint()
        loaded_model.load_checkpoint("/tmp/nebula_checkpoint_16bit/",
                                    tag=latest_ckpt.tag)

        compare_model_states(ds_model,
                            loaded_model,
                            compare_optimizer=True,
                            load_module_only=False)