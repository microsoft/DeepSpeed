'''Copyright The Microsoft DeepSpeed Team'''

import glob
import shutil
from unit.common import DistributedTest
from unit.checkpoint.common import *
from unit.simple_model import *
import json


class TestNebulaCheckpoint(DistributedTest):
    world_size = 1

    def test_save_checkpoint(self, tmpdir):
        print("list /dev/shm origin status: ", os.listdir("/dev/shm/"))
        for filename in glob.glob("/dev/shm/shm_name_partition_*"):
            os.remove(filename)
        print("list /dev/shm before save: ", os.listdir("/dev/shm/"))

        isExist = os.path.exists("/tmp/nebula_checkpoint/")
        if isExist:
            shutil.rmtree("/tmp/nebula_checkpoint/")

        #judge if nebula service is running
        if is_service_launched("redis-server") or is_service_launched(
                "n2e0b2u2la_saturn") or is_service_launched(
                    "n2e0b2u2la_mars") or is_service_launched(
                        "n2e0b2u2la_replica_server"):
            shut_down_nebula_service()

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
        dtype = torch.float32
        ds_model = create_deepspeed_model(config_dict=config_dict,
                                          model=models[0],
                                          base_optimizer=None)

        data_loader = random_dataloader(model=ds_model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=ds_model.device,
                                        dtype=dtype)

        for _, batch in enumerate(data_loader):
            loss = ds_model(batch[0], batch[1])
            ds_model.backward(loss)
            ds_model.step()

        trained_model = ds_model

        save_folder = os.path.join(tmpdir, 'saved_checkpoint')
        save_tag = None

        trained_model.save_checkpoint(save_folder, tag=save_tag)
        print("list /dev/shm after save: ", os.listdir("/dev/shm/"))
        dist.barrier()

        # loaded_model = create_deepspeed_model(config_dict=config_dict,
        #                                       model=models[1],
        #                                       base_optimizer=None)

        # assert list(trained_model.parameters())[0].dtype == list(
        #     loaded_model.parameters())[0].dtype

        import torch_nebula as tn
        tn.flush_persistence()

        global_tag_ckpt = tn.list_checkpoints()
        js = json.dumps(global_tag_ckpt, sort_keys=True, indent=4, separators=(",", ":"))
        print(js)

        # get latest checkpoints by name
        latest_ckpt = tn.get_latest_checkpoint()
        print("latest_ckpt: ", latest_ckpt.tag)

        for root, dirs, files in os.walk("/tmp/nebula_checkpoint/"):
            if len(dirs) != 1:
                print("dirs: ", dirs)
                raise ValueError("Persist dir should only have one dir")
            if dirs[0] != latest_ckpt.tag:
                print("dirs[0]: ", dirs[0])
                raise ValueError(f"Dir name should be {latest_ckpt.tag}")
            for file in files:
                if file == "complete.txt":
                    comlete_file = os.path.join(root, file)
                    #read complete.txt
                    with open(comlete_file, "r") as f:
                        if f.read() != "OK":
                            raise ValueError("complete.txt should be OK")
        # loaded_model.load_checkpoint(save_folder, tag=save_tag)

        # compare_model_states(trained_model,
        #                      loaded_model,
        #                      compare_optimizer=True,
        #                      load_module_only=False)
        shut_down_nebula_service()
