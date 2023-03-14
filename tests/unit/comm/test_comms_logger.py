import torch
import deepspeed.comm as dist
import deepspeed

from unit.common import DistributedTest
from unit.simple_model import SimpleModel

import time, logging, os


def within_range(val, target, tolerance):
    print(f'prof_on: {val}, prof_off: {target}')
    return val - target / target < tolerance


# This tolerance seems tight enough to catch comm logging overhead while loose enough to allow for comm instability.
# Can increase if github runner comm instability leads to many false negatives.
TOLERANCE = 0.05


class TestCommsLoggingOverhead(DistributedTest):
    world_size = [2, 4]

    def test(self):
        # Need comm warmups, or else whoever communicates first loses
        NUM_WARMUPS = 5
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "comms_logger": {
                "enabled": False,
                "verbose": True,
                "prof_all": True,
                "debug": False
            }
        }

        # dummy model
        model = SimpleModel(4)
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                            model=model,
                                            model_parameters=model.parameters())
        x = torch.ones(4, 2**15).cuda() * (dist.get_rank() + 1)

        for i in range(NUM_WARMUPS):
            dist.all_reduce(x)

        # Time allreduce without logging
        start = time.time()
        dist.all_reduce(x, prof=False)
        torch.cuda.synchronize()
        time_prof_off = time.time() - start
        dist.all_reduce(torch.Tensor([time_prof_off]).cuda(),
                        prof=False,
                        op=dist.ReduceOp.AVG)

        # Time allreduce with logging
        start = time.time()
        dist.all_reduce(x, prof=True)
        torch.cuda.synchronize()
        time_prof_on = time.time() - start
        dist.all_reduce(torch.Tensor([time_prof_on]).cuda(),
                        prof=False,
                        op=dist.ReduceOp.AVG)

        # Ensure logging doesn't add significant overhead
        assert within_range(time_prof_on, time_prof_off, tolerance=TOLERANCE)


class TestNumLoggingCalls(DistributedTest):
    world_size = [2, 4]

    def test(self, class_tmpdir):
        num_all_reduce_calls = 4
        num_broadcast_calls = 2

        # Have the DeepSpeed logger output to both stdout and file so that we can verify log output
        file_path = os.path.join(class_tmpdir,
                                 f"comm_output_{int(os.environ['WORLD_SIZE'])}.log")
        DSLogger = logging.getLogger('DeepSpeed')
        fileHandler = logging.FileHandler(file_path)
        DSLogger.addHandler(fileHandler)

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "comms_logger": {
                "enabled": True,
                "verbose": True,
                "prof_all": True,
                "debug": False
            }
        }

        # dummy model so that config options are picked up
        model = SimpleModel(4)
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                            model=model,
                                            model_parameters=model.parameters())

        x = torch.ones(1, 3).cuda() * (dist.get_rank() + 1)

        # Make comm calls
        for i in range(num_all_reduce_calls):
            dist.all_reduce(x, log_name="all_reduce_test")
        for i in range(num_broadcast_calls):
            dist.broadcast(x, 0, log_name="broadcast_test")

        # Count the number of logs
        with open(file_path, 'r') as f:
            log_output = f.read()
        num_all_reduce_logs = log_output.count('all_reduce_test')
        num_broadcast_logs = log_output.count('broadcast_test')

        # Ensure all comm calls are logged
        assert num_all_reduce_logs == num_all_reduce_calls
        assert num_broadcast_logs == num_broadcast_calls
