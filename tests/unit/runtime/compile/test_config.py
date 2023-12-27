# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest, get_master_port
from unit.simple_model import SimpleModel

import torch
import torch.multiprocessing as mp

# A test on its own
import deepspeed

TIMEOUT = 600


@pytest.fixture
def base_config():
    config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015
            }
        },
        "fp16": {
            "enabled": True
        },
        "compile": {
            "backend": "inductor"
        }
    }
    return config_dict


custom_backend_called = False


def custom_backend(gm: torch.fx.GraphModule, example_inputs):
    global custom_backend_called
    custom_backend_called = True
    return gm.forward


class DistributedCompileTest(DistributedTest):
    """
    This class runs tests with non-daemonic processes while DistributedTest launches daemon processes.
    torch.compile creates a new process to compile the model, but daemonic processes is not allowed to create new processes.
    """

    def _dist_run_queue(self, local_rank, num_procs, master_port, queue):
        queue.put(self._dist_run(local_rank, num_procs, master_port))

    def _launch_procs(self, num_procs):
        # Verify we have enough accelerator devices to run this test
        if get_accelerator().is_available() and get_accelerator().device_count() < num_procs:
            pytest.skip(
                f"Skipping test because not enough GPUs are available: {num_procs} required, {get_accelerator().device_count()} available"
            )

        # Set start method to `forkserver` (or `fork`)
        mp.set_start_method('forkserver', force=True)

        master_port = get_master_port()

        # Run the test
        queue = mp.Queue()
        procs = [
            mp.Process(target=self._dist_run_queue, args=(local_rank, num_procs, master_port, queue))
            for local_rank in range(num_procs)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        try:
            skip_msgs = [queue.get(timeout=TIMEOUT) for _ in range(num_procs)]
        except mp.Empty:
            pytest.exit("Test hanged, exiting", returncode=0)

        # If we skipped a test, propagate that to this process
        if any(skip_msgs):
            assert len(set(skip_msgs)) == 1, "Multiple different skip messages received"
            pytest.skip(skip_msgs[0])


class TestConfigLoad(DistributedCompileTest):
    world_size = 1

    def _init_engine(self, config):
        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
        return engine

    def _run_model(self, engine):
        train_batch_size = 1
        device = torch.device(get_accelerator().current_device_name())
        dtype = engine.module.linears[0].weight.dtype
        hidden_dim = engine.module.linears[0].weight.shape[1]
        x = torch.rand(train_batch_size, hidden_dim, device=device, dtype=dtype)
        y = torch.randn_like(x)
        engine(x, y)

    def test_compile(self, base_config):
        engine = self._init_engine(base_config)
        self._run_model(engine)

    def test_custom_backend(self, base_config):
        global custom_backend_called
        custom_backend_called = False

        engine = self._init_engine(base_config)
        engine.set_backend(f"{__name__}.custom_backend")
        self._run_model(engine)
        assert custom_backend_called
