# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import random
import os
import numpy as np
from copy import deepcopy

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero import GatheredParameters

from unit.simple_model import SimpleModel
from unit.common import DistributedTest, get_master_port

TIMEOUT = 600


def enable_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    get_accelerator().manual_seed(seed)
    get_accelerator().manual_seed_all(seed)

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)


def write_to_file(msg):
    with open("debug_msg.txt", "a") as f:
        f.write(f"{msg}\n")


def compare_with_ddp(self, config, dtype):
    iteration = 5
    hidden_dim = 10
    RTOL = 1e-1
    ATOL = 1e-3

    enable_determinism(123)

    device = torch.device(get_accelerator().current_device_name())
    model = SimpleModel(hidden_dim)

    i = get_accelerator().current_device()
    ddp_model = DDP(deepcopy(model).to(device=device, dtype=torch.float), device_ids=[i], output_device=i)
    ddp_optimizer = torch.optim.Adam(ddp_model.parameters(), lr=config["optimizer"]["params"]["lr"])

    if config["zero_optimization"]["stage"] == 3:
        with deepspeed.zero.Init(config_dict_or_path=config):
            ds_model = SimpleModel(hidden_dim)
        with GatheredParameters(ds_model.parameters(), modifier_rank=0):
            for p1, p2 in zip(ds_model.parameters(), model.parameters()):
                p1.data.copy_(p2.data)
    else:
        ds_model = deepcopy(model)

    ds_engine, ds_optimizer, _, _ = deepspeed.initialize(config=config,
                                                         model=ds_model,
                                                         model_parameters=ds_model.parameters())

    train_batch_size = config["train_micro_batch_size_per_gpu"]

    xs = [torch.randn(train_batch_size, hidden_dim, device=device, dtype=dtype) for _ in range(iteration)]
    ys = [torch.randn_like(x) for x in xs]

    for x, y in zip(xs, ys):
        ddp_loss = ddp_model(x.float(), y.float())
        ds_loss = ds_engine(x, y)

        write_to_file(f"ddp_loss: {ddp_loss} ds_loss: {ds_loss}")
        assert torch.allclose(ddp_loss.to(dtype), ds_loss, rtol=RTOL, atol=ATOL)

        ddp_loss.backward()
        ds_engine.backward(ds_loss)

        ddp_optimizer.step()
        ds_optimizer.step()

        with GatheredParameters(ds_model.parameters()):
            for p1, p2 in zip(ddp_model.parameters(), ds_model.parameters()):
                assert torch.allclose(p1.to(dtype), p2, rtol=RTOL, atol=ATOL)


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
