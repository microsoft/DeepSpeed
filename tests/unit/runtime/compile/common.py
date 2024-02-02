# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import random
import os
import numpy as np
from copy import deepcopy
import queue

import torch
import torch.multiprocessing as mp

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero import GatheredParameters

from unit.simple_model import SimpleModel
from unit.common import DistributedTest, get_master_port
from typing import Callable, Any

TIMEOUT = 600


class EnableDeterminism:

    def __init__(self, seed: int):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        self.seed = seed + local_rank
        self.saved_random_state = None
        self.saved_np_random_state = None
        self.saved_cuda_launch_blocking = None
        self.saved_cublas_workspace_config = None
        self.saved_deterministic_algorithms = None

    def __enter__(self):
        self.saved_random_state = random.getstate()
        self.saved_np_random_state = np.random.get_state()
        self.saved_acc_rng_state = get_accelerator().get_rng_state()
        self.saved_cuda_launch_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING", "")
        self.saved_cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
        self.saved_deterministic_algorithms = torch.are_deterministic_algorithms_enabled()

        random.seed(self.seed)
        np.random.seed(self.seed)
        get_accelerator().manual_seed(self.seed)
        get_accelerator().manual_seed_all(self.seed)

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)

    def __exit__(self, type, value, traceback):
        random.setstate(self.saved_random_state)
        np.random.set_state(self.saved_np_random_state)
        get_accelerator().set_rng_state(self.saved_acc_rng_state)
        os.environ["CUDA_LAUNCH_BLOCKING"] = self.saved_cuda_launch_blocking
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.saved_cublas_workspace_config
        torch.use_deterministic_algorithms(self.saved_deterministic_algorithms)


def enable_determinism(seed: int):

    def decorator(func: Callable) -> Callable:

        def wrapper(*args: Any, **kwargs: Any):
            with EnableDeterminism(seed):
                return func(*args, **kwargs)

        return wrapper

    return decorator


@enable_determinism(123)
def compare_loss(self, config, dtype):
    iteration = 5
    hidden_dim = 10
    RTOL = 1e-1
    ATOL = 1e-3

    device = torch.device(get_accelerator().current_device_name())
    model = SimpleModel(hidden_dim)

    i = get_accelerator().current_device()
    baseline_model = deepcopy(model)
    baseline_config = deepcopy(config)
    baseline_config["zero_optimization"]["stage"] = 0
    baseline_config["zero_optimization"]["offload_optimizer"] = {}
    baseline_config["compile"]["enabled"] = False
    baseline_engine, baseline_optimizer, _, _ = deepspeed.initialize(config=baseline_config,
                                                                     model=baseline_model,
                                                                     model_parameters=baseline_model.parameters())

    if config["zero_optimization"]["stage"] == 3:
        with deepspeed.zero.Init(config_dict_or_path=config):
            target_model = SimpleModel(hidden_dim)
        with GatheredParameters(target_model.parameters(), modifier_rank=0):
            for p1, p2 in zip(target_model.parameters(), model.parameters()):
                p1.data.copy_(p2.data)
    else:
        target_model = deepcopy(model)

    target_engine, target_optimizer, _, _ = deepspeed.initialize(config=config,
                                                                 model=target_model,
                                                                 model_parameters=target_model.parameters())

    train_batch_size = config["train_micro_batch_size_per_gpu"]

    xs = [torch.randn(train_batch_size, hidden_dim, device=device, dtype=dtype) for _ in range(iteration)]
    ys = [torch.randn_like(x) for x in xs]

    for x, y in zip(xs, ys):
        baseline_loss = baseline_engine(x, y)
        target_loss = target_engine(x, y)

        assert torch.allclose(baseline_loss, target_loss, rtol=RTOL, atol=ATOL)

        baseline_engine.backward(baseline_loss)
        target_engine.backward(target_loss)

        baseline_optimizer.step()
        target_optimizer.step()

        with GatheredParameters(target_engine.parameters()):
            for p1, p2 in zip(baseline_engine.parameters(), target_engine.parameters()):
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
        result_queue = mp.Queue()
        procs = [
            mp.Process(target=self._dist_run_queue, args=(local_rank, num_procs, master_port, result_queue))
            for local_rank in range(num_procs)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        try:
            skip_msgs = [result_queue.get(timeout=TIMEOUT) for _ in range(num_procs)]
        except queue.Empty:
            pytest.exit("Test hanged, exiting", returncode=0)

        # If we skipped a test, propagate that to this process
        if any(skip_msgs):
            assert len(set(skip_msgs)) == 1, "Multiple different skip messages received"
            pytest.skip(skip_msgs[0])
