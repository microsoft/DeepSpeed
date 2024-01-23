# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

import torch.multiprocessing as mp
from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest, get_master_port

TIMEOUT = 600


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
