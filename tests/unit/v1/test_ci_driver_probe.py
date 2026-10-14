# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Diagnostic probe: surface host GPU driver details in CI logs.

The modal-torch-latest Sandbox report step does not print the host driver
version, which hid a fleet-wide driver update (610.57.04) that intermittently
poisons CUDA contexts. This probe makes the driver version visible in test
logs so future infra regressions are attributable at a glance.
"""

import subprocess

import pytest


def _driver_version() -> str:
    result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
                            capture_output=True,
                            text=True,
                            check=True)
    return result.stdout.strip()


def test_report_host_driver_version():
    print(f"DS_CI_DRIVER_PROBE: {_driver_version()}")


def test_cuda_context_stays_healthy_across_allocations():
    import torch

    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("CUDA not available")
    # Repeated small allocations plus a kernel launch catch a wedged context
    # early instead of letting it poison every later test in the suite.
    for _ in range(8):
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        assert y.isfinite().all().item()
    torch.cuda.synchronize()  #ignore-cuda
