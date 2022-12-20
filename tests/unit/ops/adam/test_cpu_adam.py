import torch
import numpy as np
import pytest
from cpuinfo import get_cpu_info

import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.op_builder import CPUAdamBuilder
from unit.common import DistributedTest

if not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
    pytest.skip("cpu-adam is not compatible", allow_module_level=True)

pytest.cpu_vendor = get_cpu_info()["vendor_id_raw"].lower()


def check_equal(first, second, atol=1e-2, verbose=False):
    x = first.detach().numpy()
    y = second.detach().numpy()
    print("ATOL", atol)
    if verbose:
        print("x = {}".format(x.flatten()))
        print("y = {}".format(y.flatten()))
        print('-' * 80)
    np.testing.assert_allclose(x, y, err_msg="param-update mismatch!", atol=atol)


def _create_param(model_size, dtype, device, rng_state):
    torch.set_rng_state(rng_state)
    return torch.nn.Parameter(torch.randn(model_size, device=device).to(dtype))


def _compare_optimizers(model_size, param1, optimizer1, param2, optimizer2):
    for i in range(10):
        rng_state = torch.get_rng_state()
        param1.grad = torch.randn(model_size, device=param1.device).to(param1.dtype)
        torch.set_rng_state(rng_state)
        param2.grad = torch.randn(model_size, device=param2.device).to(param2.dtype)

        optimizer1.step()
        optimizer2.step()

    tolerance = param1.float().norm().detach().numpy() * 1e-2
    check_equal(param1.float().norm(),
                param2.float().cpu().norm(),
                atol=tolerance,
                verbose=True)


@pytest.mark.parametrize('dtype', [torch.half, torch.float], ids=["fp16", "fp32"])
@pytest.mark.parametrize('model_size',
                         [
                             (64),
                             (22),
                             #(55),
                             (128),
                             (1024),
                             (1048576),
                         ]) # yapf: disable
class TestCPUAdam(DistributedTest):
    world_size = 1
    requires_cuda_env = False
    if not torch.cuda.is_available():
        init_distributed = False
        set_dist_env = False

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="only supported in CUDA environments.")
    def test_fused_adam_equal(self, dtype, model_size):
        if ("amd" in pytest.cpu_vendor) and (dtype == torch.half):
            pytest.skip("cpu-adam with half precision not supported on AMD CPUs")

        from deepspeed.ops.adam import DeepSpeedCPUAdam

        rng_state = torch.get_rng_state()
        cpu_param = _create_param(model_size, dtype, 'cpu', rng_state)
        cpu_optimizer = DeepSpeedCPUAdam([cpu_param])

        ref_param = _create_param(model_size, dtype, 'cuda', rng_state)
        ref_optimizer = FusedAdam([ref_param])

        _compare_optimizers(model_size=model_size,
                            param1=cpu_param,
                            optimizer1=cpu_optimizer,
                            param2=ref_param,
                            optimizer2=ref_optimizer)

    def test_torch_adamw_equal(self, dtype, model_size):
        if torch.cuda.is_available():
            if ("amd" in pytest.cpu_vendor) and (dtype == torch.half):
                pytest.skip("cpu-adam with half precision not supported on AMD CPUs")
            ref_param_device = 'cuda'
        else:
            if dtype == torch.half:
                pytest.skip(
                    "torch.optim.AdamW with half precision only supported in CUDA environments."
                )
            ref_param_device = 'cpu'

            from deepspeed.ops.adam import DeepSpeedCPUAdam

            rng_state = torch.get_rng_state()
            cpu_param = _create_param(model_size, dtype, 'cpu', rng_state)
            cpu_optimizer = DeepSpeedCPUAdam([cpu_param])

            ref_param = _create_param(model_size, dtype, ref_param_device, rng_state)
            ref_optimizer = torch.optim.AdamW([ref_param])

            _compare_optimizers(model_size=model_size,
                                param1=cpu_param,
                                optimizer1=cpu_optimizer,
                                param2=ref_param,
                                optimizer2=ref_optimizer)


class TestCPUAdamGPUError(DistributedTest):
    def test_cpu_adam_gpu_error(self):
        model_size = 64
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        device = 'cuda:0'
        param = torch.nn.Parameter(torch.randn(model_size, device=device))
        optimizer = DeepSpeedCPUAdam([param])

        param.grad = torch.randn(model_size, device=device)
        with pytest.raises(AssertionError):
            optimizer.step()
