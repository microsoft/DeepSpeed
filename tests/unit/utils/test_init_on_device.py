import torch
import pytest
from unit.simple_model import SimpleModel
from deepspeed import OnDevice
from packaging import version as pkg_version
from unit.common import DistributedTest

if not torch.cuda.is_available():
    pytest.skip("Only supported on CUDA environments", allow_module_level=True)


@pytest.mark.parametrize('device', ['meta', 'cuda:0'])
class TestOnDevice(DistributedTest):
    world_size = 1

    def test_on_device(self, device):
        if device == "meta" and pkg_version.parse(
                torch.__version__) < pkg_version.parse("1.10"):
            pytest.skip("meta tensors only became stable after torch 1.10")

        with OnDevice(dtype=torch.half, device=device):
            model = SimpleModel(4)

        for p in model.parameters():
            assert p.device == torch.device(device)
            assert p.dtype == torch.half
