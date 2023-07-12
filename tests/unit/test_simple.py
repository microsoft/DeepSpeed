import pytest
import deepspeed
import torch
from unit.common import DistributedTest


@pytest.fixture()
def invalid_test(model_w_task, dtype, enable_cuda_graph, enable_triton):
    model, task = model_w_task
    msg = ""
    if "gpt-j-6b" in model:
        if dtype != torch.half:
            msg = f"Not enough GPU memory to run {model} with dtype {dtype}"
        elif enable_cuda_graph:
            msg = f"Not enough GPU memory to run {model} with CUDA Graph enabled"
    elif "gpt-neox-20b" in model:  # TODO: remove this when neox issues resolved
        msg = "Skipping gpt-neox-20b for now"
    elif ("gpt-neox-20b" in model) and (dtype != torch.half):
        msg = f"Not enough GPU memory to run {model} with dtype {dtype}"
    elif ("bloom" in model) and (dtype != torch.half):
        msg = f"Bloom models only support half precision, cannot use dtype {dtype}"
    elif ("bert" not in model.lower()) and enable_cuda_graph:
        msg = "Non bert/roberta models do no support CUDA Graph"
    elif enable_triton and not (dtype in [torch.half]):
        msg = "Triton is for fp16"
    elif enable_triton and not deepspeed.HAS_TRITON:
        msg = "triton needs to be installed for the test"
    elif ("bert" not in model.lower()) and enable_triton:
        msg = "Triton kernels do not support Non bert/roberta models yet"
    return msg


@pytest.fixture(
    params=[
        ("EleutherAI/gpt-neo-1.3B", "text-generation"),
        ("EleutherAI/gpt-neox-20b", "text-generation"),
        ("bigscience/bloom-3b", "text-generation"),
        ("EleutherAI/gpt-j-6b", "text-generation"),
    ],
    ids=["gpt-neo", "gpt-neox", "bloom", "gpt-j"],
)
def model_w_task(request):
    return request.param


@pytest.fixture(params=[torch.float16, torch.float32], ids=["fp16", "fp32"])
def dtype(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["CG", "noCG"])
def enable_cuda_graph(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["Triton", "noTriton"])
def enable_triton(request):
    return request.param


class TestSimple(DistributedTest):
    world_size = 1

    def test_all(self, model_w_task, dtype, enable_cuda_graph, enable_triton, invalid_test):
        if invalid_test:
            pytest.skip(invalid_test)
        pass

    def test_some(self, model_w_task, dtype, invalid_test):
        if invalid_test:
            pytest.skip(invalid_test)
        pass
