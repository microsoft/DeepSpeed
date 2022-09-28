import os
import time
import pytest
import torch
import deepspeed
from transformers import pipeline
from unit.common import DistributedTest


@pytest.mark.inference
@pytest.mark.parametrize("cuda_graphs", [True, False])
class TestModelProfiling(DistributedTest):
    world_size = 1

    def test(self, cuda_graphs):
        model = "bert-base-cased"
        task = "fill-mask"
        query = "I am a [MASK] model"
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        dtype = torch.float16

        pipe = pipeline(task, model, framework="pt", device=local_rank)
        pipe.model = deepspeed.init_inference(pipe.model,
                                              dtype=dtype,
                                              mp_size=world_size,
                                              replace_with_kernel_inject=True,
                                              replace_method="auto",
                                              enable_cuda_graph=cuda_graphs)
        pipe.model.profile_model_time()

        e2e_times = []
        model_times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()

            r = pipe(query)

            torch.cuda.synchronize()
            end = time.perf_counter_ns()

            e2e_times.append((end - start) / 1e6)  # convert ns to ms
            model_times.extend(pipe.model.model_times())

        for e2e_t, model_t in zip(e2e_times, model_times):
            assert e2e_t >= model_t
