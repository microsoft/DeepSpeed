import os
import torch
import deepspeed
from transformers import pipeline
from .common import distributed_test


def test_gpt2_inject():
    @distributed_test(world_size=[1])
    def _go():
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        generator = pipeline("text-generation", model="gpt2", device=local_rank)

        generator.model = deepspeed.init_inference(
            generator.model,
            mp_size=world_size,
            dtype=torch.float,
            replace_method="auto",
            replace_with_kernel_inject=True,
        )

        prompt = "DeepSpeed is"
        string_1 = generator(prompt, do_sample=False, max_length=128)
        string_2 = generator(prompt, do_sample=False, max_length=128)
        assert string_1 == string_2

    _go()
