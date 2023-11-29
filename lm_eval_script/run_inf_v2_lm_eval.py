import lm_eval
import lm_eval.models
import lm_eval.tasks
import lm_eval.evaluator

from deepspeed.accelerator import get_accelerator
from torch import nn
import os
import time
import torch
import itertools
import deepspeed

from mii.api import _parse_kwargs_to_model_config

# TODO (lekurile): Update to OPT for Inf V2
task = "lambada_standard"
model_family = "opt"
#model_name = "EleutherAI/gpt-neo-2.7B"
model_name = "facebook/opt-1.3b"

# The bootstrap_stderr function in lm_eval.metrics uses a
# multiprocessing Pool to increase performance. Since we use a Pool for
# our distributed tests and cannot nest Pools, we must redefine and
# patch this function with a version that does not use Pool.
def no_pool_bootstrap_stderr(f, xs, iters):
    from lm_eval.metrics import _bootstrap_internal
    from lm_eval.metrics import sample_stddev
    res = []
    chunk_size = min(1000, iters)
    for i in range(iters // chunk_size):
        res.extend(_bootstrap_internal(f, chunk_size)((i, xs)))
    return sample_stddev(res)

lm_eval.metrics.bootstrap_stderr = no_pool_bootstrap_stderr

local_rank = os.getenv("LOCAL_RANK", "0")
device = torch.device(get_accelerator().device_name(local_rank))
dtype = torch.float
task_dict = lm_eval.tasks.get_task_dict([task])

if 'gpt-j-6b' in model_name:
    dtype = torch.half
    lm = lm_eval.models.get_model(model_family).create_from_arg_string(f"pretrained={model_name}",
                                                                        {"device": "cpu"})
    setattr(lm, model_family, getattr(lm, model_family).half().to(device))
    lm._device = device
else:
    lm = lm_eval.models.get_model(model_family).create_from_arg_string(
        f"pretrained={model_name}", {"device": get_accelerator().device_name()})

get_accelerator().synchronize()
start = time.time()
#bs_output = lm_eval.evaluator.evaluate(lm=lm, task_dict=task_dict)
get_accelerator().synchronize()
bs_time = time.time() - start

getattr(lm, model_family).to("cpu")


#ds_model = deepspeed.init_inference(
#    getattr(lm, model_family),
#    mp_size=1,
#    dtype=dtype,
#    replace_with_kernel_inject=True,
#    enable_cuda_graph=False,
#)
# TODO (lekurile): Test v2 engine build code
from deepspeed.inference import build_hf_engine

#init_distributed(model_config)
#provider = model_config.provider
#if provider == ModelProvider.HUGGING_FACE:

import pdb; pdb.set_trace()

model_config, remaining_kwargs = _parse_kwargs_to_model_config(
    model_name_or_path=model_name
)

import pdb; pdb.set_trace()

ds_model = build_hf_engine(
    path=model_name,
    engine_config=model_config
)

import pdb; pdb.set_trace()
#ds_model = build_hf_engine(
#    path=getattr(lm, model_family),
#    engine_config=
#)

# TODO (lekurile): =========================
#check_injection(ds_model)
setattr(lm, model_family, ds_model)
get_accelerator().synchronize()
start = time.time()
ds_output = lm_eval.evaluator.evaluate(lm=lm, task_dict=task_dict)
get_accelerator().synchronize()
ds_time = time.time() - start

ppl_diff = abs(bs_output["results"][task]["ppl"] - ds_output["results"][task]["ppl"])
#assert ds_time <= bs_time
assert ppl_diff < 0.01



#===========================================================================
#    PYTEST
#===========================================================================

#@pytest.mark.nightly
#@pytest.mark.parametrize(
#    "model_family, model_name",
#    (
#        ["gpt2", "EleutherAI/gpt-neo-2.7B"],
#        #["gpt2", "EleutherAI/gpt-j-6b"], # Causing OOM for this test
#        ["gpt2", "gpt2-xl"],
#    ),
#)
#@pytest.mark.parametrize("task", ["lambada_standard"])
#class TestLMCorrectnessInfV2(DistributedTest):
#    world_size = 1
#    exec_timeout = 1200  # Give these tests longer to complete
#
#    def test(self, model_family, model_name, task):
#        # imports here to avoid import errors when pytest collects tests
#        import lm_eval
#        import lm_eval.models
#        import lm_eval.tasks
#        import lm_eval.evaluator
#
#        # The bootstrap_stderr function in lm_eval.metrics uses a
#        # multiprocessing Pool to increase performance. Since we use a Pool for
#        # our distributed tests and cannot nest Pools, we must redefine and
#        # patch this function with a version that does not use Pool.
#        def no_pool_bootstrap_stderr(f, xs, iters):
#            from lm_eval.metrics import _bootstrap_internal
#            from lm_eval.metrics import sample_stddev
#            res = []
#            chunk_size = min(1000, iters)
#            for i in range(iters // chunk_size):
#                res.extend(_bootstrap_internal(f, chunk_size)((i, xs)))
#            return sample_stddev(res)
#
#        lm_eval.metrics.bootstrap_stderr = no_pool_bootstrap_stderr
#
#        local_rank = os.getenv("LOCAL_RANK", "0")
#        device = torch.device(get_accelerator().device_name(local_rank))
#        dtype = torch.float
#        task_dict = lm_eval.tasks.get_task_dict([task])
#
#        if 'gpt-j-6b' in model_name:
#            dtype = torch.half
#            lm = lm_eval.models.get_model(model_family).create_from_arg_string(f"pretrained={model_name}",
#                                                                               {"device": "cpu"})
#            setattr(lm, model_family, getattr(lm, model_family).half().to(device))
#            lm._device = device
#        else:
#            lm = lm_eval.models.get_model(model_family).create_from_arg_string(
#                f"pretrained={model_name}", {"device": get_accelerator().device_name()})
#
#        get_accelerator().synchronize()
#        start = time.time()
#        bs_output = lm_eval.evaluator.evaluate(lm=lm, task_dict=task_dict)
#        get_accelerator().synchronize()
#        bs_time = time.time() - start
#
#        getattr(lm, model_family).to("cpu")
#        ds_model = deepspeed.init_inference(
#            getattr(lm, model_family),
#            mp_size=1,
#            dtype=dtype,
#            replace_with_kernel_inject=True,
#            enable_cuda_graph=False,
#        )
#        # TODO (lekurile): Test v2 engine build code
#        from deepspeed.inference import build_hf_engine
#
#        #init_distributed(model_config)
#        #provider = model_config.provider
#        #if provider == ModelProvider.HUGGING_FACE:
#        #    inference_engine = build_hf_engine(
#        #        path=model_config.model_name_or_path,
#        #        engine_config=model_config.inference_engine_config)
#
#        #import pdb; pdb.set_trace()
#        pytest.set_trace()
#        #ds_model = build_hf_engine(
#        #    path=getattr(lm, model_family),
#        #    engine_config=
#        #)
#
#        # TODO (lekurile): =========================
#        check_injection(ds_model)
#        setattr(lm, model_family, ds_model)
#        get_accelerator().synchronize()
#        start = time.time()
#        ds_output = lm_eval.evaluator.evaluate(lm=lm, task_dict=task_dict)
#        get_accelerator().synchronize()
#        ds_time = time.time() - start
#
#        ppl_diff = abs(bs_output["results"][task]["ppl"] - ds_output["results"][task]["ppl"])
#        #assert ds_time <= bs_time
#        assert ppl_diff < 0.01
