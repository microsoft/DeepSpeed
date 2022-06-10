import os
import torch
import pytest
import itertools
import deepspeed
from deepspeed.git_version_info import torch_info
from collections import defaultdict
from huggingface_hub import HfApi
from transformers import pipeline
from .common import distributed_test
from packaging import version as pkg_version

_bert_models = [
    "bert-base-cased",
    "bert-base-uncased",
    "bert-large-cased",
    "bert-large-uncased",
    "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased",
    "deepset/minilm-uncased-squad2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "dslim/bert-base-NER",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "distilbert-base-cased-distilled-squad",
]
_roberta_models = [
    "roberta-large",
    "roberta-base",
    "deepset/roberta-base-squad2",
    "j-hartmann/emotion-english-distilroberta-base",
    "Jean-Baptiste/roberta-large-ner-english",
]
_gpt_models = [
    "gpt2",
    "distilgpt2",
    "Norod78/hebrew-bad_wiki-gpt_neo-tiny",
    "EleutherAI/gpt-j-6B"
]
_all_models = HfApi().list_models()

test_models = set(_bert_models + _roberta_models + _gpt_models)
test_tasks = [
    "fill-mask",
    "question-answering",
    "text-classification",
    "token-classification",
    "text-generation",
]
pytest.all_models = {
    task: [m.modelId for m in _all_models if m.pipeline_tag == task]
    for task in test_tasks
}

_model_w_tasks = itertools.product(*[test_models, test_tasks])


def _valid_model_task(model_task):
    m, t = model_task
    return m in pytest.all_models[t]


pytest.models_w_tasks = list(filter(_valid_model_task, _model_w_tasks))
"""
These fixtures iterate all combinations of tasks and models, dtype, & cuda_graph
"""


@pytest.fixture(params=pytest.models_w_tasks)
def model_w_task(request):
    return request.param


@pytest.fixture(params=[torch.float, torch.half])
def dtype(request):
    return request.param


@pytest.fixture(params=[True, False])
def enable_cuda_graph(request):
    return request.param


"""
This fixture will validate the configuration
"""


@pytest.fixture()
def invalid_model_task_config(model_w_task, dtype, enable_cuda_graph):
    model, task = model_w_task
    if pkg_version.parse(torch.__version__) <= pkg_version.parse("1.2"):
        msg = "DS inference injection doesn't work well on older torch versions"
    elif model not in pytest.all_models[task]:
        msg = f"Not a valid model / task combination: {model} / {task}"
    elif enable_cuda_graph and (torch_info['cuda_version'] == "0.0"):
        msg = "CUDA not detected, cannot use CUDA Graph"
    elif enable_cuda_graph and pkg_version.parse(
            torch.__version__) < pkg_version.parse("1.10"):
        msg = "CUDA Graph is only available in torch versions >= 1.10"
    elif ('gpt-j-6B' in model) and (dtype == torch.float):
        msg = f"Not enough GPU memory to run {model} with dtype {dtype}"
    else:
        msg = ''
    return msg


"""
These fixtures can be used to customize the query, inference args, and assert
statement for each combination of model /task
"""


@pytest.fixture
def query(model_w_task):
    model, task = model_w_task
    if task == "fill-mask":
        if "roberta" in model:
            return "Hello I'm a <mask> model."
        else:
            return "Hell I'm a [MASK] model."
    elif task == "question-answering":
        return {
            "question": "What's my name?",
            "context": "My name is Clara and I live in Berkeley",
        }
    elif task == "text-classification":
        return "DeepSpeed is the greatest"
    elif task == "token-classification":
        return "My name is jean-baptiste and I live in montreal."
    elif task == "text-generation":
        return "DeepSpeed is the greatest"
    else:
        NotImplementedError(f'query for task "{task}" is not implemented')


@pytest.fixture
def inf_kwargs(model_w_task):
    model, task = model_w_task
    if task == "text-generation":
        return {"do_sample": False}
    else:
        return {}


@pytest.fixture
def assert_fn(model_w_task):
    model, task = model_w_task
    if task == "fill-mask":
        return lambda x, y: set(res["token_str"] for res in x) == set(
            res["token_str"] for res in y
        )
    elif task == "question-answering":
        return lambda x, y: x["answer"] == y["answer"]
    elif task == "text-classification":
        return lambda x, y: set(res["label"] for res in x) == set(
            res["label"] for res in y
        )
    elif task == "token-classification":
        return lambda x, y: set(ent["word"] for ent in x) == set(
            ent["word"] for ent in y
        )
    elif task == "text-generation":
        return lambda x, y: set(res["generated_text"] for res in x) == set(
            res["generated_text"] for res in y
        )
    else:
        NotImplementedError(f'assert_fn for task "{task}" is not implemented')


"""
Tests
"""


def test_model_task(model_w_task,
                    dtype,
                    enable_cuda_graph,
                    query,
                    inf_kwargs,
                    assert_fn,
                    invalid_model_task_config):
    if invalid_model_task_config:
        pytest.skip(invalid_model_task_config)

    model, task = model_w_task

    @distributed_test(world_size=[1])
    def _go():
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        pipe = pipeline(task, model=model, device=local_rank, framework="pt")
        bs_output = pipe(query, **inf_kwargs)

        pipe.model = deepspeed.init_inference(
            pipe.model,
            mp_size=1,
            dtype=dtype,
            replace_method="auto",
            replace_with_kernel_inject=True,
            enable_cuda_graph=enable_cuda_graph,
        )
        ds_output = pipe(query, **inf_kwargs)

        if task == 'text-generation':
            bs_output = pipe(query, **inf_kwargs)

        assert assert_fn(bs_output, ds_output)

    _go()
