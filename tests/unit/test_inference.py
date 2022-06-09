import os
import torch
import pytest
import deepspeed
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
    "distilgpt2",
    "Norod78/hebrew-bad_wiki-gpt_neo-tiny",
    "EleutherAI/gpt-j-6B"
]
_all_models = HfApi().list_models()

pytest.test_models = set(_bert_models + _roberta_models + _gpt_models)
pytest.test_tasks = [
    "fill-mask",
    "question-answering",
    "text-classification",
    "token-classification",
    "text-generation",
]
pytest.all_models = {
    task: [m.modelId for m in _all_models if m.pipeline_tag == task]
    for task in pytest.test_tasks
}
"""
These fixtures will iterate over all combinations of tasks and models (and
dtype), only returning valid combinations in valid_model_task
"""


@pytest.fixture(params=pytest.test_tasks)
def task(request):
    return request.param


@pytest.fixture(params=pytest.test_models)
def model(request):
    return request.param


@pytest.fixture(params=[torch.float, torch.half])
def dtype(request):
    return request.param


@pytest.fixture(params=[True, False])
def enable_cuda_graph(request):
    return request.param


@pytest.fixture()
def valid_model_task(model, task, dtype):
    if model in pytest.all_models[task]:
        model_task = (model, task)
    else:
        pytest.skip(f"Not a valid model / task combination: {model} / {task}")
    ''' model specific checks '''
    if ('gpt-j-6B' in model) and (dtype == torch.float):
        pytest.skip(f"Not enough GPU memory to run {model} with dtype {dtype}")

    return model_task


"""
These fixtures can be used to customize the query, inference args, and assert
statement for each combination of model /task
"""


@pytest.fixture
def query(task, model):
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
def inf_kwargs(task, model):
    if task == "text-generation":
        return {"do_sample": False}
    else:
        return {}


@pytest.fixture
def assert_fn(task, model):
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


def test_model_task(valid_model_task,
                    dtype,
                    enable_cuda_graph,
                    query,
                    inf_kwargs,
                    assert_fn):
    model, task = valid_model_task

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


def test_gpt2_inject(dtype):
    if pkg_version.parse(torch.__version__) <= pkg_version.parse("1.2"):
        pytest.skip("DS inference injection doesn't work well on older torch versions")

    @distributed_test(world_size=[1])
    def _go():
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        generator = pipeline("text-generation",
                             model="gpt2",
                             device=local_rank,
                             framework="pt")

        generator.model = deepspeed.init_inference(
            generator.model,
            mp_size=world_size,
            dtype=dtype,
            replace_method="auto",
            replace_with_kernel_inject=True,
        )

        prompt = "DeepSpeed is"
        string_1 = generator(prompt, do_sample=False, max_length=128)
        string_2 = generator(prompt, do_sample=False, max_length=128)
        assert string_1 == string_2

    _go()
