# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch


@pytest.fixture(params=[torch.float, torch.half], ids=["fp32", "fp16"])
def dtype(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["CG", "noCG"])
def enable_cuda_graph(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["Triton", "noTriton"])
def enable_triton(request):
    return request.param


@pytest.fixture
def query(model_w_task):
    model, task = model_w_task
    angle_bracket_mask_models = ["roberta", "camembert", "esm", "ibert", "luke", "mpnet", "yoso", "mpnet"]

    if task == "fill-mask":
        if any(map(lambda x: x in model, angle_bracket_mask_models)):
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
    elif task == "text2text-generation":
        return "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy"
    elif task == "translation" or task == "summarization":
        return "Hello, my dog is cute"
    else:
        NotImplementedError(f'query for task "{task}" is not implemented')


@pytest.fixture
def inf_kwargs(model_w_task):
    model, task = model_w_task
    if task == "text-generation":
        if model == "EleutherAI/gpt-j-6b":
            # This model on V100 is hitting memory problems that limit the number of output tokens
            return {"do_sample": False, "max_length": 12}
        return {"do_sample": False, "max_length": 20}
    else:
        return {}


def fill_mask_assert(x, y):
    return set(res["token_str"] for res in x) == set(res["token_str"] for res in y)


def question_answering_assert(x, y):
    return x["answer"] == y["answer"]


def text_classification_assert(x, y):
    return set(res["label"] for res in x) == set(res["label"] for res in y)


def token_classification_assert(x, y):
    return set(ent["word"] for ent in x) == set(ent["word"] for ent in y)


def text_generation_assert(x, y):
    return set(res["generated_text"] for res in x) == set(res["generated_text"] for res in y)


def text2text_generation_assert(x, y):
    return set(res["generated_text"] for res in x) == set(res["generated_text"] for res in y)


def translation_assert(x, y):
    return set(res["translation_text"] for res in x) == set(res["translation_text"] for res in y)


def summarization_assert(x, y):
    return set(res["summary_text"] for res in x) == set(res["summary_text"] for res in y)


@pytest.fixture
def assert_fn(model_w_task):
    model, task = model_w_task
    assert_fn_dict = {
        "fill-mask": fill_mask_assert,
        "question-answering": question_answering_assert,
        "text-classification": text_classification_assert,
        "token-classification": token_classification_assert,
        "text-generation": text_generation_assert,
        "text2text-generation": text2text_generation_assert,
        "translation": translation_assert,
        "summarization": summarization_assert
    }
    assert_fn = assert_fn_dict.get(task, None)
    if assert_fn is None:
        NotImplementedError(f'assert_fn for task "{task}" is not implemented')
    return assert_fn
