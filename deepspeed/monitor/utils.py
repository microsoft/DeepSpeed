# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team


def check_tb_availability():
    try:
        # torch.utils.tensorboard will fail if `tensorboard` is not available,
        # see their docs for more details: https://pytorch.org/docs/1.8.0/tensorboard.html
        import tensorboard  # noqa: F401 # type: ignore
    except ImportError:
        print('If you want to use tensorboard logging, please `pip install tensorboard`')
        raise


def check_wandb_availability():
    try:
        import wandb  # noqa: F401 # type: ignore
    except ImportError:
        print(
            'If you want to use wandb logging, please `pip install wandb` and follow the instructions at https://docs.wandb.ai/quickstart'
        )
        raise


def check_comet_availability():
    try:
        import comet_ml
    except ImportError:
        print(
            'If you want to use comet logging, please `pip install "comet_ml>=3.41.0"`'
        )
        raise