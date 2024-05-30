import argparse
import logging
import os
import time
from functools import partial
from typing import Optional

import torch
from pytorch_lightning.callbacks import Callback
from transformers.models.llama.tokenization_llama import LlamaTokenizer

from data_utils import prepare_dataloader
from llama_pl_model import LLama
logger = logging.getLogger(__name__)


def tokenize_batch(batch, tokenizer: Optional[LlamaTokenizer] = None, max_length: int = 2048):
    texts = [sample['text'] for sample in batch]
    data = tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
    data['labels'] = data['input_ids'].clone()
    return data


def get_dataloader(args):
    retry = 0
    while retry < 50:
        retry += 1
        try:
            tokenizer = LlamaTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')
            break
        except Exception as e:
            print(f'Fail to load tokenizer, retry! {e}')
    # follows fast chat: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py#L257
    tokenizer.pad_token = tokenizer.unk_token
    from datasets import load_dataset
    dataset = load_dataset(args.dataset)
    train_ds = dataset['train']
    dataloader = prepare_dataloader(train_ds,
                                    batch_size=args.per_device_train_batch_size,
                                    shuffle=True,
                                    drop_last=True,
                                    collate_fn=partial(tokenize_batch, tokenizer=tokenizer, max_length=args.max_length))
    return dataloader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_of_model_param', default='13b', type=str)
    parser.add_argument('--gpu_nums', default=1, type=int)
    parser.add_argument('--num_nodes', default=1, type=int)
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        default='togethercomputer/RedPajama-Data-1T-Sample',
                        help='Data set path')
    parser.add_argument('--ds_config', default='ds_config.json', type=str)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--logging_steps', default=10, type=int)
    parser.add_argument('--gradient_checkpointing', default='true', type=str)
    parser.add_argument('--strategy', default='ds', type=str)
    parser.add_argument('--fsdp_sharding_strategy', default='HYBRID_SHARD', type=str)

    parser.add_argument('--flash_attn', default='false', type=str)
    parser.add_argument('--core_checkpointing', default='false', type=str)
    parser.add_argument('--skip_checkpointing_layer', default='false', type=str)

    parser.add_argument('--public_cloud', default='false', type=str)

    parser.add_argument('--learning_rate', type=float, default=3e-6)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--adam-eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)

    parser.add_argument('--num_train_epochs', default=1, type=int)
    parser.add_argument('--val_check_interval', default=2000, type=int)
    parser.add_argument('--per_device_train_batch_size', default=1, type=int)
    parser.add_argument('--per_device_val_batch_size', default=1, type=int)

    parser.add_argument('--max_length', default=1024, type=int)

    parser.add_argument("--output_dir", default='./output', type=str)

    args = parser.parse_args()
    return args, parser


class CudaMemUsageCallback(Callback):

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 10 == 0:
            allocated_memory = torch.cuda.memory_allocated() / 1024 ** 3
            max_allocated_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
            memory_reserved = torch.cuda.memory_reserved() / 1024 ** 3
            max_memory_reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
            logger.info(
                f'=== Cuda Mem Usage: batch_idx={batch_idx}, mem={allocated_memory} GB, max_mem={max_allocated_memory} GB, mem_reserved={memory_reserved} GB, max_mem_reserved={max_memory_reserved} GB')


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    from pytorch_lightning import seed_everything

    seed = 42
    seed_everything(seed)
    try:
        from deepspeed.runtime.zero.utils import patch_pl

        patch_pl()
    except:
        print(f'======= Fail to patch pl ==========')
        pass
    args, parser = get_args()

    if args.flash_attn == 'true':
        from patch_utils import patch_modeling_llama_flash_attn

        patch_modeling_llama_flash_attn()

    if args.core_checkpointing == 'true':
        from patch_utils import patch_modeling_llama_core_attn_checkpointing

        patch_modeling_llama_core_attn_checkpointing()

    if args.skip_checkpointing_layer == 'true':
        from patch_utils import patch_modeling_llama_checkpointing

        patch_modeling_llama_checkpointing()

    accumulate_grad_batches = args.gradient_accumulation_steps
    if args.strategy == 'fsdp':
        accumulate_grad_batches = 1
        from torch.distributed.fsdp import ShardingStrategy
        import pytorch_lightning as pl

        sharding_strategy = ShardingStrategy.HYBRID_SHARD
        if args.fsdp_sharding_strategy == '_HYBRID_SHARD_ZERO2':
            sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2


        @property
        def _process_group(self):
            return self._process_group


        pl.strategies.DDPFullyShardedNativeStrategy.process_group = _process_group

        strategy = pl.strategies.DDPFullyShardedNativeStrategy(
            sharding_strategy=sharding_strategy,
            # auto_wrap_policy=size_based_auto_wrap_policy,
            # activation_checkpointing=LlamaDecoderLayer
        )
        strategy.num_nodes = args.num_nodes
        print('----- strategy ----- ', strategy)
    else:
        import pytorch_lightning as pl
        from pytorch_lightning.strategies import DeepSpeedStrategy

        strategy = DeepSpeedStrategy(config=args.ds_config)
        print('----- strategy ----- ', strategy.config)

    from datetime import datetime

    now = datetime.now()
    # 添加模型保存支持oss与pangu

    callbacks = [CudaMemUsageCallback()]

    # 模型保存的callback
    from pytorch_lightning.callbacks import ModelCheckpoint

    # 自定义checkpoint策略
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        verbose=True,
        save_last=True,
    )

    trainer = pl.Trainer(max_epochs=args.num_train_epochs,
                         log_every_n_steps=args.logging_steps,
                         accelerator='gpu',
                         devices=args.gpu_nums,
                         num_nodes=args.num_nodes,
                         strategy=strategy,
                         # val_check_interval=args.val_check_interval,
                         callbacks=callbacks,
                         # profiler=profiler,
                         # logger=ant_logger,
                         # plugins=plugins,
                         enable_progress_bar=False,  # 关闭官方进度条
                         accumulate_grad_batches=accumulate_grad_batches,
                         precision='bf16',
                         #  max_steps=40,
                         )
    print(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()) + " config dataset and model ")

    train_dataloader = get_dataloader(args)
    len_dataset = len(train_dataloader)
    parser.add_argument('--len_dataset', type=int, default=len_dataset)
    parsed_args = parser.parse_args()
    model = LLama(parsed_args)
    print(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()) + " config dataset and model done ")
    print("PYTORCH_CUDA_ALLOC_CONF:", os.getenv("PYTORCH_CUDA_ALLOC_CONF"))
    logging.getLogger("pytorch_lighting").setLevel(logging.INFO)
    logging.getLogger("DeepSpeed").setLevel(logging.INFO)
    trainer.fit(model=model, train_dataloaders=train_dataloader)
