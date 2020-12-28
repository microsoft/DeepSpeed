import os
import sys
import time
import logging
import numpy as np
import random
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from turing.logger import Logger
from turing.utils import get_sample_writer
from turing.models import BertMultiTask
from turing.dataset import PreTrainingDataset, PretrainBatch, PretrainDataType
from turing.sources import PretrainingDataCreator, WikiPretrainingDataCreator, TokenInstance
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear, warmup_linear_decay_exp, warmup_exp_decay_exp, warmup_exp_decay_poly
from utils import get_argument_parser, is_time_to_exit

from bing_bert_dataset_provider import BingBertDatasetProvider
from nvidia_bert_dataset_provider import NvidiaBertDatasetProvider

import deepspeed

global_step = 0
global_data_samples = 0
last_global_step_from_restore = 0


def checkpoint_model(PATH, ckpt_id, model, epoch, last_global_step,
                     last_global_data_samples, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        'epoch': epoch,
        'last_global_step': last_global_step,
        'last_global_data_samples': last_global_data_samples
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.network.save_checkpoint(PATH, ckpt_id,
                                            checkpoint_state_dict)
    status_msg = 'checkpointing: PATH={}, ckpt_id={}'.format(PATH, ckpt_id)
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    return


def load_training_checkpoint(args, model, PATH, ckpt_id):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    logger = args.logger
    _, checkpoint_state_dict = model.network.load_checkpoint(PATH, ckpt_id)
    epoch = checkpoint_state_dict['epoch']
    last_global_step = checkpoint_state_dict['last_global_step']
    last_global_data_samples = checkpoint_state_dict[
        'last_global_data_samples']
    del checkpoint_state_dict
    return (epoch, last_global_step, last_global_data_samples)


def get_dataloader(args, dataset: Dataset, eval_set=False):
    if args.local_rank == -1:
        train_sampler = RandomSampler(dataset)
    else:
        train_sampler = DistributedSampler(dataset)
    return (x for x in
            DataLoader(dataset,
                       batch_size=args.train_micro_batch_size_per_gpu //
                       2 if eval_set else args.train_micro_batch_size_per_gpu,
                       sampler=train_sampler,
                       num_workers=args.config['training']['num_workers']))


def pretrain_validation(args, index, model):
    if args.validation_data_path_prefix is None:
        return

    config = args.config
    logger = args.logger

    logger.info(
        f"Validation micro batch size: {args.train_micro_batch_size_per_gpu}")

    model.eval()
    dataset = PreTrainingDataset(
        args.tokenizer,
        os.path.join(args.validation_data_path_prefix,
                     config['validation']['path']), args.logger,
        args.max_seq_length, index, PretrainDataType.VALIDATION,
        args.max_predictions_per_seq)
    data_batches = get_dataloader(args, dataset, eval_set=True)
    eval_loss = 0
    nb_eval_steps = 0
    for batch in tqdm(data_batches):
        batch = tuple(t.to(args.device) for t in batch)
        tmp_eval_loss = model.network(batch, log=False)
        dist.reduce(tmp_eval_loss, 0)
        # Reduce to get the loss from all the GPU's
        tmp_eval_loss = tmp_eval_loss / dist.get_world_size()
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    logger.info(f"Validation Loss for epoch {index + 1} is: {eval_loss}")
    if (not args.no_cuda
            and dist.get_rank() == 0) or (args.no_cuda
                                          and args.local_rank == -1):
        args.summary_writer.add_scalar(f'Validation/Loss', eval_loss,
                                       index + 1)
    return


def master_process(args):
    return (not args.no_cuda
            and dist.get_rank() == 0) or (args.no_cuda
                                          and args.local_rank == -1)


from deepspeed.utils.logging import logger


def train(args,
          index,
          model,
          optimizer,
          pretrain_dataset_provider,
          finetune=False):
    global global_step
    global global_data_samples
    global last_global_step_from_restore

    dataset_iterator, total_length = pretrain_dataset_provider.get_shard(index)
    current_data_sample_count = global_data_samples

    config = args.config
    logger = args.logger
    logger.info(
        f'worker-{dist.get_rank()}: begin epoch {index+1} current_sample_count {current_data_sample_count} shard_length {total_length} global_data_samples {global_data_samples}'
    )

    pretrain_dataset_provider.prefetch_shard(index + 1)

    model.train()

    epoch_step = 0
    rounds = 20
    all_step_time = 0.0
    step_counts = 0

    for _, batch_index in enumerate(tqdm(dataset_iterator, smoothing=1)):
        try:
            step_start = time.time()
            batch = pretrain_dataset_provider.get_batch(batch_index)
            batch = tuple(t.to(args.device) for t in batch)  # Move to GPU

            # Calculate forward pass
            loss = model.network(batch)
            unscaled_loss = loss.item()
            current_data_sample_count += (args.train_micro_batch_size_per_gpu *
                                          dist.get_world_size())

            # Prefetch training data
            pretrain_dataset_provider.prefetch_batch()

            model.network.backward(loss)

            loss = None

            if model.network.is_gradient_accumulation_boundary():
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = update_learning_rate(
                        args, config, global_step, optimizer)

                report_step_metrics(args, lr_this_step, unscaled_loss,
                                    global_step, current_data_sample_count)

                model.network.step()

                report_lamb_coefficients(args, optimizer)
                global_step += 1
                epoch_step += 1
            else:
                # Call DeepSpeed engine step on micro steps
                model.network.step()

        except StopIteration:
            continue

        current_global_step = global_step - last_global_step_from_restore
        if is_time_to_exit(args=args,
                           epoch_steps=epoch_step,
                           global_steps=current_global_step):
            print(
                f'Warning: Early epoch termination due to max steps limit, epoch step ={epoch_step}, global step = {current_global_step}, epoch = {index+1}'
            )
            break
        step_time = time.time() - step_start
        all_step_time += step_time
        if global_step % rounds == 0 and global_step != 0 and model.network.is_gradient_accumulation_boundary(
        ) and dist.get_rank() == 0:
            one_step_bs = args.train_micro_batch_size_per_gpu * args.gradient_accumulation_steps * dist.get_world_size(
            ) * rounds
            print(' At step {}, the throughput is {:2f} Samples/s'.format(
                global_step * args.gradient_accumulation_steps,
                one_step_bs / all_step_time),
                  flush=True)
            all_step_time = 0.0

    pretrain_dataset_provider.release_shard(index)

    global_data_samples = current_data_sample_count

    # Run Validation Loss
    if not finetune and args.max_seq_length == 512:
        pretrain_validation(args, index, model)


def update_learning_rate(args, config, current_global_step, optimizer):
    global last_global_step_from_restore

    global_step_for_lr = current_global_step - last_global_step_from_restore

    if args.lr_schedule == "EE":
        #print(f'LR Schedule is {args.lr_schedule} EE')
        lr_this_step = config["training"][
            "learning_rate"] * warmup_exp_decay_exp(
                global_step_for_lr, config["training"]["decay_rate"],
                config["training"]["decay_step"],
                config["training"]["total_training_steps"],
                config["training"]["warmup_proportion"])
    elif args.lr_schedule == "EP":
        print(f'LR Schedule is {args.lr_schedule} EP')
        lr_this_step = config["training"][
            "learning_rate"] * warmup_exp_decay_poly(
                global_step_for_lr, config["training"]["total_training_steps"],
                config["training"]["warmup_proportion"])
    else:
        lr_this_step = config["training"][
            "learning_rate"] * warmup_linear_decay_exp(
                global_step_for_lr, config["training"]["decay_rate"],
                config["training"]["decay_step"],
                config["training"]["total_training_steps"],
                config["training"]["warmup_proportion"])
    lr_this_step += args.lr_offset

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step

    return lr_this_step


def report_step_metrics(args, lr, loss, step, data_sample_count):
    ##### Record the LR against global_step on tensorboard #####
    if (not args.no_cuda
            and dist.get_rank() == 0) or (args.no_cuda
                                          and args.local_rank == -1):
        args.summary_writer.add_scalar(f'Train/lr', lr, step)

        args.summary_writer.add_scalar(f'Train/Samples/train_loss', loss,
                                       data_sample_count)

        args.summary_writer.add_scalar(f'Train/Samples/lr', lr,
                                       data_sample_count)
    ##### Recording  done. #####

    if (step + 1) % args.print_steps == 0 and master_process(args):
        print('bing_bert_progress: step={}, loss={}, lr={}, sample_count={}'.
              format(step + 1, loss, lr, data_sample_count))


def report_lamb_coefficients(args, optimizer):
    if master_process(args):
        if (args.fp16 and args.use_lamb):
            #print("Lamb Coeffs", optimizer.optimizer.get_lamb_coeffs())
            lamb_coeffs = optimizer.optimizer.get_lamb_coeffs()
            lamb_coeffs = np.array(lamb_coeffs)
            if lamb_coeffs.size > 0:
                args.summary_writer.add_histogram(f'Train/lamb_coeffs',
                                                  lamb_coeffs, global_step)


def get_arguments():
    parser = get_argument_parser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    # no cuda mode is not supported
    args.no_cuda = False

    return args


def construct_arguments():
    args = get_arguments()

    # Prepare Logger
    logger = Logger(cuda=torch.cuda.is_available() and not args.no_cuda)
    args.logger = logger
    config = json.load(open(args.config_file, 'r', encoding='utf-8'))

    # choose dataset and training config based on the given sequence length
    seq_len = str(args.max_seq_length)

    datasets = config["data"]["mixed_seq_datasets"][seq_len]
    del config["data"]["mixed_seq_datasets"]
    training = config["mixed_seq_training"][seq_len]
    del config["mixed_seq_training"]
    config["data"]["datasets"] = datasets
    config["training"] = training
    args.config = config

    args.job_name = config['name'] if args.job_name is None else args.job_name
    print("Running Config File: ", args.job_name)
    # Setting the distributed variables
    print("Args = {}".format(args))

    # Setting all the seeds so that the task is random but same accross processes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    args.saved_model_path = os.path.join(args.output_dir, "saved_models/",
                                         args.job_name)

    args.n_gpu = 1

    # Loading Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config["bert_token_file"])
    args.tokenizer = tokenizer

    # Set validation dataset path
    if args.validation_data_path_prefix is None:
        logging.warning(
            'Skipping validation because validation_data_path_prefix is unspecified'
        )

    # Issue warning if early exit from epoch is configured
    if args.max_steps < sys.maxsize:
        logging.warning(
            'Early training exit is set after {} global steps'.format(
                args.max_steps))

    if args.max_steps_per_epoch < sys.maxsize:
        logging.warning('Early epoch exit is set after {} global steps'.format(
            args.max_steps_per_epoch))

    return args


def prepare_optimizer_parameters(args, model):
    config = args.config

    param_optimizer = list(model.network.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if "weight_decay" in config["training"].keys():
        weight_decay = config["training"]["weight_decay"]
    else:
        weight_decay = 0.01

    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        weight_decay
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    return optimizer_grouped_parameters


def prepare_model_optimizer(args):
    # Initialize torch distributed
    # torch.distributed.init_process_group(backend="nccl")

    # Loading Model
    model = BertMultiTask(args)

    # Optimizer parameters
    optimizer_grouped_parameters = prepare_optimizer_parameters(args, model)

    # DeepSpeed initializer handles FP16, distributed, optimizer automatically.
    model.network, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model.network,
        model_parameters=optimizer_grouped_parameters)

    # Overwrite application configs with DeepSpeed config
    args.train_micro_batch_size_per_gpu = model.network.train_micro_batch_size_per_gpu(
    )
    args.gradient_accumulation_steps = model.network.gradient_accumulation_steps(
    )

    # Set DeepSpeed info
    args.local_rank = model.network.local_rank
    args.device = model.network.device
    model.set_device(args.device)
    args.fp16 = model.network.fp16_enabled()
    args.use_lamb = model.network.optimizer_name(
    ) == deepspeed.runtime.config.LAMB_OPTIMIZER

    # Prepare Summary Writer and saved_models path
    if dist.get_rank() == 0:
        summary_writer = get_sample_writer(name=args.job_name,
                                           base=args.output_dir)
        args.summary_writer = summary_writer
        os.makedirs(args.saved_model_path, exist_ok=True)

    return model, optimizer


def load_checkpoint(args, model):
    global global_step
    global global_data_samples
    global last_global_step_from_restore

    config = args.config
    logger = args.logger

    logger.info(
        f"Restoring previous training checkpoint from PATH={args.load_training_checkpoint}, CKPT_ID={args.load_checkpoint_id}"
    )
    start_epoch, global_step, global_data_samples = load_training_checkpoint(
        args=args,
        model=model,
        PATH=args.load_training_checkpoint,
        ckpt_id=args.load_checkpoint_id)
    logger.info(
        f"The model is loaded from last checkpoint at epoch {start_epoch} when the global steps were at {global_step} and global data samples at {global_data_samples}"
    )

    if args.rewarmup:
        logger.info(
            f"Rewarmup learning rate with last_global_step_from_restore = {global_step}"
        )
        last_global_step_from_restore = global_step

    lr_this_step = config["training"][
        "learning_rate"] * warmup_linear_decay_exp(
            global_step, config["training"]["decay_rate"],
            config["training"]["decay_step"],
            config["training"]["total_training_steps"],
            config["training"]["warmup_proportion"])
    logger.info(f"Restart training with lr = {lr_this_step}")

    # Run validation for checkpoint before training
    if not args.finetune and args.max_seq_length == 512:
        logger.info(
            f"Validation Loss of Checkpoint {start_epoch} before pretraining")
        index = start_epoch - 1 if start_epoch > 0 else start_epoch
        pretrain_validation(args, index, model)

    return start_epoch


def run(args, model, optimizer, start_epoch):
    global global_step
    global global_data_samples
    global last_global_step_from_restore

    config = args.config
    logger = args.logger

    if args.use_nvidia_dataset:
        pretrain_dataset_provider = NvidiaBertDatasetProvider(args)
    else:
        pretrain_dataset_provider = BingBertDatasetProvider(args)

    for index in range(start_epoch, config["training"]["num_epochs"]):
        logger.info(f"Training Epoch: {index + 1}")
        pre = time.time()
        train(args, index, model, optimizer, pretrain_dataset_provider)

        # Save ckpts according to "--ckpt_to_save" option,
        # e.g. "--ckpt_to_save 160 161" to save epoch 160 and 161.
        if args.ckpt_to_save is None or (index + 1) in args.ckpt_to_save:
            logger.info(
                f"Saving a checkpointing of the model for epoch: {index+1}")

            checkpoint_model(PATH=args.saved_model_path,
                             ckpt_id='epoch{}_step{}'.format(
                                 index + 1, global_step),
                             model=model,
                             epoch=index + 1,
                             last_global_step=global_step,
                             last_global_data_samples=global_data_samples)

        post = time.time()
        logger.info(f"Time for shard {index + 1}: {post-pre} seconds")

        current_global_step = global_step - last_global_step_from_restore
        if is_time_to_exit(args=args, global_steps=current_global_step):
            print(
                f'Warning: Early training termination due to max steps limit, epoch={index+1}, global_step={current_global_step}'
            )
            break


def main():
    start = time.time()
    args = construct_arguments()
    model, optimizer = prepare_model_optimizer(args)
    start_epoch = 0
    if not None in [args.load_training_checkpoint, args.load_checkpoint_id]:
        start_epoch = load_checkpoint(args, model)
    run(args, model, optimizer, start_epoch)
    elapsed = time.time() - start
    logger = args.logger
    logger.info(f"Elapsed time: {elapsed} seconds")


if __name__ == "__main__":
    main()
