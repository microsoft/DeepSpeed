#!/usr/bin/env python3

import os
import sys
import argparse
import logging

import torch
import torch.distributed as dist

import deepspeed
from deepspeed.pt.pipe.PipelineParallelGrid import PipeDataParallelTopology, PipeModelDataParallelTopology

# All models
from SimpleNet import *


def cifar_trainset():
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,
                               0.5,
                               0.5),
                              (0.5,
                               0.5,
                               0.5))])

    # Only one rank per machine downloads.
    dist.barrier()
    if args.local_rank != 0:
        dist.barrier()
    trainset = torchvision.datasets.CIFAR100(root='/tmp/cifar100-data',
                                             train=True,
                                             download=True,
                                             transform=transform)
    if args.local_rank == 0:
        dist.barrier()
    return trainset


def cifar_loader(batch_size):
    """Construct DataLoader for CIFAR10 train data. """
    trainset = cifar_trainset()

    sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=False)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              sampler=sampler,
                                              batch_size=batch_size,
                                              pin_memory=True)
    return trainloader


def get_cmd_args():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=1000,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=4,
                        help='pipeline parallelism')
    parser.add_argument('-t',
                        '--tied',
                        action="store_true",
                        help='use a model with a tied linear layer (affects quality)')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def go(args):
    torch.manual_seed(args.seed)
    #net = SimpleNet()
    #net = AlexNet()
    #net = AlexNet(num_classes=100)
    if args.tied:
        net = SimpleNetTied(num_classes=100)
    else:
        net = AlexNet(num_classes=100)

    engine, opt, dataloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=cifar_trainset())

    num_steps = 0
    num_micros = 0
    epoch_idx = 0
    while True:
        #dataloader.data_sampler.set_epoch(epoch_idx)
        for batch_idx, data in enumerate(dataloader):
            inputs = data[0].to(engine.device)
            labels = data[1].to(engine.device)

            if engine.fp16_enabled():
                inputs = inputs.half()

            loss = engine.train_batch(inputs, labels)

            num_micros += 1
            if num_micros % engine.gradient_accumulation_steps() == 0:
                num_steps += 1

            if (num_steps % 100 == 0) or (num_steps == args.steps):
                print(
                    f'  RANK={dist.get_rank()} LOSS={loss.item()} global_steps={num_steps} '
                )

            if num_steps == args.steps:
                return

        epoch_idx = 1


def go_pipeline(args):
    torch.manual_seed(args.seed)
    world = dist.get_world_size()
    assert world % args.pipeline_parallel_size == 0
    dp = world // args.pipeline_parallel_size
    topology = PipeDataParallelTopology(num_dp=dp, num_pp=args.pipeline_parallel_size)

    #net = SimpleNetPipe(topology=topology)
    #net = AlexNetPipe(topology=topology)
    if args.tied:
        net = SimpleNetPipeTied(num_classes=100, topology=topology)
    else:
        net = AlexNetPipe(num_classes=100, topology=topology)

    if args.pipeline_parallel_size == 1:
        net.save_state_dict(os.path.join('checkpoint', 'init'))
    else:
        net.load_state_dir(os.path.join('checkpoint', 'init'))

    trainset = cifar_trainset()

    engine, opt, dataloader, scheduler = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    for step in range(args.steps):
        loss = engine.train_batch()


if __name__ == '__main__':
    args = get_cmd_args()

    torch.cuda.set_device(args.local_rank)

    dist.init_process_group(backend=args.backend)

    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s %(asctime)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    if args.pipeline_parallel_size == 0:
        go(args)
    else:
        go_pipeline(args)
