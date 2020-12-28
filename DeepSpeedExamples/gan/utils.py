import argparse
import random
import os
import torch
import numpy as np

def get_argument_parser():
    parser = argparse.ArgumentParser(description="GAN")

    # Other parameters
    parser.add_argument('--backend', type=str, default='nccl', help='distributed backend')

    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist | imagenet | folder | lfw | fake | celeba')
    parser.add_argument('--dataroot', type=str, required=False, default='/data/celeba/', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=16)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=16, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='./gan_output', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, default=999, help='manual seed')
    parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
    parser.add_argument('--tensorboard_path', default='./runs/deepspeed', help='tensorboard log dir')
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    return parser

def set_seed(value):
    print("Random Seed: ", value)
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    np.random.seed(value)

def create_folder(folder):
    try:
        os.makedirs(folder)
    except OSError:
        pass
