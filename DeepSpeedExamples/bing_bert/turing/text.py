import torch

PAD = 0


def mask(x):
    return x != PAD


def torch_long(x):
    return torch.LongTensor(x)
