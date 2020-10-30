import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import register_module_forward_hook
from functools import partial
import numpy as np
import sys
from ptflops import get_model_complexity_info


class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=6,
                      kernel_size=5,
                      stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16,
                      out_channels=120,
                      kernel_size=5,
                      stride=1),
            nn.Tanh())

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120,
                      out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84,
                      out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs


if __name__ == "__main__":
    mod = LeNet5(10)
    input = torch.randn(3, 1, 32, 32)
    macs, params = get_model_complexity_info(mod,
                                             tuple(input.shape)[1:],
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
