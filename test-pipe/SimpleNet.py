import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import deepspeed.pt.pipe as pipe
import deepspeed.pt.pipe.PipelineModule as PipelineModule
from deepspeed.pt.pipe.PipelineModule import LayerSpec


class VerboseLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,
                      64,
                      kernel_size=11,
                      stride=4,
                      padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(64,
                      192,
                      kernel_size=5,
                      padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(192,
                      384,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,
                      256,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,
                      256,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.loss_fn(x, y)


class AlexNetPipe(PipelineModule.PipelineModule):
    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        super().__init__(**kwargs)
        self.loss_fn = nn.CrossEntropyLoss()

    def layer_specs(self):
        specs = [
            LayerSpec(nn.Conv2d, 3, 64, kernel_size=11, stride=4, padding=5),
            F.relu,
            LayerSpec(nn.MaxPool2d, kernel_size=2, stride=2),
            LayerSpec(nn.Conv2d, 64, 192, kernel_size=5, padding=2),
            F.relu,
            LayerSpec(nn.MaxPool2d, kernel_size=2, stride=2),
            LayerSpec(nn.Conv2d, 192, 384, kernel_size=3, padding=1),
            F.relu,
            LayerSpec(nn.Conv2d, 384, 256, kernel_size=3, padding=1),
            F.relu,
            LayerSpec(nn.Conv2d, 256, 256, kernel_size=3, padding=1),
            F.relu,
            LayerSpec(nn.MaxPool2d, kernel_size=2, stride=2),

            lambda x: x.view(x.size(0), -1),
            LayerSpec(nn.Linear, 256, self.num_classes), # classifier
        ]
        return specs


# Taken from CIFAR example
class SimpleNet(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=64, layers=8):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = layers

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.linears = nn.ModuleList([nn.Linear(16 * 5 * 5, self.hidden_dim), nn.ReLU()])
        # Configurable sequence of linear layers
        for l in range(self.num_layers):
            self.linears.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()])

        self.fc3 = nn.Linear(self.hidden_dim, self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.linears[0](x)
        for layer in self.linears[1:]:
            x = layer(x)

        x = self.fc3(x)
        return self.loss_fn(x, y)


# Taken from CIFAR example
class SimpleNetTied(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=64, layers=8):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = layers

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.linears = nn.ModuleList([nn.Linear(16 * 5 * 5, self.hidden_dim), nn.ReLU()])
        self.tied = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        # Configurable sequence of linear layers
        for l in range(self.num_layers):
            self.linears.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()])

        self.fc3 = nn.Linear(self.hidden_dim, self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        # Do the first Linear/ReLU
        x = self.linears[0](x)
        x = self.linears[1](x)
        # Insert the tied layer
        x = self.tied(x)
        for layer in self.linears[1:]:
            x = layer(x)
        x = F.linear(x, self.tied.weight)
        x = self.fc3(x)
        return self.loss_fn(x, y)


class SimpleNetPipe(PipelineModule.PipelineModule):
    def __init__(self, num_classes=10, hidden_dim=64, layers=8, **kwargs):
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = layers
        self._specs = None
        super().__init__(**kwargs)

        self.loss_fn = nn.CrossEntropyLoss()

    def layer_specs(self):
        """ Required for PipelineModule (abstract method). """
        if self._specs is None:
            self._specs = self._build_specs()
        return self._specs

    def _build_specs(self):
        """ Define the module layers. """

        # Note that this will be allocated on each pipeline stage because it is not in a
        # LayerSpec.
        pool = nn.MaxPool2d(2, 2)

        specs = []

        # First conv nets and prepare for linear layers
        specs.extend([
            PipelineModule.LayerSpec(nn.Conv2d,
                                     3,
                                     6,
                                     5),
            F.relu,
            pool,
            PipelineModule.LayerSpec(nn.Conv2d,
                                     6,
                                     16,
                                     5),
            F.relu,
            pool,
            lambda x: x.view(-1,
                             16 * 5 * 5),
            PipelineModule.LayerSpec(nn.Linear,
                                     16 * 5 * 5,
                                     self.hidden_dim),
            F.relu,
        ])

        # Configurable sequence of linear layers
        for l in range(self.num_layers):
            specs.extend([
                PipelineModule.LayerSpec(nn.Linear,
                                         self.hidden_dim,
                                         self.hidden_dim),
                F.relu,
            ])

        # Map to classification
        specs.append(
            PipelineModule.LayerSpec(nn.Linear,
                                     self.hidden_dim,
                                     self.num_classes))
        return specs


class SimpleNetPipeTied(PipelineModule.PipelineModule):
    def __init__(self, num_classes=10, hidden_dim=64, layers=8, **kwargs):
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = layers
        self._specs = None
        super().__init__(**kwargs)

        self.loss_fn = nn.CrossEntropyLoss()

    def layer_specs(self):
        """ Required for PipelineModule (abstract method). """
        if self._specs is None:
            self._specs = self._build_specs()
        return self._specs

    def _build_specs(self):
        """ Define the module layers. """

        # Note that this will be allocated on each pipeline stage because it is not in a
        # LayerSpec.
        pool = nn.MaxPool2d(2, 2)

        specs = []

        # First conv nets and prepare for linear layers
        specs.extend([
            PipelineModule.LayerSpec(nn.Conv2d,
                                     3,
                                     6,
                                     5),
            F.relu,
            pool,
            PipelineModule.LayerSpec(nn.Conv2d,
                                     6,
                                     16,
                                     5),
            F.relu,
            pool,
            lambda x: x.view(-1,
                             16 * 5 * 5),
            PipelineModule.LayerSpec(nn.Linear,
                                     16 * 5 * 5,
                                     self.hidden_dim),
            F.relu,
        ])
        specs.append(
            PipelineModule.TiedLayerSpec('tied_linear',
                                         VerboseLinear,
                                         self.hidden_dim,
                                         self.hidden_dim))

        # Configurable sequence of linear layers
        for l in range(self.num_layers):
            specs.extend([
                PipelineModule.LayerSpec(nn.Linear,
                                         self.hidden_dim,
                                         self.hidden_dim),
                F.relu,
            ])
        '''
        # XXX just testing list send/recv
        specs.append(lambda x: [x, torch.tensor(data=[1.38, 11.38], requires_grad=True).cuda()])
        for l in range(8 * self.num_layers):
            specs.extend([
                lambda xs: [xs[0], 2 * xs[1]],
                lambda xs: [xs[0], 2 * xs[1]],
            ])
        specs.append(lambda xs: xs[0])
        '''

        specs.append(
            PipelineModule.TiedLayerSpec('tied_linear',
                                         VerboseLinear,
                                         self.hidden_dim,
                                         self.hidden_dim,
                                         forward_fn=lambda mod,
                                         x: F.linear(x,
                                                     mod.weight)))

        # Map to classification
        specs.append(
            PipelineModule.LayerSpec(nn.Linear,
                                     self.hidden_dim,
                                     self.num_classes))
        return specs
