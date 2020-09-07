---
title: "Getting Started with Pipeline Parallelism"
---

DeepSpeed provides a pipeline parallel engine as of `v0.3`!

## Pipeline Parallelism in DeepSpeed

*Placeholder intro taken from press release:*

Pipeline parallelism divides the layers of the model into stages
that can be processed in parallel. As one stage completes the forward pass
for a micro-batch, the activation memory is communicated to the next stage in
the pipeline. Similarly, as the next stage completes its backward
propagation, gradients are communicated backwards through the pipeline.
Multiple micro-batches must be kept in flight to ensure pipeline stages
compute in parallel. Several approaches, such as PipeDream, have been
developed to trade off memory and compute efficiency as well as convergence
behavior. DeepSpeed’s approach extracts parallelism through gradient
accumulation to maintain the same convergence behavior as traditional data-
and model-parallel training with the same total batch size.

### Hybrid Parallelism

![Pipeline](/assets/images/3d-parallelism.png)


## Pipeline Parallel Models

### AlexNet
We start from an AlexNet implementation:

```python
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
```
