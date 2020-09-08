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
class AlexNetPipe(torchvision.models.alexnet.AlexNet):
    def to_layers(self):
        layers = [
            *self.features,
            self.avgpool,
            lambda x: torch.flatten(x, 1),
            *self.classifier
        ]
        return layers
```
