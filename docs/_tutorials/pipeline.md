---
title: "Pipeline Parallelism"
---

DeepSpeed v0.3 includes new support for pipeline parallelism! Pipeline
parallelism improves both the memory and compute efficiency of deep learning
training by partitioning the layers of a model into stages that can be
processed in parallel.
DeepSpeed's training engine provides hybrid data and pipeline parallelism and
can be further combined model parallelism such as
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM).
An illustration of
3D parallelism is shown below. Our latest [results](linklinklink)
demonstrate that this 3D parallelism enables training models with over a
**trillion** parameters.

![3D parallelism in DeepSpeed](/assets/images/3d-parallelism.png)

DeepSpeed uses *gradient accumulation* to extract pipeline parallelism (shown
below). Each batch of training data is divided into micro-batches that can be
processed in parallel by the pipeline stages. Once a stage completes the
forward pass for a micro-batch, the activation memory is communicated to the
next stage in the pipeline. Similarly, as the next stage completes its
backward pass on a micro-batch, the gradient with respect to the activation
is communicated backwards through the pipeline. Each backward pass
accumulates gradients locally. Next, all data parallel groups perform
reductions of the gradients in parallel. Lastly, the optimizer updates the
model weights.

![Pipeline Schedule](/assets/images/pipe-schedule.png)

*TODO: working on a real illustration of pipeline scheduling. This is just from a
spreadsheet I had laying around – I think Samyam and I made it live one
afternoon over a call. It doesn’t show gradient communication or the
optimizer step.*


## Getting Starting with Pipeline Parallelism

DeepSpeed strives to accelerate *and* simplify the process of pipeline
parallel training. This section provides first steps with hybrid data and
pipeline parallel training by preparing `torchvision`'s
[AlexNet](https://pytorch.org/docs/1.2.0/_modules/torchvision/models/alexnet.html)
model.

### Expressing Pipleline Models
Pipeline parallelism requires models be expressed as a sequence of layers.
In the forward pass, each layer consumes the output of the previous
layer. In fact, there is no need to specify a `forward()` for a pipeline
parallel model! The forward pass of a pipeline parallel model implicitly
takes the form:
```python
def forward(self, inputs):
    x = inputs
    for layer in self.layers:
        x = layer(x)
    return x
```

PyTorch's
[`torch.nn.Sequential`](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
is a convenient container for expressing pipeline parallel models and can be
parallelized by DeepSpeed with no modification:
```python
net = nn.Sequential(
    nn.Linear(in_features, hidden_dim),
    nn.ReLU(inplace=True),
    nn.Linear(hidden_dim, out_features)
)
from deepspeed.pipe import PipelineModule
net = PipelineModule(layers=net, num_stages=2)
```
`PipelineModule` uses its `layers` argument as the sequence of layers that
comprise the model. After initialization, `net` is divided into two pipeline
stages and its layers moved to the correpsonding GPUs. If more than two GPUs
are present, DeepSpeed will also use hybrid data parallelism.

**Note:** The total number of GPUs must be divisible by the number of pipeline
stages.
{: .notice--info}

### AlexNet
Let's look at an abbreviated implementation of `torchvision`'s
[AlexNet](https://pytorch.org/docs/1.2.0/_modules/torchvision/models/alexnet.html).

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            ...
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            ...
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```
`AlexNet` is the composition of several `Sequential` submodules. We can turn this into
a `PipelineModule` by



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

DeepSpeed's `PipelineModule` supports a superset of PyTorch’s
`torch.nn.Sequential` container. In addition

DeepSpeed provides several conveniences for expressing tools as pipeline
parallel modules in addition to tensors in addition to modules whose forward
passes are executed users may also provide arbitrary functions such as
reshaping tensors or converting to precision so on.below is an example of a
module which provides three layers with a reshape in between .


### Inputs and Outputs
Following `torch.nn.Sequential`, the inputs and outputs of each layer must be
either a single `torch.Tensor` or a `tuple` of tensors. In practice, some
models may need to modify their forward pass to pack and unpack arguments to
`forward()`. Consider an abbreviated implementation of a Transformer block in a GPT
model:
```python
class TransformerBlock(nn.Module)
    ...
    def forward(self, hidden, mask):
        output = self.compute(hidden, mask)
        return output
    ...

stack = [ TransformerBlock() for _ in range(num_layers) ]
```
Two modifications to `TransformerBlock` are required:

1. The arguments must be collected into a `tuple`.
2. `mask` must also be returned from `forward()` so the next layer can use it.

These modifications can be accomplished with a short subclass:
```python
class TransformerBlockPipe(TransformerBlock)
    def forward(self, inputs):
        hidden, mask = inputs
        outputs = super().forward(hidden, mask)
        return output, mask
```


### Dealing with Data


```python
trainloader = deepspeed.utils.RepeatingLoader(trainloader)
trainiter = iter(trainloader)
```

### Training AlexNet

For example, forward and backward passes are interleaved and thus the training
loop cannot be divided into separate stages of `forward()`, `backward()` and `step()`.


```python
def join_layers(vision_model):
    """Joins the a torchvision AlexNet or VGG model into a single sequence."""
    return [
        *vision_model.features,
        vision_model.avgpool,
        lambda x: torch.flatten(x, 1),
        *vision_model.classifier,
    ]

net = torchvision.models.alexnet(num_classes=10)
net = PipelineModule(layers=join_layers(net),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     num_stages=2)

engine, _, _, _ = deepspeed.initialize(
    args=args,
    model=net,
    model_parameters=[p for p in net.parameters() if p.requires_grad],
    training_data=cifar_trainset())

for step in range(args.steps):
    loss = engine.train_batch()
```


## Advanced Topics

### Custom Pipeline Schedules


<!--
Commented out scratch space


Pipeline parallelism has several strengths:

* Enables larger models than with data
parallelism alone because workers only store a subset of the model and
optimizer locally.
* Accelerates training in bandwidth-bound scenarios through a significant reduction in  communication volume.



including [ZeRO](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

-->
