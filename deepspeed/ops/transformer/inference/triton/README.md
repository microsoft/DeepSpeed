# DeepSpeed with Triton compiler

# 1. Overview

We integrate Triton to Deepspeed, which further accelerates inference in BERT-like models in float16 precision. In other words, Triton kernels can be used in DeepSpeed which leverages a recent open source compiler (https://github.com/openai/triton). Depending on the model, task-query (e.g., different sequence lengths in fill-mask task) and underlying hardware (i.e., A100), it has been shown to give an average latency reduction of 4~18% as shown in Table 1.

<div align="center">

| Hardware | Bert-base | Bert-large | Roberta-large |
|----------|:------:|:------:|:------:|
| A100 | 17% | 18% | 17% |
| A6000 | 10% | 11% | 8% |
| V100 | 4% | 5% | 2% |

Table 1. Average latency reduction in percentage when compared to the latency with CUDA kernels in Deepspeed.

</div>

Specifically, with the latest Triton (https://pypi.org/project/triton/2.0.0.post1/), it gives a larger performance gain for a longer sequence length as shown in Table 2 and performs better with GPUs with ampre architectures.


<div align="center">

| Sequence length range | Bert-base | Bert-large | Roberta-large |
|----------|:------:|:------:|:------:|
| short (8 ~ 64) | 9% | 9% | 9% |
| medium (64 ~ 256) | 11% | 11% | 10% |
| long (256 ~ 512) | 23% | 25% | 23% |

Table 2. Latency reduction in percentage with different sequence lengths in A100.

</div>


Next, we dive deeper into this new feature in DeepSpeed.

# 2. How to use Triton in Deepspeed

For those transformer operators in float16 (such as matmul, softmax and layer-norm), there are kernels written in Triton language that replaces ordinary CUDA or torch operators. From DeepSpeed config, it can be enabled to use Triton compilers to optimize the kernels.

```
pipe = pipeline('fill-mask', model='bert-base-cased', framework='pt', device=0)
pipe.model = deepspeed.init_inference(pipe.model,
                                        dtype=torch.float16,
                                        replace_with_kernel_inject=True,
                                        replace_method='auto',
                                        enable_cuda_graph=True,
                                        use_triton=True,
                                        triton_autotune=True,
                                        max_out_tokens=pipe.tokenizer.model_max_length)
```


## Running BERT inference with Triton kernels

We use an example of Bert-base here.

```python
pip install deepspeed

git clone https://github.com/microsoft/DeepSpeedExamples.git
cd DeepSpeedExamples/inference/huggingface/fill-mask

deepspeed --num_gpus 1 test-bert.py --triton
```

Also, you can run a performance benchmark.

```python
pip install deepspeed

git clone https://github.com/microsoft/DeepSpeedExamples.git
cd DeepSpeedExamples/benchmarks/inference

deepspeed --num_gpus 1 triton-bert-benchmark.py --model bert-base-cased --dtype fp16 --kernel-inject --deepspeed --graphs --triton
```

# NOTE
<!-- **_NOTE:_** -->
* To get started, please visit our github page for DeepSpeed: [GitHub Landing Page](https://github.com/microsoft/DeepSpeedExamples)

* We will continue to improve DeepSpeed-Triton with your feedback and support.

* Please visit our [website](https://www.deepspeed.ai/) for detailed blog posts, tutorials, and helpful documentation.
* This is primarily for BERT, Roberta and other BERT-like models and is not enabled for text-generation yet.

* Sequence length ranges from 8 to 512 and batch-size is set to 1 in the experiments shown in Table 1 and 2.

* It also should be noted the cuda-graph has to be enabled to benefit from Triton. Otherwise, there will be rather larger overhead from JIT compilation and a deep call stack in Triton.

* 'triton_autotune' in the config also needs to be on for the best performance. It initially goes through Triton autotuning step to build the optimal autotune table for Triton kernels and it will take some time.
