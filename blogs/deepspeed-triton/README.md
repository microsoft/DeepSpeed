# DeepSpeed with Triton compiler

# 1. Overview

We have integrated [Triton](https://github.com/openai/triton), an open source compiler for GPU programming, into DeepSpeed, which further boosts the inference speed of BERT-like models in float16 precision.
With Triton kernels, DeepSpeed can achieve 1.14~1.68x performance gain (in other words, average latency reduction of 12~41%), depending on the model and the underlying hardware.
Table 1 shows the average performance improvement for different models and GPUs.

<div align="center">

| Hardware | Bert-base | Bert-large | Roberta-base | Roberta-large |
|----------|:------:|:------:|:------:|:------:|
| A100 |1.65x | 1.68x | 1.53x | 1.61x |
| V100 | 1.29x | 1.14x | 1.23x | 1.21x |

Table 1. The average performance gain when compared to the baseline (see NOTE below)


</div>

For those transformer operators in float16, we have implemented kernels written in Triton language that replace ordinary CUDA kernels or torch operators.
Specifically, softmax, activation, layer-normalization, residual-addition and all the matmuls except MLP layers are replaced by Triton kernels (refer NOTE below for details).
In our experiments, Triton kernels helped to reduce the average (over difference sequence lengths) latency by 6~24% (depending on model and hardware) when compared to the latency with CUDA-only kernels.


Figures below further show performance profiles in detail.
Figure 1 visualizes latency reduction in different sequence lengths in A100 GPU for Bert-base model.
The baseline (blue) is from Huggingface transformers without any kernel injection, the orange is from Deepspeed with CUDA-only kernels and the gray is from Deepspeed with Triton kernels.
Figure 2 shows again the normalized latency in A100 but for Bert-large model.

<div align="center">

<img src="../assets/images/triton-bert-base-latency.png" width="500px" alt="triton-bert-base-latency"/>

*Figure 1: Sequence length ranges versus normalized P90 latency in A100 for Bert-base model*

<img src="../assets/images/triton-bert-large-latency.png" width="500px" alt="triton-bert-large-latency"/>

*Figure 2: Sequence length ranges versus normalized P90 latency in A100 for Bert-large model*

</div>


Next, we dive deeper into this new feature in DeepSpeed.

# 2. How to use Triton in Deepspeed

You can enable Triton compilers to optimize these kernels by setting a flag in the DeepSpeed config file.

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

To run a performance benchmark, you can use the following command:

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

* This feature is currently only available for BERT, Roberta and other BERT-like models and is not supported for text-generation yet.

* It is important to note that CUDA graph has to be enabled to benefit from Triton. Otherwise, there will be a significant overhead from JIT compilation and a deep call stack in Triton.
Thus, we enabled CUDA graph for all cases.

* 'triton_autotune' in the config also needs to be on for the best performance. It will run an initial Triton autotuning step to build the optimal autotune table for Triton kernels, which will take some time.

* In our experiments, sequence length in query ranged from 8 to 512 and batch-size was set to 1, while the task is set to 'fill-mask'.
Table 1 compares the P90 model latencies averaged over the entire sequence length range (i.e., 8~512), while Figures 1 and 2 compare the P90 model latencies over specific sub-ranges (i.e. sequence lengths in the range shown in x-axis).
The baseline was set to the Huggingface transformers and the performance gain is defined to be (reference P90 latency)/(Deepspeed-Triton P90 Latency).
One thing to note is that the cuda kernel in MLP exhibited a better performance than the triton kernel in our experiments so we chose the hybrid approach that both triton and cuda kernels are used when Triton kernel is enabled from the Deepspeed config.

* We used [the latest Triton release](https://pypi.org/project/triton/2.0.0.post1/) in our experiments.
