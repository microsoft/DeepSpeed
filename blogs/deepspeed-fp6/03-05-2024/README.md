<div align="center">

# DeepSpeed-FP6: The Power of FP6-Centric Serving for Large Language Models 

</div>

<div align="center">

<img src="./assets/hero-figure.png" width="1000px" alt="DeepSpeed-VisualChat!"/>

</div>


To cite DeepSpeed-FP6, please cite the following two arxiv reports:

```
@article{wu2023zeroquant,
  title={Zeroquant (4+ 2): Redefining llms quantization with a new fp6-centric strategy for diverse generative tasks},
  author={Wu, Xiaoxia and Xia, Haojun and Youn, Stephen and Zheng, Zhen and Chen, Shiyang and Bakhtiari, Arash and Wyatt, Michael and Aminabadi, Reza Yazdani and He, Yuxiong and Ruwase, Olatunji and Song, Leon and others},
  journal={arXiv preprint arXiv:2312.08583},
  year={2023}
}

@article{xia2024fp6,
  title={FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design},
  author={Xia, Haojun and Zheng, Zhen and Wu, Xiaoxia and Chen, Shiyang and Yao, Zhewei and Youn, Stephen and Bakhtiari, Arash and Wyatt, Michael and Zhuang, Donglin and Zhou, Zhongzhu and others},
  journal={arXiv preprint arXiv:2401.14112},
  year={2024}
}
```
# Table of Contents
1. [Why 6-bit Floating Point (FP6)?](#introduction)
2. [System Support for FP6](#system-fp6)
3. [LLMs serving with FP6](#serving-llm)
4. [How to start](#how-to-start)
5. [Software Improvements](#software-improvements)
6. [Acknowledgments and Contributions](#ac)

# 1.Why 6-bit Floating Point (FP6) <a name="introduction"></a>
The realm of Large Language Models (LLMs) like GPT has been evolving rapidly, with a focus on enhancing performance while managing the computational and storage demands.

*Diving Deep into 4-Bit Quantization's Challenges.* In our [recent research](https://arxiv.org/abs/2312.08583), we examine the drawbacks of using 4-bit quantization techniques such as GPTQ in large language models (LLMs). While these techniques hold the potential to decrease model size and computational requirements, they often fall short in critical more general tasks due to overfitting issues.  We extend the examination to include more generative tasks like code generation and summarization, areas where standard quantization methods have not been thoroughly explored. We found that INT4 weight quantization does not perform well in these broader applications, underscoring the urgent need for new approaches that improve both the efficiency and effectiveness of LLMs.

*Breakthrough with FP6.* Our exploration of different quantization methods brought us to the FP6 precision standard. Despite the difficulties in integrating and speeding up FP6 with current AI hardware—a challenge we will address in the following section—this format excels in performance and flexibility for a variety of tasks. Notably, models quantized with FP6, like the StarCoder-15B, achieve results comparable to their FP16 equivalents in code generation, and smaller models (like BART-406M) meet standard FP16 performance levels in summarization. To improve the efficiency of AI hardware and equal the best performance seen with INT4 quantization, we propose a novel 4+2 FP6 scheme. This innovation makes FP6 a promising avenue for enhancing the efficiency of LLMs, marking a significant leap in the progress of AI technologies.  For more details, please refer to our [research paper](https://arxiv.org/abs/2312.08583). 


# 2 System Support for FP6 <a name="system-fp6"></a>

*Pioneering Full-Stack GPU Kernel Design*. One challenge of FP6 quantization is that there lacks an efficient GPU kernel design for this irregular bit-width. In our [recent research](https://arxiv.org/abs/2401.14112), we introduce TC-FPx, the first full-stack GPU system design scheme with unified Tensor Core support of float-point weights for FP6 and various quantization bit-width (6-bit, 5-bit, 3-bit, etc.), mitigating the "memory wall" issues during LLM inference. TC-FPx breaks the limitations of the underlying GPU hardware, allowing the GPU to support linear layer calculations involving model weights of arbitrary bit width. In TC-FPx, Tensor Cores are utilized for intensive computation of matrix multiplications, while SIMT cores are effectively leveraged for weight dequantization, transforming the x-bit model weights to FP16 type during runtime before feeding them to Tensor Cores. It has the following key innovations:
<div align="center">
  <img src="./assets/fp6-design.png" alt="fp6 design" width="600"/>

</div>

* *Ahead-of-time Bit-level Pre-packing*, to resolve the challenge of unfriendly memory access for weights with irregular bit-width, enabling optimal GPU memory access.

* *SIMT-Efficient GPU Runtime*, to minimize the runtime overhead of weight de-quantization.

* *The software pipeline of TC-FPx kernel*, where SIMT cores, Tensor Cores, and the GPU memory hierarchy cooperate efficiently with high performance.



On average, the TC-FPx kernel demonstrates a 2.1-fold enhancement in processing speed over the FP16 cuBLAS benchmark during memory-intensive matrix-matrix multiplications on NVIDIA A100 GPUs. Notably, the implementation of the FP6 kernel through FP6 quantization facilitates the operation of LLaMA-70b on a solitary A100 GPU. This remarkable feat results in a normalized inference throughput that is 1.69 to 2.65 times superior to the FP16 benchmark when conducting inference tasks with batch-size under 32.


# 3. LLMs serving with FP6 <a name="serving-llm"></a>

We have successfully integrated the FP6 quantization kernel into DeepSpeed-FastGen, facilitating on-the-fly, weight-only quantization. This enhancement permits the efficient quantization and deployment of large language models (LLMs) through a unified configuration option within DeepSpeed-FastGen. Detailed information regarding this feature will be provided in due course. Via our interface, users have the flexibility to input either a HuggingFace model name or a local checkpoint directory. Upon input, our system initiates the loading of the specified checkpoint, implements FP6 round-to-nearest quantization across each linear layer, and transforms the quantized weights into 6-bit prepacked tensors. These tensors then serve as the updated weights, while the original FP16 weights are discarded to optimize memory usage. Throughout the inference stage, the FP6 kernels leverage these 6-bit prepacked weights, ensuring a seamless experience for users engaging with our platform.

We assessed the LLaMA-70b model's serving performance using FP6 quantization on two A100 GPUs-80G, achieving a *1.5x* decrease in inference latency and a *3.5x* increase in inference throughput compared to the FP16 baseline. FP6 quantization offers two key benefits for model inference: it enables the deployment of large language models (LLMs) on fewer GPUs—for instance, LLaMA-70b fits on a single A100-80G GPU with FP6, versus at least two GPUs required for the FP16 baseline. Additionally, it significantly accelerates linear layers in memory-bound scenarios, common in LLM inference. Moreover, FP6 quantization reduces GPU memory requirements for model weights, allowing for more queries to be served simultaneously, leading to higher serving throughputs.

Our system demonstrates exceptional efficiency in handling long generation sequences. As illustrated in Figure 1, for generation lengths surpassing the prompt length, our system exhibits a notable performance superiority. The disparity in performance between FP6 and the FP16 baseline widens with the extension of the generation sequence length. This trend is primarily attributed to the inference process becoming increasingly memory-constrained as the decoding length expands, favoring our weight-quantized GPU kernels by facilitating greater kernel speed enhancements relative to the FP16 baseline. It is important to highlight two factors contributing to the increased memory constraints in longer decoding scenarios. 
 - Firstly, the memory usage for the KV cache escalates with the sequence length, reducing the feasible batch sizes and leading to memory-bound General Matrix Multiply (GEMM) operations. 
 - Secondly, within the context of DeepSpeed-FastGen's prefill-decoding-mixed-batch technique, scenarios involving extended token generation encounter a reduction in prefill-chunks available for mixing with decodings. This results in a higher frequency of batches dedicated solely to decodings, further intensifying the memory-bound conditions.

<p align="center">
  <img src="./assets/servingllm/100-250.png" alt="Caption1" width="30%">
  <img src="./assets/servingllm/100-500.png" alt="Caption2" width="30%">
  <img src="./assets/servingllm/100-1000.png" alt="Caption3" width="30%">
</p>

  *Figure 1*:  End-to-end serving performances in DeepSpeed-MII with 128 number of requests and 32 clients, for LLaMA-2-70B model on 2xA100-80g with two-way tensor parallelism. We experimented with different number of requests between 128, 256 and 512 and found that the speedup is simillar. 

Despite the significant benefits of FP6 quantization, the current implementation faces limitations. Notably, in scenarios where GEMM operations become compute-bound due to large batch sizes or sufficient GPU memory, our weight-only quantization kernel may not sustain its latency advantage, especially against optimized libraries like cuBlas. However, our system's memory efficiency remains a key benefit. Currently, support is limited to Non-Mixture of Experts (Non-MoE) structures, with efforts underway to extend support to MoE structures. Additionally, the system is compatible only with FP16 input models, as the FP6 kernel processes FP16 activations exclusively.

</div>

# 4. How to begin with DeepSpeed-FP6  <a name="how-to-start"></a>

The quantization-and-inference experience of DeepSpeed-FP6 is straightforward and convenient. Here we give an example based on LLaMa-2-70B model:

```python
import mii
pipe = mii.pipeline("NousResearch/Llama-2-70b-hf", quantization_mode='wf6af16')
response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
print(response)
```

You need to install the following:
```
pip install deepspeed
pip install deepspeed-mii
pip install qtorch
```

To benchmark with our DeepSpeed-FP6, please visit the following script: 
```bash
https://github.com/microsoft/DeepSpeedExamples/blob/master/benchmarks/inference/mii/run_all.sh
```

Please also visit the [FP6-LLM github](https://github.com/usyd-fsalab/fp6_llm) for the standalone kernel of FP6.  Don't forget to star the repo to show your support!


# 5. Software Improvement  <a name="software-improvements"></a>



* Our DeepSpeed-FP6 currently only support for linear GEMM. We are looking forward the support for  MoE GEMM. We will continue to improve DeepSpeed-FP6 with your feedback and support. DeepSpeed-FP6 is a component of the larger DeepSpeed ecosystem, which includes a range of Deep Learning systems and modeling technologies. To learn more,

* Please visit our [website](https://www.deepspeed.ai/) for detailed blog posts, tutorials, and helpful documentation.
* Follow us on our [English X(Twitter)](https://twitter.com/MSFTDeepSpeed), [Japanese X(Twitter)](https://twitter.com/MSFTDeepSpeedJP), and [Chinese Zhihu](https://www.zhihu.com/people/deepspeed) for latest news on DeepSpeed.

We welcome your contributions to DeepSpeed! We encourage you to report issues, contribute PRs, and join discussions on the [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed/) page. Please see our [contributing guide](https://github.com/microsoft/DeepSpeed/blob/master/CONTRIBUTING.md) for more details. We are open to collaborations with universities, research labs, companies, such as those working together on deep learning research, applying DeepSpeed to empower real-world AI models and applications, and so on. For such requests (and other requests unsuitable for GitHub), please directly email to deepspeed-info@microsoft.com.

* "Star" our [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed/) and [DeepSpeed-MII GitHub](https://github.com/microsoft/DeepSpeed-MII/) and [DeepSpeedExamples GitHub](https://github.com/microsoft/DeepSpeedExamples/)  repositories if you like our work!


# 6. Acknowledgments and Contributions <a name="ac"></a>
We thank the collaboration of the University of Sydney and Rutgers University. We also thank the open-source library [aspuru-guzik-group/qtorch](https://github.com/aspuru-guzik-group/qtorch).

Contributions:  
Xiaoxia Wu\* $^1$, Zhen Zheng\* $^1$, Haojun Xia\* $^2$, Arash Bakhtiari $^1$, Michael Wyatt $^1$, Shiyang Chen $^3$, Stephen Youn $^1$, Yuxiong He, Tunji Ruwase $^1$,  Zhewei Yao, Leon Song $^1$ $^2$  

\* Equal Contribution  
1: Microsoft  
2: University of Sydney  
3: Rutgers University
