<div align="center">

# DeepSpeed-FastGen：通过 MII 和 DeepSpeed-Inference 实现 LLM 高吞吐量文本生成

</div>

<div align="center">
 <img src="assets/images/fastgen-hero-light.png#gh-light-mode-only" width="850px">
 <img src="assets/images/fastgen-hero-dark.png#gh-dark-mode-only" width="850px">
</div>

## 目录
1. [引言](#introduction)
2. [关键的 LLM 服务技术](#background)
3. [动态 SplitFuse：一种新颖的提示和生成组合策略](#technical-approach)
4. [性能评估](#performance-evaluation)
5. [DeepSpeed-FastGen：实现与使用](#using-deepspeed-fastgen)
6. [尝试 DeepSpeed-FastGen](#try)
7. [致谢](#acknowledgements)


## 1. 引言 <a name="introduction"></a>

像 GPT-4 和 LLaMA 这样的大型语言模型（LLMs）已成为在各个层面融入 AI 的广泛应用服务的主导工作负载。从常规聊天模型到文档摘要，从自动驾驶到软件栈的各个层面的辅助功能，这些模型的部署和服务需求正在迅速增加。尽管像 DeepSpeed、PyTorch 和其他几个框架可以在 LLM 训练期间定期实现良好的硬件利用率，但这些应用的交互性质以及像开放式文本生成这样的任务的低算术强度已成为现有系统中推理吞吐量的瓶颈。

为此，像 [vLLM](https://arxiv.org/pdf/2309.06180.pdf) 这样由 PagedAttention 驱动的框架和像 [Orca](https://www.usenix.org/system/files/osdi22-yu.pdf) 这样的研究系统显著提高了 LLM 推理的性能。然而，这些系统在为长提示的工作负载提供一致的服务质量方面仍然存在挑战。随着越来越多的模型，如 [MPT-StoryWriter](https://www.mosaicml.com/blog/mpt-7b)，和系统，例如 [DeepSpeed Ulysses](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-ulysses)，支持延伸到数万个令牌的上下文窗口，这些长提示工作负载变得越来越重要。为了更好地理解问题空间，我们提供了详细的示例来说明 LLM 的文本生成是如何在称为提示处理和生成的两个不同阶段中工作的。当系统将它们视为不同的阶段时，生成将被提示处理所抢占，这可能会破坏服务级别协议（SLAs）。

今天，我们很高兴地介绍 DeepSpeed-FastGen，一个通过利用所提出的动态 SplitFuse 技术并提供比 vLLM 等最先进系统高达 2.3 倍的有效吞吐量的系统。DeepSpeed-FastGen 利用 DeepSpeed-MII 和 DeepSpeed-Inference 的组合，提供了一个易于使用的服务系统。

**快速开始：** 尝试 DeepSpeed-FastGen 就像安装最新的 [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII) 发行版一样简单：

```bash
pip install deepspeed-mii
```

要使用简单的非持久性管道部署生成文本，请运行以下代码。有关更多详情，请参见[第 5 节](#using-deepspeed-fastgen)。

```python
from mii import pipeline
pipe = pipeline("mistralai/Mistral-7B-v0.1")
output = pipe(["Hello, my name is", "DeepSpeed is"], max_new_tokens=128)
print(output)
```

## 2. 文献中的现有 LLM 服务技术 <a name="background"></a>

单个序列的文本生成工作负载由两个阶段组成：1）提示处理，即用户提供的文本作为一批令牌有效处理以构建注意力的键值（KV）缓存，以及 2）令牌生成，它将单个令牌添加到该缓存并生成新令牌。在生成文本序列的过程中，模型将对模型进行多次前向调用以生成完整的文本序列。文献和系统中已经提出了两种主要技术，它们解决了这些阶段中可能出现的各种限制和瓶颈。

_<b> 阻塞 KV 缓存：</b>_

vLLM 识别到由于大型单体 KV 缓存导致的内存碎片化显著降低了 LLM 服务系统的并发性，并提出了 [Paged Attention](https://arxiv.org/pdf/2309.06180.pdf) 以启用非连续缓存并增加总系统吞吐量。而不是分配个别可变大小的连续内存块，KV 缓存中的底层存储是固定大小的块（也称为页面）。阻塞 KV 缓存通过消除 KV 缓存引起的内存碎片化，增加了潜在的序列并发量，从而增加了系统吞吐量。非连续 KV 缓存的实现也包括在 [HuggingFace TGI](https://github.com/huggingface/text-generation-inference) 和 [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) 中。

_<b> 连续批处理：</b>_

过去，动态批处理（服务器等待多个请求同步处理）被用来提高 GPU 利用率。然而，这种方法有缺点，因为它通常需要将输入填充到相同长度或使系统等待以构建更大的批次。

近期大型语言模型（LLM）推理和服务的进步一直专注于细粒度调度和优化内存效率。例如，Orca 提出了 _迭代级调度_（也称为连续批处理），它在模型的每次前向传递时作出独特的调度决策。这允许请求根据需要加入/离开批次，从而消除了填充请求的需要，提高了总体吞吐量。除了 Orca，NVIDIA TRT-LLM、HuggingFace TGI 和 vLLM 也实现了连续批处理。

在当前系统中，有两种主要方法来实现连续批处理。在 TGI 和 vLLM 中，生成阶段被抢占以执行提示处理（在 TGI 中称为填充）然后继续生成。在 Orca 中，这些阶段不被区分；相反，只要总序列数没有达到固定限制，Orca 就会将提示加入正在运行的批次中。这两种方法都在不同程度上需要暂停生成以处理长提示（参见[第 3B 节](#splitfuse)）。

为了解决这些缺点，我们提出了一种新颖的提示和生成组合策略，动态 SplitFuse。

## 3. 动态 SplitFuse：一种新颖的提示和生成组合策略<a name="technical-approach"></a>

DeepSpeed-FastGen 旨在利用连续批处理和非连续 KV 缓存，以提高数据中心服务 LLM 的占用率和响应性，类似于现有的框架如 TRT-LLM、TGI 和 vLLM。为了达到新的性能水平，DeepSpeed-FastGen 引入了 SplitFuse，它利用动态提示和生成分解与统一来进一步改善连续批处理和系统吞吐量。

### A. 三个性能见解
在描述动态 SplitFuse 之前，我们回答三个关键的性能问题，这些问题共同激发了其设计。

*__1. 哪些因素影响单个 LLM 的前向传递？__* 为了有效地调度，必须了解调度循环应控制的相关独立变量。我们观察到，在前向传递中序列的组成（序列中的批次大小）与前向传递中的令牌数量相比对性能的影响可以忽略不计。这意味着可以围绕单一信号——前向传递中的令牌数量——构建一个有效的调度器。

<div align="center">
<img src="assets/images/observation-prompt-v-latency.png" alt="" width="480"/><br>
</div>

*__2. 模型的吞吐量如何响应前向传递中令牌数量的变化？__* 一个 LLM 有两个关键的运行区域，并且过渡相对陡峭。当令牌数量较少时，GPU 的瓶颈是从内存中读取模型，因此吞吐量会随着令牌数量的增加而扩展，而当令牌数量很多时，模型的吞吐量受计算限制并看到近乎恒定的吞吐量。如果所有前向传递都在吞吐量饱和区域，则模型应高效运行。

<div align="center">
<img src="assets/images/observation-prompt-v-flops.png" alt="" width="480"/><br>
</div>

*__3. 如何在多个前向传递中调度一组令牌？__* 我们在上面观察到，对于对齐良好的输入，令牌吞吐量曲线是凹的，这意味着第二导数必定小于或等于 0。例如，让 $f(x)$ 是给定模型的延迟至吞吐量的凹函数。对于凹函数 $f(x)$，以下成立：

  $$0 \geq \lim_{h \to 0} \frac{f(x + h) - 2f(x) + f(x - h)}{h^2}$$

  $$0 \geq f(x + h) - 2f(x) + f(x - h)$$

  $$2f(x) \geq f(x + h) + f(x - h)$$

这表明，对于给定的 `2x` 令牌池来说，最大化吞吐量的方式是将它们均匀分割到两个批次之间。更一般地说，在一个系统中，必须在 F 个前向传递中消费和处理 P 个令牌，理想的分区方案将均匀分配它们。

### B. 动态分割融合（Dynamic SplitFuse） <a name="splitfuse"></a>

动态分割融合是一种用于提示处理和令牌生成的新型令牌组成策略。DeepSpeed-FastGen 利用动态分割融合策略，在保持一致的前向大小（forward size）运行，通过从提示中取出部分令牌并与生成过程相结合。特别是，动态分割融合执行两个关键行为：

1. 将长提示分解成更小的块，并在多个前向传递（迭代）中进行调度，只有在最后一个传递中才执行任何生成。
2. 短提示将被组合以精确填满目标令牌预算。即使是短提示也可能被分解，以确保预算被精确满足，前向大小（forward sizes）保持良好对齐。

这两种技术共同为所有用户指标提供了具体的好处：

1. **更好的响应性：** 由于长提示不再需要极长的前向传递来处理，模型将提供更低的客户端延迟。在同一时间窗口内执行的前向传递更多。
2. **更高的效率：** 短提示的融合到更大的令牌预算使模型能够持续运行在高吞吐量状态。
3. **更低的变异性和更好的一致性：** 由于前向传递的大小一致，且前向传递大小是性能的主要决定因素，每个前向传递的延迟比竞争系统更加一致，生成频率的感知也是如此。没有像其他先前工作那样的抢占或长时间运行的提示来增加延迟。

因此，DeepSpeed-FastGen 将以允许快速持续生成的速率消耗来自提示的令牌，同时向系统添加令牌，提高系统利用率，提供更低的延迟和更高的吞吐量流式生成给所有客户端，与其他最先进的服务系统相比。

<div align="center">
  <img src="assets/images/fastgen-overview-light.png#gh-light-mode-only" width="640">
  <img src="assets/images/fastgen-overview-dark.png#gh-dark-mode-only" width="640"><br>

  *图 1: 连续批处理策略的插图。每个块显示一个前向传递的执行。箭头表示前向传递有一个或多个生成的令牌序列。vLLM 在一个前向传递中要么生成令牌，要么处理提示；令牌生成抢占提示处理。Orca 在生成过程中完整长度运行提示。动态分割融合执行固定大小批次的动态组合，包括生成和提示令牌。*

</div>

## 4. 性能评估 <a name="performance-evaluation"></a>

DeepSpeed-FastGen 利用其阻塞 KV 缓存和动态分割融合连续批处理，提供了最先进的 LLM 服务性能。我们按照下面讨论的基准测试方法对 DeepSpeed-FastGen 和 vLLM 在一系列模型和硬件配置上进行评估。

### A. 基准测试方法论

我们使用两个主要的定量方案来衡量性能。

**吞吐量-延迟曲线：** 生产准备的两个关键指标是吞吐量（以每秒请求计）和延迟（每个请求的响应性）。为了衡量这一点，我们实例化多个客户端（从 1 到 32 个不等）同时发送请求（总共 512 个）给服务器。每个请求的结果延迟在端点测量，吞吐量通过完成实验的端到端时间来测量。

**有效吞吐量：** 诸如聊天应用程序之类的交互式应用程序可能有比顶层指标如端到端延迟更严格和复杂的要求。特别是，我们关注越来越受欢迎的聊天用户场景：

  1. 用户通过发送提示来启动任务。
  2. 系统处理提示并返回第一个令牌。
  3. 随着生成的进行，后续令牌被流式传输给用户。

在这个过程的每个点，系统都有可能提供不利的用户体验；例如，如果第一个令牌到达得太慢或生成似乎停止了一段时间。我们提出了一个考虑这两个维度的 SLA 框架。

由于提示和生成文本的长度差异很大，影响计算成本，因此设定固定的 SLA 值对于吞吐量和延迟是不切实际的。因此，我们将提示延迟的 SLA 定义为 |提示中的令牌| / 512 秒（= 512 令牌/秒）。此外，考虑到人类的阅读速度，我们将生成延迟的 SLA 设置在指数移动平均（EMA）上为 2、4 或 6 令牌/秒。坚持这些 SLA 的请求被认为是成功的，这些成功请求的吞吐量被称为**有效吞吐量**。

我们在 NVIDIA A100、H100 和 A6000 上对 Llama-2 7B、Llama-2 13B 和 Llama-2 70B 评估了 vLLM 和 DeepSpeed-FastGen。

### B. 吞吐量-延迟分析

在这个实验中，DeepSpeed-FastGen 在吞吐量和延迟方面都优于 vLLM，提供了等同的延迟和更大的吞吐量或更快的响应延迟和相同的吞吐量。在 Llama-2 70B 配备 4 个 A100x80GB 的情况下，DeepSpeed-FastGen 展示了高达 2 倍的吞吐量（1.36 rps 对比 0.67 rps）在相同的延迟（9 秒）或高达 50% 的延迟减少（7 秒对比 14 秒）同时实现相同的吞吐量（1.2 rps），如图 2 所示。评估 Llama-2 13B 时也呈现了这些趋势，如图 3 所示。

<div align="center">
  <img src="assets/images/throughput_latency.png" alt="" width="850"/><br>

  *图 2: 使用 Llama 2 70B 进行文本生成的吞吐量和延迟（通过 4 个 A100-80GB GPU 的张量并行）。对提示和生成长度应用了正态分布，平均值分别为 1200/2600 和 128/60，变异性为 30%*
</div><br>

<div align="center">
  <img src="assets/images/throughput_latency_13B.png" alt="" width="850"/><br>

  *图 3: 使用 Llama 2 13B 进行文本生成的吞吐量和延迟（A100-80GB GPU，无张量并行）。对提示和生成长度施加了正态分布，平均值分别为 1200/2600 和 60/128，并且有 30% 的方差*
</div>

### C. 有效吞吐量分析

在考虑了首个令牌的延迟和生成速率的有效吞吐量分析下，DeepSpeed-FastGen 提供的吞吐量比 vLLM 高出多达 2.3 倍。图 4 展示了 DeepSpeed-FastGen 和 vLLM 的有效吞吐量的比较分析。每个绘制的点表示从特定数量的客户端得出的有效吞吐量。当我们扩大客户端数量时，我们最初观察到有效吞吐量的增加。然而，当客户端数量接近系统容量时，延迟也显著增加，导致许多请求未能满足 SLA。因此，有效吞吐量将在某个点上饱和或减少。从可用性角度来看，达到最大有效吞吐量所需的客户端数量并不特别重要；线条的最高点是最优的服务点。

<div align="center">
  <img src="assets/images/effective_throughput.png" alt="" width="1200" />

  *图 4: DeepSpeed-FastGen 和 vLLM 的有效吞吐量（Llama 2 70B/A100-80GB 使用张量并行在 4 个 A100-80GB GPU 上。对提示和生成长度施加了正态分布，平均值分别为 2600 和 60，并且有 30% 的方差)*
</div><br>

当 vLLM 抢占正在进行的先前请求的生成时，生成延迟会明显增加。这导致 vLLM 的有效吞吐量看起来低于其直接测量的吞吐量。在 vLLM 的峰值时，有效吞吐量为 0.63 查询/秒，大约 28% 的请求未能满足 4 令牌/秒的 SLA。在相同的 SLA 下，DeepSpeed-FastGen 达到了 1.42 查询/秒（不到 1% 的请求未能满足 SLA），这是 vLLM 的 2.3 倍。

### D. 令牌级时间分析

图 5 显示了生成过程的 P50、P90 和 P95 延迟。vLLM 和 DeepSpeed-FlexGen 展示了类似的 P50 延迟，但 vLLM 的 P90 和 P95 延迟显著更高。

这种差异是由于 vLLM 在抢占正在进行的生成以处理新提示时，生成延迟出现显著增加所导致的。
相比之下，DeepSpeed-FastGen 通常会同时处理之前请求的提示和生成，导致生成延迟更加一致。

<div align="center">
  <img src="assets/images/token_latency.png" alt="" width="400"/><br>

  *图 5: 使用张量并行在 4 个 A100-80GB GPU 上的 Llama 2 70B/A100-80GB 的每令牌生成延迟，16 客户端。对提示和生成长度施加了正态分布，平均值分别为 2600 和 128，并且有 30% 的方差。对提示和生成长度施加了正态分布，平均值分别为 2600 和 128，并且有 30% 的方差*
</div><br>


### E. 使用负载均衡的可扩展性

DeepSpeed-FastGen 提供了副本级负载均衡，可以将请求均匀分布在多个服务器上，让您轻松扩展应用程序。

图 6 说明了在使用负载均衡器和多达 16 个副本时 DeepSpeed-FastGen 的可扩展性。请注意，我们使用了 4 个 A100 GPU 来计算 Llama 2 70B 模型。总共，我们使用了 8 个节点来运行 16 个副本。结果展示了 DeepSpeed-FastGen 几乎完美的可扩展性。
鉴于单个副本的吞吐量为 1.46 查询/秒，16 个副本的吞吐量达到了 23.7 查询/秒，与单个副本相比标志着线性的 16 倍增长。

<div align="center">
  <img src="assets/images/repl_scale_llama70b_tp4_p2600g60.png" alt="" width="400"/><br>

  *图 6: 使用负载均衡功能的可扩展性。对提示和生成长度施加了正态分布，平均值分别为 2600 和 60，并且有 30% 的方差*<br>
</div>

### F. 其他硬件平台

除了对 A100 的深入分析，我们还提供了 H100 和 A6000 的额外基准测试结果。在 A6000 和 H100 上观察到的性能趋势与 A100 相同。

<div align="center">
  <img src="assets/images/H100_benchmark.png" alt="" width="800"/><br>

  *图 7: 使用 8 个 H100 GPU 的 Llama 2 70b 的吞吐量-延迟曲线和有效吞吐量。对提示和生成长度施加了正态分布，平均值分别为 2600 和 60，并且有 30% 的方差*<br>
</div>

<div align="center">
  <img src="assets/images/A6000_benchmark.png" alt="" width="800"/><br>

  *图 8: 使用 A6000 的 Llama 2 7b 的吞吐量-延迟曲线和有效吞吐量。对提示和生成长度施加了正态分布，平均值分别为 2600 和 60，并且有 30% 的方差*<br>
</div>

## 5. DeepSpeed-FastGen：实现与使用 <a name="using-deepspeed-fastgen"></a>

DeepSpeed-FastGen 是 [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII) 和 [DeepSpeed-Inference](https://github.com/microsoft/DeepSpeed) 的协同组合，如下图所示。这两个软件包共同提供了系统的各个组成部分，包括前端 API、用于使用动态 SplitFuse 调度批次的主机和设备基础设施、优化的内核实现，以及构建新模型实现的工具。

<div align="center">
<img src="assets/images/fastgen-arch-light.png#gh-light-mode-only" width="800px">
<img src="assets/images/fastgen-arch-dark.png#gh-dark-mode-only" width="800px">
</div>


使用我们的 alpha 版 DeepSpeed-FastGen 最快的入门方式是：`pip install deepspeed-mii`。

请按照我们的 [入门指南](https://github.com/microsoft/deepspeed-mii#getting-started-with-mii) 获取更多细节。如需使用和报告问题，请使用 [DeepSpeed-MII Github 仓库](https://github.com/microsoft/DeepSpeed-MII)。

### A. 支持的模型

在 DeepSpeed-FastGen 的这个 alpha 版本中，我们目前支持以下模型架构：

* [LLaMA](https://huggingface.co/models?other=llama) 和 [LLaMA-2](https://huggingface.co/models?other=llama-2)
* [Mistral](https://huggingface.co/models?other=mistral)
* [OPT](https://huggingface.co/models?other=opt)

所有当前模型都利用了我们后端的 [HuggingFace](https://github.com/huggingface) API 来提供模型权重和模型对应的分词器。

> 我们计划在最初发布后的几周和几个月内添加更多模型。如果您希望支持特定的模型架构，请[提交问题](https://github.com/microsoft/DeepSpeed-MII/issues)并让我们知道。

### B. 部署选项
以下所有示例均可在 [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/tree/master/inference/mii) 中运行。安装后，您有两种部署方式：交互式非持久管道或持久化服务部署：

#### 非持久管道

非持久管道部署是快速入门的好方法，只需几行代码即可完成。非持久模型只在您运行的 python 脚本期间存在，但适用于临时交互式会话。

```python
from mii import pipeline
pipe = pipeline("mistralai/Mistral-7B-v0.1")
output = pipe(["Hello, my name is", "DeepSpeed is"], max_new_tokens=128)
print(output)
```

#### 持久部署

持久部署非常适合用于长时间运行和生产应用。持久部署使用了轻量级的 GRPC 服务器，可以使用以下两行代码创建：

```python
import mii
mii.serve("mistralai/Mistral-7B-v0.1")
```

上述服务器可以同时被多个客户端查询，这要归功于 DeepSpeed-MII 内置的负载平衡器。创建客户端也只需要两行代码：

```python
client = mii.client("mistralai/Mistral-7B-v0.1")
output = client.generate("Deepspeed is", max_new_tokens=128)
print(output)
```

持久部署可以在不再需要时终止：

```python
client.terminate_server()
```

### C. 高级安装信息

为了使用方便并显著减少许多此类项目所需的冗长编译时间，我们通过名为 [DeepSpeed-Kernels](https://github.com/microsoft/DeepSpeed-Kernels) 的新库分发了覆盖我们大部分自定义内核的预编译 Python 轮子。我们发现这个库在环境中非常便携，这些环境具有 NVIDIA GPU 的计算能力 8.0+（Ampere+）、CUDA 11.6+ 和 Ubuntu 20+。在大多数情况下，您甚至不需要知道这个库的存在，因为它是 DeepSpeed-MII 的依赖项，并将与之一起安装。然而，如果您因任何原因需要手动编译我们的内核，请参阅我们的[高级安装文档](https://github.com/microsoft/DeepSpeed-Kernels#source)。


# 6. 尝试 DeepSpeed-FastGen <a name="try"></a>
我们非常高兴分享这个 DeepSpeed-FastGen alpha 版本。

* 要开始，请访问我们的 DeepSpeed-MII GitHub 页面： [GitHub 登陆页面](https://github.com/microsoft/DeepSpeed-MII)

DeepSpeed-FastGen 是更大的 DeepSpeed 生态系统的一部分，该生态系统包含了多种深度学习系统和建模技术。要了解更多，

* 请访问我们的[网站](https://www.deepspeed.ai/)，详细查看博客文章、教程和有用的文档。
* 您也可以通过我们的[英文 Twitter](https://twitter.com/MSFTDeepSpeed)、[日本 Twitter](https://twitter.com/MSFTDeepSpeedJP) 和[中文知乎](https://www.zhihu.com/people/deepspeed) 关注我们，获取 DeepSpeed 的最新消息。

DeepSpeed 欢迎您的贡献！我们鼓励您在 [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed/) 页面上报告问题、贡献 PR，并参与讨论。有关更多详细信息，请参见我们的[贡献指南](https://github.com/microsoft/DeepSpeed/blob/master/CONTRIBUTING.md)。我们愿意与大学、研究实验室和公司合作，比如那些在深度学习研究上共同工作，应用 DeepSpeed 来赋能真实世界的 AI 模型和应用等。对于那些不适合在 GitHub 上提出的请求（以及其他请求），请直接发送电子邮件至 deepspeed-info@microsoft.com。

以下项目在我们的路线图上，我们计划通过我们的 GitHub 问题和 PR 与我们的社区在这些项目上进行交流：

- 性能改进
- 更广泛的模型支持
- 通过与合作伙伴的合作支持新硬件后端
- 发布性能基准（用于在此博客中生成图表）

如果您喜欢我们的工作，请为我们的 [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed/) 和 [DeepSpeedMII GitHub](https://github.com/microsoft/DeepSpeed-MII/) 仓库打上“星标”！

# 7. 致谢 <a name="acknowledgements"></a>

我们要感谢包括 HuggingFace、vLLM 和 HuggingFace TGI 在内的各种开源社区项目。我们在 alpha 版本中利用 HF API 来支持模型和分词器，并将继续添加更多模型。我们特别感谢 [Flash Attention](https://github.com/Dao-AILab/flash-attention) 开发者的出色工作。我们在系统中广泛利用了 FlashAttention 内核，并在适当的文件头部已经在我们的代码库中进行了认可。最后，我们要感谢我们在 MoE 内核中使用的 [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) 内核的开发者（作为 DeepSpeed-Kernels 仓库的一部分发布）。

