---
layout: single
title: "DeepSpeed Sparse Transformer - experimental results)"
excerpt: ""
categories: news
new_post: true
date: 2020-08-00 00:00:00
---

* **Power 10x longer sequences**
We have run a pre-training experiment using BERT model, both base and large, with batch size of 1 on a single Nvidia V100 GPU-32GB memory. Following table shows maximum possible sequence length for:
	* original dense model
	* + adding checkpointing
	* and replacing self-attention with sparse self-attention

|Max SeqLen	|Dense	|Dense\_chpt|Sparse\_chpt|
|---------------|-------|-----------|------------|
|BERT Base	|4k	|14k	    |39k         |
|BERT Large	|2k	|12k	    |32k         |

* **up to 6.3x faster computation**
Further, we ran a pre-training experiment for different batch sizes and sequence lengths, using [BERT base/large](https://github.com/microsoft/DeepSpeedExamples/tree/fd869ae1c9de686f8cb92413efeba83fc989027c/bing_bert) model and [Megatron GPT2](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM). In this experiment we let the experiment to continue for 100 iteration and recorded the average time per last 30 iterations. Following charts shows these results in which we can see up to `6.3X`, `5.3X`, and `6.1X` speed up for BERT Base, BERT large and GPT2 models respectively.

![st_bert_base_result](/assets/images/st_bert_base_result.png){: .align-center}
![st_bert_large_result](/assets/images/st_bert_result_result.png){: .align-center}
![st_megatron_gpt2_result](/assets/images/st_megatron_gpt2_result.png){: .align-center}

* **higher accuracy**
In addition to lower memory overhead and faster computation, interestingly we have seen higher accuracy and faster convergence. As mentioned in the original paper, this may point to a useful inductive bias introduced by sparsity or an underlying optimization issue in full attention. In the following chart we show accuracy of fine-tuning three models for long document comprehension. In this application we use sequence length of 2k. What we have observed, pre-training from scratch, sparse transformer is much faster and converges with the higher accuracy. Further, fine-tuning from a pre-trained checkpoint, we gain better performance in the point of time and accuracy. In how to use section we will describe how to extend a pre-trained model with smaller sequence lengths, to be used fine-tuning with sparse transformer.

![st_deepthink_result](/assets/images/st_deepthink_result.png){: .align-center}

* **flexibility to handle any block-sparse structure**
DeepSpeed sparse kernels support efficient execution of flexible sparse format and empower users to innovate on their customized sparse structures. Currently DeepSpeed has added the following sparsity structures:
	* `Fixed`
	* `BigBird`
	* `BSLongformer` (Block-Sparse Longformer)
	* `Variable` (Can be used to simply customize any block-sparse random/local/global attention pattern.)
In addition to this list, user can add any other sparsity structure as described in [tutorial](https://github.com/microsoft/DeepSpeed-internal/tree/master/docs/_tutorials/sparse_transformer.md) section.


# Comparison with Longformer
[Longformer](https://arxiv.org/abs/2004.05150) is another sparse transformer work that leverages the idea of local and global notion sparsifying the attention matrix. Local attention is satisfied through local sliding window attention, and global attention through global task specific attention. Longformer has been applied to multiple downstream tasks where it shows superior result compared to SOTA. Considering that these kind of works sparsify the attention matrix similarly, we expect comparable accuracy result. This is what we have observed through few experiments that we will show in the following. However, we have noticed higher speed in DeepSpeed Sparse Self-Attention compared to Longformer.
In the first experiment, we have followed the [notebook](https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb) offered by Longformer. In this experiment, we pre-train an MLM model using RoBERTa-base checkpoint. This is done on 8 GPU V100-SXM2 node. Following table shows the result in which using DeepSpeed Sparse Self-Attention, we see up to **1.47X** speed-up.

|Model 	            |Window - Stride |BPC     |Train Step  |Time Per Iteration  |Time Improvement  |Accuracy improvement  |
|-------------------|----------------|--------|------------|--------------------|------------------|----------------------|
|RoBERTa Checkpoint |                |2.5326  |                                                                           |
|Longformer 	    |512 	     |2.6535  |0           |                    |1.47              |1.01                  |
|Sparse Transformer |	             |2.6321  |		   |		        |                  |                      |
|Longformer 	    |                |1.6708  |3k	   |1.6280		|                  |1.01                  |
|Sparse Transformer |		     |1.6613  |            |1.1059              |                  |	                  |
|Longformer         |64              |5.7840  |0           |                    |1.31              |1.46                  |
|Sparse Transformer |                |3.9737  |            |                    |                  |                      |
|Longformer         |                |2.0466  |3k          |1.4855              |                  |1.09                  |
|Sparse Transformer |                |1.8693  |            |1.1372              |                  |                      |


Through our DeepThink application we described above, we also checked the inference time for different window sizes testing BERT model on a `2k` Sequence Length and batch size `1`. In this experiment, we noticed up to `3.13X` speed up replacing Bert Self-Attention with DeepSpeed Sparse Self-Attention instead of Longformer Self-Attention. Following table shows the complete result.

|Window Size / Stride  |Time Improvement|
|----------------------|----------------|
|512                   |3.13            |
|256                   |2.29            |
|128                   |2.16            |
|64                    |1.5             |
|32                    |1.24            |
|16                    |1.23            |
