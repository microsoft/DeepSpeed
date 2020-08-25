---
layout: single
title: "DeepSpeed Sparse Transformer (Sparse Self Attention)"
excerpt: ""
categories: news
new_post: true
date: 2020-08-00 00:00:00
---

We introduce DeepSpeed Sparse Transformer, an instrumental technology to support long sequence of model inputs, whether for text, image or sound. Comparing with the classic dense transformers, it powers input sequence with an order-of-magnitude longer and obtains up to 6x faster execution with comparable accuracy; it also outperforms state-of-arts sparse implementations with 1.5 - 3x faster execution. Furthermore, our sparse kernels support efficient execution of flexible sparse format and empower users to innovate on their customized sparse structures.
This library is PyTorch based and develops required kernels through [Triton](https://github.com/ptillet/triton) platform; kernels are not written in CUDA, which leaves the door open for CPU/OpenCL/Vulkan support in the future. The library is an extension to DeepSpeed and can be used through DeepSpeed as well as stand alone.
DeepSpeed Sparse Attention kernels, handles the following block-sparse computations:
(`S` stands for a `sparse matrix` and `D` a `dense matrix`. Sparse matrix is `blocked sparse matrix`.)

* Forward:

![sparse_self_attention_forward_pass](/assets/images/sparse_self_attention_forward_pass.png){: .align-center}

	* `S = D X D          =>  w = q X trans(k)`
	* `S = softmax(S)     =>  w = softmax(w)`
	* `D = S X D          =>  a = w X v`


* Backward:

![sparse_self_attention_backward_pass](/assets/images/sparse_self_attention_backward_pass.png){: .align-center}

	* `D = S X D           =>  v  = trans(w) X da`
	* `S = D X D           =>  w  = da X trans(v)`
	* `S = bwd_softmax(S)  =>  dw = w X (dw - sum(dw X w))`
	* `D = D X S           =>  k  = q X trans(dw)`
	* `D = S X D           =>  q  = dw X k`

For Sparsity Config, and also how to use this library, please check our [tutorial](https://github.com/microsoft/DeepSpeed-internal/tree/master/docs/_tutorials/sparse_transformer.md) that provides detailed information about it.
In [result](https://github.com/microsoft/DeepSpeed-internal/tree/master/docs/_tutorials/sparse_transformer_results.md) section, we provide a summary of our experiments using this library. This section also contains experiments comparing our library with [Longformer](https://arxiv.org/abs/2004.05150); state-of-arts sparse implementation of sparsity pattern.
