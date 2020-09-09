---
layout: single
title: "Up to 5x less communication and 3.4x faster training through 1-bit Adam"
excerpt: ""
categories: news
new_post: true
date: 2020-09-09 00:00:00
---


Adam is an effective and probably the most well-utilized optimizer for
training many large-scale deep learning models.  However, Adam is generally
not compatible with communication-efficient optimization algorithms, and
therefore the communication cost could become a bottleneck while scaling
across distributed devices. We introduce a new algorithm - 1-bit Adam - and
its efficient implementation, which reduces communication volume by 5x while
achieving similar convergence efficiency as Adam. We observe over 3.4x faster
distributed training for communication intensive models on bandwidth-limited
clusters.

* Brief overview, see our [press release](https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/).
* Detailed technology deep dive, see our [blog post](https://www.deepspeed.ai/news/2020/09/09/onebit-adam-blog-post.md).
* Tutorial on how to reproduce our results, see our [BERT pre-training tutorial](https://www.deepspeed.ai/tutorials/onebit-adam/).
* The source code for 1-bit Adam can be found in the [DeepSpeed repo](https://github.com/microsoft/deepspeed) and example codes to try this feature can be found in the [DeepSpeedExamples repo](https://github.com/microsoft/deepspeedexamples).
