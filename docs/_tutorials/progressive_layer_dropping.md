---
title: "Accelerating Training of Transformer-Based Language Models with Progressive Layer Dropping"

---

In this tutorial, we are going to introduce the progressive layer dropping (PLD) in DeepSpeed. PLD allows to train Transformer networks such as BERT 24% faster under the same number of samples and 2.5 times faster to get similar accuracy on downstream tasks. Detailed description of PLD is available from our [technical report](XXX).

To illustrate the benefits and usage of PLD in DeepSpeed, we first use PLD to pre-train a BERT model and fine-tune the pre-trained model on the GLUE datasets.

## Running Pre-training with DeepSpeed and PLD

For data downloading and pre-processing, please refer to the [BERT Pre-training](/tutorials/bert-pretraining/) post.

The main part of pre-training is done in `deepspeed_train.py`, which has
already been modified to use DeepSpeed. The  `ds_train_bert_progressive_layer_drop_bsz4k_seq128.sh` is the shell script that launches the pre-training with DeepSpeed and PLD.

```shell
bash ds_train_bert_progressive_layer_drop_bsz4k_seq128.sh
```

Most of the flags in the above script should be familiar if you have stepped through the BERT pre-training [tutorial](/tutorials/bert-pretraining/). To enable PLD, one needs to add the following flags.  The first flag enables progressive layer dropping on Transformer blocks. The second flag determines the progressive drop schedule. We recommend 0.5, a value that worked well in our experiments. 

    --progressive_layer_drop --layerdrop_theta 0.5

Setting these flags should print a message as below:

    Enabled progressive layer dropping (theta = 0.5). 

The `deepspeed_bsz4k_progressive_layer_drop_config_seq128.json` file allows users to specify DeepSpeed
options in terms of batch size, micro batch size, optimizer, learning rate, and other parameters.

Below is the DeepSpeed configuration file we use for running BERT with PLD.

```json
{
  "train_batch_size": 4096,
  "train_micro_batch_size_per_gpu": 16,
  "steps_per_print": 1000,
  "prescale_gradients": true,
  "gradient_predivide_factor": 8,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-3,
      "weight_decay": 0.01,
      "bias_correction": false
    }
  },
  "gradient_clipping": 1.0,

  "wall_clock_breakdown": false,

  "fp16": {
    "enabled": true,
    "loss_scale": 0
  }
}
```

Table 1 shows the hyperparameters we use for pre-training with PLD enabled.

| Parameters                     | Value                   |
| ------------------------------ | ----------------------- |
| Effective batch size           | 4K                      |
| Train micro batch size per GPU | 16                      |
| Optimizer                      | Adam                    |
| Peak learning rate             | 1e-3                    |
| Sequence-length                | 128                     |
| Learning rate scheduler        | Warmup linear decay exp |
| Warmup ratio                   | 0.02                    |
| Decay rate                     | 0.99                    |
| Decay step                     | 1000                    |
| Weight decay                   | 0.01                    |
| Gradient clipping              | 1.0                     |

Table 1. Pre-training hyperparameters

## Fine-tuning with DeepSpeed

We use GLUE for fine-tuning tasks. GLUE (General Language Understanding Evaluation benchmark) (https://gluebenchmark.com/) is a  collection of sentence or sentence-pair natural language understanding tasks including question answering, sentiment analysis, and textual entailment.  It is designed to favor sample-efficient learning and knowledge-transfer across a range of different linguistic tasks in different domains. 

You can download all GLUE data using the helper [script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e).

We use the PLD pre-trained BERT model checkpoint to run the fine-tuning.

The main part of fine-tuning is done in `run_glue_classifier_bert_base.py`, which has
already been modified to use DeepSpeed. The `run_glue_classifier_bert_base.sh` script
invokes pre-training and setup several hyperparameters relevant
to fine-tuning.