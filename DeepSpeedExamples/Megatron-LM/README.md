Megatron is a large, powerful transformer. This repo is for ongoing research on training large, powerful transformer language models at scale. Currently, we support model-parallel, multinode training of [GPT2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and [BERT](https://arxiv.org/pdf/1810.04805.pdf) in mixed precision. 

Our codebase is capable of efficiently training a 72-layer, 8.3 Billion Parameter GPT2 Language model with 8-way model and 64-way data parallelism across 512 GPUs. We find that bigger language models are able to surpass current GPT2-1.5B wikitext perplexities in as little as 5 epochs of training.

For BERT training our repository trains BERT Large on 64 V100 GPUs in 3 days. We achieved a final language modeling perplexity of 3.15 and SQuAD F1-score of 90.7.
<!--
do we want to make any claims about GPT2 speed, convergence, or model release
-->

# Setup
We officially support only python3.6.

To use this repo please install the latest supported versions of PyTorch with GPU support. 

Additionally, part of this codebase leverages tensorflow-cpu to (optionally) perform dataloading of TFRecords for BERT training. We recommend either utilizing the provided Dockerfile in [`./docker/`](./docker) or creating a virtual environment (to avoid breaking existing tf installations) and install our `requirements.txt`. 

```
python -m pip install virtualenv
virtualenv bert_env
source bert_env/bin/activate
pip install -r requirements.txt
```


# Usage
We've provided 5 scripts that pretrain BERT and 3 scripts that pretrain GPT2. Save and load model checkpoints with `--save` and `--load`. Additionally we provide GPT2 scripts for interactive text generation and zero shot evaluation of GPT2 on wikitext and LAMBADA.

## BERT Pretraining
`bash scripts/pretrain_bert.sh`

This script runs single gpu BERT pretraining and is mainly for debugging purposes. The optimization arguments are set with 64-way distributed training in mind.

To use this script place your `--train-data` in loose json format with one json per line. The text field of your json dictionaries should correspond to `--text-key`. 

```
python pretrain_bert.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 4 \
       --seq-length 512 \
       --max-preds-per-seq 80 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --save checkpoints/bert_345m \
       --load checkpoints/bert_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-large-uncased \
       --presplit-sentences \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
       --fp32-embedding
```

## GPT2 Pretraining
`bash scripts/pretrain_gpt2.sh`

This script runs single gpu gpt2 pretraining and is mainly for debugging purposes. The optimization arguments are set with 64-way distributed training in mind. 

It follows largely the same format as the previous script with a few notable differences: the `--tokenizer-type` has been switched to a `GPT2BPETokenizer`, the `--lr-decay-style` has been switched to cosine decay, and activation checkpointing has been turned on with `--checkpoint-activations` and `--checkpoint-num-layers` set to checkpoint every `1` layers.

Additionally GPT2 uses a different parameter initialization from BERT designed for training deep residual networks. To train BERT with this initialization use `--deep-init`.

```
python pretrain_gpt2.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 8 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 320000 \
       --save checkpoints/gpt2_345m \
       --load checkpoints/gpt2_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --fp16
```

## GPT2 Text Generation
`bash scripts/generate_text.sh`

Starts an interactive terminal session that generates text either conditionally or unconditionally depending on what the user enters into the prompt. Specify the model in the script by setting the `CHECKPOINT_PATH` variable and the appropriate model configuration. 

The script is capable of greedy sampling, top-k, or top-p sampling as specified by the appropriate variables within the script.

## GPT2 Evaluation
We support 3 modes of GPT2 evaluation with [`./scripts/run_gpt2_eval.py`](./scripts/run_gpt2_eval.py): wikitext ppl evaluation, lambada cloze accuracy, large corpora ppl evaluation.

### Wikitext PPL evaluation
For even comparison with prior works we evaluate wikitext perplexity on the word-level wikitext test dataset, which can be downloaded [here](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip), and appropriately compute perplexity given the change in tokens when using our subword tokenizer.

We use the following command to run wikitext evaluation:

```
python scripts/run_gpt2_eval.py \
  --model-parallel-size 1 \
  --num-layers 24 \
  --hidden-size 1024 \
  --num-attention-heads 16 \
  --model-path <gpt2_345_path> \
  --data-path <wikitext_tokens_test_path> \
  --batch-size 16 \
  --cache-dir cache
```

### Lambada Cloze Accuracy
To compute Lambada cloze accuracy (the accuracy of predicting the last token given the preceding tokens) we utilize a detokenized, processed version of the Lambada dataset we sourced from [here](https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl).

We use the following command to run lambada evaluation:

```
python scripts/run_gpt2_eval.py \
  --model-parallel-size 1 \
  --num-layers 24 \
  --hidden-size 1024 \
  --num-attention-heads 16 \
  --model-path <gpt2_345_path> \
  --data-path <lambada_test_path> \
  --batch-size 16 \
  --cloze-eval \
  --cache-dir cache
```

### Large Corpora PPL evaluation
This functionality allows one to evaluate the gpt2 model on a loose json file. With the following command we evaluate the gpt2 model for 5000 iterations at a batch size of 16 on a webtext test data split. We recommend that the user presplit their dataset before training a model according to the procedure outlined [below](#partitioning-datasets-into-train-val-test).

```
python scripts/run_gpt2_eval.py \
  --model-parallel-size 1 \
  --num-layers 24 \
  --hidden-size 1024 \
  --num-attention-heads 16 \
  --model-path <gpt2_345_path> \
  --data-path <webtext_test_path> \
  --batch-size 16 \
  --eval-iters 5000 \
  --webtext-eval \
  --cache-dir cache
```

## Distributed BERT or GPT2 Pretraining
`bash scripts/pretrain_bert_distributed.sh` or `bash scripts/pretrain_gpt2_distributed.sh`

To use these scripts, follow the same data preparation procedure as in earlier sections. This script uses the pytorch distributed launcher to launch distributed training. As such, multinode training can be achieved by properly setting environment variables for the `env://` init method. See the official pytorch [documentation](https://pytorch.org/docs/stable/distributed.html#launch-utility) for further description of these [environment variables](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization). By default multinode training uses the nccl distributed backend.

## Model Parallel BERT or GPT2 Pretraining
`bash scripts/pretrain_bert_model_parallel.sh` or `bash scripts/pretrain_gpt2_model_parallel.sh`

These scripts build upon the distributed training scripts and are identical in setup. They differ in use of the `--model-parallel-size` flag. For model parallelism of 2 and a world size of 8, the scripts will launch training with 4-way distributed data parallelism and 2-way model parallelism.

We note that we have experimented with multiple distributed data parallel implementations: a simple one of our own which performs gradient all-reduce at the end of back propagation step, and torch's distributed data parallel wrapper which overlaps gradient reduction with back propagation computation. To switch between these two options toggle the `USE_TORCH_DDP` flag (the default is set to `False` and uses our DDP implementation) at the top of `pretrain_bert.py` and `pretrain_gpt2.py`. We find that torch distributed data parallelism is more efficient at larger model parallel sizes. For example, for the 8.3 billion parameters model running on 512 GPUs, the scaling increases from 60% to 74% when torch's distributed data parallel is used. However, the overlapping method requires more memory and for some configurations (e.g., 2.5 billion parameters using 2-way model parallel and 1.2 billion parameters with no model parallel) can make the overall training slower as a result. We empirically found that using a smaller model in those cases improves the training time.

## Distributed BERT Pretraining with TFRecords
`bash scripts/pretrain_bert_tfrecords_distributed.sh`

This script takes advantage of TensorFlow BERT's [`create_pretraining.py`](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/create_pretraining_data.py) script to pre-cache the dataset in the TFRecord format. To convert the data to pytorch tensors we use a `TFRecordDataset` and tensorflow eager mode to turn the TFRecords into numpy matrices before loading them into pytorch gpu tensors. This greatly reduces the overhead of dataprocessing and speeds up training. Pass a whitespace-separated list of TFRecord paths to `--train-data` and enable the `--use-tfrecords` flag. Multinode training can be achieved as described in the [previous section](#distributed-bert-pretraining).

## Train Custom Sentence Piece Tokenizer and Pretrain BERT
`bash scripts/pretrain_bert_sentencepiece.sh`

This script runs BERT pretraining with a `sentencepiece` tokenizer. If no sentencepiece tokenizer exists at `--tokenizer-path` one will be trained automatically. The sentencepiece tokenizer can be used with the previous scripts (NOTE: sentencepiece training can only happen during single gpu pretraining). `<--tokenizer-path>.vocab` can be used with [`create_pretraining_data.py`](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/create_pretraining_data.py) to make a TFRecord dataset with the given tokenization.


# Data sets
We do not host any datasets for GPT2 or BERT training, however, we detail their collection so that our results may be reproduced.

## Collecting Wikipedia Training Data
We recommend following the wikipedia data extraction process specified by google research: "the recommended pre-processing is to download [the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2), extract the text with [WikiExtractor.py](https://github.com/attardi/wikiextractor), and then apply any necessary cleanup to convert it into plain text." 

We recommend using the `--json` argument when using WikiExtractor, which will dump the wikipedia data into loose json format (one json per line), making it more manageable and readily consumable by our codebase. We recommend further preprocessing this json dataset by preprocessing the dataset with nltk punctuation standardization, and presplitting each document into newline separated sentences. This can be done with the provided script `./scripts/presplit_sentences_json.py` and will allow for faster data processing during training time. Pretraining with presplit data should be run with the `--presplit-sentences` flag as shown above. (Note that if you'd like to use wikipedia data for GPT2 training you should still clean it with nltk/spacy/ftfy, but do not split it into newline seperated sentences)

Once the json dataset is ready make sure to set the path in line 27 of `data_utils/corpora.py`.

If your system is memory limited we also recommend running pretraining with the `--lazy-loader` argument as we've done. After preprocessing the dataset once, this will allow the dataset to be lazily loaded from disk, as opposed to storing it in memory. Make sure to run the code once on a 

## Collecting GPT2 Webtext Data
We utilize the publicly available [OpenWebText](https://github.com/eukaryote31/openwebtext) library from [jcpeterson](https://github.com/jcpeterson/openwebtext) and [eukaryote31's](https://github.com/eukaryote31/openwebtext) work to download urls. We then filtered, cleaned, and deduplicated all downloaded content according to the procedure described in our [openwebtext](./openwebtext) directory. For reddit URLS corresponding to content upto october 2018 we arrived at approximately 37GB of content.

We recommend creating an alias for this dataset as described below.

## Aliasing datasets with corpora.py
As mentioned in the previous Wikipedia data section we recommend aliasing datasets with human readable names (eg. `--train-data wikipedia`). This helps avoid forgetting arguments when submitting jobs, and allows one to combine datasets that would otherwise require different commandline options/data structures.

Examples of how to create these dataset objects can be found in [`./data_utils/corpora.py`](./data_utils/corpora.py). We recommend that the objects inherit from or adhere to the interface laid out by `torch.utils.data.Dataset` objects.

Any created datasets should be then added to the `NAMED_CORPORA` dictionary object in [`./data_utils/corpora.py`](./data_utils/corpora.py). At runtime one can specify one or more corpora from the commandline with `--train-data corpus1 corpus2 corpus3`, `--valid-data corpus1 corpus2 corpus3`, or `--test-data ...`.

## Partitioning datasets into Train/Val/Test
We support multiple ways to partition corpora into train/val/test splits. By specifying a `--split 95,5` commandline argument, the corpora specified by `--train-data` will have it's documents split proportionally into a 95%, 5% train/val split. The split is performed lazily on the fly and is efficient and deterministic from run to run given the same `--seed`. Note that if `--valid-data` or `--test-data` is specified then the train data will still be split accordingly, but `--valid-data`/`--test-data` will still be used as the validation/test source.

We do realize that this method, while effective, introduces noise into the development process, since different seeds will change the dataset and outcome. To have fixed training/validation/test sets across all your runs please utilize our script [`./scripts/split_json.py`](./scripts/split_json.py)
