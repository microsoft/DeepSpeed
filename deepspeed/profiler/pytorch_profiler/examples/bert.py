from functools import partial

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from deepspeed.profiler.pytorch_profiler import get_model_profile


def bert_input_constructor(input_shape, tokenizer):
    inp_seq = ""
    for _ in range(input_shape[1] - 2):  # there are two special tokens [CLS] and [SEP]
        inp_seq += tokenizer.pad_token  # let's use pad token to form a fake
    # sequence for subsequent flops calculation

    inputs = tokenizer([inp_seq] * input_shape[0],
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    labels = torch.tensor([1] * input_shape[0])
    # Batch size input_shape[0], sequence length input_shape[128]
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    return inputs


if __name__ == '__main__':
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    macs, params, steps = get_model_profile(
        model,
        (2, 128),
        input_constructor=partial(bert_input_constructor,
                                  tokenizer=bert_tokenizer),
        print_profile=True,
        print_aggregated_profile=True,
    )
    print("{:<30}  {:<8}".format("Number of multiply-adds: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
    print("{:<30}  {:<8}".format("Number of steps profiled: ", steps))

# Output:
# Number of multiply-adds:        21.74 GMACs
# Number of parameters:           109.48 M
