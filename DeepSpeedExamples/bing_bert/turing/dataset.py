# __AUTHOR__    :   SAKSHAM SINGHAL
# __EMAIL__     :   SAKSINGH@MICROSOFT.COM

import torch
import os
from torch.utils.data import DataLoader, Dataset
from enum import IntEnum
from random import choice
import random
import collections
import time

from turing.utils import namedtorchbatch
from turing.text import mask, torch_long, PAD
from turing.sources import QueryPassageDataset, QueryInstanceDataset, \
    PretrainingDataCreator, TokenInstance, QueryPassageFineTuningDataset, \
    WikiNBookCorpusPretrainingDataCreator, CleanBodyDataCreator, \
    NumpyPretrainingDataCreator
from turing.sources import WikiPretrainingDataCreator
from pytorch_pretrained_bert.tokenization import BertTokenizer


class BatchType(IntEnum):
    RANKING_BATCH = 0
    QP_BATCH = 1
    PRETRAIN_BATCH = 2


class PretrainDataType(IntEnum):
    NUMPY = 0
    VALIDATION = 1


MaskedLMInstance = collections.namedtuple("MaskedLMInstance",
                                          ["index", "label"])

QABatch = collections.namedtuple(
    'QABatch', ['input_ids', 'input_mask', 'sequence_ids', 'label'])

RankingBatch = collections.namedtuple(
    'RankingBatch', ['input_ids', 'input_mask', 'sequence_ids', 'label'])

PretrainBatch = collections.namedtuple('PreTrainBatch', [
    'input_ids', 'input_mask', 'sequence_ids', 'is_next_label',
    'masked_lm_output'
])


class BertJobType(IntEnum):
    """Enumerates the various tasks that we will be running
    """
    QA_TASK = 0  # This is Q-P pair prediction
    MLM = 1  # Masking LM for captions data
    NSP = 1  # Next Sentence Prediction task


def get_random_partition(data_directory, index):
    partitions = [
        os.path.join(data_directory, x) for x in os.listdir(data_directory)
    ]
    partitions = sorted(partitions)
    i = index % len(partitions)
    return partitions[i]


def map_to_torch(encoding):
    encoding = torch_long(encoding)
    encoding.requires_grad_(False)
    return encoding


def map_to_torch_float(encoding):
    encoding = torch.FloatTensor(encoding)
    encoding.requires_grad_(False)
    return encoding


def map_to_torch_half(encoding):
    encoding = torch.HalfTensor(encoding)
    encoding.requires_grad_(False)
    return encoding


def encode_sequence(seqA, seqB, max_seq_len, tokenizer):
    seqA = ["[CLS]"] + seqA + ["[SEP]"]
    seqB = seqB + ["[SEP]"]

    input_tokens = seqA + seqB
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    sequence_ids = [0] * len(seqA) + [1] * len(seqB)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_len:
        input_ids.append(PAD)
        sequence_ids.append(PAD)
        input_mask.append(PAD)

    return (map_to_torch(input_ids), map_to_torch(input_mask),
            map_to_torch(sequence_ids))


def truncate_input_sequence(tokens_a, tokens_b, max_num_tokens):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


class QADataset(Dataset):
    def __init__(self, tokenizer: BertTokenizer, folder: str, logger,
                 max_seq_len, index):
        self.tokenizer = tokenizer
        self.dir_path = folder
        self.max_seq_len = max_seq_len
        self.len = 0

        path = get_random_partition(self.dir_path, index)

        logger.info(f"Loading Query-Passage Pairs from {path}")
        self.data = QueryPassageDataset(path)
        self.len = len(self.data)
        logger.info(
            f"Data Loading Completed for Query-Passage Pairs from {path} with {self.len} samples."
        )

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        i = index % self.len

        query, passage, label = self.data.all_pairs[i]
        label = float(label)

        # sample_choice = choice([0, 1])

        # if sample_choice == 0:  # generate negative sample
        #     query, passage = self.data.all_pairs[i][0], self.data.all_pairs[i+1][1]
        # else:  # generate positive sample
        #     query, passage = self.data.all_pairs[i]

        query_tokens = self.tokenizer.tokenize(query)
        passage_tokens = self.tokenizer.tokenize(passage)

        if (len(query_tokens) > self.max_seq_len // 2):
            query_tokens = query_tokens[0:self.max_seq_len // 2]

        max_passage_tokens = self.max_seq_len - \
            len(query_tokens) - 3  # Removing 3 for SEP and CLS

        if (len(passage_tokens) > max_passage_tokens):
            passage_tokens = passage_tokens[0:max_passage_tokens]

        input_ids, input_mask, sequence_ids = encode_sequence(
            query_tokens, passage_tokens, self.max_seq_len, self.tokenizer)
        return tuple([
            map_to_torch([BatchType.QP_BATCH]), input_ids, input_mask,
            sequence_ids,
            map_to_torch_float([label])
        ])
        # return QABatch(input_ids=input_ids, input_mask=input_mask, sequence_ids=sequence_ids, label=map_to_torch([label]))


class QAFinetuningDataset(QADataset):
    def __init__(self, tokenizer: BertTokenizer, file_path, logger,
                 max_seq_len):
        self.tokenizer = tokenizer
        self.path = file_path
        self.max_seq_len = max_seq_len
        self.len = 0

        logger.info(f"Loading Query-Passage Pairs from {self.path}")
        self.data = QueryPassageFineTuningDataset(self.path)
        self.len = len(self.data)
        logger.info(
            f"Data Loading Completed for Finetuning Query-Passage Pairs from {self.path} with {self.len} samples."
        )


class RankingDataset(Dataset):
    def __init__(self,
                 tokenizer: BertTokenizer,
                 folder: str,
                 logger,
                 max_seq_len,
                 index,
                 fp16=False):
        self.tokenizer = tokenizer
        self.dir_path = folder
        self.max_seq_len = max_seq_len
        self.len = 0
        self.fp16 = fp16

        path = get_random_partition(self.dir_path, index)

        logger.info(f"Loading Query-Instance Pairs from {path}")
        self.data = QueryInstanceDataset(path)
        self.len = len(self.data)
        logger.info(
            f"Data Loading Completed for Query-Instance Pairs from {path} with {self.len} samples."
        )

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        i = index % self.len

        query, instance, label = self.data.all_pairs[i]
        label = float(label)

        instances = instance.split('<sep>')

        query_tokens = self.tokenizer.tokenize(query)
        instances = [self.tokenizer.tokenize(x) for x in instances]
        instance_tokens = []
        for x in instances:
            instance_tokens.extend(x)
            instance_tokens.append('[SEP]')

        instance_tokens = instance_tokens[:-1]
        # instance_tokens = self.tokenizer.tokenize(instance)

        if (len(query_tokens) > self.max_seq_len // 2):
            query_tokens = query_tokens[0:self.max_seq_len // 2]

        max_instance_tokens = self.max_seq_len - \
            len(query_tokens) - 3  # Removing 3 for SEP and CLS

        if (len(instance_tokens) > max_instance_tokens):
            instance_tokens = instance_tokens[0:max_instance_tokens]

        input_ids, input_mask, sequence_ids = encode_sequence(
            query_tokens, instance_tokens, self.max_seq_len, self.tokenizer)
        return tuple([
            map_to_torch([BatchType.RANKING_BATCH]), input_ids, input_mask,
            sequence_ids,
            map_to_torch_float([label])
        ])


class PreTrainingDataset(Dataset):
    def __init__(self,
                 tokenizer: BertTokenizer,
                 folder: str,
                 logger,
                 max_seq_length,
                 index,
                 data_type: PretrainDataType = PretrainDataType.NUMPY,
                 max_predictions_per_seq: int = 20):
        self.tokenizer = tokenizer
        self.dir_path = folder
        self.max_seq_length = max_seq_length
        self.len = 0
        self.masked_lm_prob = 0.15
        self.max_predictions_per_seq = max_predictions_per_seq
        self.vocab_words = list(tokenizer.vocab.keys())

        path = get_random_partition(self.dir_path, index)

        logger.info(f"Loading Pretraining Data from {path}")
        start = time.time()
        # logger.info(f"Loading Pretraining Data from {path}")
        # if data_type == PretrainDataType.CLEAN_BODY:
        #     self.data = CleanBodyDataCreator.load(path)
        # elif data_type == PretrainDataType.WIKIPEDIA or data_type == PretrainDataType.BOOK_CORPUS:
        #     self.data = WikiNBookCorpusPretrainingDataCreator.load(path)
        if data_type == PretrainDataType.VALIDATION:
            self.data = WikiPretrainingDataCreator.load(path)
        elif data_type == PretrainDataType.NUMPY:
            self.data = NumpyPretrainingDataCreator.load(path)
        self.len = len(self.data)
        logger.info(
            f"Data Loading Completed for Pretraining Data from {path} with {self.len} samples took {time.time()-start:.2f}s."
        )

        self.len = len(self.data)
        logger.info(
            f"Data Loading Completed for Pretraining Data from {path} with {self.len} samples."
        )

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        i = index % self.len

        instance: TokenInstance = self.data.instances[i]
        return self.create_training_instance(instance)

    def create_training_instance(self, instance: TokenInstance):
        tokens_a, tokens_b, is_next = instance.get_values()
        # print(f'is_next label:{is_next}')
        # Create mapper
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)

        tokens.append("[SEP]")
        segment_ids.append(1)

        # Get Masked LM predictions
        tokens, masked_lm_output = self.create_masked_lm_predictions(tokens)

        # Convert to Ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_seq_length:
            input_ids.append(PAD)
            segment_ids.append(PAD)
            input_mask.append(PAD)
            masked_lm_output.append(-1)
        return ([
            map_to_torch([BatchType.PRETRAIN_BATCH]),
            map_to_torch(input_ids),
            map_to_torch(input_mask),
            map_to_torch(segment_ids),
            map_to_torch([is_next]),
            map_to_torch(masked_lm_output)
        ])

    def create_masked_lm_predictions(self, tokens):
        cand_indexes = []
        for i, token in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            cand_indexes.append(i)

        random.shuffle(cand_indexes)
        output_tokens = list(tokens)

        num_to_predict = min(
            self.max_predictions_per_seq,
            max(1, int(round(len(tokens) * self.masked_lm_prob))))

        masked_lms = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% mask
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% Keep Original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% replace w/ random word
                else:
                    masked_token = self.vocab_words[random.randint(
                        0,
                        len(self.vocab_words) - 1)]

            output_tokens[index] = masked_token
            masked_lms.append(
                MaskedLMInstance(index=index, label=tokens[index]))

        masked_lms = sorted(masked_lms, key=lambda x: x.index)
        masked_lm_output = [-1] * len(output_tokens)
        for p in masked_lms:
            masked_lm_output[p.index] = self.tokenizer.vocab[p.label]

        return (output_tokens, masked_lm_output)
