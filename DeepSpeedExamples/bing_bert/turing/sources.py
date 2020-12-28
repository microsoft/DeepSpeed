from tqdm import tqdm
from typing import Tuple
from random import shuffle
import pickle
import random
import numpy as np
from pathlib import Path

from pytorch_pretrained_bert.tokenization import BertTokenizer


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


class TokenInstance:
    """ This TokenInstance is a obect to have the basic units of data that should be
        extracted from the raw text file and can be consumed by any BERT like model.
    """
    def __init__(self, tokens_a, tokens_b, is_next, lang="en"):
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # 0 is if in continuation, 1 if is random
        self.lang = lang

    def get_values(self):
        return (self.tokens_a, self.tokens_b, self.is_next)

    def get_lang(self):
        return self.lang


class QueryPassageDataset:
    def __init__(self, path, readin=20000000):
        all_pairs = []
        with open(path, encoding="utf-8") as fd:
            for i, line in enumerate(tqdm(fd)):
                line = line.replace('\n', '')
                qpl_tuple: Tuple[str, str, str] = line.split('\t')
                all_pairs.append(qpl_tuple)
                if i > readin:
                    break

        shuffle(all_pairs)
        self.all_pairs = all_pairs
        self.len = len(self.all_pairs)

    def __len__(self):
        return self.len


class QueryPassageFineTuningDataset:
    def __init__(self, path, readin=20000000):
        all_pairs = []
        with open(path, encoding="utf-8") as fd:
            for i, line in enumerate(tqdm(fd)):
                line = line.replace('\n', '')
                entities = line.split('\t')
                qpl_tuple: Tuple[str, str,
                                 str] = (entities[0], entities[2], entities[4])
                all_pairs.append(qpl_tuple)
                if i > readin:
                    break

        shuffle(all_pairs)
        self.all_pairs = all_pairs
        self.len = len(self.all_pairs)

    def __len__(self):
        return self.len


class QueryInstanceDataset:
    def __init__(self, path, readin=20000000):
        all_pairs = []
        with open(path, encoding="utf-8") as fd:
            for i, line in enumerate(tqdm(fd)):
                line = line.replace('\n', '')
                qpl_tuple: Tuple[str, str, str] = line.split('\t')
                all_pairs.append(qpl_tuple)
                if i > readin:
                    break

        shuffle(all_pairs)
        self.all_pairs = all_pairs
        self.len = len(self.all_pairs)

    def __len__(self):
        return self.len


class PretrainingDataCreator:
    def __init__(self,
                 path,
                 tokenizer: BertTokenizer,
                 max_seq_length,
                 readin: int = 2000000,
                 dupe_factor: int = 5,
                 small_seq_prob: float = 0.1):
        self.dupe_factor = dupe_factor
        self.max_seq_length = max_seq_length
        self.small_seq_prob = small_seq_prob

        documents = []
        instances = []
        with open(path, encoding='utf-8') as fd:
            for i, line in enumerate(tqdm(fd)):
                line = line.replace('\n', '')
                # Expected format (Q,T,U,S,D)
                # query, title, url, snippet, document = line.split('\t')
                # ! remove this following line later
                document = line
                if len(document.split("<sep>")) <= 3:
                    continue
                lines = document.split("<sep>")
                document = []
                for seq in lines:
                    document.append(tokenizer.tokenize(seq))
                # document = list(map(tokenizer.tokenize, lines))
                documents.append(document)

        documents = [x for x in documents if x]

        self.documents = documents
        for _ in range(self.dupe_factor):
            for index in range(len(self.documents)):
                instances.extend(self.create_training_instance(index))

        shuffle(instances)
        self.instances = instances
        self.len = len(self.instances)
        self.documents = None
        documents = None

    def __len__(self):
        return self.len

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save(self, filename):
        with open(filename, 'wb') as outfile:
            pickle.dump(self, outfile)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def create_training_instance(self, index):
        document = self.documents[index]
        # l = 0
        # for s in document:
        #     l+=len(s)
        # print(l)
        # print(document)

        # Need to add [CLS] + 2*[SEP] tokens
        max_num_tokens = self.max_seq_length - 3

        # We want to maximize the inp sequence but also want inputs similar
        # to our generic task inputs which will be compartively smaller
        # than the data on which we intend to pre-train.
        target_seq_length = max_num_tokens
        if random.random() < self.small_seq_prob:
            target_seq_length = random.randint(5, max_num_tokens)

        # Need to make the sequences split for NSP task for interesting
        # rather than choosing some arbitrary point. If not the NSP
        # task might become way too easy.
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []

                    # Random Next
                    is_random_next = False
                    if len(current_chunk) == 1 or random.random() < 0.5:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # Pick a random document
                        for _ in range(10):
                            random_doc_index = random.randint(
                                0,
                                len(self.documents) - 1)
                            if random_doc_index != index:
                                break

                        random_doc = self.documents[random_doc_index]
                        random_start = random.randint(0, len(random_doc) - 1)
                        for j in range(random_start, len(random_doc)):
                            tokens_b.extend(random_doc[j])
                            if len(tokens_b) >= target_b_length:
                                break

                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments

                    # Actual Next
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    truncate_input_sequence(tokens_a, tokens_b, max_num_tokens)

                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    instances.append(
                        TokenInstance(tokens_a, tokens_b, int(is_random_next)))
                    # print(instances[-1])
                current_chunk = []
                current_length = 0
            i += 1
        # print(len(instances))
        return instances


class CleanBodyDataCreator(PretrainingDataCreator):
    def __init__(self,
                 path,
                 tokenizer: BertTokenizer,
                 max_seq_length: int = 512,
                 readin: int = 2000000,
                 dupe_factor: int = 5,
                 small_seq_prob: float = 0.1):
        self.dupe_factor = dupe_factor
        self.max_seq_length = max_seq_length
        self.small_seq_prob = small_seq_prob

        documents = []
        instances = []
        with open(path, encoding='utf-8') as fd:
            for i, line in enumerate(tqdm(fd)):
                line = line.replace('\n', '')
                url, cleanbody, rand_int = line.rstrip("\n").split("\t")
                cleanbody = cleanbody.replace("#TAB#", " ").replace(
                    "#NULL#", "").replace("#HASH#", "#")
                cleanbody_parts = cleanbody.split("#R##N#")
                for document in cleanbody_parts:
                    lines = document.split("#N#")
                    document = []
                    document_len = 0
                    for seq in lines:
                        tok_seq = tokenizer.tokenize(seq)
                        if len(tok_seq) != 0:
                            document.append(tok_seq)
                            document_len += len(tok_seq)
                    if document_len >= 200:
                        documents.append(document)

        documents = [x for x in documents if x]

        self.documents = documents
        for _ in range(self.dupe_factor):
            for index in range(len(self.documents)):
                instances.extend(self.create_training_instance(index))

        shuffle(instances)
        self.instances = instances
        self.len = len(self.instances)
        self.documents = None
        documents = None


class WikiNBookCorpusPretrainingDataCreator(PretrainingDataCreator):
    def __init__(self,
                 path,
                 tokenizer: BertTokenizer,
                 max_seq_length: int = 512,
                 readin: int = 2000000,
                 dupe_factor: int = 6,
                 small_seq_prob: float = 0.1):
        self.dupe_factor = dupe_factor
        self.max_seq_length = max_seq_length
        self.small_seq_prob = small_seq_prob

        documents = []
        instances = []
        with open(path, encoding='utf-8') as fd:
            document = []
            for i, line in enumerate(tqdm(fd)):
                line = line.replace('\n', '')
                # document = line
                # if len(document.split("<sep>")) <= 3:
                #     continue
                if len(line) == 0:  # This is end of document
                    documents.append(document)
                    document = []
                if len(line.split(' ')) > 2:
                    document.append(tokenizer.tokenize(line))
            if len(document) > 0:
                documents.append(document)

        documents = [x for x in documents if x]
        print(documents[0])
        print(len(documents))
        self.documents = documents
        for _ in range(self.dupe_factor):
            for index in range(len(self.documents)):
                instances.extend(self.create_training_instance(index))

        shuffle(instances)
        self.instances = instances
        self.len = len(self.instances)
        self.documents = None
        documents = None


class WikiPretrainingDataCreator(PretrainingDataCreator):
    def __init__(self,
                 path,
                 tokenizer: BertTokenizer,
                 max_seq_length: int = 512,
                 readin: int = 2000000,
                 dupe_factor: int = 6,
                 small_seq_prob: float = 0.1):
        self.dupe_factor = dupe_factor
        self.max_seq_length = max_seq_length
        self.small_seq_prob = small_seq_prob

        documents = []
        instances = []
        with open(path, encoding='utf-8') as fd:
            document = []
            for i, line in enumerate(tqdm(fd)):
                line = line.replace('\n', '')
                # document = line
                # if len(document.split("<sep>")) <= 3:
                #     continue
                if len(line
                       ) > 0 and line[:2] == "[[":  # This is end of document
                    documents.append(document)
                    document = []
                if len(line.split(' ')) > 2:
                    document.append(tokenizer.tokenize(line))
            if len(document) > 0:
                documents.append(document)

        documents = [x for x in documents if x]
        # print(len(documents))
        # print(len(documents[0]))
        # print(documents[0][0:10])
        self.documents = documents
        for _ in range(self.dupe_factor):
            for index in range(len(self.documents)):
                instances.extend(self.create_training_instance(index))

        shuffle(instances)
        self.instances = instances
        self.len = len(self.instances)
        self.documents = None
        documents = None


class NumpyByteInstances:
    TOKEN_SEP_VAL = int.from_bytes(b'\x1f', byteorder='big')

    def __init__(self, data_creator):
        self.data_creator = data_creator
        self.getitem_fixed = self.sep_getitem_fixed if self.data_creator.use_separators else self.data_creator.nosep_getitem_fixed
        # if self.data_creator.multilingual:
        #     self.__getitem__ = self.getitem_multilingual
        # else:
        #     self.__getitem__ = self.getitem_monolingual

    def getitem_multilingual(self, i):
        tokens_a, tokens_b, is_next = self.getitem_fixed(i)
        return TokenInstance(tokens_a,
                             tokens_b,
                             is_next,
                             lang=self.data_creator.lang[i])

    def getitem_monolingual(self, i):
        return TokenInstance(*self.getitem_fixed(i))

    def __getitem__(self, i):
        if self.data_creator.multilingual:
            return self.getitem_multilingual(i)
        else:
            return self.getitem_monolingual(i)

    def nosep_getitem_fixed(self, i):
        if i > self.data_creator.len:
            raise IndexError
        if i < 0:
            i += self.data_creator.len
        instance_start, instance_end = self.data_creator.instance_offsets[i:i +
                                                                          2]
        tok_offsets_start, tok_offsets_end = self.data_creator.instance_token_offsets[
            i:i + 2]
        token_offsets = self.data_creator.token_offsets[
            tok_offsets_start:tok_offsets_end]
        tokens_split = self.data_creator.tokens_split[i]
        token_arrs = np.split(
            self.data_creator.data[instance_start:instance_end], token_offsets)
        tokens = [t.tostring().decode('utf8') for t in token_arrs]

        return tokens[:tokens_split], tokens[
            tokens_split:], self.data_creator.is_next[i]

    def sep_getitem_fixed(self, i):
        if i > self.data_creator.len:
            raise IndexError
        if i < 0:
            i += self.data_creator.len

        instance_start, instance_end = self.data_creator.instance_offsets[i:i +
                                                                          2]
        instance_data = self.data_creator.data[instance_start:instance_end]

        tokens_split = self.data_creator.tokens_split[i]
        token_arrs = np.split(
            instance_data,
            np.where(instance_data == NumpyByteInstances.TOKEN_SEP_VAL)
            [0])  # split on the token separator
        tokens = [
            (t[1:] if i > 0 else t).tostring().decode('utf8')
            for i, t in enumerate(token_arrs)
        ]  # ignore first byte, which will be separator, for tokens after the first

        return tokens[:tokens_split], tokens[
            tokens_split:], self.data_creator.is_next[i]

    def __len__(self):
        return self.data_creator.len


class NumpyPretrainingDataCreator:
    def __init__(self, path, mmap=False):
        path = Path(path)
        self.path = path

        mmap_mode = 'r' if mmap else None

        self.data = np.load(str(path / 'data.npy'), mmap_mode=mmap_mode)
        self.is_next = np.load(str(path / 'is_next.npy'), mmap_mode=mmap_mode)
        self.tokens_split = np.load(str(path / 'tokens_split.npy'),
                                    mmap_mode=mmap_mode)
        self.instance_offsets = np.load(str(path / 'instance_offsets.npy'),
                                        mmap_mode=mmap_mode)

        if (path / 'instance_token_offsets.npy').is_file():
            self.use_separators = False
            self.instance_token_offsets = np.load(str(
                path / 'instance_token_offsets.npy'),
                                                  mmap_mode=mmap_mode)
            self.token_offsets = np.load(str(path / 'token_offsets.npy'),
                                         mmap_mode=mmap_mode)
        else:
            self.use_separators = True
            self.instance_token_offsets = None
            self.token_offsets = None

        if (path / 'lang.npy').is_file():
            self.multilingual = True
            self.lang = np.load(str(path / 'lang.npy'), mmap_mode=mmap_mode)
        else:
            self.multilingual = False
            self.lang = None

        self.instances = NumpyByteInstances(self)

        self.len = len(self.is_next)

    def __len__(self):
        return self.len

    @classmethod
    def load(cls, path):
        return cls(path)
