from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import os
import logging
from multiprocessing import Pool
import numpy as np
import multiprocessing

import os
import logging
import shutil
import tempfile
import argparse
import json
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional, Tuple, Union, IO, Callable, Set
from hashlib import sha256
from functools import wraps
import random
from tqdm import tqdm
from random import shuffle
import pickle

import boto3
from botocore.exceptions import ClientError
import requests

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
}
VOCAB_NAME = 'vocab.txt'

PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))


def split_s3_path(url: str) -> Tuple[str, str]:
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # Remove '/' at beginning of path.
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def s3_request(func: Callable):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url: str, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise FileNotFoundError("file {} not found".format(url))
            else:
                raise

    return wrapper


@s3_request
def s3_etag(url: str) -> Optional[str]:
    """Check ETag on S3 object."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def http_get(url: str, temp_file: IO) -> None:
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    # progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            # progress.update(len(chunk))
            temp_file.write(chunk)
    # progress.close()


def url_to_filename(url: str, etag: str = None) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()

    return filename


def get_from_cache(url: str, cache_dir: str = None) -> str:
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE

    os.makedirs(cache_dir, exist_ok=True)

    # Get eTag to add to filename, if it exists.
    if url.startswith("s3://"):
        etag = s3_etag(url)
    else:
        response = requests.head(url, allow_redirects=True)
        if response.status_code != 200:
            raise IOError("HEAD request failed for url {} with status code {}"
                          .format(url, response.status_code))
        etag = response.headers.get("ETag")

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    if not os.path.exists(cache_path):
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info("%s not found in cache, downloading to %s",
                        url, temp_file.name)

            # GET file object
            if url.startswith("s3://"):
                s3_get(url, temp_file)
            else:
                http_get(url, temp_file)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info("copying %s to cache at %s",
                        temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info("creating metadata file for %s", cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w') as meta_file:
                json.dump(meta, meta_file)

            logger.info("removing temp file %s", temp_file.name)

    return cache_path


def cached_path(url_or_filename: Union[str, Path], cache_dir: str = None) -> str:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ('http', 'https', 's3'):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == '':
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError(
            "unable to parse {} as a URL or as a local path".format(url_or_filename))


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, do_lower_case=True):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        if pretrained_model_name in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name]
        else:
            vocab_file = pretrained_model_name
        if os.path.isdir(vocab_file):
            vocab_file = os.path.join(vocab_file, VOCAB_NAME)
        # redirect to the cache, if necessary
        try:
            resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()),
                    vocab_file))
            return None
        if resolved_vocab_file == vocab_file:
            logger.info("loading vocabulary file {}".format(vocab_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                vocab_file, resolved_vocab_file))
        # Instantiate tokenizer.
        tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)
        return tokenizer


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def truncate_input_sequence(tokens_a, tokens_b, max_num_tokens):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        # assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


class TokenInstance:
    def __init__(self, tokens_a, tokens_b, is_next):
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # 0 is if in continuation, 1 if is random

    def get_values(self):
        return (self.tokens_a, self.tokens_b, self.is_next)


class PretrainingDataCreator:
    def __init__(self, path, tokenizer: BertTokenizer,  max_seq_length: int = 512, readin: int = 200000000, dupe_factor: int = 6, small_seq_prob: float = 0.1):
        self.dupe_factor = dupe_factor
        self.max_seq_length = max_seq_length
        self.small_seq_prob = small_seq_prob

        documents = []
        instances = []
        with open(path, encoding='utf-8') as fd:
            for i, line in enumerate(tqdm(fd)):
                line = line.replace('\n', '')
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

    def save(self, filename):
        with open(filename, 'wb') as outfile:
            pickle.dump(self, outfile)

    def save_npy(self, output_path, raw_bytes=True, use_separators=True, multilingual=False):
        # Make Sure the directory exists
        os.makedirs(output_path, exist_ok=True)
        data = []  # list of numpy arrays containing data for each instance
        lengths = []  # lengths of the data for each instance
        tokens_split = []  # number of tokens in tokens_a for each instance
        is_next = []  # is_next value for each instance

        if multilingual:
            lang = []  # language for each instance

        if not use_separators:
            token_offsets = []
            instance_token_counts = []  #
        else:
            token_sep = b'\x1f'

        for instance in tqdm(self.instances, desc='instances'):
            tokens_a, tokens_b, instance_is_next = instance.get_values()

            if raw_bytes:
                tokens_a = [t.encode('utf8') for t in tokens_a]
                tokens_b = [t.encode('utf8') for t in tokens_b]
                if use_separators:
                    instance_data = np.array(
                        list(token_sep.join(tokens_a + tokens_b)), dtype='b')

                    # sanity check, make sure the separators didn't appear in the data
                    assert np.count_nonzero(instance_data == int.from_bytes(
                        token_sep, byteorder='big')) == len(tokens_a) + len(tokens_b) - 1
                else:
                    instance_data = np.array(
                        list(b''.join(tokens_a+tokens_b)), dtype='b')
            else:
                instance_data = np.array(
                    list(''.join(tokens_a+tokens_b)), dtype='U1')

            data.append(instance_data)
            lengths.append(len(instance_data))
            tokens_split.append(len(tokens_a))
            is_next.append(instance_is_next)

            if multilingual:
                lang.append(instance.get_lang())

            if not use_separators:
                token_offsets.append(
                    np.cumsum([len(t) for t in tokens_a] + [len(t) for t in tokens_b])[:-1])
                instance_token_counts.append(len(tokens_a)+len(tokens_b)-1)

        data = np.concatenate(data)
        tokens_split = np.array(tokens_split)
        is_next = np.array(is_next)
        instance_offsets = np.insert(np.cumsum(lengths), 0, 0)
        np.save(os.path.join(output_path, 'data.npy'),
                data, allow_pickle=False)
        np.save(os.path.join(output_path, 'tokens_split.npy'),
                tokens_split, allow_pickle=False)
        np.save(os.path.join(output_path, 'is_next.npy'),
                is_next, allow_pickle=False)
        np.save(os.path.join(output_path, 'instance_offsets.npy'),
                instance_offsets, allow_pickle=False)

        if multilingual:
            lang = np.array(lang)
            np.save(os.path.join(output_path, 'lang.npy'),
                    lang, allow_pickle=False)

        if not use_separators:
            instance_token_offsets = np.insert(
                np.cumsum(instance_token_counts), 0, 0)
            token_offsets = np.concatenate(token_offsets)

            np.save(os.path.join(output_path, 'instance_token_offsets.npy'),
                    instance_token_offsets, allow_pickle=False)
            np.save(os.path.join(output_path, 'token_offsets.npy'),
                    token_offsets, allow_pickle=False)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def create_training_instance(self, index):
        document = self.documents[index]

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
            if i == len(document)-1 or current_length >= target_seq_length:
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
                    label_next = 0
                    rand_num = random.random()
                    if len(current_chunk) == 1 or rand_num < 0.5:
                        label_next = 1
                        target_b_length = target_seq_length - len(tokens_a)

                        # Pick a random document
                        for _ in range(10):
                            random_doc_index = random.randint(
                                0, len(self.documents) - 1)
                            if random_doc_index != index:
                                break

                        random_doc = self.documents[random_doc_index]
                        random_start = random.randint(0, len(random_doc)-1)
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
                        label_next = 0
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])
                        # # Actual Previous
                        # if rand_num > 0.666:
                        #     tmp = tokens_a
                        #     tokens_a = tokens_b
                        #     tokens_b = tmp
                        #     label_next = 2

                    truncate_input_sequence(tokens_a, tokens_b, max_num_tokens)

                    # assert len(tokens_a) >= 1
                    # assert len(tokens_b) >= 1

                    instances.append(TokenInstance(
                        tokens_a, tokens_b, int(label_next)))
                current_chunk = []
                current_length = 0
            i += 1
        return instances


class WikiNBookCorpusPretrainingDataCreator(PretrainingDataCreator):
    def __init__(self, path, tokenizer: BertTokenizer,  max_seq_length: int = 512, readin: int = 200000000, dupe_factor: int = 6, small_seq_prob: float = 0.1):
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


class BookCorpusPretrainingDataCreator(PretrainingDataCreator):
    def __init__(self, path, tokenizer: BertTokenizer,  max_seq_length: int = 512, readin: int = 200000000, dupe_factor: int = 6, small_seq_prob: float = 0.1):
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
        self.documents = documents
        for _ in range(self.dupe_factor):
            for index in range(len(self.documents)):
                instances.extend(self.create_training_instance(index))

        shuffle(instances)
        self.instances = instances
        self.len = len(self.instances)
        self.documents = None
        documents = None


def parse_data(input_file, output_file):
    if not os.path.exists(output_file):
        print(input_file)
        dataset = WikiNBookCorpusPretrainingDataCreator(
            input_file, tokenizer, dupe_factor=10, max_seq_length=128)
        dataset.save_npy(output_file)
        print(f"Completed Pickling: {output_file}")
    else:
        print(f'Already parsed: {output_file}')


parser = argparse.ArgumentParser(
    description="Give initial arguments for parsing")

parser.add_argument("--input_dir", "--id", type=str)
parser.add_argument("--output_dir", "--od", type=str)
parser.add_argument("--token_file", default="bert-large-cased", type=str)

args = parser.parse_args()
tokenizer = BertTokenizer.from_pretrained(args.token_file, do_lower_case=False)

input_files = []
output_files = []


# parse_data("sample.txt", "test_sample")
# # data = WikiNBookCorpusPretrainingDataCreator.load("test_sample.bin")
# # print(len(data))

for filename in os.listdir(args.input_dir):
    input_file = os.path.join(args.input_dir, filename)
    outfilename = "_".join(filename.split('.')[:-1])
    output_file = os.path.join(args.output_dir, outfilename)
    input_files.append(input_file)
    output_files.append(output_file)
    # parse_data(input_file, output_file)

with Pool(processes=multiprocessing.cpu_count()) as pool:
    pool.starmap(parse_data, zip(input_files, output_files))

