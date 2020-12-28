# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain BERT"""

import os
import json
import math
import random
import numpy as np
import torch


from arguments import get_args
from configure_data import configure_data
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
from learning_rates import AnnealingLR
from model import GPT2Model
from model import gpt2_get_params_for_weight_decay_optimization
from model import DistributedDataParallel as DDP
import mpu
from apex.optimizers import FusedAdam as Adam
from utils import Timers
from utils import load_checkpoint
from utils import report_memory
from utils import print_params_min_max_norm
from utils import print_rank_0

from data_utils import make_tokenizer

from detokenizer import *

def get_model(args):
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(num_layers=args.num_layers,
                      vocab_size=args.vocab_size,
                      hidden_size=args.hidden_size,
                      num_attention_heads=args.num_attention_heads,
                      embedding_dropout_prob=args.hidden_dropout,
                      attention_dropout_prob=args.attention_dropout,
                      output_dropout_prob=args.hidden_dropout,
                      max_sequence_length=args.max_position_embeddings,
                      checkpoint_activations=args.checkpoint_activations,
                      checkpoint_num_layers=args.checkpoint_num_layers,
                      parallel_output=not args.cloze_eval)

    print_rank_0(' > number of parameters: {}'.format(
        sum([p.nelement() for p in model.parameters()])))

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    model = DDP(model)

    return model


def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args)

    if args.load is not None:
        _ = load_checkpoint(
            model, None, None, args)

    return model

def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask):

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i+1):, :(i+1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i+1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    return attention_mask, loss_mask, position_ids

def get_batch(data_iterator, args, timers):
    ''' get_batch subdivides the source data into chunks of
    length args.seq_length. If source is equal to the example
    output of the data loading example, with a seq_length limit
    of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the data loader. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM. A Variable representing an appropriate
    shard reset mask of the same dimensions is also returned.
    '''
    # Items and their type.
    keys = ['text', 'pad_mask']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    lm_labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    padding_mask = data_b['pad_mask'].byte()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        args.reset_position_ids,
        args.reset_attention_mask)

    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, lm_labels, attention_mask, position_ids, padding_mask


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    batch = get_batch(data_iterator, args, timers)
    if batch is None:
        return None
    tokens, lm_labels, attention_mask, position_ids, loss_mask = batch
    timers('batch generator').stop()
    # Forward model.
    if args.eval_hf:
        output, _ = model(tokens)
    else:
        output = model(tokens, position_ids, attention_mask)

    if not args.cloze_eval:
        #losses = torch.nn.CrossEntropyLoss(reduce=False)(
        losses = mpu.vocab_parallel_cross_entropy(
            output.contiguous().float(), lm_labels.contiguous())
        loss_mask = loss_mask.contiguous()
        loss_mask = loss_mask.view(-1)
        lm_loss = torch.sum(
            losses.view(-1) * loss_mask.float())
    else:
        outputs = torch.argmax(output, -1).contiguous().view(-1)
        acc = (outputs == lm_labels.contiguous().view(-1)).float()
        loss_mask = loss_mask.contiguous().view(-1).float()
        lm_loss = torch.sum(acc * loss_mask)

    return lm_loss


def evaluate(data_loader, model, args, timers,
             num_iterations=None):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_lm_loss = 0
    if num_iterations is not None:
        max_iters = num_iterations
    else:
        if mpu.get_model_parallel_rank() == 0:
            max_iters_gpu = torch.cuda.LongTensor([len(data_loader)])
        else:
            max_iters_gpu = torch.cuda.LongTensor([0])
        torch.distributed.broadcast(max_iters_gpu,
                                    mpu.get_model_parallel_src_rank(),
                                    group=mpu.get_model_parallel_group())
        max_iters = max_iters_gpu[0].item()
        print_rank_0('global rank: {} | max iters: {}'.format(
            torch.distributed.get_rank(), max_iters))

    if data_loader is not None:
        data_iterator = iter(data_loader)
    else:
        data_iterator = None

    with torch.no_grad():
        iteration = 0
        while iteration < max_iters:
            if iteration % args.log_interval == 0:
                print_rank_0('global rank: {} | iteration: {}'.format(
                    torch.distributed.get_rank(), iteration))
            # Forward evaluation.
            lm_loss = forward_step(data_iterator, model, args, timers)
            if lm_loss is None:
                break
            # Reduce across processes.
            if isinstance(model, DDP):
                torch.distributed.all_reduce(lm_loss.data)
                if args.cloze_eval:
                    lm_loss.data = lm_loss.data / args.world_size
                else:
                    lm_loss.data = lm_loss.data / args.model_parallel_size

            if not args.cloze_eval:
                total_lm_loss += lm_loss.data.detach().float().item()/(args.num_tokenized_tokens-1)
            else:
                total_lm_loss += lm_loss.data.detach().float().item()

            iteration += 1

    # Move model back to the train mode.
    model.train()

    return total_lm_loss


def evaluate_and_print_results(prefix, data_iterator, model,
                               args, timers, num_iterations=None):
    """Helper function to evaluate and dump results on screen."""
    if not args.cloze_eval:
        lm_loss = evaluate(data_iterator, model, args, timers, num_iterations)
        val_loss = lm_loss
        ppl = math.exp(min(20, val_loss))
        token_ratio = (args.num_tokenized_tokens-1)/(args.num_original_tokens-1)
        adjusted_ppl = math.exp(min(20, val_loss*token_ratio))
        print_rank_0('-' * 100)
        string = ' validation results on {} | '.format(prefix)
        string += 'avg loss: {:.4E} | '.format(val_loss)
        string += 'ppl: {:.4E} | '.format(ppl)
        string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
        string += 'token ratio: {} |'.format(token_ratio)
        length = len(string) + 1
        print_rank_0('-' * length)
        print_rank_0(string)
        print_rank_0('-' * length)

        return val_loss
    else:
        num_correct = evaluate(data_iterator, model, args, timers, num_iterations)
        acc = num_correct / args.num_examples
        print_rank_0('-' * 100)
        string = ' validation results on {} | '.format(prefix)
        string += 'number correct: {:.4E} | '.format(num_correct)
        string += 'total examples: {:.4E} | '.format(args.num_examples)
        string += 'avg accuracy: {:.4E}'.format(acc)
        length = len(string) + 1
        print_rank_0('-' * length)
        print_rank_0(string)
        print_rank_0('-' * length)
        return acc


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)


class LM_Eval_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokens, seq_len, pad_idx, overalapping_eval=None):
        self.tokens = tokens
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.overalapping_eval = overalapping_eval
        if self.overalapping_eval is None:
            self.overalapping_eval = self.seq_len
        self.overalapping_eval = max(1, self.overalapping_eval)

        self.total_targets = len(self.tokens) - 1
        # remove first sequence tokens
        targets = max(self.total_targets - self.overalapping_eval, 0)
        self.total_sequences = max(math.ceil(targets / self.overalapping_eval)+1, 1)

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.overalapping_eval
        end_idx = start_idx + self.seq_len
        tokens = self.tokens[start_idx:end_idx+1]
        num_tokens = len(tokens)
        pad_mask = [1]*num_tokens
        if num_tokens < self.seq_len+1:
            num_pad = (self.seq_len+1-num_tokens) 
            pad_mask += [0]*(num_pad)
            tokens += [self.pad_idx] * num_pad
        pad_mask = np.array(pad_mask[1:])
        if self.overalapping_eval != self.seq_len and idx!=0:
            pad_mask[:-self.overalapping_eval] *= 0

        return {'text': np.array(tokens), 'pad_mask': pad_mask}

class Lambada_Eval_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, seq_len):
        self.seq_len = seq_len
        self.pad_idx = tokenizer.get_command('pad').Id

        self.tokens = []
        with open(path, 'r') as f:
            for line in f.readlines():
                text = json.loads(line)['text']
                self.tokens.append(tokenizer.EncodeAsIds(text).tokenization)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):

        tokens = self.tokens[idx]
        num_tokens = len(tokens)
        pad_mask = [0]*num_tokens
        pad_mask[-1] = 1
        if num_tokens < self.seq_len+1:
            num_pad = (self.seq_len+1-num_tokens) 
            pad_mask += [0]*(num_pad)
            tokens += [self.pad_idx] * num_pad
        pad_mask = np.array(pad_mask[1:])

        return {'text': np.array(tokens), 'pad_mask': pad_mask}

def get_tokenizer(args):
    tokenizer_args = {
        'tokenizer_type': args.tokenizer_type,
        'corpus': None,
        'model_path': args.tokenizer_path,
        'vocab_size': args.vocab_size,
        'model_type': args.tokenizer_model_type,
        'cache_dir': args.cache_dir}
    return make_tokenizer(**tokenizer_args) 

def get_eval_data(args):
    val_dataloader = None
    if mpu.get_model_parallel_rank() == 0:
        eval_batch_size = args.eval_batch_size
        eval_batch_size = args.batch_size if eval_batch_size is None else eval_batch_size
        seq_len = args.seq_length
        valid_data = args.valid_data
        valid_data = valid_data[0] if isinstance(valid_data, list) else valid_data

        tokenizer = get_tokenizer(args)

        if not args.cloze_eval:

            with open(valid_data, "rb") as reader:
                entire_data = reader.read().decode('utf-8')
            num_original_tokens = len(entire_data.strip().split(" "))
            entire_data = get_detokenizer(valid_data)(entire_data)
            tokenized_data = tokenizer.EncodeAsIds(entire_data).tokenization
            num_tokenized_tokens = len(tokenized_data)
            string = 'Original Tokens: %d, Detokenized tokens: %d' % (num_tokenized_tokens, num_original_tokens)
            print_rank_0(string)

            eod_token = tokenizer.get_command('pad').Id
            val_dataset = LM_Eval_Dataset(tokenized_data, seq_len, eod_token,
                                          args.overlapping_eval)
        else:
            val_dataset = Lambada_Eval_Dataset(valid_data, tokenizer, seq_len)
            num_tokenized_tokens = 0
            num_original_tokens = 0
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=eval_batch_size, drop_last=False)

        before = tokenizer.num_tokens
        after = before
        while after % mpu.get_model_parallel_world_size() != 0:
            after += 1
        print_rank_0('> padded vocab (size: {}) with {} dummy tokens (new size: {})'.
              format(before, after - before, after))
        eod_token = tokenizer.get_command('pad').Id
        num_examples = len(val_dataset)
        token_counts = torch.cuda.LongTensor([after, eod_token, num_examples,
                                              num_original_tokens,
                                              num_tokenized_tokens])
    else:
        token_counts = torch.cuda.LongTensor([0, 0, 0, 0, 0])
    torch.distributed.broadcast(token_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    args.vocab_size = token_counts[0].item()
    args.eod_token = token_counts[1].item()
    args.num_examples = token_counts[2].item()
    args.num_original_tokens = token_counts[3].item()
    args.num_tokenized_tokens = token_counts[4].item()

    print('global rank: {} | vocab size: {} | eod token: {} | '
          'num_examples: {} | num_original_tokens: {} | '
          'num_tokenized_tokens: {}'.format(
              torch.distributed.get_rank(), args.vocab_size,
              args.eod_token, args.num_examples, args.num_original_tokens,
              args.num_tokenized_tokens ))
    return val_dataloader

def main():
    """Main training program."""

    print('Evaluate GPT2 model')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # Data stuff.
    eval_data = get_eval_data(args)

    # Model, optimizer, and learning rate.
    if args.eval_hf:
        from pytorch_pretrained_bert import GPT2LMHeadModel
        from pytorch_pretrained_bert import GPT2Model as HFGPT2Model
        if args.num_layers == 24:
            model_path = args.load
            #model_path = '/home/universal-lm-data.cosmos549/repos/gpt2_mp/models/345M'
            hfmodel = HFGPT2Model.from_pretrained(model_path, cache_dir='gpt2_weights', from_tf=True).cuda()
            model = GPT2LMHeadModel(hfmodel.config)
            model.transformer.load_state_dict(hfmodel.state_dict())
            model.cuda()
        else:
            model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='gpt2_weights').cuda()
    else:
        if args.load_openai:
            from utils import move_weights
            model_path = args.load
            args.load = None
            model = setup_model(args)
            from pytorch_pretrained_bert import GPT2LMHeadModel
            from pytorch_pretrained_bert import GPT2Model as HFGPT2Model

            model_path = 'gpt2'
            from_tf = False
            print('loading openai weights')
            model.cpu()
            if args.num_layers == 24:
                #model_path = '/home/universal-lm-data.cosmos549/repos/gpt2_mp/models/345M'
                hfmodel = HFGPT2Model.from_pretrained(model_path, cache_dir='gpt2_weights', from_tf=True)
                gpt2model = GPT2LMHeadModel(hfmodel.config)
                gpt2model.transformer.load_state_dict(hfmodel.state_dict())
                gpt2model
            else:
                gpt2model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='gpt2_weights')
            model2fill = model
            while isinstance(model2fill, (DDP, FP16_Module)):
                model2fill = model2fill.module
            move_weights(model2fill, gpt2model)
            model.cuda()
        else:
            model = setup_model(args)

    # Run on test data.
    prefix = "wiki" #os.path.basename(args.valid_data)
    evaluate_and_print_results(prefix, eval_data,
                               model, args, timers)


if __name__ == "__main__":
    main()
