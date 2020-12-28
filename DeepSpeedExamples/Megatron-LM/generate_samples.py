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

"""Sample Generate GPT2"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import time
from arguments import get_args
from utils import Timers
from pretrain_gpt2 import initialize_distributed
from pretrain_gpt2 import set_random_seed
from pretrain_gpt2 import get_train_val_test_data
from pretrain_gpt2 import get_masks_and_position_ids
from utils import load_checkpoint
from data_utils import make_tokenizer
from configure_data import configure_data
import mpu

from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0

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
                      parallel_output=False)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

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


def get_batch(context_tokens, device, args):
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()
    tokens = tokens.to(device)

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        args.reset_position_ids,
        args.reset_attention_mask)

    return tokens, attention_mask, position_ids

def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        
    if top_p > 0.0:
        #convert to 1D
        logits=logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        #going back to 2D
        logits=logits.view(1, -1).contiguous()
	
    return logits


def generate_samples(model, tokenizer, args, device):
    
    context_count=0
    model.eval()
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs=0

            if mpu.get_model_parallel_rank() == 0:
                raw_text = input("\nContext prompt (stop to exit) >>> ")
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input("\nContext prompt (stop to exit) >>> ")
           
                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.EncodeAsIds(raw_text).tokenization
                    context_length = len(context_tokens)

                    if context_length >=args.seq_length//2:
                        print("\nContext length", context_length, \
                            "\nPlease give smaller context (half of the sequence length)!")
                        continue
            else:
                context_tokens = tokenizer.EncodeAsIds("EMPTY TEXT").tokenization
                context_length = len(context_tokens)
            
            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(terminate_runs_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
            terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                return

            pad_id = tokenizer.get_command('pad').Id
            if context_length < args.seq_length:
                context_tokens.extend([pad_id] * (args.seq_length - context_length))

            context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
            context_length_tensor = torch.cuda.LongTensor([context_length])

            torch.distributed.broadcast(context_length_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
            torch.distributed.broadcast(context_tokens_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())

            context_length = context_length_tensor[0].item()
            tokens, attention_mask, position_ids=get_batch(context_tokens_tensor, device, args)

            start_time = time.time()

            counter = 0
            org_context_length = context_length

            while counter < (org_context_length + args.out_seq_length):
                logits = model(tokens, position_ids, attention_mask)
                logits = logits[:, context_length - 1, :] / args.temperature
                logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)            
                log_probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1)
                tokens[0, context_length] = prev[0] 
                context_length += 1
                counter += 1

                output_tokens_list = tokens.view(-1).contiguous()
                decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())
                token_end = decode_tokens.find("<|endoftext|>")


                if mpu.get_model_parallel_rank() == 0 and (counter % 16 == 0 or token_end != -1):
                   os.system('clear')
                   print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
                   print("\nContext:", raw_text, flush=True)
                   trim_decode_tokens = decode_tokens[len(raw_text):decode_tokens.find("<|endoftext|>")]
                   print("\nGPT2:", trim_decode_tokens, flush=True)
                if token_end != -1:
                   break
                
            if mpu.get_model_parallel_rank() == 0:
                os.system('clear')
                print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
                print("\nContext:", raw_text, flush=True)
                output_tokens_list = tokens.view(-1).contiguous()
                decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())
                trim_decode_tokens = decode_tokens[len(raw_text):decode_tokens.find("<|endoftext|>")]
                print("\nGPT2:", trim_decode_tokens, flush=True)
            raw_text = None

            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1

def prepare_tokenizer(args):

    tokenizer_args = {
        'tokenizer_type': args.tokenizer_type,
        'corpus': None,
        'model_path': args.tokenizer_path,
        'vocab_size': args.vocab_size,
        'model_type': args.tokenizer_model_type,
        'cache_dir': args.cache_dir}
    tokenizer = make_tokenizer(**tokenizer_args)

    args.tokenizer_num_tokens = tokenizer.num_tokens
    args.tokenizer_num_type_tokens = tokenizer.num_type_tokens
    args.eod_token = tokenizer.get_command('eos').Id

    after = tokenizer.num_tokens
    while after % mpu.get_model_parallel_world_size() != 0:
        after += 1

    args.vocab_size = after
    print("prepare tokenizer done", flush=True)

    return tokenizer

def main():
    """Main training program."""

    print('Generate Samples')

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

    #get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    #setting default batch size to 1
    args.batch_size = 1

    #generate samples
    generate_samples(model, tokenizer, args, torch.cuda.current_device())
    

if __name__ == "__main__":
    main()



