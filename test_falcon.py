from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import torch
import deepspeed
import time
from deepspeed.accelerator import get_accelerator
import json
import io
import os
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--save_mp_sharded_ckpt", required=False, action='store_true')
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--model-name", type=str, default='falcon-40b')
parser.add_argument("--ckpt-root", type=str, default='falcon-40b')
args = parser.parse_args()
repo_root = args.ckpt_root 
model = "tiiuae/"+args.model_name
#AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
if args.save_mp_sharded_ckpt:
    checkpoints_json = "checkpoints.json"
    with io.open(checkpoints_json, "w", encoding="utf-8") as f:
        file_list = [str(entry).split('/')[-1] for entry in Path(repo_root).rglob("*.[bp][it][n]") if entry.is_file()]
        data = {"type": "ds_model", "checkpoints": file_list, "version": 1.0}
        json.dump(data, f)
else:
    checkpoints_json = "/tmp/" + args.model_name + "/ds_inference_config.json"

tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
config = AutoConfig.from_pretrained(model, trust_remote_code=True)

with deepspeed.OnDevice(dtype=torch.bfloat16, device="meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

model = deepspeed.init_inference(model, 
                                 mp_size=int(os.getenv("WORLD_SIZE", "1")), 
                                 replace_with_kernel_inject=True, 
                                 base_dir=repo_root, 
                                 dtype=torch.bfloat16,
                                 checkpoint=checkpoints_json, 
                                 save_mp_checkpoint_path='/tmp/'+args.model_name if args.save_mp_sharded_ckpt else None
                                 )

input_prompt = [
   "Deep learning involves the use of neural networks, which are computational models inspired by the structure and functioning of the human brain. These networks consist of interconnected nodes called neurons"
   ]
input_tokens = tokenizer.batch_encode_plus(input_prompt, return_tensors="pt",)
token_num = input_tokens['input_ids'].size(-1)
for t in input_tokens:
    if torch.is_tensor(input_tokens[t]):
        input_tokens[t] = input_tokens[t].to(get_accelerator().current_device_name())
input_tokens.pop('token_type_ids')
sequences = model.generate(**input_tokens, min_length=200, max_length=300, do_sample=True)

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(f"Result: {tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]}")
