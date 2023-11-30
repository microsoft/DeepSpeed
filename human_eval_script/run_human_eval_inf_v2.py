import os
import torch
from transformers import pipeline
from human_eval.data import write_jsonl, read_problems
from deepspeed.accelerator import get_accelerator

#pipe = pipeline(model="facebook/opt-6.7b", torch_dtype=torch.bfloat16, device_map="auto", max_length=252)

local_rank = os.getenv("LOCAL_RANK", "0")
#device = torch.device(get_accelerator().device_name(local_rank))

print("Initializing Pipeline")
pipe = pipeline(model="EleutherAI/gpt-j-6b", device=torch.device(get_accelerator().device_name(local_rank)), max_length=252)

def generate_one_completion(problem_prompt: str) -> str:
    #import pdb; pdb.set_trace()
    #print(f"Generating code for: {problem_prompt}")
    #output = pipe(problem_prompt, do_sample=True)
    #import pdb; pdb.set_trace()
    return pipe(problem_prompt, do_sample=True)[0]["generated_text"]

print("Loading Problems")
problems = read_problems("HumanEvalTest.jsonl.gz")
#import pdb; pdb.set_trace()

#num_samples_per_task = 200
num_samples_per_task = 100
print("Generating Samples")
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
#import pdb; pdb.set_trace()
print("Writing Samples")
write_jsonl("samples.jsonl", samples)
