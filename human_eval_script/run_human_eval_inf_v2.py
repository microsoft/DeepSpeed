import torch
from transformers import pipeline
from human_eval.data import write_jsonl, read_problems

model_name = "facebook/opt-125m"

pipe = pipeline(model="facebook/opt-1.3b", torch_dtype=torch.bfloat16, device_map="auto", max_length=252)

def generate_one_completion(problem_prompt: str) -> str:
    #import pdb; pdb.set_trace()
    print(f"Generating code for: {problem_prompt}")
    #output = pipe(problem_prompt, do_sample=True)
    #import pdb; pdb.set_trace()
    return pipe(problem_prompt, do_sample=True)[0]["generated_text"]

problems = read_problems("HumanEvalTest.jsonl.gz")
#import pdb; pdb.set_trace()

#num_samples_per_task = 200
num_samples_per_task = 3
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
import pdb; pdb.set_trace()
write_jsonl("samples.jsonl", samples)
