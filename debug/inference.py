from transformers import AutoTokenizer, pipeline
import torch

from deepspeed.inference import RaggedInferenceEngineConfig, build_hf_engine

local_rank = 0
world_size = 1

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# TODO: modify the deepspeed.init_inference to support the inference_v2 functions with the new config.
engine_config = RaggedInferenceEngineConfig()
engine_config.tensor_parallel.tp_size = world_size
engine_config.quantization.quantization_mode = 'wf6af16'
ds_engine = build_hf_engine(path = model_name, engine_config = engine_config)
model = ds_engine.module.to(f'cuda:{local_rank}')

input_string = ['DeepSpeed is']
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=local_rank, torch_dtype=torch.float16)
output = pipe(input_string)

print(output)
