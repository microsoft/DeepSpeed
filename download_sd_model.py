import torch
from diffusers import DiffusionPipeline

model = "prompthero/midjourney-v4-diffusion"
print("Started downloading SD model")
pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.half)
print("Finsihed downloading SD model")
