import random
import torch
from transformers import LlamaForCausalLM



PATH = "debug/priceton-nlp/Sheared-LLaMA-1.3B"


if __name__ == '__main__':
    model = LlamaForCausalLM.from_pretrained(
        "princeton-nlp/Sheared-LLaMA-1.3B").half()
    state_dict = model.cpu().state_dict()
    for key in state_dict.keys():
        if 'proj' in key:
            name = '.'.join(key.split('.')[:-1])
            print(name)
            weight = torch.load(f"{PATH}/{name}.fp6_quant_output.pt", map_location='cpu').half()
            scales = torch.load(f"{PATH}/{name}.fp6_quant_scales.pt", map_location='cpu')
            weight.scales = scales.half()
            state_dict[key] = weight
    torch.save(state_dict, "debug/models--princeton-nlp--Sheared-LLaMA-1.3B/snapshots/dbdd43e8018c14ec50f2f569564b328b28f254b9/pytorch_model.bin")