from .optimized_linear import LoRAOptimizedLinear, OptimizedLinear

import torch


try:
    import transformers
except ImportError:
    transformers = None


class Init(object):
    def __init__(self, lora_config=None, quant_config=None):
        self._orig_nn_linear = torch.nn.Linear
        self._orig_causallm_pretrained = None
        if transformers != None:
            self._orig_causallm_pretrained = transformers.AutoModelForCausalLM.from_pretrained
        self.lora_config = lora_config
        self.quant_config = quant_config

    def __enter__(self):
        class OptLinearWrapper:
            _orig_nn_linear = self._orig_nn_linear
            _lora_config = self.lora_config
            _quant_config = self.quant_config
            def __new__(self, *args, **kwargs):
                self._lora_config.delay_lora_init = True
                kwargs['lora_config'] = self._lora_config
                kwargs['quantization_config'] = self._quant_config
                kwargs['linear_cls'] = self._orig_nn_linear
                return OptimizedLinear(*args, **kwargs)

        def post_model_hook(*args, **kwargs):
            model = self._orig_causallm_pretrained(*args, **kwargs)

            if self.lora_config != None:
                model.requires_grad_(False)
                for m in model.modules():
                    if isinstance(m, LoRAOptimizedLinear):
                        m.init_lora()
            return model

        torch.nn.Linear = OptLinearWrapper
        if transformers != None:
            transformers.AutoModelForCausalLM.from_pretrained = post_model_hook

    def __exit__(self, *args, **kwargs):
        torch.nn.Linear = self._orig_nn_linear
        if transformers is None and self.lora_config != None:
            print('WARNING: Not using transformers, you must call `init_lora` on each module in order to use DeepSpeed LoRA')
        else:
            transformers.AutoModelForCausalLM.from_pretrained = self._orig_causallm_pretrained
