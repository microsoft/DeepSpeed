import random
import torch
from torch import nn
from qtorch.quant import float_quantize
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from safetensors.torch import save_file, load_file
import mii


def recursive_getattr(model, module_name):
    """
    Recursively get the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to get the attribute from.
        module_name (`str`)
            The name of the module to get the attribute from.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output


def recursive_setattr(model, module_name, module):
    """
    Recursively set the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to set the attribute in.
        module_name (`str`)
            The name of the module to set the attribute in.
        module (`torch.nn.Module`)
            The module to set the attribute to.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_list[-1], module)


class FPQuantizer(torch.autograd.Function):
    """
    Floating point quantization
    Note that this is based on qtorch, which may have different format as H100 or others
    """
    @staticmethod
    def forward(ctx, input, num_bits,  min_value=None, max_value=None, group_size=-1, exp_bits=3):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input which needs to be quantized
            num_bits (int, >=4)
                Number of bits to use for quantization
            exp/man_bits:
                fp exp/man_bits, if they are both 0, means use a default setting for now
            min_value/max_vlue (torch.FloatTensor)
                Used for static activation quantization
            group_size (int) N
                The quantization block size, each N numbers has its own scaling factor and off-site
                -1 means use the last dim as the group_size 
        Returns:
            quantized_input (`torch.FloatTensor`)
                Quantized input
        """
        assert (min_value is None
                and max_value is None) or (min_value is not None
                                           and max_value is not None)

        ori_dtype = input.dtype
        input = input.to(torch.float32)
        if num_bits == 6:
            if exp_bits == 3:  # this is defulat
                q_range = 28
            else:
                raise NotImplementedError

        man_bits = num_bits - exp_bits - 1
        assert exp_bits + man_bits == num_bits - 1, "exp + man == num_bits-1"
        input_shape = input.shape

        if group_size == -1:
            group_size = input_shape[-1]
        else:
            raise NotImplementedError
        num_groups = input.numel() // group_size
        input = input.reshape(num_groups, -1)

        if min_value is None:
            max_input = torch.amax(
                torch.abs(input), dim=-1).view(num_groups, -1)
        else:
            max_input = torch.max(min_value.abs(), max_value)  # .view(-1)
        scale = max_input / q_range  # q_range + 1
        scale[scale == 0] = 1  # avoid zero scale
        scaled_input = input / scale
        # torch.cuda.synchronize()for some reason this is needed to avoid the output being 0
        output_fp6 = float_quantize(
            scaled_input, exp_bits, man_bits, rounding="nearest")
        output = output_fp6 * scale
        output = output.reshape(input_shape).contiguous().to(ori_dtype)
        return output, output_fp6, scale

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None


class LinearLayer_Compress(nn.Linear):
    """
    Linear layer with compression.
    """

    def __init__(self, *kargs, bias=True):
        super(LinearLayer_Compress, self).__init__(*kargs, bias=bias)
        self.weight_quantizer = None
        self.weight_quantizer = FPQuantizer.apply
        weight, output_fp6, scale = self.weight_quantizer(
            self.weight, 6, None, None, -1, 3)
        scale_param = nn.Parameter(scale)
        output_fp6 = nn.Parameter(output_fp6)
        # self.weight = nn.Parameter(weight)
        self.weight = nn.Parameter(output_fp6)
        self.register_buffer('scales', scale_param)
        # self.register_buffer('weight_fp6', output_fp6)

    def forward(self, input):
        bias = self.bias
        weight = self.weight * self.scales
        if self.bias is not None:
            return nn.functional.linear(input.float(), weight.float(), bias.float()).half()
        else:
            return nn.functional.linear(input.float(), weight.float()).half()


@torch.no_grad()
def quantize_model_rtn(model_id: str):
    model = LlamaForCausalLM.from_pretrained(model_id).cuda().half()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'model.layers.' in name:
            old_module = recursive_getattr(model, name)
            # print(f"{name} is quantized to fp6.")
            need_bias = False
            if hasattr(old_module, 'bias') and old_module.bias is not None:
                need_bias = True
            new_module = LinearLayer_Compress(old_module.in_features,
                                              old_module.out_features,
                                              bias=need_bias).to(
                device=old_module.weight.device,
                dtype=old_module.weight.dtype)
            new_module.weight.data = old_module.weight.data
            new_module.name = name
            recursive_setattr(model, name, new_module)
            # print(new_module)
            del old_module
    return model


def run_ground_truth(model, tokenizer, prompts):
    ...


def fake_request_texts(batch_size: int):
    request_texts = ["Ha ha ha"] * batch_size
    return request_texts


def run_pipeline(deployment_name: str, prompts: list):
    pipe = mii.pipeline(model_name_or_path=deployment_name)
    response = pipe(prompts, max_new_tokens=2)
    print(f"{len(response)} responses.")
    # print(f"Response: {response}.")


if __name__ == '__main__':
    model_id = "meta-llama/Llama-2-7b-hf"
    # model_id = "princeton-nlp/Sheared-LLaMA-1.3B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    batch_size = 32
    prompts = fake_request_texts(batch_size)

    # Quantize the model.
    print(f"\nQuantizing model {model_id}... It may take a while.")
    quantized_model = quantize_model_rtn(model_id)

    # Run the quantized model. This can help to check the correctness of the FP6 integration.
    print(f"\nRunning the quantized model...")
    run_ground_truth(quantized_model, tokenizer, prompts)
    # model(torch.randint(
    #     0, 20000, (1, model.config.max_position_embeddings//2)).long().cuda())

    # Save the quantized model.
    save_path = f"quant/{model_id}"
    print(f"\nSaving the quantized model to {save_path}...")
    quantized_model.cpu().save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    # torch.save(model.cpu().state_dict(), "./pytorch_model.bin")
    # state_dict = model.state_dict()
    # print(f"generated state_dict: {model.state_dict().keys()}")

    # Run MII.
    print(f"\nRunning MII given the quantized model checkpoint...")
    run_pipeline(save_path, prompts)

    # TODO: a function to check the correctness of the MII result.

    print("\nFinished.")

    if False:
        # The AutoModel.from_pretrained() will skip the newly added `scales` when loading the model.
        # But luckly, the DeepSpeed MII will use AutoConfig.from_pretrained() to load the config.
        # It will reserve the `scales` tensors.
        model = AutoModel.from_pretrained(save_path)
        model.load_state_dict()
        print(f"loaded state_dict: {model.state_dict().keys()}")
