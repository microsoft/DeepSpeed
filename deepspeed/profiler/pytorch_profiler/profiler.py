import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

module_flop_count = []


def prod(dims):
    p = 1
    for v in dims:
        p *= v
    return p


def linear_flops_compute(input, weight, bias=None):
    out_features = weight.shape[0]
    return torch.numel(input) * out_features


def relu_flops_compute(input, inplace=False):
    return torch.numel(input)


def pool_flops_compute(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    return torch.numel(input)


def conv_flops_compute(input,
                       weight,
                       bias=None,
                       stride=1,
                       padding=0,
                       dilation=1,
                       groups=1):
    assert weight.shape[1] * groups == input.shape[1]

    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[-2:])
    input_dims = list(input.shape[2:])

    paddings = padding if type(padding) is tuple else (padding, padding)
    strides = stride if type(stride) is tuple else (stride, stride)
    dilations = dilation if type(dilation) is tuple else (dilation, dilation)

    output_dims = [0, 0]
    output_dims[0] = (input_dims[0] + 2 * paddings[0] -
                      (dilations[0] * (kernel_dims[0] - 1) + 1)) // strides[0] + 1
    output_dims[1] = (input_dims[1] + 2 * paddings[1] -
                      (dilations[1] * (kernel_dims[1] - 1) + 1)) // strides[1] + 1

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(prod(output_dims))
    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    return int(overall_flops)


def conv_trans_flops_compute(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[-2:])
    input_dims = list(input.shape[2:])

    paddings = padding if type(padding) is tuple else (padding, padding)
    strides = stride if type(stride) is tuple else (stride, stride)
    dilations = dilation if type(dilation) is tuple else (dilation, dilation)

    output_dims = [0, 0]
    output_dims[0] = (input_dims[0] + 2 * paddings[0] -
                      (dilations[0] * (kernel_dims[0] - 1) + 1)) // strides[0] + 1
    output_dims[1] = (input_dims[1] + 2 * paddings[1] -
                      (dilations[1] * (kernel_dims[1] - 1) + 1)) // strides[1] + 1

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(prod(input_dims))
    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * batch_size * int(prod(output_dims))

    overall_flops = overall_conv_flops + bias_flops

    return int(overall_flops)


def batch_norm_flops_compute(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    # assume affine is true
    flops = 2 * torch.numel(input)
    return flops


def upsample_flops_compute(input,
                           size=None,
                           scale_factor=None,
                           mode="nearest",
                           align_corners=None):
    if size is not None:
        return int(prod(size))
    assert scale_factor is not None
    flops = torch.numel(input)
    if len(scale_factor) == len(input):
        flops * int(prod(scale_factor))
    else:
        flops * scale_factor**len(input)
    return flops


def softmax_flops_compute(input, dim=None, _stacklevel=3, dtype=None):
    return torch.numel(input)


def embedding_flops_compute(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    return 0


def dropout_flops_compute(input, p=0.5, training=True, inplace=False):
    return 0


def wrapFunc(func, funcFlopCompute):
    oldFunc = func

    def newFunc(*args, **kwds):
        flops = funcFlopCompute(*args, **kwds)
        name = "nn.functional." + func.__name__
        module_flop_count.append((name, flops))
        return oldFunc(*args, **kwds)

    return newFunc


# FC
F.linear = wrapFunc(F.linear, linear_flops_compute)

# convolutions
F.conv1d = wrapFunc(F.conv1d, conv_flops_compute)
F.conv2d = wrapFunc(F.conv2d, conv_flops_compute)
F.conv3d = wrapFunc(F.conv3d, conv_flops_compute)

# conv transposed
F.conv_transpose1d = wrapFunc(F.conv_transpose1d, conv_trans_flops_compute)
F.conv_transpose2d = wrapFunc(F.conv_transpose2d, conv_trans_flops_compute)
F.conv_transpose3d = wrapFunc(F.conv_transpose3d, conv_trans_flops_compute)

# activations
F.relu = wrapFunc(F.relu, relu_flops_compute)
F.prelu = wrapFunc(F.prelu, relu_flops_compute)
F.elu = wrapFunc(F.elu, relu_flops_compute)
F.leaky_relu = wrapFunc(F.leaky_relu, relu_flops_compute)
F.relu6 = wrapFunc(F.relu6, relu_flops_compute)

# BatchNorms
F.batch_norm = wrapFunc(F.batch_norm, batch_norm_flops_compute)

# poolings
F.avg_pool1d = wrapFunc(F.avg_pool1d, pool_flops_compute)
F.avg_pool2d = wrapFunc(F.avg_pool2d, pool_flops_compute)
F.avg_pool3d = wrapFunc(F.avg_pool3d, pool_flops_compute)
F.max_pool1d = wrapFunc(F.max_pool1d, pool_flops_compute)
F.max_pool2d = wrapFunc(F.max_pool2d, pool_flops_compute)
F.max_pool3d = wrapFunc(F.max_pool3d, pool_flops_compute)
F.adaptive_avg_pool1d = wrapFunc(F.adaptive_avg_pool1d, pool_flops_compute)
F.adaptive_avg_pool2d = wrapFunc(F.adaptive_avg_pool2d, pool_flops_compute)
F.adaptive_avg_pool3d = wrapFunc(F.adaptive_avg_pool3d, pool_flops_compute)
F.adaptive_max_pool1d = wrapFunc(F.adaptive_max_pool1d, pool_flops_compute)
F.adaptive_max_pool2d = wrapFunc(F.adaptive_max_pool2d, pool_flops_compute)
F.adaptive_max_pool3d = wrapFunc(F.adaptive_max_pool3d, pool_flops_compute)

# upsample
F.upsample = wrapFunc(F.upsample, upsample_flops_compute)
F.interpolate = wrapFunc(F.interpolate, upsample_flops_compute)

# softmax
F.softmax = wrapFunc(F.softmax, softmax_flops_compute)

# embedding
F.embedding = wrapFunc(F.embedding, embedding_flops_compute)


def rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    flops += w_ih.shape[0] * w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    flops += w_hh.shape[0] * w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size * 3
        # last two hadamard product and add
        flops += rnn_module.hidden_size * 3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size * 4
        # two hadamard product and add for C state
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def rnn_forward_hook(rnn_module, input, output):
    """
    Takes into account batch goes at first position, contrary
    to pytorch common rule (but actually it doesn't matter).
    IF sigmoid and tanh are made hard, only a comparison FLOPS should be accurate
    """
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__("weight_ih_l" + str(i))
        w_hh = rnn_module.__getattr__("weight_hh_l" + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__("bias_ih_l" + str(i))
            b_hh = rnn_module.__getattr__("bias_hh_l" + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def rnn_cell_forward_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__("weight_ih")
    w_hh = rnn_cell_module.__getattr__("weight_hh")
    input_size = inp.shape[1]
    flops = rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__("bias_ih")
        b_hh = rnn_cell_module.__getattr__("bias_hh")
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


MODULE_HOOK_MAPPING = {
    # RNN
    nn.RNN: rnn_forward_hook,
    nn.GRU: rnn_forward_hook,
    nn.LSTM: rnn_forward_hook,
    nn.RNNCell: rnn_cell_forward_hook,
    nn.LSTMCell: rnn_cell_forward_hook,
    nn.GRUCell: rnn_cell_forward_hook,
}


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_profile_methods(model):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    model.start_profile = start_profile.__get__(model)
    model.stop_profile = stop_profile.__get__(model)
    model.reset_profile = reset_profile.__get__(model)
    model.get_total_flops = get_total_flops.__get__(model)
    model.get_total_duration = get_total_duration.__get__(model)
    model.get_total_params = get_total_params.__get__(model)
    model.print_model_profile = print_model_profile.__get__(model)
    model.print_model_aggregated_profile = print_model_aggregated_profile.__get__(model)

    model.reset_profile()

    return model


def start_profile(self, **kwargs):
    def register_module_hooks(module, ignore_list):
        if ignore_list and type(module) in ignore_list:
            return

        # if compute the flops of a module directly
        if type(module) in MODULE_HOOK_MAPPING:
            module.__flops_handle__ = module.register_forward_hook(
                MODULE_HOOK_MAPPING[type(module)])
            return

        # if compute the flops of the functionals in a module
        def pre_hook(module, input):
            module_flop_count.clear()
            batch_size = 1
            if len(input) > 0:
                # Can have multiple inputs, getting the first one
                input = input[0]
                batch_size = len(input)
            module.__batch__ += batch_size

        module.__pre_hook_handle__ = module.register_forward_pre_hook(pre_hook)

        def post_hook(module, input, output):
            module.__flops__ = sum([elem[1] for elem in module_flop_count])
            module_flop_count.clear()

        has_children = len(module._modules.items()) != 0
        if not has_children:
            module.__post_hook_handle__ = module.register_forward_hook(post_hook)

        def start_time_hook(module, input):
            module.__start_time__ = time.time()

        module.__start_time_hook_handle__ = module.register_forward_pre_hook(
            start_time_hook)

        def end_time_hook(module, input, output):
            module.__end_time__ = time.time()

        module.__end_time_hook_handle__ = module.register_forward_hook(end_time_hook)

    self.apply(partial(register_module_hooks, **kwargs))


def add_or_reset_attrs(module):
    module.__flops__ = 0
    module.__batch__ = 0
    module.__params__ = get_model_parameters_number(module)
    module.__start_time__ = 0
    module.__end_time__ = 0


def reset_profile(self):
    self.apply(add_or_reset_attrs)


def remove_profile_attrs(module):
    if hasattr(module, "__batch__"):
        del module.__batch__
    if hasattr(module, "__flops__"):
        del module.__flops__
    if hasattr(module, "__params__"):
        del module.__params__
    if hasattr(module, "__start_time__"):
        del module.__start_time__
    if hasattr(module, "__end_time__"):
        del module.__end_time__
    if hasattr(module, "__pre_hook_handle__"):
        module.__pre_hook_handle__.remove()
        del module.__pre_hook_handle__
    if hasattr(module, "__post_hook_handle__"):
        module.__post_hook_handle__.remove()
        del module.__post_hook_handle__
    if hasattr(module, "__flops_handle__"):
        module.__flops_handle__.remove()
        del module.__flops_handle__
    if hasattr(module, "__start_time_hook_handle__"):
        module.__start_time_hook_handle__.remove()
        del module.__start_time_hook_handle__
    if hasattr(module, "__end_time_hook_handle__"):
        module.__end_time_hook_handle__.remove()
        del module.__end_time_hook_handle__


def get_total_flops(self):
    sum = 0
    for module in self.modules():
        sum += module.__flops__
    return sum


def get_total_duration(self):
    return self.__end_time__ - self.__start_time__


def get_total_params(self):
    return self.__params__


def stop_profile(self):
    self.apply(remove_profile_attrs)


def flops_to_string(flops, units=None, precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.0**9, precision)) + " GMACs"
        elif flops // 10**6 > 0:
            return str(round(flops / 10.0**6, precision)) + " MMACs"
        elif flops // 10**3 > 0:
            return str(round(flops / 10.0**3, precision)) + " KMACs"
        else:
            return str(flops) + " MACs"
    else:
        if units == "GMACs":
            return str(round(flops / 10.0**9, precision)) + " " + units
        elif units == "MMACs":
            return str(round(flops / 10.0**6, precision)) + " " + units
        elif units == "KMACs":
            return str(round(flops / 10.0**3, precision)) + " " + units
        else:
            return str(flops) + " MACs"


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10**6 > 0:
            return str(round(params_num / 10**6, 2)) + " M"
        elif params_num // 10**3:
            return str(round(params_num / 10**3, 2)) + " k"
        else:
            return str(params_num)
    else:
        if units == "M":
            return str(round(params_num / 10.0**6, precision)) + " " + units
        elif units == "K":
            return str(round(params_num / 10.0**3, precision)) + " " + units
        else:
            return str(params_num)


def duration_to_string(duration, units=None, precision=2):
    if units is None:
        if duration > 1:
            return str(round(duration, precision)) + " s"
        elif duration * 10**3 > 1:
            return str(round(duration * 10**3, precision)) + " ms"
        elif duration * 10**6 > 1:
            return str(round(duration * 10**6, precision)) + " us"
        else:
            return str(duration)
    else:
        if units == "us":
            return str(round(duration * 10.0**6, precision)) + " " + units
        elif units == "ms":
            return str(round(duration * 10.0**3, precision)) + " " + units
        else:
            return str(round(duration, precision)) + " s"


def print_model_profile(self):
    total_flops = self.get_total_flops()
    total_duration = self.get_total_duration()
    total_params = self.__params__

    def accumulate_flops(self):
        has_children = len(self._modules.items()) != 0
        if not has_children:
            return self.__flops__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_flops()
        return sum

    def flops_repr(self):
        params = self.__params__
        flops = self.accumulate_flops()
        items = [
            params_to_string(params),
            "{:.3%} Params".format(params / total_params),
            flops_to_string(flops),
            "{:.3%} MACs".format(0 if total_flops == 0 else flops / total_flops),
        ]
        duration = self.__end_time__ - self.__start_time__
        items.append(duration_to_string(duration))
        items.append("{:.2%} time".format(0 if total_duration == 0 else duration /
                                          total_duration))
        # flops = 2 * MACs
        items.append("0" if duration ==
                     0 else str(round(2 * flops / duration / 10**12,
                                      2)) + " TFLOPS")
        items.append(self.original_extra_repr())
        return ", ".join(items)

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, "original_extra_repr"):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, "accumulate_flops"):
            del m.accumulate_flops

    self.apply(add_extra_repr)
    print(self)
    self.apply(del_extra_repr)


def print_model_aggregated_profile(self, depth=-1, top_num=3):
    info = {}
    if not hasattr(self, "__flops__"):
        print(
            "no __flops__ attribute in the model, call this function after start_profile and before stop_profile"
        )
        return

    def walk_module(module, curr_depth, info):
        if curr_depth not in info:
            info[curr_depth] = {}
        if module.__class__.__name__ not in info[curr_depth]:
            info[curr_depth][module.__class__.__name__] = [
                0,
                0,
                0,
            ]  # flops, params, time
        info[curr_depth][module.__class__.__name__][0] += module.__flops__
        info[curr_depth][module.__class__.__name__][1] += module.__params__
        info[curr_depth][module.__class__.__name__][2] += (module.__end_time__ -
                                                           module.__start_time__)
        has_children = len(module._modules.items()) != 0
        if has_children:
            for child in module.children():
                walk_module(child, curr_depth + 1, info)

    walk_module(self, 0, info)

    max_depth = len(info)
    if depth == -1:
        depth = max_depth - 1

    num_items = min(top_num, len(info[depth]))

    sort_flops = {
        k: flops_to_string(v[0])
        for k,
        v in sorted(info[depth].items(),
                    key=lambda item: item[1][0],
                    reverse=True)[:num_items]
    }
    sort_params = {
        k: params_to_string(v[1])
        for k,
        v in sorted(info[depth].items(),
                    key=lambda item: item[1][1],
                    reverse=True)[:num_items]
    }
    sort_time = {
        k: duration_to_string(v[2])
        for k,
        v in sorted(info[depth].items(),
                    key=lambda item: item[1][2],
                    reverse=True)[:num_items]
    }
    print(f"Top {num_items} modules in flops at depth {depth} are {sort_flops}")
    print(f"Top {num_items} modules in params at depth {depth} are {sort_params}")
    print(f"Top {num_items} modules in time at depth {depth} are {sort_time}")


def get_model_profile(
    model,
    input_res,
    input_constructor=None,
    print_profile=True,
    print_aggregated_profile=True,
    depth=-1,
    top_num=3,
    warm_up=10,
    as_strings=True,
    ignore_modules=[],
):
    assert type(input_res) is tuple
    assert len(input_res) >= 1
    assert isinstance(model, nn.Module)
    model = add_profile_methods(model)
    model.eval()
    model.start_profile(ignore_list=ignore_modules)
    for _ in range(warm_up):
        if input_constructor:
            input = input_constructor(input_res)
            _ = model(**input)
        else:
            try:
                batch = torch.ones(()).new_empty(
                    (1,
                     *input_res),
                    dtype=next(model.parameters()).dtype,
                    device=next(model.parameters()).device,
                )
            except StopIteration:
                batch = torch.ones(()).new_empty((1, *input_res))
            _ = model(batch)
    model.reset_profile()

    if input_constructor:
        input = input_constructor(input_res)
        _ = model(**input)
    else:
        try:
            batch = torch.ones(()).new_empty(
                (1,
                 *input_res),
                dtype=next(model.parameters()).dtype,
                device=next(model.parameters()).device,
            )
        except StopIteration:
            batch = torch.ones(()).new_empty((1, *input_res))
        _ = model(batch)

    flops = model.get_total_flops()
    duration = model.get_total_duration()
    params = model.__params__
    if print_profile:
        model.print_model_profile()
    if print_aggregated_profile:
        model.print_model_aggregated_profile(depth=depth, top_num=top_num)
    model.stop_profile()
    if as_strings:
        return flops_to_string(flops), params_to_string(params)

    return flops, params


class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=6,
                      kernel_size=5,
                      stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16,
                      out_channels=120,
                      kernel_size=5,
                      stride=1),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120,
                      out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84,
                      out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs


if __name__ == "__main__":
    mod = LeNet5(10)
    input = torch.randn(3, 1, 32, 32)
    macs, params = get_model_profile(
        mod,
        tuple(input.shape)[1:],
        print_profile=True,
        print_aggregated_profile=True,
        depth=-1,
        top_num=3,
        warm_up=10,
        as_strings=True,
        ignore_modules=None,
    )
    print("{:<30}  {:<8}".format("Number of multiply-adds: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
