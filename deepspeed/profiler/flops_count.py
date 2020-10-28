from numpy.core.fromnumeric import prod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import register_module_forward_hook
from functools import partial
import numpy as np
import sys

module_flop_count = []

# https://pytorch.org/docs/stable/nn.functional.html


def linear_flops_compute(input, weight, bias=None):
    """
    input: (N, *, in_features) where * means any number of additional dimensions
    weight: (out_features, in_features)
    bias: (out_features)
    output: (N, *, out_features)
    """
    out_features = weight.shape[0]
    return torch.numel(input) * out_features


def relu_flops_compute(input, inplace=False):
    return torch.numel(input)


def pool_flops_compute(input,
                       kernel_size,
                       stride=None,
                       padding=0,
                       ceil_mode=False,
                       count_include_pad=True,
                       divisor_override=None):
    return torch.numel(input)


def conv_flops_compute(input,
                       weight,
                       bias=None,
                       stride=1,
                       padding=0,
                       dilation=1,
                       groups=1):
    """
    input – input tensor of shape (minibatch , in_channels , iH , iW)
    weight – filters of shape (out_channels , in_channels/groups , kH , kW)
    bias – optional bias tensor of shape(out_channels) . Default: None
    stride – the stride of the convolving kernel. Can be a single number or a tuple (sH, sW). Default: 1
    padding – implicit paddings on both sides of the input. Can be a single number or a tuple (padH, padW). Default: 0
    dilation – the spacing between kernel elements. Can be a single number or a tuple (dH, dW). Default: 1
    groups – split input into groups, in_channelsin_channels should be divisible by the number of groups. Default: 1
    """
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
    conv_per_position_flops = int(
        np.prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(np.prod(output_dims))
    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    return int(overall_flops)


def conv_trans_flops_compute(input,
                             weight,
                             bias=None,
                             stride=1,
                             padding=0,
                             output_padding=0,
                             groups=1,
                             dilation=1):
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
    conv_per_position_flops = int(
        np.prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(np.prod(input_dims))
    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * batch_size * int(prod(output_dims))

    overall_flops = overall_conv_flops + bias_flops

    return int(overall_flops)


def batch_norm_flops_compute(input,
                             running_mean,
                             running_var,
                             weight=None,
                             bias=None,
                             training=False,
                             momentum=0.1,
                             eps=1e-05):
    # assume affine is true
    flops = 2 * torch.numel(input)
    return flops


def upsample_flops_compute(input,
                           size=None,
                           scale_factor=None,
                           mode='nearest',
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


def embedding_flops_compute(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
    return 0


def dropout_embedding_flops_compute(input, p=0.5, training=True, inplace=False)
    return 0

def wrapFunc(func, funcFlopCompute):
    oldFunc = func

    def newFunc(*args, **kwds):
        flops = funcFlopCompute(*args, **kwds)
        name = "nn.functional." + func.__name__
        module_flop_count.append((name, flops))
        return oldFunc(*args, **kwds)

    return newFunc


def wrapModule(func, funcFlopCompute):
    oldFunc = func

    def newFunc(*args, **kwds):
        flops = funcFlopCompute(*args, **kwds)
        name = "nn.functional." + func.__name__
        module_flop_count.append((name, flops))
        return oldFunc(*args, **kwds)

    return newFunc


# FUNCS_MAPPING = {
#     nn.functional.avg_pool2d: pool_flops_compute,
#     nn.functional.linear: linear_flops_compute,
#     nn.functional.conv2d: conv_flops_compute
# }
#
# for f in FUNCS_MAPPING:
#     print(f.__name__)
#     setattr(sys.modules[f.__module__], f.__name__, wrapFunc(f, FUNCS_MAPPING[f]))

# FC
nn.functional.linear = wrapFunc(nn.functional.linear, linear_flops_compute)

# convolutions
nn.functional.conv1d = wrapFunc(nn.functional.conv1d, conv_flops_compute)
nn.functional.conv2d = wrapFunc(nn.functional.conv2d, conv_flops_compute)
nn.functional.conv3d = wrapFunc(nn.functional.conv3d, conv_flops_compute)

# conv transposed
nn.functional.conv_transpose1d = wrapFunc(nn.functional.conv_transpose1d,
                                          conv_trans_flops_compute)
nn.functional.conv_transpose2d = wrapFunc(nn.functional.conv_transpose2d,
                                          conv_trans_flops_compute)
nn.functional.conv_transpose3d = wrapFunc(nn.functional.conv_transpose3d,
                                          conv_trans_flops_compute)

# activations
nn.functional.relu = wrapFunc(nn.functional.relu, relu_flops_compute)
nn.functional.prelu = wrapFunc(nn.functional.prelu, relu_flops_compute)
nn.functional.elu = wrapFunc(nn.functional.elu, relu_flops_compute)
nn.functional.leaky_relu = wrapFunc(nn.functional.leaky_relu, relu_flops_compute)
nn.functional.relu6 = wrapFunc(nn.functional.relu6, relu_flops_compute)

# BatchNorms
nn.functional.batch_norm = wrapFunc(nn.functional.batch_norm, batch_norm_flops_compute)

# poolings
nn.functional.avg_pool1d = wrapFunc(nn.functional.avg_pool1d, pool_flops_compute)
nn.functional.avg_pool2d = wrapFunc(nn.functional.avg_pool2d, pool_flops_compute)
nn.functional.avg_pool3d = wrapFunc(nn.functional.avg_pool3d, pool_flops_compute)
nn.functional.max_pool1d = wrapFunc(nn.functional.max_pool1d, pool_flops_compute)
nn.functional.max_pool2d = wrapFunc(nn.functional.max_pool2d, pool_flops_compute)
nn.functional.max_pool3d = wrapFunc(nn.functional.max_pool3d, pool_flops_compute)
nn.functional.adaptive_avg_pool1d = wrapFunc(nn.functional.adaptive_avg_pool1d,
                                             pool_flops_compute)
nn.functional.adaptive_avg_pool2d = wrapFunc(nn.functional.adaptive_avg_pool2d,
                                             pool_flops_compute)
nn.functional.adaptive_avg_pool3d = wrapFunc(nn.functional.adaptive_avg_pool3d,
                                             pool_flops_compute)
nn.functional.adaptive_max_pool1d = wrapFunc(nn.functional.adaptive_max_pool1d,
                                             pool_flops_compute)
nn.functional.adaptive_max_pool2d = wrapFunc(nn.functional.adaptive_max_pool2d,
                                             pool_flops_compute)
nn.functional.adaptive_max_pool3d = wrapFunc(nn.functional.adaptive_max_pool3d,
                                             pool_flops_compute)

# upsample
nn.functional.upsample = wrapFunc(nn.functional.upsample, upsample_flops_compute)
nn.functional.interpolate = wrapFunc(nn.functional.interpolate, upsample_flops_compute)

# softmax
nn.functional.softmax = wrapFunc(nn.functional.softmax, softmax_flops_compute)

# embedding
nn.functional.embedding = wrapFunc(nn.functional.embedding, embedding_flops_compute)

# dropout
nn.functional.dropout = wrapFunc(nn.functional.dropout, dropout_flops_compute)
nn.functional.dropout2d = wrapFunc(nn.functional.dropout2d, dropout_flops_compute)
nn.functional.dropout3d = wrapFunc(nn.functional.dropout3d, dropout_flops_compute)

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
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
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
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    flops = rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
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
    nn.GRUCell: rnn_cell_forward_hook
}


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_flops_counting_methods(model):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    model.start_flops_count = start_flops_count.__get__(model)
    model.stop_flops_count = stop_flops_count.__get__(model)
    model.reset_flops_count = reset_flops_count.__get__(model)
    model.compute_total_flops_count = compute_total_flops_count.__get__(model)

    model.reset_flops_count()

    return model


def start_flops_count(self, **kwargs):
    def register_module_hooks(module, verbose, ost, ignore_list):
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
            # module.__flops__ += sum([child.__flops__ for child in module.children()])
            module_flop_count.clear()

        has_children = len(module._modules.items()) != 0
        if not has_children:
            module.__post_hook_handle__ = module.register_forward_hook(post_hook)

    self.apply(partial(register_module_hooks, **kwargs))


def add_variable_or_reset(module):
    if hasattr(module,
               '__flops__') or hasattr(module,
                                       '__params__') or hasattr(module,
                                                                '__batch__'):
        print('Warning: variables __flops__ or __params__ or __batch__'
              ' are already defined for the module' + type(module).__name__ +
              ' the profiler can affect your code!')
    module.__flops__ = 0
    module.__batch__ = 0
    module.__params__ = get_model_parameters_number(module)


def reset_flops_count(self):
    self.apply(add_variable_or_reset)


def remove_flops_count_attrs(module):
    if hasattr(module, '__batch__'):
        del module.__batch__
    if hasattr(module, '__flops__'):
        del module.__flops__
    if hasattr(module, '__params__'):
        del module.__params__
    if hasattr(module, '__pre_hook_handle__'):
        module.__pre_hook_handle__.remove()
        del module.__pre_hook_handle__
    if hasattr(module, '__post_hook_handle__'):
        module.__post_hook_handle__.remove()
        del module.__post_hook_handle__
    if hasattr(module, '__flops_handle__'):
        module.__flops_handle__.remove()
        del module.__flops_handle__


def compute_total_flops_count(self):
    sum = 0
    for module in self.modules():
        sum += module.__flops__
    return sum


def stop_flops_count(self):
    self.apply(remove_flops_count_attrs)


def flops_to_string(flops, units=None, precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10**6 > 0:
            return str(round(params_num / 10**6, 2)) + ' M'
        elif params_num // 10**3:
            return str(round(params_num / 10**3, 2)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)


def print_model_with_flops(model,
                           total_flops,
                           total_params,
                           units=None,
                           precision=3,
                           ost=sys.stdout):
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
        accumulated_flops_cost = self.accumulate_flops()
        return ', '.join([
            params_to_string(params,
                             units=units,
                             precision=precision),
            '{:.3%} Params'.format(params / total_params),
            flops_to_string(accumulated_flops_cost,
                            units=units,
                            precision=precision),
            '{:.3%} MACs'.format(0 if total_flops == 0 else accumulated_flops_cost /
                                 total_flops),
            self.original_extra_repr()
        ])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(model, file=ost)
    model.apply(del_extra_repr)


def get_model_complexity_info(model,
                              input_res,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None,
                              ost=sys.stdout,
                              verbose=False,
                              ignore_modules=[],
                              custom_funcs_hooks={}):
    assert type(input_res) is tuple
    assert len(input_res) >= 1
    assert isinstance(model, nn.Module)
    model = add_flops_counting_methods(model)
    model.eval()
    model.start_flops_count(ost=ost, verbose=verbose, ignore_list=ignore_modules)
    if input_constructor:
        input = input_constructor(input_res)
        _ = model(**input)
    else:
        try:
            batch = torch.ones(()).new_empty((1,
                                              *input_res),
                                             dtype=next(model.parameters()).dtype,
                                             device=next(model.parameters()).device)
        except StopIteration:
            batch = torch.ones(()).new_empty((1, *input_res))

        _ = model(batch)
        # _ = model(batch)
    flops_count = model.compute_total_flops_count()
    params_count = model.__params__
    if print_per_layer_stat:
        print_model_with_flops(model, flops_count, params_count, ost=ost)
    model.stop_flops_count()

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count


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
            nn.Tanh())

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
    macs, params = get_model_complexity_info(mod,
                                             tuple(input.shape)[1:],
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
