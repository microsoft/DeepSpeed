import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import register_module_forward_hook
from functools import partial
import numpy as np
import sys

module_flop_count = []


def linear_flops_compute(input, weight, bias=None):
    """
    Input: (N, *, in_features) where * means any number of additional dimensions
    Weight: (out_features, in_features)
    Bias: (out_features)
    Output: (N, *, out_features)
    """
    # print(input.shape)
    # print((weight.shape))
    output_last_dim = weight.shape[-1]
    return "nn.functional.linear", 111
    return "nn.functional.linear", int(np.prod(input.shape) * output_last_dim)


def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('Warning! No positional inputs found for a module,'
              ' assuming batch size is 1.')
    module.__batch__ += batch_size


def register_batch_counter_hook(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def is_supported_functional(func):
    if func in FUNCS_MAPPING:
        return True
    return False


FUNCS_MAPPING = {
    # # convolutions
    # nn.Conv1d: conv_flops_counter_hook,
    # nn.Conv2d: conv_flops_counter_hook,
    # nn.Conv3d: conv_flops_counter_hook,
    # # activations
    # nn.ReLU: relu_flops_counter_hook,
    # nn.PReLU: relu_flops_counter_hook,
    # nn.ELU: relu_flops_counter_hook,
    # nn.LeakyReLU: relu_flops_counter_hook,
    # nn.ReLU6: relu_flops_counter_hook,
    # # poolings
    # nn.MaxPool1d: pool_flops_counter_hook,
    # nn.AvgPool1d: pool_flops_counter_hook,
    # nn.AvgPool2d: pool_flops_counter_hook,
    # nn.MaxPool2d: pool_flops_counter_hook,
    # nn.MaxPool3d: pool_flops_counter_hook,
    # nn.AvgPool3d: pool_flops_counter_hook,
    # nn.AdaptiveMaxPool1d: pool_flops_counter_hook,
    # nn.AdaptiveAvgPool1d: pool_flops_counter_hook,
    # nn.AdaptiveMaxPool2d: pool_flops_counter_hook,
    # nn.AdaptiveAvgPool2d: pool_flops_counter_hook,
    # nn.AdaptiveMaxPool3d: pool_flops_counter_hook,
    # nn.AdaptiveAvgPool3d: pool_flops_counter_hook,
    # # BNs
    # nn.BatchNorm1d: bn_flops_counter_hook,
    # nn.BatchNorm2d: bn_flops_counter_hook,
    # nn.BatchNorm3d: bn_flops_counter_hook,
    # FC
    nn.functional.linear:
    linear_flops_compute,
    # nn.functional.conv2d:
    # conv_flops_counter_hook,
    # # Upscale
    # nn.Upsample: upsample_flops_counter_hook,
    # # Deconvolution
    # nn.ConvTranspose2d: deconv_flops_counter_hook,
    # # RNN
    # nn.RNN: rnn_flops_counter_hook,
    # nn.GRU: rnn_flops_counter_hook,
    # nn.LSTM: rnn_flops_counter_hook,
    # nn.RNNCell: rnn_cell_flops_counter_hook,
    # nn.LSTMCell: rnn_cell_flops_counter_hook,
    # nn.GRUCell: rnn_cell_flops_counter_hook
}


def wrapFunc(func, funcFlopCompute):
    oldFunc = func

    def newFunc(*args, **kwds):
        name, flops = funcFlopCompute(*args, **kwds)
        module_flop_count.append((name, flops))
        return oldFunc(*args, **kwds)

    return newFunc


for f in FUNCS_MAPPING:
    setattr(sys.modules[f.__module__], f.__name__, wrapFunc(f, FUNCS_MAPPING[f]))

# nn.functional.linear = wrapFunc(nn.functional.linear,
#                                 FUNCS_MAPPING[nn.functional.linear])


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
        _ = model(batch)
        _ = model(batch)
    flops_count, params_count = model.compute_total_flops()
    if print_per_layer_stat:
        print_model_with_flops(model, flops_count, params_count, ost=ost)
    model.stop_flops_count()

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count


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
    def accumulate_params(self):
        has_children = len(self._modules.items()) != 0
        if not has_children:
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()
            return sum

    def accumulate_flops(self):
        has_children = len(self._modules.items()) != 0
        if not has_children:
            return self.__flops__ / self.__batch__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_flops()
            return sum

    def flops_repr(self):
        accumulated_params_num = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops()
        return ', '.join([
            params_to_string(accumulated_params_num,
                             units=units,
                             precision=precision),
            '{:.3%} Params'.format(accumulated_params_num / total_params),
            flops_to_string(accumulated_flops_cost,
                            units=units,
                            precision=precision),
            '{:.3%} MACs'.format(accumulated_flops_cost / total_flops),
            self.original_extra_repr()
        ])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
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


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_flops_counting_methods(model):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    model.start_flops_count = start_flops_count.__get__(model)
    model.stop_flops_count = stop_flops_count.__get__(model)
    model.reset_flops_count = reset_flops_count.__get__(model)
    model.compute_total_flops = compute_total_flops.__get__(model)

    model.reset_flops_count()

    return model


def compute_total_flops(self):
    flops_sum = 0
    params_sum = 0
    for name, module in self.named_children():
        _ = name
        flops_sum += module.__flops__
    params_sum = get_model_parameters_number(self)
    return flops_sum / self.__batch__, params_sum


def start_flops_count(self, **kwargs):
    def register_module_hooks(module, verbose, ost, ignore_list):
        has_children = len(module._modules.items()) != 0

        def pre_hook(module, input):
            module_flop_count.clear()
            module.__batch__ += 1

        if has_children:

            def post_hook(module, input, output):
                module.__flops__ = sum([child.__flops__ for child in module.children()
                                        ]) + sum([elem[1] for elem in module_flop_count])
                # print("class ",
                #       module.__class__.__name__,
                #       " = ",
                #       len(mod._modules.items()))
                # print("flops={}".format(module.__flops__))
                module_flop_count.clear()

            module.register_forward_pre_hook(pre_hook)
            module.register_forward_hook(post_hook)
        else:

            def post_hook(module, input, output):
                # print("module_flop_count = {}".format(module_flop_count))
                flops = sum([elem[1] for elem in module_flop_count])
                module.__flops__ += flops
                # print("class ",
                #       module.__class__.__name__,
                #       " = ",
                #       len(mod._modules.items()))
                # print("flops={}".format(module.__flops__))
                module_flop_count.clear()

            module.register_forward_pre_hook(pre_hook)
            module.register_forward_hook(post_hook)

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


def stop_flops_count(self):
    if hasattr(self, '__batch__'):
        del self.__batch__
    if hasattr(self, '__flops__'):
        del self.__flops__
    if hasattr(self, '__params__'):
        del self.__params__


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
