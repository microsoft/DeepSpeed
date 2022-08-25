import torch
global device_type


def set_device_type(device_string):
    global device_type

    if device_string == 'cuda' or device_string == 'xpu':
        device_type = device_string
    else:
        print("Warning: unrecognized device type, default to 'cuda' device")
        device_type = 'cuda'


def literal_device(ordinal=None):
    global device_type

    if ordinal == None:
        return device_type
    else:
        return "{}:{}".format(device_type, ordinal)


'''
Check whether the tensor is on the designated device type
'''


def on_accel_device(tensor):
    global device_type

    device_str = str(tensor.device)
    if device_str.startswith('{}:'.format(device_type)):
        return True
    else:
        return False


# because xpu may not be imported, catch all possible errors
try:
    import intel_extension_for_pytorch as ipex  # noqa: F401
    import oneccl_bindings_for_pytorch  # noqa: F401
    if torch.xpu.is_available():
        device = 'xpu'
    else:
        device = ''
except AttributeError:
    device = ''
except ModuleNotFoundError:
    device = ''

if device == '':
    set_device_type('cuda')
else:
    set_device_type(device)
