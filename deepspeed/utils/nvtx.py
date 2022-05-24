import torch
from deepspeed.accelerator import literal_device
from deepspeed.accelerator import runtime as accel_runtime


def instrument_w_nvtx(func):
    """decorator that causes an NVTX range to be recorded for the duration of the
    function call."""
    if ((literal_device() == 'cuda' and hasattr(torch.cuda.nvtx,
                                                "range_push"))
            or (literal_device() == 'xpu' and hasattr(torch.xpu.itt,
                                                      "range_push"))):

        def wrapped_fn(*args, **kwargs):
            accel_runtime.range_push(func.__qualname__)
            ret_val = func(*args, **kwargs)
            accel_runtime.range_pop()
            return ret_val

        return wrapped_fn
    else:
        return func
