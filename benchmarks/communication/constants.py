from deepspeed.accelerator.real_accelerator import get_accelerator

DEFAULT_WARMUPS = 5
DEFAULT_TRIALS = 50
DEFAULT_TYPE = 'float'
DEFAULT_BACKEND = 'ccl' if get_accelerator().device_name() == 'xpu' else 'nccl'
DEFAULT_UNIT = 'Gbps'
DEFAULT_DIST = 'deepspeed'
DEFAULT_MAXSIZE = 24
