from deepspeed.accelerator import literal_device

DEFAULT_WARMUPS = 5
DEFAULT_TRIALS = 50
DEFAULT_TYPE = 'float'
DEFAULT_BACKEND = 'ccl' if literal_device() == 'xpu' else 'nccl'
DEFAULT_UNIT = 'Gbps'
DEFAULT_DIST = 'deepspeed'
DEFAULT_MAXSIZE = 24
