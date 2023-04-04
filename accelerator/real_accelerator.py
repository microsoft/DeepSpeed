# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

try:
    from accelerator.abstract_accelerator import DeepSpeedAccelerator as dsa1
except ImportError as e:
    dsa1 = None
try:
    from deepspeed.accelerator.abstract_accelerator import DeepSpeedAccelerator as dsa2
except ImportError as e:
    dsa2 = None

ds_accelerator = None


def _validate_accelerator(accel_obj):
    # because abstract_accelerator has different path during
    # build time (accelerator.abstract_accelerator)
    # and run time (deepspeed.accelerator.abstract_accelerator)
    # and extension would import the
    # run time abstract_accelerator/DeepSpeedAccelerator as its base
    # class, so we need to compare accel_obj with both base class.
    # if accel_obj is instance of DeepSpeedAccelerator in one of
    # accelerator.abstractor_accelerator
    # or deepspeed.accelerator.abstract_accelerator, consider accel_obj
    # is a conforming object
    if not ((dsa1 != None and isinstance(accel_obj, dsa1)) or (dsa2 != None and isinstance(accel_obj, dsa2))):
        raise AssertionError(f'{accel_obj.__class__.__name__} accelerator is not subclass of DeepSpeedAccelerator')

    # TODO: turn off is_available test since this breaks tests
    #assert accel_obj.is_available(), \
    #    f'{accel_obj.__class__.__name__} accelerator fails is_available() test'


def get_accelerator():
    global ds_accelerator
    if ds_accelerator is None:
        try:
            from intel_extension_for_deepspeed import XPU_Accelerator
        except ImportError as e:
            pass
        else:
            ds_accelerator = XPU_Accelerator()
            _validate_accelerator(ds_accelerator)
            print(f"Setting ds_accelerator to {ds_accelerator._name} (auto-detect)")
            return ds_accelerator

        # We need a way to choose between CUDA_Accelerator and CPU_Accelerator
        # Currently we detect whether intel_etension_for_pytorch is installed
        # in the environment and use CPU_Accelerator if the answewr is True.
        # An alternative might be detect whether CUDA device is installed on
        # the system but this comes with two pitfalls:
        # 1. the system may not have torch pre-installed, so
        #    get_accelerator().is_avaiable() may not work.
        # 2. Some scenario like install on login node (without CUDA device)
        #    and run on compute node (with CUDA device) may cause mismatch
        #    between installation time and runtime.
        try:
            import intel_extension_for_pytorch  # noqa: F401
        except ImportError as e:
            from .cuda_accelerator import CUDA_Accelerator
            ds_accelerator = CUDA_Accelerator()
            _validate_accelerator(ds_accelerator)
            print(f"Setting ds_accelerator to {ds_accelerator._name} (auto-detect)")
            return ds_accelerator
        else:
            from .cpu_accelerator import CPU_Accelerator
            ds_accelerator = CPU_Accelerator()
            _validate_accelerator(ds_accelerator)
            print(f"Setting ds_accelerator to {ds_accelerator._name} (auto-detect)")
            return ds_accelerator
    else:
        return ds_accelerator


def set_accelerator(accel_obj):
    global ds_accelerator
    _validate_accelerator(accel_obj)
    print(f"Setting ds_accelerator to {accel_obj._name}")
    ds_accelerator = accel_obj


'''
-----------[code] test_get.py -----------
from deepspeed.accelerator import get_accelerator
my_accelerator = get_accelerator()
print(f'{my_accelerator._name=}')
print(f'{my_accelerator._communication_backend=}')
print(f'{my_accelerator.HalfTensor().device=}')
print(f'{my_accelerator.total_memory()=}')
-----------[code] test_get.py -----------

---[output] python test_get.py---------
my_accelerator.name()='cuda'
my_accelerator.communication_backend='nccl'
my_accelerator.HalfTensor().device=device(type='cuda', index=0)
my_accelerator.total_memory()=34089730048
---[output] python test_get.py---------

**************************************************************************
-----------[code] test_set.py -----------
from deepspeed.accelerator.cuda_accelerator import CUDA_Accelerator
cu_accel = CUDA_Accelerator()
print(f'{id(cu_accel)=}')
from deepspeed.accelerator import set_accelerator, get_accelerator
set_accelerator(cu_accel)

my_accelerator = get_accelerator()
print(f'{id(my_accelerator)=}')
print(f'{my_accelerator._name=}')
print(f'{my_accelerator._communication_backend=}')
print(f'{my_accelerator.HalfTensor().device=}')
print(f'{my_accelerator.total_memory()=}')
-----------[code] test_set.py -----------


---[output] python test_set.py---------
id(cu_accel)=139648165478304
my_accelerator=<deepspeed.accelerator.cuda_accelerator.CUDA_Accelerator object at 0x7f025f4bffa0>
my_accelerator.name='cuda'
my_accelerator.communication_backend='nccl'
my_accelerator.HalfTensor().device=device(type='cuda', index=0)
my_accelerator.total_memory()=34089730048
---[output] python test_set.py---------
'''
