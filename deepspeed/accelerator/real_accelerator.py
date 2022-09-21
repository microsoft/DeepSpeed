ds_accelerator = None


def get_accelerator():
    global ds_accelerator
    if ds_accelerator is None:
        from deepspeed.accelerator.cuda_accelerator import CUDA_Accelerator
        ds_accelerator = CUDA_Accelerator()

    assert ds_accelerator.is_available(), \
            f'CUDA_Accelerator fails is_available() test (import was successful)'
    return ds_accelerator


def set_accelerator(accel_obj):
    global ds_accelerator
    ds_accelerator = accel_obj


'''
-----------[code] test_get.py -----------
from deepspeed.accelerator.real_accelerator import get_accelerator
my_accelerator = get_accelerator()
print(f'{my_accelerator.name=}')
print(f'{my_accelerator.communication_backend=}')
print(f'{my_accelerator.HalfTensor().device=}')
print(f'{my_accelerator.total_memory()=}')
-----------[code] test_get.py -----------

---[output] python test_get.py---------
my_accelerator.name='cuda'
my_accelerator.communication_backend='nccl'
my_accelerator.HalfTensor().device=device(type='cuda', index=0)
my_accelerator.total_memory()=34089730048
---[output] python test_get.py---------

**************************************************************************
-----------[code] test_set.py -----------
from deepspeed.accelerator.cuda_accelerator import CUDA_Accelerator
cu_accel = CUDA_Accelerator()
print(f'{id(cu_accel)=}')
from deepspeed.accelerator.real_accelerator import set_accelerator, get_accelerator
set_accelerator(cu_accel)

my_accelerator = get_accelerator()
print(f'{id(my_accelerator)=}')
print(f'{my_accelerator.name=}')
print(f'{my_accelerator.communication_backend=}')
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
