#!/usr/bin/env python
# run the benchmark under timeit (-t), cProfile (-c), line_profiler (-l)
#
# usage:
# ./tmap_bench.py -t
# ./tmap_bench.py -c
# kernprof -l tmap_bench.py -l; python -m line_profiler  tmap_bench.py.lprof
import argparse
import gc
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import TMapBuilder

tmap_ops = TMapBuilder().load()
tensor_map = tmap_ops.TensorMap()

torch.manual_seed(0)
# Simulate random 2D input signature used to index cuda graph
inputKey = torch.randint(1, 1000, (1000, )).to(get_accelerator().device_name())
inputValue = torch.randint(1, 1000, (1000, )).to(get_accelerator().device_name())

# Warmup and test same output is produced
pyTestDict = {}
pyTestDict[inputKey] = inputValue
tensor_map.insert(inputKey, inputValue)

pyValue = pyTestDict[inputKey]
cppValue = tensor_map.read(inputKey)
assert torch.eq(pyValue, cppValue).all(), "both produce the same tensor"

TIMES = 1000


# the programs being tested
def py():
    for i in range(TIMES):
        pyTestDict[inputKey] = inputValue
        pyValue = pyTestDict[inputKey]


def cpp():
    for i in range(TIMES):
        tensor_map.insert(inputKey, inputValue)
        cppValue = tensor_map.read(inputKey)


#### cProfile ####

import cProfile


def cprofileme():
    print("--------------- cProfile -----------------")
    print("py")
    cProfile.run("py()", sort=-1)
    gc.collect()
    get_accelerator().empty_cache()
    print("cpp")
    cProfile.run("cpp()", sort=-1)
    gc.collect()
    get_accelerator().empty_cache()


#### timeit ####

import timeit


def timeme():
    print("--------------- timeit -----------------")
    print(f'py  ={timeit.Timer("py()", globals=globals()).timeit(number=1)}')
    gc.collect()
    get_accelerator().empty_cache()
    print(f'cpp ={timeit.Timer("cpp()", globals=globals()).timeit(number=1)}')
    gc.collect()
    get_accelerator().empty_cache()


#### line_profiler ####
# this one requires a special way to be called
# pip install line_profiler
# kernprof -l tmap_bench.py -l; python -m line_profiler  tmap_bench.py.lprof


def line_profileme():
    print("--------------- line_profiler -----------------")
    print("py")
    profile(py)()  # noqa: F821
    gc.collect()
    get_accelerator().empty_cache()
    print("cpp")
    profile(cpp)()  # noqa: F821
    gc.collect()
    get_accelerator().empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", action='store_true')
    parser.add_argument("-c", action='store_true')
    parser.add_argument("-t", action='store_true')
    args = parser.parse_args()
    if args.l:
        line_profileme()
    elif args.c:
        cprofileme()
    elif args.t:
        timeme()
