import argparse
import torch
import time
import numpy as np
import pytest
import copy

import deepspeed
from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
from deepspeed.ops.op_builder import CPUAdagradBuilder

if not deepspeed.ops.__compatible_ops__[CPUAdagradBuilder.NAME]:
  pytest.skip("cpu-adagrad is not compatible")


def check_equal(first, second, atol=1e-2, verbose=False):
  x = first.detach().numpy()
  y = second.detach().numpy()
  if verbose:
    print("x = {}".format(x.flatten()))
    print("y = {}".format(y.flatten()))
    print('-' * 80)
  np.testing.assert_allclose(x, y, err_msg="param-update dismatch!", atol=atol)


@pytest.mark.parametrize('model_size',
                         [
                             (64),
                             (22),
                             (55),
                             (127),
                             (1024),
                             (1048576),
                             (30000000),
                         ]) # yapf: disable
def test_cpu_adam_opt(model_size):
  device = 'cpu'
  rng_state = torch.get_rng_state()
  param = torch.nn.Parameter(torch.randn(model_size, device=device))
  torch.set_rng_state(rng_state)
  param1 = torch.nn.Parameter(torch.randn(model_size, device=device))
  torch.set_rng_state(rng_state)

  optimizer = DeepSpeedCPUAdagrad([param])
  optimizer1 = torch.optim.Adagrad([param1])

  opt_timer = 0.
  opt_timer1 = 0.
  for i in range(10):
    rng_state = torch.get_rng_state()
    param.grad = torch.randn(model_size, device=device)
    torch.set_rng_state(rng_state)
    param1.grad = torch.randn(model_size, device=device)
    start = time.time()
    optimizer.step()
    opt_timer += time.time() - start

    start = time.time()
    optimizer1.step()
    opt_timer1 += time.time() - start

  check_equal(param, param1, atol=1e-2, verbose=True)
  print(f"opt_timer: {opt_timer}\n" f"opt_timer1: {opt_timer1}\n")


@pytest.mark.parametrize('model_size',
                         [
                             # (10),
                             # (22),
                             # (55),
                             # (127),
                             # (1024),
                             # (1048576),
                             (22440000),
                         ]) # yapf: disable
def test_cpu_adam_opt_sparse(model_size):
  device = 'cpu'
  rng_state = torch.get_rng_state()

  def gen_sparse_tensor(vocabulary_size, dim, num_indices, dtype, device):
    i = torch.randint(vocabulary_size,
                      size=(1, num_indices),
                      dtype=torch.int64,
                      device=device)
    v = torch.randn(num_indices, dim, dtype=dtype, device=device)
    t = torch.sparse_coo_tensor(i, v, (vocabulary_size, dim), device=device)
    t = t.coalesce()
    new_i = (t.indices().view(-1, 1).repeat(1, dim) * dim +
             torch.tensor(range(dim))).flatten().unsqueeze(0)
    new_v = t.values().flatten()
    new_t = torch.sparse_coo_tensor(new_i,
                                    new_v, (vocabulary_size * dim,),
                                    device=device)
    new_t = new_t.coalesce()
    new_t.requires_grad = False
    return new_t

  voc_size = (120000000 // 8)
  dim = 25
  num_indices = int(model_size // dim)
  dtype = torch.float32

  param = torch.nn.Parameter(torch.randn((voc_size * dim,),
                                         dtype=dtype,
                                         device=device),
                             requires_grad=True)
  torch.set_rng_state(rng_state)
  param1 = torch.nn.Parameter(torch.randn((voc_size * dim,),
                                          dtype=dtype,
                                          device=device),
                              requires_grad=True)
  torch.set_rng_state(rng_state)

  optimizer = DeepSpeedCPUAdagrad([param])
  optimizer1 = torch.optim.Adagrad([param1])

  opt_timer = 0.
  opt_timer1 = 0.
  for i in range(10):
    torch.set_rng_state(rng_state)
    param.grad = gen_sparse_tensor(voc_size,
                                   dim,
                                   num_indices,
                                   dtype=dtype,
                                   device=device)
    torch.set_rng_state(rng_state)
    param1.grad = gen_sparse_tensor(voc_size,
                                    dim,
                                    num_indices,
                                    dtype=dtype,
                                    device=device)
    start = time.time()
    optimizer.step()
    opt_timer += time.time() - start

    start = time.time()
    optimizer1.step()
    opt_timer1 += time.time() - start

  check_equal(param, param1, atol=1e-2, verbose=True)
  print(f"opt_timer: {opt_timer}\n" f"opt_timer1: {opt_timer1}\n")
