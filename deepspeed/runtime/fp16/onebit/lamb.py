'''
Copyright 2020 The Microsoft DeepSpeed Team
'''
import types
import torch
import importlib
import numpy as np
import time
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from deepspeed.utils.logging import logger


class Lamb(torch.optim.Optimizer):
    """Implements the 1-bit Lamb algorithm. Currently GPU-only.
    For usage example please see, https://www.deepspeed.ai/tutorials/onebit-adam/
    It has been proposed in APMSqueeze (https://arxiv.org/abs/2008.11343)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        freeze_step (int, optional): Number of steps for warmup (uncompressed)
            stage before we start using compressed communication. (default 100000)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        max_coeff(float, optional): maximum value of the lamb coefficient (default: 10.0)
        min_coeff(float, optional): minimum value of the lamb coefficient (default: 0.01)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in 1-bit Adam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
        cuda_aware (boolean, required): Set True if the underlying MPI implementation
            supports CUDA-Aware communication. (default: False)
        comm_backend_name (string, optional): Set to 'mpi' if needed. (default: 'nccl')
            from cupy. (default: 'deepspeed')
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(self,
                 params,
                 deepspeed=None,
                 lr=1e-3,
                 freeze_step=100000,
                 bias_correction=True,
                 betas=(0.9,
                        0.999),
                 eps=1e-8,
                 eps_inside_sqrt=False,
                 weight_decay=0.,
                 max_grad_norm=0.,
                 max_coeff=10.0,
                 min_coeff=0.01,
                 amsgrad=False,
                 cuda_aware=False,
                 comm_backend_name='nccl',
                 coeff_beta=0.99,
                 compress_mode=0,
                 ratio_max=2.5,
                 ratio_min=0.5,
                 ratio_threshold=0.1,
                 linear_step=1000,
                 extra_stats=0):

        if amsgrad:
            raise RuntimeError('1-bit Lamb does not support the AMSGrad variant.')

        defaults = dict(lr=lr,
                        bias_correction=bias_correction,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm,
                        max_coeff=max_coeff,
                        min_coeff=min_coeff)

        super(Lamb, self).__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1
        assert (dist.is_initialized())

        self.comm_time = 0.0
        self.step_time = 0.0
        self.ave_step = 1
        self.bk_time = 0.0

        self.deepspeed = deepspeed
        self.adam_freeze_key = False
        self.initialize = False
        self.freeze_step = freeze_step
        self.cuda_aware = cuda_aware
        self.coeff_beta = coeff_beta
        self.compress_mode = int(compress_mode)
        self.ratio_max = ratio_max
        self.ratio_min = ratio_min
        self.ratio_threshold = ratio_threshold
        self.linear_step = int(linear_step)
        self.extra_stats = int(extra_stats)

        self.comm_backend_name = comm_backend_name

        # Empty initializer. Set handle based on the comm backend as follows.
        self.comm_backend_handle = None

        if self.comm_backend_name == 'nccl':
            assert torch.__version__.startswith("1.8."), "Please use torch 1.8 or greater to enable NCCL backend in 1-bit Adam. Alternatively, please specify 'mpi' as the 'comm_backend_name' in config file to proceed with the MPI backend"
            assert dist.is_initialized() == True, "Please initialize the torch distributed backend."
            from deepspeed.runtime.comm.nccl import NcclBackend
            self.comm_backend_handle = NcclBackend()

        elif self.comm_backend_name == 'mpi':
            from deepspeed.runtime.comm.mpi import MpiBackend
            self.comm_backend_handle = MpiBackend(cuda_aware)

        self.size = self.comm_backend_handle.size

        self.divider = int(self.size * 8 / np.gcd(self.size, 8))

        self.exp_avg_flat = []
        self.dummy_exp_avg = {}
        self.corrected_tensor_sizes = []
        self.server_chunk_sizes = []
        self.worker_errors = []
        self.server_errors = []

        self.lamb_coeffs = []

    def step(self, closure=None, grads=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced recision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients. (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        if grads is None:
            grads_group = [None] * len(self.param_groups)
        # backward compatibility
        # assuming a list/generator of parameter means single group
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif type(grads[0]) != list:
            grads_group = [grads]
        else:
            grads_group = grads

        #remove the previous stats
        del self.lamb_coeffs[:]

        if self.adam_freeze_key and self.compress_mode == 0:
            exp_avg_back_list = []
            for group in self.param_groups:
                exp_avg_back_list.append([])
                for p in group['params']:
                    exp_avg_back_list[-1].append(
                        self.state[p]['exp_avg'].detach().clone())

        for group, grads_this_group in zip(self.param_groups, grads_group):
            if grads_this_group is None:
                grads_this_group = [None] * len(group['params'])

            bias_correction = 1 if group['bias_correction'] else 0

            for p, grad in zip(group['params'], grads_this_group):
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'FusedAdam does not support sparse gradients, please consider SparseAdam instead'
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['lamb_coeff_freeze'] = 0.0
                    state['last_ratio'] = 1.0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if self.compress_mode == 0:
                        state['exp_avg_sq_back'] = torch.zeros_like(p.data)

                if not self.initialize:
                    self.adam_freeze_key = True

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                max_coeff = group['max_coeff']
                min_coeff = group['min_coeff']
                if self.compress_mode == 0:
                    exp_avg_sq_back = state['exp_avg_sq_back']

                state['step'] += 1

                if self.adam_freeze_key is False:
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    if self.compress_mode == 0 and state['step'] == self.freeze_step:
                        exp_avg_sq_back.data = exp_avg_sq.detach().clone()
                    grad = None
                    if self.initialize:
                        weight_norm = p.data.pow(2).sum().sqrt()
                        update = exp_avg / (exp_avg_sq.sqrt() + group['eps'])
                        if group['weight_decay'] > 0.0:
                            update += group['weight_decay'] * p.data
                        update_norm = update.pow(2).sum().sqrt()
                        lamb_coeff = 1.0
                        if weight_norm != 0 and update_norm != 0:
                            lamb_coeff = (weight_norm / update_norm).item()
                            if lamb_coeff > max_coeff:
                                lamb_coeff = max_coeff
                            if lamb_coeff < min_coeff:
                                lamb_coeff = min_coeff
                        if lamb_coeff != 1.0:
                            state['lamb_coeff_freeze'] = self.coeff_beta * state[
                                'lamb_coeff_freeze'] + (1 - self.coeff_beta) * lamb_coeff
                        self.lamb_coeffs.append(lamb_coeff)
                        with torch.no_grad():
                            p.add_(-group['lr'] * lamb_coeff * update)
                else:
                    if self.initialize:
                        exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    grad = None

        # init flattened momentums, worker/server error sizes
        if len(self.exp_avg_flat) == 0:
            for i, param_group in enumerate(self.param_groups):
                momentum_groups = [
                    self.state[p]['exp_avg'] for p in param_group['params']
                ]
                tensor_size = sum([torch.numel(p.data) for p in momentum_groups])
                corrected_tensor_size = tensor_size
                if tensor_size % (self.size * self.divider) != 0:
                    difference = ((self.size * self.divider) -
                                  (tensor_size % (self.size * self.divider)))
                    corrected_tensor_size += difference
                    self.dummy_exp_avg[i] = torch.zeros(
                        difference,
                        device=momentum_groups[0].data.device)
                    momentum_groups.append(self.dummy_exp_avg[i])
                self.corrected_tensor_sizes.append(corrected_tensor_size)
                self.server_chunk_sizes.append(corrected_tensor_size // self.size)

                self.exp_avg_flat.append(
                    # _flatten_dense_tensors([p.detach().clone()
                    _flatten_dense_tensors([p.clone().detach()
                                            for p in momentum_groups]))
                updated_params = _unflatten_dense_tensors(self.exp_avg_flat[i],
                                                          momentum_groups)
                for p, q in zip(momentum_groups, updated_params):
                    p.data = q.data

        if self.initialize and len(self.worker_errors) == 0:
            torch.cuda.empty_cache()
            for i in range(len(self.exp_avg_flat)):
                self.worker_errors.append(
                    torch.zeros(self.corrected_tensor_sizes[i],
                                device=self.exp_avg_flat[i].device))
                self.server_errors.append(
                    torch.zeros(self.server_chunk_sizes[i],
                                device=self.exp_avg_flat[i].device))
            torch.cuda.empty_cache()

        if self.adam_freeze_key:
            if self.size > 1 and self.linear_step != 0:
                for i in range(len(self.exp_avg_flat)):
                    if not self.initialize:
                        torch.cuda.empty_cache()
                        self.worker_errors.append(
                            torch.zeros(self.corrected_tensor_sizes[i],
                                        device=self.exp_avg_flat[i].device))
                        self.server_errors.append(
                            torch.zeros(self.server_chunk_sizes[i],
                                        device=self.exp_avg_flat[i].device))
                        torch.cuda.empty_cache()
                        if torch.distributed.get_rank() == 0:
                            print("Cupy Buffers Initialized Successfully.")

                        self.comm_backend_handle.compressed_allreduce(
                            self.exp_avg_flat[i],
                            self.worker_errors[0],
                            self.server_errors[0],
                            self.deepspeed.local_rank)

                        if torch.distributed.get_rank() == 0:
                            print('Pop out errors', flush=True)
                        del self.worker_errors[:]
                        del self.server_errors[:]
                    else:
                        self.comm_backend_handle.compressed_allreduce(
                            self.exp_avg_flat[i],
                            self.worker_errors[i],
                            self.server_errors[i],
                            self.deepspeed.local_rank)

        if self.adam_freeze_key and self.initialize:
            for i, group in enumerate(self.param_groups):
                bias_correction = 1 if group['bias_correction'] else 0

                for j, p in enumerate(group['params']):
                    state = self.state[p]
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']
                    max_coeff = group['max_coeff']
                    min_coeff = group['min_coeff']

                    if self.compress_mode == 0:
                        exp_avg_back = exp_avg_back_list[i][j]
                        exp_avg_sq_back = state['exp_avg_sq_back']
                        grad_recover = ((exp_avg - exp_avg_back * beta1) / (1 - beta1))
                        exp_avg_sq_back.mul_(beta2).addcmul_(1 - beta2,
                                                             grad_recover,
                                                             grad_recover)
                    denom = exp_avg_sq.sqrt() + group['eps']
                    update_prelim = exp_avg / denom

                    if group['weight_decay'] > 0.0:
                        update = update_prelim + group['weight_decay'] * p.data
                    else:
                        update = update_prelim

                    lamb_coeff = 1.0
                    if self.compress_mode == 0:
                        update_norm = update.pow(2).sum().sqrt()
                        denom_real = exp_avg_sq_back.sqrt() + group['eps']
                        ratio = (denom / denom_real).max().item()
                        if group['weight_decay'] > 0.0:
                            update_ratio = (update_prelim.pow(2).sum().sqrt() /
                                            update_norm).item()
                            update_ratio = min(1.0, update_ratio)
                            ratio = ratio * update_ratio + (1.0 - update_ratio)
                        if ratio > self.ratio_max:
                            ratio = self.ratio_max
                        if ratio < self.ratio_min:
                            ratio = self.ratio_min
                        if ratio > state['last_ratio'] * (1.0 + self.ratio_threshold):
                            ratio = state['last_ratio'] * (1.0 + self.ratio_threshold)
                        if ratio < state['last_ratio'] * (1.0 - self.ratio_threshold):
                            ratio = state['last_ratio'] * (1.0 - self.ratio_threshold)
                        state['last_ratio'] = ratio
                        lamb_coeff = state['lamb_coeff_freeze'] * ratio
                    elif self.compress_mode == 1:
                        ratio = min(
                            1.0,
                            float(state['step'] - self.freeze_step) /
                            (self.linear_step - self.freeze_step))
                        factor = 1.0 + self.ratio_max * ratio
                        lamb_coeff = state['lamb_coeff_freeze'] * factor
                    else:
                        lamb_coeff = min_coeff
                    self.lamb_coeffs.append(lamb_coeff)
                    with torch.no_grad():
                        p.add_(-group['lr'] * lamb_coeff * update)
            if self.compress_mode == 0:
                del exp_avg_back_list[:]
                exp_avg_back_list = None

        if not self.initialize:
            self.adam_freeze_key = False
            self.initialize = True
            print(
                f"Finished the initialization step at rank {torch.distributed.get_rank()}"
            )
            return loss

        if self.adam_freeze_key is False:
            if state['step'] >= self.freeze_step:
                self.adam_freeze_key = True
                self.deepspeed.enable_backward_allreduce = False

        return loss

    def get_lamb_coeffs(self):
        return self.lamb_coeffs
