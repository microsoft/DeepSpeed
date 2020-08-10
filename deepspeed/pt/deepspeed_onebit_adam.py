'''
Copyright 2019 The Microsoft DeepSpeed Team test
Copyright NVIDIA/apex
This file is adapted from NVIDIA/apex/optimizer/fused_adam and implements the LAMB optimizer
'''
import types
import torch
import importlib
import numpy as np
import time
import cupy
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

class OnebitAdam(torch.optim.Optimizer):
    """Implements LAMB algorithm. Currently GPU-only.  Requires DeepSpeed adapted Apex to be installed via
    ``python setup.py install --cuda_ext --cpp_ext``.
    For usage example please see, TODO DeepSpeed Tutorial
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes.
    https://arxiv.org/abs/1904.00962
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        max_coeff(float, optional): maximum value of the lamb coefficient (default: 10.0)
        min_coeff(float, optional): minimum value of the lamb coefficient (default: 0.01)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(self,
                 params,
                 deepspeed=None,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9,
                        0.999),
                 eps=1e-8,
                 eps_inside_sqrt=False,
                 weight_decay=0.,
                 max_grad_norm=0.,
                 amsgrad=False,
                 threshold = 0.001):

        if amsgrad:
            raise RuntimeError('FusedLamb does not support the AMSGrad variant.')
        defaults = dict(lr=lr,
                        bias_correction=bias_correction,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(OnebitAdam, self).__init__(params, defaults)
        from mpi4py import MPI
        self.eps_mode = 0 if eps_inside_sqrt else 1

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.comm_time = 0.0
        self.step_time = 0.0
        self.ave_step = 1
        self.bk_time = 0.0
        self.divider = int(self.size * 8 / np.gcd(self.size, 8))
        self.deepspeed = deepspeed
        self.adam_freeze_key = False
        self.threshold = threshold

    def myIgather(self, rank, size, comm, sendbuf, recbuf, root):
        req = []
        if rank == root:
            for idx in range(size):
                if idx != rank:
                    req.append(comm.Irecv(recbuf[idx], source=idx))
                else:
                    recbuf[rank] = sendbuf
        else:
            req.append(comm.Isend(sendbuf, dest=root))

        return req

    def torch2cupy(self, tensor):
        return cupy.fromDlpack(to_dlpack(tensor))

    def cupy2torch(self, cupy_tensor):
        return from_dlpack(cupy_tensor.toDlpack())

    def compress_by_chunk(self, cupy_bool_tensor, num_chunks):
        packed_sign = cupy.packbits(cupy_bool_tensor)
        sign_list_packed = cupy.split(packed_sign, num_chunks)
        return sign_list_packed

    def Compressed_Allreduce(self, buffer_m: torch.tensor, worker_error, server_error, rank, world_size, comm):
        from mpi4py import MPI
        all_start_time = time.time()
        original_size = buffer_m.numel()
        cupy.cuda.Device(rank % torch.cuda.device_count()).use()
        if torch.numel(buffer_m) != torch.numel(worker_error):
            empty_tensor = torch.zeros(torch.numel(worker_error) - torch.numel(buffer_m), device=buffer_m.device)
            buffer_m = torch.cat([buffer_m, empty_tensor])
        buffer_m.add_(worker_error)
        worker_scale = torch.norm(buffer_m) / np.sqrt(torch.numel(buffer_m))
        buffer_m_bk = buffer_m
        worker_error_bk = worker_error
        partition = buffer_m_bk.numel() // 8
        for i in range(8):
            start = i * partition
            buffer_m = buffer_m_bk.narrow(0, start, partition)
            worker_error = worker_error_bk.narrow(0,start, partition)
            worker_error.set_(buffer_m - worker_scale * buffer_m.sign())

        compensated_buffer_m = buffer_m_bk
        compensated_buffer_m.sign_()
        compensated_buffer_m = compensated_buffer_m.add_(1).bool()
        cupy_worker_scale = self.torch2cupy(worker_scale)
        cupy_compensated_buffer_m = self.torch2cupy(compensated_buffer_m)
        compensated_buffer_m = None
        # del compensated_buffer_m
        # del buffer_m
        # print(cupy_compensated_buffer_m)

        cupy_sign_list_packed = self.compress_by_chunk(cupy_compensated_buffer_m, world_size)
        cupy_compensated_buffer_m = None
        # del cupy_compensated_buffer_m

        cupy_recvbuf_sign = cupy.zeros([world_size, cupy_sign_list_packed[rank].size],
                                       dtype=cupy_sign_list_packed[0].dtype)
        cupy_recvbuf_scale = cupy.zeros([world_size, 1], dtype=cupy_worker_scale.dtype)
        requests = []

        gather_start = time.time()
        for idx in range(world_size):
            req_sign = self.myIgather(rank, world_size, comm, cupy_sign_list_packed[idx], cupy_recvbuf_sign, root=idx)
            requests += req_sign
        for idx in range(world_size):
            req_scale = self.myIgather(rank, world_size, comm, cupy_worker_scale, cupy_recvbuf_scale, root=idx)
            requests += req_scale
        MPI.Request.Waitall(requests)
        gather_end = time.time()

        cupy_unpacked_sign = (cupy.unpackbits(cupy_recvbuf_sign.flatten())).reshape(world_size, -1)
        cupy_recvbuf_sign = None
        # del cupy_recvbuf_sign
        unpacked_sign = self.cupy2torch(cupy_unpacked_sign).float()
        cupy_unpacked_sign = None
        # del cupy_unpacked_sign
        unpacked_sign = unpacked_sign.add_(-0.5).mul_(2.0)
        worker_scale = self.cupy2torch(cupy_recvbuf_scale).mul_(1/world_size)
        compensated_server_m = unpacked_sign.mul_(worker_scale).sum(0)
        unpacked_sign = None
        # del unpacked_sign
        compensated_server_m.add_(server_error)
        server_scale = torch.norm(compensated_server_m) / np.sqrt(compensated_server_m.numel())
        server_error.set_(compensated_server_m - server_scale * compensated_server_m.sign())

        compensated_server_m.sign_()
        compensated_server_m = compensated_server_m.add_(1).bool()
        cupy_server_scale = self.torch2cupy(server_scale)
        cupy_compensated_server_m = self.torch2cupy(compensated_server_m)
        compensated_server_m = None
        # del compensated_server_m

        cupy_server_sign_packed = self.compress_by_chunk(cupy_compensated_server_m, 1)

        cupy_recvbuf_sign_server = cupy.zeros([world_size, cupy_server_sign_packed[0].size],
                                              dtype=cupy_sign_list_packed[0].dtype)
        cupy_recvbuf_scale_server = cupy.zeros([world_size, 1], dtype=cupy_worker_scale.dtype)

        allgather_start = time.time()
        comm.Allgather(cupy_server_sign_packed[0], cupy_recvbuf_sign_server)
        comm.Allgather(cupy_server_scale, cupy_recvbuf_scale_server)
        allgather_end = time.time()

        cupy_server_unpacked_sign = (cupy.unpackbits(cupy_recvbuf_sign_server.flatten())).reshape(world_size, -1)
        cupy_recvbuf_sign_server = None
        # del cupy_recvbuf_sign_server
        server_unpacked_sign = self.cupy2torch(cupy_server_unpacked_sign)
        cupy_server_unpacked_sign = None
        # del cupy_server_unpacked_sign
        server_unpacked_sign = server_unpacked_sign.float().add_(-0.5).mul_(2.0)
        server_scale = self.cupy2torch(cupy_recvbuf_scale_server)
        buffer_m = server_unpacked_sign.mul_(server_scale).flatten()[0:original_size]

        cupy._default_memory_pool.free_all_blocks()

        return buffer_m


    def step(self,
             closure=None,
             grads=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients. (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        gather_time = 0
        allgather_time = 0
        all_time = 0


        if self.adam_freeze_key is False:
            v_diff_buffer = 0.0

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
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data) + 10000

                    state['tensor_size'] = torch.numel(p.data)
                    state['corrected_tensor_size'] = state['tensor_size']

                    if state['tensor_size'] % (self.size * self.divider) != 0:
                        state['corrected_tensor_size'] += (
                                (self.size * self.divider) - (state['tensor_size'] % (self.size * self.divider)))
                    state['server_chunk_size'] = state['corrected_tensor_size'] // self.size
                    state['worker_error'] = torch.zeros(state['corrected_tensor_size'], device=p.device)
                    state['server_error'] = torch.zeros(state['server_chunk_size'], device=p.device)


                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # print('I am Here')
                if self.adam_freeze_key is False:
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    # v_diff = -beta2 * exp_avg_sq + beta2 * grad * grad
                    # v_diff_buffer += v_diff.norm() / exp_avg_sq.norm() / state['tensor_size']
                    # exp_avg_sq.add_(v_diff).addcmul_(1 - beta2, grad, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    grad = None
                    # v_diff = None

                else:
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    grad = None
                    torch.cuda.synchronize()
                    cupy.cuda.get_current_stream().synchronize()
                    if self.size > 1:
                        exp_avg = self.Compressed_Allreduce(
                            exp_avg,
                            state['worker_error'],
                            state['server_error'],
                            self.rank,
                            self.size, self.comm)
                        # print('Rank is {}, Inside the optimizer the step is: {}'.format(self.rank, state['step']))
                        cupy._default_memory_pool.free_all_blocks()
                        torch.cuda.synchronize()
                        cupy.cuda.get_current_stream().synchronize()


                update = exp_avg / (exp_avg_sq.sqrt() + group['eps'])
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data
                with torch.no_grad():
                    p.add_(-group['lr'] * update)
                # torch.cuda.synchronize()

        # if self.adam_freeze_key is True:
        #     print('Using Mavapich2 the communication time is {:.2f}ms, compression takes {:.2f}ms'.format((gather_time + allgather_time)*1000, (all_time - (gather_time + allgather_time) )* 1000))

        if self.adam_freeze_key is False:
            # if False:
            if state['step'] > 200:
            # if v_diff_buffer >= self.threshold:
                self.adam_freeze_key = True
                self.deepspeed.enable_backward_allreduce = False

        return loss
