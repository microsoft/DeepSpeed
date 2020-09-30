import math
import torch


class DeepSpeedCPUAdam(torch.optim.Optimizer):
    optimizer_id = 0
    ds_opt_adam = None

    def __init__(self,
                 model_params,
                 lr=1e-3,
                 betas=(0.9,
                        0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False):

        default_args = dict(lr=lr,
                            betas=betas,
                            eps=eps,
                            weight_decay=weight_decay,
                            amsgrad=amsgrad)
        super(DeepSpeedCPUAdam, self).__init__(model_params, default_args)

        self.opt_id = DeepSpeedCPUAdam.optimizer_id
        DeepSpeedCPUAdam.optimizer_id = DeepSpeedCPUAdam.optimizer_id + 1

        #global ds_opt_adam
        #ds_opt_adam = importlib.import_module('deepspeed.ops.adam.cpu_adam_op')
        ds_opt_adam = DeepSpeedCPUAdam.load_op()
        ds_opt_adam.create_adam(self.opt_id, lr, betas[0], betas[1], eps, weight_decay)

    @staticmethod
    def load_op():
        if DeepSpeedCPUAdam.ds_opt_adam is None:
            from torch.utils.cpp_extension import load
            DeepSpeedCPUAdam.ds_opt_adam = load(
                name='ds_cpu_adam',
                sources=['csrc/adam/cpu_adam.cpp',
                         'csrc/adam/custom_cuda_kernel.cu'],
                extra_include_paths=['csrc/includes/',
                                     '/usr/local/cuda/include'],
                extra_cflags=[
                    '-O3',
                    '-march=native',
                    '-std=c++14',
                    '-L/usr/local/cuda/lib64',
                    '-lcudart',
                    '-lcublas',
                    '-g',
                    '-Wno-reorder',
                    '-fopenmp',
                    '-D__AVX256__'
                ],
                extra_cuda_cflags=[
                    '-O3',
                    '--use_fast_math',
                    '-gencode',
                    'arch=compute_61,code=compute_61',
                    '-gencode',
                    'arch=compute_70,code=compute_70',
                    '-std=c++14',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__'
                ],
                verbose=True)
        return DeepSpeedCPUAdam.ds_opt_adam

    def __setstate__(self, state):
        super(DeepSpeedCPUAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None, fp16_param_groups=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):

                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    print(f'group {group_id} param {param_id} = {p.numel()}')
                    state['step'] = 0
                    # gradient momentums
                    state['exp_avg'] = torch.zeros_like(p.data, device='cpu')
                    # gradient variances
                    state['exp_avg_sq'] = torch.zeros_like(p.data, device='cpu')

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                ds_opt_adam = DeepSpeedCPUAdam.load_op()
                if fp16_param_groups is not None:
                    p_fp16 = fp16_param_groups[group_id][param_id]
                    ds_opt_adam.adam_update_copy(self.opt_id,
                                                 p.data,
                                                 grad,
                                                 exp_avg,
                                                 exp_avg_sq,
                                                 p_fp16)
                else:
                    ds_opt_adam.adam_update(self.opt_id,
                                            p.data,
                                            grad,
                                            exp_avg,
                                            exp_avg_sq)
        return loss
