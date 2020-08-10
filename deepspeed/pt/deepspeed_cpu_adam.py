import math
import torch


class CPUAdam(torch.optim.Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9,
                        0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        amsgrad=amsgrad)
        super(CPUAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CPUAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step_with_cpuoffload(self,
                             closure=None,
                             fp32_params=None,
                             fp32_params_grad=None,
                             exp_avg=None,
                             exp_avg_sq=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            fp32_params: fp32 params on CPU
            grads
        """
        loss = None
        if closure is not None:
            loss = closure()

        if fp32_params is None:
            raise RuntimeError('params is None')

        index = 0

        for group in self.param_groups:
            group_size = sum([t.numel() for t in group['params']])
            p = torch.zeros(group_size, device=torch.device('cpu'), requires_grad=True)
            p = fp32_params[index:index + group_size].detach()
            p_grad = torch.zeros(group_size, device=torch.device('cpu'))
            p_grad = fp32_params_grad[index:index + group_size].detach()
            p.grad = p_grad
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError(
                    'Adam does not support sparse gradients, please consider SparseAdam instead'
                )
            #amsgrad = group['amsgrad']

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0

            beta1, beta2 = group['betas']

            state['step'] += 1

            if group['weight_decay'] != 0:
                grad.add_(group['weight_decay'], p.data)

            # Decay the first and second moment running average coefficient
            exp_avg[index:index + group_size].mul_(beta1).add_(1 - beta1, grad)
            exp_avg_sq[index:index + group_size].mul_(beta2).addcmul_(
                1 - beta2,
                grad,
                grad)

            denom = exp_avg_sq[index:index + group_size].sqrt().add_(group['eps'])

            bias_correction1 = 1 - beta1**state['step']
            bias_correction2 = 1 - beta2**state['step']
            step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

            p.data.addcdiv_(-step_size, exp_avg[index:index + group_size], denom)

            index += group_size

        return loss
        '''
        for i, p in emulrate(fp32_params)
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            #amsgrad = group['amsgrad']

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0

            beta1, beta2 = group['betas']

            state['step'] += 1

            if group['weight_decay'] != 0:
                grad.add_(group['weight_decay'], p.data)

            # Decay the first and second moment running average coefficient
            exp_avg[i].mul_(beta1).add_(1 - beta1, grad)
            exp_avg_sq[i].mul_(beta2).addcmul_(1 - beta2, grad, grad)

            denom = exp_avg_sq[i].sqrt().add_(group['eps'])

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

            p.data.addcdiv_(-step_size, exp_avg[i], denom)

        returns
        '''
