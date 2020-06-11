"""
Copyright 2019 The Microsoft DeepSpeed Team

Implementation of learning rate schedules.

Taken and modified from PyTorch v1.0.1 source
https://github.com/pytorch/pytorch/blob/v1.1.0/torch/optim/lr_scheduler.py

"""

import argparse
from torch.optim import Optimizer
from typing import Union, List
import math
from deepspeed.pt.deepspeed_constants import *
from deepspeed.pt.log_utils import logger
from deepspeed.pt.deepspeed_constants import LRScheduleConstants
from deepspeed.pt.args import create_lr_tuning_parser
from deepspeed.pt.args import parse_lr_tuning_args


def add_tuning_arguments(parser):
    return create_lr_tuning_parser(parser)


def parse_arguments():
    return parse_lr_tuning_args()


def _single_override(args, key, params):
    """Override a single param if appears in args"""
    if getattr(args, key, None) is not None:
        params[key] = getattr(args, key)


def override_lr_range_test_params(args, params):
    keys = [
        LRScheduleConstants.LR_RANGE_TEST_MIN_LR.name,
        LRScheduleConstants.LR_RANGE_TEST_STEP_RATE.name,
        LRScheduleConstants.LR_RANGE_TEST_STEP_SIZE.name,
        LRScheduleConstants.LR_RANGE_TEST_STAIRCASE.name,
    ]
    for key in keys:
        _single_override(args, key, params)


def override_1cycle_params(args, params):
    keys = [
        LRScheduleConstants.CYCLE_FIRST_STEP_SIZE.name,
        LRScheduleConstants.CYCLE_FIRST_STAIR_COUNT.name,
        LRScheduleConstants.CYCLE_SECOND_STEP_SIZE.name,
        LRScheduleConstants.CYCLE_SECOND_STAIR_COUNT.name,
        LRScheduleConstants.DECAY_STEP_SIZE.name,
        LRScheduleConstants.CYCLE_MIN_LR.name,
        LRScheduleConstants.CYCLE_MAX_LR.name,
        LRScheduleConstants.CYCLE_MIN_MOM.name,
        LRScheduleConstants.CYCLE_MAX_MOM.name,
        LRScheduleConstants.DECAY_MOM_RATE.name,
        LRScheduleConstants.DECAY_LR_RATE.name,
    ]
    for key in keys:
        _single_override(args, key, params)


def override_warmupLR_params(args, params):
    keys = [
        LRScheduleConstants.WARMUP_MAX_LR.name,
        LRScheduleConstants.WARMUP_MIN_LR.name,
        LRScheduleConstants.WARMUP_NUM_STEPS.name,
    ]
    for key in keys:
        _single_override(args, key, params)


def override_params(args, params):
    # LR range test params
    override_lr_range_test_params(args, params)

    # 1Cycle params
    override_1cycle_params(args, params)

    # WarmupLR params
    override_warmupLR_params(args, params)


def get_config_from_args(args):
    if getattr(args, LRScheduleConstants.LR_SCHEDULE.name, None) is None:
        return None, '--{} not specified on command line'.format(LRScheduleConstants.LR_SCHEDULE.name)

    if not args.lr_schedule in LRScheduleConstants.VALID_LR_SCHEDULES:
        return None, '{} is not supported LR schedule'.format(args.lr_schedule)

    config = {}
    config['type'] = args.lr_schedule
    config['params'] = {}

    if args.lr_schedule == LRScheduleConstants.LR_RANGE_TEST.name:
        override_lr_range_test_params(args, config['params'])
    elif args.lr_schedule == LRScheduleConstants.ONE_CYCLE.name:
        override_1cycle_params(args, config['params'])
    else:
        override_warmupLR_params(args, config['params'])

    return config, None


def get_lr_from_config(config):
    if not 'type' in config:
        return None, 'LR schedule type not defined in config'

    if not 'params' in config:
        return None, 'LR schedule params not defined in config'

    lr_schedule = config['type']
    lr_params = config['params']

    if not lr_schedule in LRScheduleConstants.VALID_LR_SCHEDULES:
        return None, '{} is not a valid LR schedule'.format(lr_schedule)

    if lr_schedule == LRScheduleConstants.LR_RANGE_TEST.name:
        return lr_params[LRScheduleConstants.LR_RANGE_TEST_MIN_LR.name], ''
    if lr_schedule == LRScheduleConstants.ONE_CYCLE.name:
        return lr_params[LRScheduleConstants.CYCLE_MAX_LR.name], ''
    # Warmup LR
    return lr_params[LRScheduleConstants.WARMUP_MAX_LR.name], ''


"""
Only optimizers that are subclass of torch.optim.Optimizer are supported. So check the passed optimizer and wrapped
optimizer to see if requirement is satisfied.
TODO: Looking under the hood to examine the wrapped optimizer is a hack that requires a better long-term fix.
"""


def get_torch_optimizer(optimizer):
    if isinstance(optimizer, Optimizer):
        return optimizer

    if hasattr(optimizer, 'optimizer') and isinstance(optimizer.optimizer, Optimizer):
        return optimizer.optimizer

    raise TypeError('{} is not a subclass of torch.optim.Optimizer'.format(
        type(optimizer).__name__))


class LRRangeTest(object):
    """Sets the learning rate of each parameter group according to
    learning rate range test (LRRT) policy. The policy increases learning
    rate starting from a base value with a constant frequency, as detailed in
    the paper `A disciplined approach to neural network hyper-parameters: Part1`_.

    LRRT policy is used for finding maximum LR that trains a model without divergence, and can be used to
    configure the LR boundaries for Cylic LR schedules.

    LRRT changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_range_test_min_lr (float or list): Initial learning rate which is the
            lower boundary in the range test for each parameter group.
        lr_range_test_step_size (int): Interval of training steps to increase learning rate. Default: 2000
        lr_range_test_step_rate (float): Scaling rate for range test. Default: 1.0
        lr_range_test_staircase (bool): Scale in staircase fashion, rather than continous. Default: False.
        last_batch_iteration (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_batch_iteration=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.LRRangeTest(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()

        _A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay:
        https://arxiv.org/abs/1803.09820
"""
    def __init__(self,
                 optimizer: Optimizer,
                 lr_range_test_min_lr: float = 1e-3,
                 lr_range_test_step_size: int = 2000,
                 lr_range_test_step_rate: float = 1.0,
                 lr_range_test_staircase: bool = False,
                 last_batch_iteration: int = -1):

        self.optimizer = get_torch_optimizer(optimizer)

        if isinstance(lr_range_test_min_lr,
                      list) or isinstance(lr_range_test_min_lr,
                                          tuple):
            if len(lr_range_test_min_lr) != len(self.optimizer.param_groups):
                raise ValueError("expected {} lr_range_test_min_lr, got {}".format(
                    len(self.optimizer.param_groups),
                    len(lr_range_test_min_lr)))
            self.min_lr = list(lr_range_test_min_lr)
        else:
            self.min_lr = [lr_range_test_min_lr] * len(self.optimizer.param_groups)

        self.step_size = lr_range_test_step_size
        self.step_rate = lr_range_test_step_rate
        self.last_batch_iteration = last_batch_iteration
        self.staircase = lr_range_test_staircase
        self.interval_fn = self._staircase_interval if lr_range_test_staircase else self._continous_interval

        if last_batch_iteration == -1:
            self._update_optimizer(self.min_lr)

    def _staircase_interval(self):
        return math.floor(float(self.last_batch_iteration) / self.step_size)

    def _continous_interval(self):
        return float(self.last_batch_iteration) / self.step_size

    def _get_increase(self):
        return (1 + self.step_rate * self.interval_fn())

    def get_lr(self):
        lr_increase = self._get_increase()
        return [
            lr_range_test_min_lr * lr_increase for lr_range_test_min_lr in self.min_lr
        ]

    def _update_optimizer(self, group_lrs):
        for param_group, lr in zip(self.optimizer.param_groups, group_lrs):
            param_group['lr'] = lr

    def step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        self._update_optimizer(self.get_lr())

    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']


class OneCycle(object):
    """Sets the learning rate of each parameter group according to
    1Cycle learning rate policy (1CLR). 1CLR is a variation of the
    Cyclical Learning Rate (CLR) policy that involves one cycle followed by
    decay. The policy simultaneously cycles the learning rate (and momentum)
    between two boundaries with a constant frequency, as detailed in
    the paper `A disciplined approach to neural network hyper-parameters`_.

    1CLR policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This implementation was adapted from the github repo: `pytorch/pytorch`_

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        cycle_min_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        cycle_max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (cycle_max_lr - cycle_min_lr).
            The lr at any cycle is the sum of cycle_min_lr
            and some scaling of the amplitude; therefore
            cycle_max_lr may not actually be reached depending on
            scaling function.
        decay_lr_rate(float): Decay rate for learning rate. Default: 0.
        cycle_first_step_size (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        cycle_second_step_size (int): Number of training iterations in the
            decreasing half of a cycle. If cycle_second_step_size is None,
            it is set to cycle_first_step_size. Default: None
        cycle_first_stair_count(int): Number of stairs in first half of cycle phase. This means
        lr/mom are changed in staircase fashion. Default 0, means staircase disabled.
        cycle_second_stair_count(int): Number of stairs in second half of cycle phase. This means
        lr/mom are changed in staircase fashion. Default 0, means staircase disabled.
        decay_step_size (int): Intervals for applying decay in decay phase. Default: 0, means no decay.
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'cycle_min_mom' and 'cycle_max_mom'.
            Default: True
        cycle_min_mom (float or list): Initial momentum which is the
            lower boundary in the cycle for each parameter group.
            Default: 0.8
        cycle_max_mom (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (cycle_max_mom - cycle_min_mom).
            The momentum at any cycle is the difference of cycle_max_mom
            and some scaling of the amplitude; therefore
            cycle_min_mom may not actually be reached depending on
            scaling function. Default: 0.9
        decay_mom_rate (float): Decay rate for momentum. Default: 0.
        last_batch_iteration (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_batch_iteration=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.OneCycle(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()


    .. _A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay: https://arxiv.org/abs/1803.09820
    """
    def __init__(self,
                 optimizer,
                 cycle_min_lr,
                 cycle_max_lr,
                 decay_lr_rate=0.,
                 cycle_first_step_size=2000,
                 cycle_second_step_size=None,
                 cycle_first_stair_count=0,
                 cycle_second_stair_count=None,
                 decay_step_size=0,
                 cycle_momentum=True,
                 cycle_min_mom=0.8,
                 cycle_max_mom=0.9,
                 decay_mom_rate=0.,
                 last_batch_iteration=-1):

        self.optimizer = get_torch_optimizer(optimizer)

        # Initialize cycle shape
        self._initialize_cycle(cycle_first_step_size,
                               cycle_second_step_size,
                               cycle_first_stair_count,
                               cycle_second_stair_count,
                               decay_step_size)

        # Initialize cycle lr
        self._initialize_lr(self.optimizer,
                            cycle_min_lr,
                            cycle_max_lr,
                            decay_lr_rate,
                            last_batch_iteration)

        # Initialize cyclic momentum
        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            self._initialize_momentum(self.optimizer,
                                      cycle_min_mom,
                                      cycle_max_mom,
                                      decay_mom_rate,
                                      last_batch_iteration)

        # Initalize batch iteration tracker
        self.last_batch_iteration = last_batch_iteration

    # Configure cycle shape

    def _initialize_cycle(self,
                          cycle_first_step_size,
                          cycle_second_step_size,
                          cycle_first_stair_count,
                          cycle_second_stair_count,
                          decay_step_size):
        cycle_first_step_size = float(cycle_first_step_size)
        cycle_second_step_size = float(
            cycle_second_step_size
        ) if cycle_second_step_size is not None else cycle_first_step_size

        self.total_size = cycle_first_step_size + cycle_second_step_size
        self.step_ratio = cycle_first_step_size / self.total_size
        self.first_stair_count = cycle_first_stair_count
        self.second_stair_count = cycle_first_stair_count if cycle_second_stair_count is None else cycle_second_stair_count
        self.decay_step_size = decay_step_size

    # Configure lr schedule
    def _initialize_lr(self,
                       optimizer,
                       cycle_min_lr,
                       cycle_max_lr,
                       decay_lr_rate,
                       last_batch_iteration):
        self.min_lrs = [cycle_min_lr] * len(optimizer.param_groups)
        if last_batch_iteration == -1:
            for lr, group in zip(self.min_lrs, optimizer.param_groups):
                group['lr'] = lr

        self.max_lrs = [cycle_max_lr] * len(optimizer.param_groups)
        self.decay_lr_rate = decay_lr_rate

    # Configure momentum schedule
    def _initialize_momentum(self,
                             optimizer,
                             cycle_min_mom,
                             cycle_max_mom,
                             decay_mom_rate,
                             last_batch_iteration):
        if 'betas' not in optimizer.defaults:
            optimizer_name = type(optimizer).__name__
            logger.warn(
                f"cycle_momentum is disabled because optimizer {optimizer_name} does not support momentum, no betas attribute in defaults"
            )
            self.cycle_momentum = False
            return

        self.decay_mom_rate = decay_mom_rate
        self.min_moms = [(cycle_min_mom, 0.99)] * len(optimizer.param_groups)
        self.max_moms = [(cycle_max_mom, 0.99)] * len(optimizer.param_groups)

        if last_batch_iteration == -1:
            for momentum, group in zip(self.min_moms, optimizer.param_groups):
                group['betas'] = momentum

    def _get_cycle_lr(self):
        cycle = math.floor(1 + self.last_batch_iteration / self.total_size)
        x = 1. + self.last_batch_iteration / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        lrs = []
        for cycle_min_lr, cycle_max_lr in zip(self.min_lrs, self.max_lrs):
            base_height = (cycle_max_lr - cycle_min_lr) * scale_factor
            lr = cycle_min_lr + base_height
            lrs.append(lr)

        if self.cycle_momentum:
            momentums = []
            for base_betas, max_betas in zip(self.min_moms, self.max_moms):
                cycle_min_mom = base_betas[0]
                cycle_max_mom = max_betas[0]
                base_height = (cycle_max_mom - cycle_min_mom) * scale_factor
                momentum = cycle_max_mom - base_height
                momentums.append((momentum, base_betas[1]))
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['betas'] = momentum

        return lrs

    def _get_decay_lr(self, decay_batch_iteration):
        """Calculates the learning rate at batch index. This function is used
        after the cycle completes and post cycle decaying of lr/mom is enabled.
        This function treats `self.last_batch_iteration` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """
        decay_interval = decay_batch_iteration / self.decay_step_size

        lr_decay_factor = (1 + self.decay_lr_rate * decay_interval)
        lrs = [cycle_min_lr * lr_decay_factor for cycle_min_lr in self.min_lrs]

        if self.cycle_momentum:
            mom_decay_factor = (1 + self.decay_mom_rate * decay_interval)
            momentums = [(beta0 * mom_decay_factor,
                          beta1) for beta0,
                         beta1 in self.max_moms]
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['betas'] = momentum

        return lrs

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_batch_iteration` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """
        if self.last_batch_iteration <= self.total_size:
            return self._get_cycle_lr()
        return self._get_decay_lr(self.last_batch_iteration - self.total_size)

    def step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']


class WarmupLR(object):
    """Increase the learning rate of each parameter group from min lr to max lr
        over warmup_num_steps steps, and then fix at max lr.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_min_lr (float or list): minimum learning rate. Default: 0
            warmup_max_lr (float or list): maximum learning rate. Default: 0.001
            warmup_num_steps (int): number of steps to warm up from min_lr to max_lr. Default: 1000
            last_batch_iteration (int): The index of the last batch. Default: -1.
        Example:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            >>> scheduler = torch.optim.WarmupLR(optimizer)
            >>> data_loader = torch.utils.data.DataLoader(...)
            >>> for epoch in range(10):
            >>>     for batch in data_loader:
            >>>         train_batch(...)
            >>>         scheduler.step()

    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_min_lr: float = 0.0,
                 warmup_max_lr: float = 0.001,
                 warmup_num_steps: int = 1000,
                 last_batch_iteration: int = -1):

        self.optimizer = get_torch_optimizer(optimizer)

        self.min_lrs = self._format_param(self.optimizer, warmup_min_lr, "min_lr")
        self.max_lrs = self._format_param(self.optimizer, warmup_max_lr, "max_lr")
        self.delta_lrs = [big - small for big, small in zip(self.max_lrs, self.min_lrs)]
        self.warmup_num_steps = warmup_num_steps
        self.inverse_log_warm_up = 1.0 / math.log(warmup_num_steps)
        self.last_batch_iteration = last_batch_iteration

    def get_lr(self):
        gamma = self._get_gamma()
        return [
            min_lr + (delta_lr * gamma) for min_lr,
            delta_lr in zip(self.min_lrs,
                            self.delta_lrs)
        ]

    def step(self, last_batch_iteration=None):
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']

    def _get_gamma(self):
        if self.last_batch_iteration < self.warmup_num_steps:
            return self.inverse_log_warm_up * math.log(self.last_batch_iteration + 1)
        return 1.0

    def _format_param(self, optimizer, param_value, param_name):
        if isinstance(param_value, list) or isinstance(param_value, tuple):
            if len(param_value) != len(optimizer.param_groups):
                raise ValueError("expected {} value for {}, got {}".format(
                    len(optimizer.param_groups),
                    param_name,
                    FileNotFoundError(param_value)))
            return list(param_value)
        return [param_value] * len(optimizer.param_groups)
