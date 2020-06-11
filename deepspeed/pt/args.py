"""
Argument definitions for deepspeed
"""

import argparse

from deepspeed.pt.log_utils import logger


def _add_lr_scheduler(parser):
    group = parser.add_argument_group(title="LR Scheduler")
    group.add_argument("--lr_schedule", help="LR schedule for training")
    group.add_argument("--lr_range_test_min_lr",
                       type=float,
                       default=0.001,
                       help="Staring lr value")
    group.add_argument("--lr_range_test_step_rate",
                       type=float,
                       default=1.0,
                       help="Scaling rate for LR range test")
    group.add_argument("--lr_range_test_step_size",
                       type=int,
                       default=1000,
                       help="Training steps per LR change")
    group.add_argument("--lr_range_test_staircase",
                       action="store_true",
                       help="Use staircase scaling for LR range test")


def _add_one_cycle_scheduler(parser):
    group = parser.add_argument_group(title="One Cycle")
    group.add_argument("--cycle_first_step_size",
                       type=int,
                       default=1000,
                       help="Size of first step of 1Cycle schedule (training steps).")
    group.add_argument("--cycle_first_stair_count",
                       type=int,
                       default=1,
                       help="First stair count for 1Cycle schedule.")
    group.add_argument("--second_stair_count",
                       type=int,
                       default=1,
                       help="Second stair count for 1Cycle schedule.")
    group.add_argument(
        "--decay_step_size",
        type=int,
        default=100,
        help="Size of intervals for applying post cycle decay (training steps).")

    # 1Cycle LR
    group.add_argument("--cycle_min_lr",
                       type=float,
                       default=0.01,
                       help="1Cycle LR lower bound.")
    group.add_argument("--cycle_max_lr",
                       type=float,
                       default=0.1,
                       help="1Cycle upper bound.")
    group.add_argument("--decay_lr_rate",
                       type=float,
                       default=0.0,
                       help="Post cycle LR decay rate")

    # 1Cycle Momentum
    group.add_argument("--cycle_momentum",
                       action="store_true",
                       help="Enable 1Cycle momentum schedule.")
    group.add_argument("--cycle_min_mom",
                       type=float,
                       default=0.8,
                       help="1Cycle momentum lower bound")
    group.add_argument("--cycle_max_mom",
                       type=float,
                       default=0.9,
                       help="1Cycle momentum upper bound")
    group.add_argument("--decay_mom_rate",
                       type=float,
                       default=0.0,
                       help="Pos cycle momentum decay rate")


def _add_warmup_lr(parser):
    group = parser.add_argument_group(title="Warmup LR")
    group.add_argument("--warmup_min_lr",
                       type=float,
                       default=0,
                       help="WarmupLR minimum/initial LR value")
    group.add_argument("--warmup_max_lr",
                       type=float,
                       default=0.001,
                       help="WarmupLR maximum LR value.")
    group.add_argument("--warmup_num_steps",
                       type=int,
                       default=1000,
                       help="WarmupLR step count for LR warmup.")


def create_lr_tuning_parser(parser=None):
    """Create learning related parser"""
    parser = parser or argparse.ArgumentParser(description="tuning parser arguments")

    _add_lr_scheduler(parser)
    _add_one_cycle_scheduler(parser)
    _add_warmup_lr(parser)

    return parser


def parse_lr_tuning_args(args=None):
    """Paring arguments for lr tuning"""
    parser = create_lr_tuning_parser()
    args, _ = parser.parse_known_args(args=args)
    if _:
        logger.warning("Unknown args for lr tuning: %s", _)
    return args, _
