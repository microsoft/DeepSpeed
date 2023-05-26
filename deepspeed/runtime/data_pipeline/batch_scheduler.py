import math
from deepspeed.utils import logger
from .constants import *


class BatchScheduler(object):
    def __init__(self, config):
        super().__init__()
        self.state = {}
        assert BATCH_SCHEDULING_MIN in config, \
            f"Batch Scheduling requires the config '{BATCH_SCHEDULING_MIN}'"
        assert BATCH_SCHEDULING_MAX in config, \
            f"Batch Scheduling requires the config '{BATCH_SCHEDULING_MAX}'"
        assert BATCH_SCHEDULING_TYPE in config, \
            f"Batch Scheduling requires the config '{BATCH_SCHEDULING_TYPE}'"
        self.state[BATCH_SCHEDULING_MIN] = config[
            BATCH_SCHEDULING_MIN]
        self.state[BATCH_SCHEDULING_MAX] = config[
            BATCH_SCHEDULING_MAX]
        self.state[BATCH_SCHEDULING_CURRENT_SIZE] = config[
            BATCH_SCHEDULING_MIN]
        self.state[BATCH_SCHEDULING_TYPE] = config[
            BATCH_SCHEDULING_TYPE]
        self.first_step = True
        if config[
                BATCH_SCHEDULING_TYPE] == BATCH_SCHEDULING_FIXED_DISCRETE:
            """
            The schedule_config is a list of sizes and a list of max
            step belonging to each size. Example json config:
            "schedule_config": {
              "sizes": [1,2,3],
              "max_step": [5,10]
            }
            The "max_step" has one less element than "sizes", because
            the last size will be used for all following steps.
            The self.state[BATCH_SCHEDULING_CONFIG] is a dictionary of
            sizes : [max step for this size, next size].
            """
            assert BATCH_SCHEDULING_SIZES in config[BATCH_SCHEDULING_CONFIG], \
                f"Batch scheduling with fixed_discrete schedule requires the schedule_config '{BATCH_SCHEDULING_SIZES}'"
            assert BATCH_SCHEDULING_MAX_STEP in config[BATCH_SCHEDULING_CONFIG], \
                f"batch scheduling with fixed_discrete schedule requires the schedule_config '{BATCH_SCHEDULING_MAX_STEP}'"
            assert len(config[BATCH_SCHEDULING_CONFIG]
                       [BATCH_SCHEDULING_MAX_STEP]) > 0
            assert len(config[BATCH_SCHEDULING_CONFIG]
                       [BATCH_SCHEDULING_SIZES]) > 0
            assert len(config[BATCH_SCHEDULING_CONFIG]
                       [BATCH_SCHEDULING_SIZES]) == len(
                           config[BATCH_SCHEDULING_CONFIG]
                           [BATCH_SCHEDULING_MAX_STEP]) + 1
            self.state[BATCH_SCHEDULING_CONFIG] = config[
                BATCH_SCHEDULING_CONFIG]
        elif config[
                BATCH_SCHEDULING_TYPE] == BATCH_SCHEDULING_FIXED_ROOT:
            """
            The schedule_config includes:
            total_batch_step: how many steps the batch scheduling takes to go
            from min size to max size.
            size_step: the batch size determined every time must
            be a multiple of this size_step. This is used to determine
            the step of size increase, and to ensure the use of NVIDIA
            Tensor Core acceleration (requires multiple of 8 (FP16) or
            16 (INT8)).
            root_degree: the degree of the root function. Degree of 2 means
            square root and degree of 3 means cube root. Degree of 1 is
            equivalent to linear.
            "schedule_config": {
              "total_batch_step": 30000,
              "size_step": 8,
              "root_degree": 2
            }
            """
            assert BATCH_SCHEDULING_TOTAL_STEP in config[BATCH_SCHEDULING_CONFIG], \
                f"Batch Scheduling with fixed_root schedule requires the schedule_config '{BATCH_SCHEDULING_TOTAL_STEP}'"
            assert BATCH_SCHEDULING_SIZE_STEP in config[BATCH_SCHEDULING_CONFIG], \
                f"Batch Scheduling with fixed_root schedule requires the schedule_config '{BATCH_SCHEDULING_SIZE_STEP}'"
            assert BATCH_SCHEDULING_ROOT_DEGREE in config[BATCH_SCHEDULING_CONFIG], \
                f"Batch Scheduling with fixed_root schedule requires the schedule_config '{BATCH_SCHEDULING_ROOT_DEGREE}'"
            if config[BATCH_SCHEDULING_CONFIG][
                    BATCH_SCHEDULING_SIZE_STEP] % 8 != 0:
                logger.warning(
                    f'When using seqlen metric, the size_step for Batch Scheduling has to be multiple of 8 (for FP16 data) or 16 (for INT8 data) to enable NVIDIA Tensor Core acceleration. Disregard this warning if this is unrelated to your metric/hardware.'
                )
            self.state[BATCH_SCHEDULING_CONFIG] = config[
                BATCH_SCHEDULING_CONFIG]
        elif config[
                BATCH_SCHEDULING_TYPE] == BATCH_SCHEDULING_FIXED_LINEAR:
            """
            The schedule_config is the same as BATCH_SCHEDULING_FIXED_ROOT but without the
            root_degree.
            "schedule_config": {
              "total_batch_step": 30000,
              "size_step": 8
            }
            """
            assert BATCH_SCHEDULING_TOTAL_STEP in config[BATCH_SCHEDULING_CONFIG], \
                f"Curriculum learning with fixed_linear schedule requires the schedule_config '{BATCH_SCHEDULING_TOTAL_STEP}'"
            assert BATCH_SCHEDULING_SIZE_STEP in config[BATCH_SCHEDULING_CONFIG], \
                f"Curriculum learning with fixed_linear schedule requires the schedule_config '{BATCH_SCHEDULING_SIZE_STEP}'"
            if config[BATCH_SCHEDULING_CONFIG][
                    BATCH_SCHEDULING_SIZE_STEP] % 8 != 0:
                logger.warning(
                    f'When using seqlen metric, the size_step for curriculum learning has to be multiple of 8 (for FP16 data) or 16 (for INT8 data) to enable NVIDIA Tensor Core acceleration. Disregard this warning if this is unrelated to your metric/hardware.'
                )
            self.state[BATCH_SCHEDULING_CONFIG] = config[
                BATCH_SCHEDULING_CONFIG]
        elif config[
                BATCH_SCHEDULING_TYPE] == BATCH_SCHEDULING_CUSTOM:
            """
            Fully customized schedule. User need to provide a custom schedule
            function by using the set_custom_curriculum_learning_schedule API
            in deepspeed/runtime/engine.py
            """
            self.custom_get_size = None
        else:
            raise RuntimeError('Unsupported batch schedule type')
        

    def get_current_size(self):
        return self.state[BATCH_SCHEDULING_CURRENT_SIZE]

    def set_current_size(self, size):
        self.state[BATCH_SCHEDULING_CURRENT_SIZE] = size

    def set_custom_get_size(self, schedule_function):
        self.custom_get_size = schedule_function

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def __fixed_discrete_get_size(self, global_steps):
        s_state = self.state[BATCH_SCHEDULING_CONFIG]
        if global_steps > s_state[BATCH_SCHEDULING_MAX_STEP][-1]:
            return s_state[BATCH_SCHEDULING_SIZES][-1]
        for i in range(len(s_state[BATCH_SCHEDULING_MAX_STEP])):
            if global_steps <= s_state[BATCH_SCHEDULING_MAX_STEP][i]:
                return s_state[BATCH_SCHEDULING_SIZES][i]

    def __fixed_root_get_size(self, global_steps, root_degree=None):
        s_state = self.state[BATCH_SCHEDULING_CONFIG]
        if root_degree is None:
            root_degree = s_state[BATCH_SCHEDULING_ROOT_DEGREE]
        next_size = (float(global_steps) /
                           s_state[BATCH_SCHEDULING_TOTAL_STEP])**(
                               1.0 / root_degree)
        next_size = math.floor(next_size *
                                     (self.state[BATCH_SCHEDULING_MAX] -
                                      self.state[BATCH_SCHEDULING_MIN]) +
                                     self.state[BATCH_SCHEDULING_MIN])
        next_size -= (next_size %
                            s_state[BATCH_SCHEDULING_SIZE_STEP])
        next_size = min(next_size,
                              self.state[BATCH_SCHEDULING_MAX])
        return next_size

    def get_size(self, global_steps):
        if self.state[
                BATCH_SCHEDULING_TYPE] == BATCH_SCHEDULING_FIXED_DISCRETE:
            return self.__fixed_discrete_get_size(global_steps)
        elif self.state[
                BATCH_SCHEDULING_TYPE] == BATCH_SCHEDULING_FIXED_LINEAR:
            return self.__fixed_root_get_size(global_steps, 1)
        elif self.state[
                BATCH_SCHEDULING_TYPE] == BATCH_SCHEDULING_FIXED_ROOT:
            return self.__fixed_root_get_size(global_steps)
        elif self.state[
                BATCH_SCHEDULING_TYPE] == BATCH_SCHEDULING_CUSTOM:
            return self.custom_get_size(global_steps)
        else:
            raise RuntimeError('Unsupported batch schedule type')

    def update(self, global_steps):
        if self.state[BATCH_SCHEDULING_CURRENT_SIZE] < self.state[
                BATCH_SCHEDULING_MAX]:
            self.state[BATCH_SCHEDULING_CURRENT_SIZE] = self.get_size(
                global_steps)
        return self.state[BATCH_SCHEDULING_CURRENT_SIZE]