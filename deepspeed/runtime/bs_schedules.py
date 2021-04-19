import math
import numpy as np


class BatchSizeScheduler(object):
    """Increase the batch size linearly from int(mb_size_per_gpu * min_batch_size_multiplier) to mb_size_per_gpu
        over warmup_num_steps steps, and then fix at mb_size_per_gpu.

    TODO: documentation
    """

    def __init__(self,
                 final_batch_size,
                 min_batch_size_multiplier: float = 0.01,
                 warmup_num_steps: int = 1000,
                 num_intervals=4,
                 last_batch_iteration: int = -1,
                 deepspeed=None):

        self.warmup_num_steps = warmup_num_steps
        self.last_batch_iteration = last_batch_iteration
        self.final_batch_size = final_batch_size
        self.num_intervals = num_intervals
        self.min_batch_size_multiplier = min_batch_size_multiplier
        self.schedule = self._build_schedule()
        self.current_batch_size = None
        self.deepspeed = deepspeed

    def _build_schedule(self):
        start = math.ceil(self.min_batch_size_multiplier * self.final_batch_size)
        batch_sizes = np.linspace(start, self.final_batch_size, num=self.num_intervals, endpoint=True, retstep=False,
                                  dtype=int, axis=0)
        steps = np.linspace(0, self.warmup_num_steps, num=self.num_intervals, endpoint=True, retstep=False, dtype=int,
                            axis=0)
        schedule = {step: batch_size for step, batch_size in zip(steps, batch_sizes)}
        # deduplicate intervals with same batch size
        prev_v = None
        to_pop = []
        for k, v in schedule.items():
            if v == prev_v:
                to_pop.append(k)
            prev_v = v
        for k in to_pop:
            schedule.pop(k)
        return schedule

    def get_current_batch_size(self):
        i = None
        iterator = sorted(self.schedule.keys(), reverse=True)
        for i, v in enumerate(iterator):
            if self.last_batch_iteration >= v:
                break
            else:
                pass
        current_batch_size = self.schedule[iterator[i]]
        return current_batch_size

    def step(self, last_batch_iteration=None):
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration
        self.current_batch_size = self.get_current_batch_size()

    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']


if __name__ == "__main__":
    sched = BatchSizeScheduler(
        final_batch_size=16,
        num_intervals=8,
        warmup_num_steps=10000
    )
    print(f'SCHEDULE: {sched.schedule}')
    prev_bs = None
    for i in range(sched.warmup_num_steps + 1):
        sched.step()
        if sched.current_batch_size != prev_bs:
            print(i, sched.current_batch_size)
        prev_bs = sched.current_batch_size
