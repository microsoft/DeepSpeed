import time
import logging
import psutil
import torch


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


class SynchronizedWallClockTimer:
    """Group of timers. Borrowed from Nvidia Megatron code"""
    class Timer:
        """Timer."""
        def __init__(self, name):
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()

        def start(self):
            """Start the timer."""
            assert not self.started_, 'timer has already been started'
            torch.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self):
            """Stop the timer."""
            assert self.started_, 'timer is not started'
            torch.cuda.synchronize()
            self.elapsed_ += (time.time() - self.start_time)
            self.started_ = False

        def reset(self):
            """Reset timer."""
            self.elapsed_ = 0.0
            self.started_ = False

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self.elapsed_
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        print_rank_0(string)


class ThroughputTimer(object):
    def __init__(self, name=None, batch_size=1, num_workers=1, start_step=2):
        self.start_time = 0
        self.end_time = 0
        self.started = False
        self.count = 0
        self.total_elapsed_time = 0
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.start_step = start_step
        self.name = name

    def start(self, cond=True):
        if cond:
            self.start_time = time.time()
            self.started = True

    def stop(self, cond=True):
        if cond and self.started:
            self.end_time = time.time()
            self.started = False
            self.count += 1
            if self.count >= self.start_step:
                self.total_elapsed_time += self.end_time - self.start_time
        elif cond and not self.started:
            print("Cannot stop timer without starting ")
            exit(0)

    def avg_samples_per_sec(self):
        if self.count > 2:
            samples_per_step = self.batch_size * self.num_workers
            avg_time_per_step = self.total_elapsed_time / (self.count - 2.0)
            # training samples per second
            return samples_per_step / avg_time_per_step
        return -999

    def avg_steps_per_sec(self):
        if self.count > 2:
            return 1 / (self.total_elapsed_time / (self.count - 2.0))
        return -999

    def print_elapsed_time(self, num_ops=None):
        if self.count > 2 and self.count % 1000 == 0:
            elapsed_time = self.total_elapsed_time / (self.count - 2.0)
            if num_ops == None:
                print(self.name, " forward pass execution time: ",
                      elapsed_time)
            else:
                print(self.name, " forward pass execution time: ",
                      elapsed_time, " TFlops : ",
                      num_ops / (elapsed_time * 1000000000000))
