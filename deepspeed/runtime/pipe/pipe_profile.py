import time
import torch
import json
from datetime import datetime

# Modified SynchronizedWallClockTimer


class PipelineProfiler:
    """Record operator time and write to file"""
    class Timer:
        """Timer"""

        def __init__(self, name, iter_):
            self.name_ = name
            self.start_ = 0
            self.end_ = 0
            self.started_ = False
            self.iter_ = iter_

        def start(self):
            """Start the timer."""
            assert not self.started_, 'timer has already been started'
            torch.cuda.synchronize()
            self.start_ = time.time()
            self.started_ = True

        def stop(self):
            """Stop the timer."""
            assert self.started_, 'timer is not started'
            torch.cuda.synchronize()
            self.end_ = time.time()
            self.started_ = False

        def __str__(self):
            return json.dumps({
                "type": self.name_,
                "iteration": self.iter_,
                "start": self.start_,
                "end": self.end_
            })

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, write_to_disk_entries_=50):
        self.timers = dict()
        self.records = list()
        self.write_to_disk_entries = write_to_disk_entries_
        self.filename = "pipeline_profile_log" + \
            datetime.now().strftime("%m/%d/%Y-%H:%M:%S")
        # create empty file
        file_creator = open(self.filename, "w")
        file_creator.close()

    # content -> dict
    def write_metadata(self, content):
        """Write Metadata"""
        content["type"] = "META"
        with open(self.filename, "a") as file_appender:
            file_appender.write(json.dumps(content))
            file_appender.write("\n")

    def start(self, key, iteration):
        """Start the timer."""
        assert key not in self.timers
        self.timers[key] = self.Timer(key, iteration)
        self.timers[key].start()

    def stop(self, key):
        """Stop the timer."""
        assert key in self.timers
        assert self.timers[key].started
        self.records.append(str(self.timers[key]))
        del self.timers[key]
        if len(self.records) == self.write_to_disk_entries:
            with open(self.filename, "a") as file_appender:
                file_appender.write("\n".join(self.records))
            self.records = list()
