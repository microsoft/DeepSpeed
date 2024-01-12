# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .utils import check_aim_availability
from .monitor import Monitor

import deepspeed.comm as dist


class AimMonitor(Monitor):
    def __init__(self, aim_config):
        super().__init__(aim_config)
        check_aim_availability()

        from aim import Run

        self.Run = Run

        self.enabled = aim_config.enabled
        self.repo = aim_config.repo
        self.experiment_name = aim_config.experiment_name
        self.log_system_params = aim_config.log_system_params
        self.run_name = aim_config.run_name
        self.run_hash = aim_config.run_hash
        self.run = None

        if self.enabled and dist.get_rank() == 0:
            self.experiment()
            for key in aim_config:
                self.experiment.set(('hparams', key), aim_config.key, strict=False)

    def write_events(self, event_list):
        if self.enabled and dist.get_rank() == 0:
            for event in event_list:
                label = event[0]
                value = event[1]
                step = event[2]
                self.experiment.track(value, name=label, step=step)

    @property
    def experiment(self):
        if self.run is None:
            if self.run_hash:
                self.run = self.Run(
                    self.run_hash,
                    repo=self._repo_path,
                    system_tracking_interval=self._system_tracking_interval,
                    capture_terminal_logs=self._capture_terminal_logs,
                )
                if self.run_name is not None:
                    self.run.name = self.run_name
            else:
                self.run = self.Run(
                    repo=self._repo_path,
                    experiment=self._experiment_name,
                    system_tracking_interval=self._system_tracking_interval,
                    log_system_params=self._log_system_params,
                    capture_terminal_logs=self._capture_terminal_logs,
                )
                self.run_hash = self.run.hash
        return self.run
