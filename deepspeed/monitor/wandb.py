from .utils import check_wandb_availability
from .monitor import Monitor


class WandbMonitor(Monitor):
    def __init__(self, monitor_config):
        super().__init__(monitor_config)
        check_wandb_availability()
        import wandb
        wandb.init(project=monitor_config.wandb_project,
                   group=monitor_config.wandb_group,
                   entity=monitor_config.wandb_team)

    def log(self, data, step=None, commit=None, sync=None):
        import wandb
        return wandb.log(data, step=step, commit=commit, sync=sync)

    def write_events(self, event_list):
        for event in event_list:
            self.log({event[0]: event[1]}, step=event[2])
