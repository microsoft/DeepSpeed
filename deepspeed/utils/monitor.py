"""
 Support different forms of monitoring such as wandb and tensorboard
"""

import os

def check_tb_availability():
    try:
        # torch.utils.tensorboard will fail if `tensorboard` is not available,
        # see their docs for more details: https://pytorch.org/docs/1.8.0/tensorboard.html
        import tensorboard
    except ImportError:
        print(
            'If you want to use tensorboard logging, please `pip install tensorboard`'
        )
        raise

class TensorBoardMonitor:
    def __init__(self, config):
        self.summary_writer = None
        self.tensorboard_output_path = None
        self.tensorboard_job_name = "DeepSpeedJobName"
        check_tb_availability()

        if hasattr(config, "tensorboard_output_path"):
            self.tensorboard_output_path = config.tensorboard_output_path
        if hasattr(config, "tensorboard_job_name"):
            self.tensorboard_job_name = config.tensorboard_job_name


    def get_summary_writer(self, base=os.path.join(os.path.expanduser("~"), "tensorboard")):
        from torch.utils.tensorboard import SummaryWriter
        if self.tensorboard_output_path is not None:
            log_dir = os.path.join(self.tensorboard_output_path, self.tensorboard_job_name)
        else:

            if "DLWS_JOB_ID" in os.environ:
                infra_job_id = os.environ["DLWS_JOB_ID"]
            elif "DLTS_JOB_ID" in os.environ:
                infra_job_id = os.environ["DLTS_JOB_ID"]
            else:
                infra_job_id = "unknown-job-id"

            summary_writer_dir_name = os.path.join(infra_job_id, "logs")
            log_dir = os.path.join(base, summary_writer_dir_name, self.tensorboard_job_name)
        os.makedirs(log_dir, exist_ok=True)
        self.summary_writer = SummaryWriter(log_dir=log_dir)
        return self.summary_writer

    def write_events(self, event_list, flush=True):
        for event in event_list:
            self.summary_writer.add_scalar(*event)
        if flush:
            self.summary_writer.flush()
    def flush(self):
        self.summary_writer.flush()


def check_wandb_availability():
    try:
        import wandb
    except ImportError:
        print(
            'If you want to use wandb logging, please `pip install wandb` and follow the instructions at https://docs.wandb.ai/quickstart'
        )
        raise


class WandbMonitor:
    def __init__(self, config):
        check_wandb_availability()
        import wandb
        wandb.init(project=config.wandb_project, group=config.wandb_group, entity=config.wandb_team)

    def log(self, data, step=None, commit=None, sync=None):
        import wandb
        return wandb.log(data, step=step, commit=commit, sync=sync)

    def write_events(self, event_list):
        for event in event_list:
            self.log({event[0]: event[1]}, step=event[2])


#class csvMonitor:
#    def __init__(self, config):


#class Monitor:
#    def __init__(self, config):
#        if config.