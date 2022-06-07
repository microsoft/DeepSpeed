from .utils import check_tb_availability
from .monitor import Monitor
import os


class TensorBoardMonitor(Monitor):
    def __init__(self, monitor_config):
        super().__init__(monitor_config)
        check_tb_availability()
        self.summary_writer = None
        #self.tensorboard_output_path = None
        #self.tensorboard_job_name = "DeepSpeedJobName"

        self.tensorboard_output_path = monitor_config.tensorboard.output_path
        self.tensorboard_job_name = monitor_config.tensorboard.job_name

        self.get_summary_writer()

    def get_summary_writer(self,
                           base=os.path.join(os.path.expanduser("~"),
                                             "tensorboard")):
        from torch.utils.tensorboard import SummaryWriter
        if self.tensorboard_output_path is not None:
            log_dir = os.path.join(self.tensorboard_output_path,
                                   self.tensorboard_job_name)
        else:

            if "DLWS_JOB_ID" in os.environ:
                infra_job_id = os.environ["DLWS_JOB_ID"]
            elif "DLTS_JOB_ID" in os.environ:
                infra_job_id = os.environ["DLTS_JOB_ID"]
            else:
                infra_job_id = "unknown-job-id"

            summary_writer_dir_name = os.path.join(infra_job_id, "logs")
            log_dir = os.path.join(base,
                                   summary_writer_dir_name,
                                   self.tensorboard_job_name)
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
