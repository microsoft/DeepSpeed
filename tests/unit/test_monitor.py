import pytest

from deepspeed.monitor.constants import *

from deepspeed.monitor.monitor import MonitorMaster
from deepspeed.monitor.tensorboard import TensorBoardMonitor
from deepspeed.monitor.wandb import WandbMonitor
from deepspeed.monitor.csv_monitor import csvMonitor

from .simple_model import *
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.monitor.config import DeepSpeedMonitorConfig

try:
    import tensorboard
    _tb_available = True
except ImportError:
    _tb_available = False
tb_available = pytest.mark.skipif(not _tb_available,
                                  reason="tensorboard is not installed")

try:
    import wandb
    _wandb_available = True
except ImportError:
    _wandb_available = False
wandb_available = pytest.mark.skipif(not _wandb_available,
                                     reason="wandb is not installed")


@tb_available
def test_tensorboard(tmpdir):
    config_dict = {
        "train_batch_size": 1,
        "tensorboard": {
            "enabled": True,
            "output_path": "test_output/ds_logs/",
            "job_name": "test"
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    ds_config = DeepSpeedConfig(args.deepspeed_config)
    tb_monitor = TensorBoardMonitor(ds_config.monitor_config)
    assert tb_monitor.enabled == True
    assert tb_monitor.output_path == "test_output/ds_logs/"
    assert tb_monitor.job_name == "test"


@tb_available
def test_empty_tensorboard(tmpdir):
    config_dict = {"train_batch_size": 1, "tensorboard": {}}
    args = args_from_dict(tmpdir, config_dict)
    ds_config = DeepSpeedConfig(args.deepspeed_config)
    tb_monitor = TensorBoardMonitor(ds_config.monitor_config)
    assert tb_monitor.enabled == TENSORBOARD_ENABLED_DEFAULT
    assert tb_monitor.output_path == TENSORBOARD_OUTPUT_PATH_DEFAULT
    assert tb_monitor.job_name == TENSORBOARD_JOB_NAME_DEFAULT


@wandb_available
def test_wandb(tmpdir):
    config_dict = {
        "train_batch_size": 1,
        "wandb": {
            "enabled": False,
            "group": "my_group",
            "team": "my_team",
            "project": "my_project"
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    ds_config = DeepSpeedConfig(args.deepspeed_config)
    wandb_monitor = WandbMonitor(ds_config.monitor_config)
    assert wandb_monitor.enabled == False
    assert wandb_monitor.group == "my_group"
    assert wandb_monitor.team == "my_team"
    assert wandb_monitor.project == "my_project"


@wandb_available
def test_empty_wandb(tmpdir):
    config_dict = {"train_batch_size": 1, "wandb": {}}
    args = args_from_dict(tmpdir, config_dict)
    ds_config = DeepSpeedConfig(args.deepspeed_config)
    wandb_monitor = WandbMonitor(ds_config.monitor_config)
    assert wandb_monitor.enabled == WANDB_ENABLED_DEFAULT
    assert wandb_monitor.group == WANDB_GROUP_NAME_DEFAULT
    assert wandb_monitor.team == WANDB_TEAM_NAME_DEFAULT
    assert wandb_monitor.project == WANDB_PROJECT_NAME_DEFAULT


def test_csv_monitor(tmpdir):
    config_dict = {
        "train_batch_size": 1,
        "csv_monitor": {
            "enabled": True,
            "output_path": "test_output/ds_logs/",
            "job_name": "test"
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    ds_config = DeepSpeedConfig(args.deepspeed_config)
    csv_monitor = csvMonitor(ds_config.monitor_config)
    assert csv_monitor.enabled == True
    assert csv_monitor.output_path == "test_output/ds_logs/"
    assert csv_monitor.job_name == "test"


def test_empty_csv_monitor(tmpdir):
    config_dict = {"train_batch_size": 1, "csv_monitor": {}}
    args = args_from_dict(tmpdir, config_dict)
    ds_config = DeepSpeedConfig(args.deepspeed_config)
    csv_monitor = csvMonitor(ds_config.monitor_config)
    assert csv_monitor.enabled == CSV_MONITOR_ENABLED_DEFAULT
    assert csv_monitor.output_path == CSV_MONITOR_OUTPUT_PATH_DEFAULT
    assert csv_monitor.job_name == CSV_MONITOR_JOB_NAME_DEFAULT
