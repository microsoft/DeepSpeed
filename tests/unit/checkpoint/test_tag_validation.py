import deepspeed

from tests.unit.common import DistributedTest
from tests.unit.simple_model import *

import pytest


class TestCheckpointValidationTag(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize('valid_mode', ["FAIL", "WARN", "IGNORE"])
    def test_checkpoint_unique_tag(self, tmpdir, valid_mode):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "checkpoint": {
                "tag_validation": valid_mode
            }
        }
        hidden_dim = 10
        args = args_from_dict(tmpdir, config_dict)

        model = SimpleModel(hidden_dim)

        def _helper(args, model):
            model, _, _,_ = deepspeed.initialize(args=args,
                                                model=model,
                                                model_parameters=model.parameters())
            if valid_mode == "FAIL":
                with pytest.raises(AssertionError):
                    model.save_checkpoint(save_dir=tmpdir, tag=f"tag-{dist.get_rank()}")
            else:
                model.save_checkpoint(save_dir=tmpdir, tag=f"tag-{dist.get_rank()}")

        _helper(args=args, model=model)

    def test_checkpoint_unknown_tag_validation(self, tmpdir):

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "checkpoint": {
                "tag_validation": "foo"
            }
        }
        hidden_dim = 10
        args = args_from_dict(tmpdir, config_dict)

        model = SimpleModel(hidden_dim)

        def _helper(args, model):
            with pytest.raises(deepspeed.DeepSpeedConfigError):
                model, _, _,_ = deepspeed.initialize(args=args,
                                                    model=model,
                                                    model_parameters=model.parameters())

        _helper(args=args, model=model)
