from deepspeed.utils import RepeatingLoader
from deepspeed.runtime.dataloader import DeepSpeedDataLoader
import torch
import pytest
import deepspeed
from .common import distributed_test
from .simple_model import SimpleModel, args_from_dict, random_dataset


def test_repeating_loader():
    loader = [1, 2, 3]
    loader = RepeatingLoader(loader)

    for idx in range(50):
        assert next(loader) == 1
        assert next(loader) == 2
        assert next(loader) == 3


@pytest.mark.parametrize('train_batch_size, drop_last',
                         [(1,
                           True),
                          (4,
                           True),
                          (1,
                           False),
                          (4,
                           False)])
def test_dataloader_drop_last(tmpdir, train_batch_size, drop_last):
    config_dict = {
        "train_batch_size": train_batch_size,
        "dataloader_drop_last": drop_last,
        "steps_per_print": 1
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[1])
    def _test_dataloader_drop_last(args, model, hidden_dim):
        optimizer = torch.optim.AdamW(params=model.parameters())
        #TODO: Figure out why this breaks with cuda device
        train_dataset = random_dataset(total_samples=50,
                                       hidden_dim=hidden_dim,
                                       device=torch.device('cpu'),
                                       dtype=torch.float32)
        model, _, training_dataloader, _ = deepspeed.initialize(args=args,
                                                                model=model,
                                                                training_data=train_dataset,
                                                                optimizer=optimizer)
        for n, batch in enumerate(training_dataloader):
            x = batch[0].to(torch.cuda.current_device())
            y = batch[1].to(torch.cuda.current_device())
            loss = model(x, y)
            model.backward(loss)
            model.step()

    _test_dataloader_drop_last(args=args, model=model, hidden_dim=hidden_dim)


@pytest.mark.parametrize('world_size, total_samples, batch_size, drop_last, result',
                         [(16,
                           11788,
                           3,
                           True,
                           245),
                          (16,
                           11788,
                           3,
                           False,
                           246)])
def test_dataloader_len(world_size, total_samples, batch_size, drop_last, result):
    args = {
        "world_size": world_size,
        "total_samples": total_samples,
        "batch_size": batch_size,
        "drop_last": drop_last,
        "result": result
    }
    #args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    @distributed_test(world_size=[1])
    def _test_dataloader_len(args, hidden_dim):
        dataset = random_dataset(total_samples=args["total_samples"],
                                 hidden_dim=hidden_dim,
                                 device=torch.device('cuda'),
                                 dtype=torch.float32)
        dsloader = DeepSpeedDataLoader(dataset,
                                       batch_size=args["batch_size"],
                                       local_rank=0,
                                       pin_memory=False,
                                       tput_timer=None,
                                       data_parallel_world_size=args["world_size"],
                                       dataloader_drop_last=args["drop_last"])
        dataloader = iter(dsloader)
        assert len(dsloader) == len(dataloader)
        assert args["result"] == len(dataloader)

    _test_dataloader_len(args=args, hidden_dim=hidden_dim)
