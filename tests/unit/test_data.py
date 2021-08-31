from deepspeed.utils import RepeatingLoader
from deepspeed.runtime.dataloader import DeepSpeedDataLoader
import torch
import pytest
import deepspeed
from common import distributed_test
from simple_model import SimpleModel, args_from_dict, random_dataset


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
        "steps_per_print": 1,
        "fp16": {
            "enabled": True
        },
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)
    device = torch.cuda.curent_device()

    @distributed_test(world_size=[1])
    def _test_dataloader_drop_last(args, model, hidden_dim, device):
        optimizer = torch.optim.AdamW(params=model.parameters())
        train_dataset = random_dataset(total_samples=50,
                                       hidden_dim=hidden_dim,
                                       device=device,
                                       dtype=torch.half)
        model, _, training_dataloader, _ = deepspeed.initialize(args=args,
                                                                model=model,
                                                                training_data=train_dataset,
                                                                optimizer=optimizer)
        for n, batch in enumerate(training_dataloader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_dataloader_drop_last(args=args, model=model, hidden_dim=hidden_dim, device=device)
