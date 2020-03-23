import os
import json
import argparse
import torch


class MultiOutputModel(torch.nn.Module):
    def __init__(self, hidden_dim, num_output=2):
        super(MultiOutputModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.num_output = num_output
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        losses = []
        for x, y in zip(inputs, labels):
            hidden_dim = self.linear(x)
            losses.append(self.cross_entropy_loss(hidden_dim, y))
        return losses


def random_dataloader(model, total_samples, hidden_dim, device):
    batch_size = model.train_micro_batch_size_per_gpu()

    train_data = [torch.randn(total_samples,
                              hidden_dim,
                              device=device,
                              dtype=torch.half) for _ in range(model.num_outputs)]

    train_label = [torch.empty(total_samples,
                              dtype=torch.long,
                              device=device).random_(hidden_dim) for _ in range(model.num_output)]

    train_dataset = torch.utils.data.TensorDataset(*train_data, *train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    return train_loader


def create_config_from_dict(tmpdir, config_dict):
    config_path = os.path.join(tmpdir, 'temp_config.json')
    with open(config_path, 'w') as fd:
        json.dump(config_dict, fd)
    return config_path


def args_from_dict(tmpdir, config_dict):
    config_path = create_config_from_dict(tmpdir, config_dict)
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args='')
    args.deepspeed = True
    args.deepspeed_config = config_path
    args.local_rank = 0
    return args
