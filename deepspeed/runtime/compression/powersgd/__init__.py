import torch

from .powersgd import Aggregator, AllReduce, Config, PowerSGD
from .utils import params_in_optimizer


def optimizer_step(optimizer: torch.optim.Optimizer, aggregator: Aggregator):
    """
    Aggregate gradients across workers using `aggregator`,
    and then take an optimizer step using the aggregated gradient.
    """
    params = params_in_optimizer(optimizer)
    # if torch.distributed.get_rank() == 0:
    #     print(">> In optimizer step:")
    #     for p in params:
    #         print(p.shape) 
    #         print(p.grad.shape)
    grads = [p.grad.data for p in params]  # type: ignore
    avg_grads = aggregator.aggregate(grads)  # subtracts the approximation from grads

    # Temporarily set parameter's gradients to the aggregated values
    for (p, g) in zip(params, avg_grads):
        p.grad = g

    # Run an optimizer step
    optimizer.step()

    # Put back the error buffer as the parameter's gradient
    for (p, g) in zip(params, grads):
        p.grad = g
