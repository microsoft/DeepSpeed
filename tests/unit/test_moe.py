import torch
import pytest
import torch.distributed as dist

import deepspeed
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology

from .common import distributed_test
from .simple_model import SimplePRMoEModel, args_from_dict, SimpleMoEModel, sequence_dataloader, MoEModelPipe
from .util import required_torch_version

try:
    from apex import amp
    _amp_available = True
except ImportError:
    _amp_available = False
amp_available = pytest.mark.skip(_amp_available, reason="apex/amp is not installed")


@pytest.mark.parametrize("ep_size, use_residual",
                         [(2,
                           True),
                          (2,
                           False),
                          (4,
                           True),
                          (4,
                           False)])
def test_moe(tmpdir, ep_size, use_residual):
    if not required_torch_version():
        pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

    config_dict = {
        "train_batch_size": 8,
        "steps_per_print": 1,
        "fp16": {
            "enabled": True
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 16

    @distributed_test(world_size=[4])
    def _test_moe(args, hidden_dim, ep_size, use_residual):
        # E+D -- ep_size = 2
        # E only -- ep_size = 4
        model = SimpleMoEModel(hidden_dim, ep_size=ep_size, use_residual=use_residual)
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              optimizer=optimizer,
                                              dist_init_required=False)
        #dist_init_required=False -- parameterize to True/False?

        data_loader = sequence_dataloader(model=model,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=model.device)

        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_moe(args=args,
              hidden_dim=hidden_dim,
              ep_size=ep_size,
              use_residual=use_residual)


@pytest.mark.parametrize("ep_size, use_residual", [(2, True), (2, False)])
def test_pr_moe(tmpdir, ep_size, use_residual):
    if not required_torch_version():
        pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

    config_dict = {
        "train_batch_size": 8,
        "steps_per_print": 1,
        "fp16": {
            "enabled": True
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 16

    @distributed_test(world_size=[4])
    def _test_moe(args, hidden_dim, ep_size, use_residual):
        # E+D -- ep_size = 2
        # E only -- ep_size = 4

        model = SimplePRMoEModel(hidden_dim, ep_size=ep_size, use_residual=use_residual)
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              optimizer=optimizer,
                                              dist_init_required=False)

        data_loader = sequence_dataloader(model=model,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=model.device)

        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_moe(args=args,
              hidden_dim=hidden_dim,
              ep_size=ep_size,
              use_residual=use_residual)


def test_moe_pipeline_parallel(tmpdir):
    if not required_torch_version():
        pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

    config_dict = {
        "train_batch_size": 8,
        "steps_per_print": 1,
        "fp16": {
            "enabled": True
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 16

    @distributed_test(world_size=[4])
    def _test_moe(args, hidden_dim):
        pp_world_size = 2
        tp_world_size = 1
        dp_world_size = 2
        ep_size = 2

        topo = PipeModelDataParallelTopology(num_pp=pp_world_size,
                                             num_mp=tp_world_size,
                                             num_dp=dp_world_size)
        model = MoEModelPipe(input_dim=hidden_dim,
                             hidden_dim=hidden_dim,
                             ep_size=ep_size,
                             topology=topo)
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              optimizer=optimizer,
                                              dist_init_required=False)

        num_steps = 3
        total_samples = num_steps * model.micro_batch_size * model.micro_batches
        data_loader = sequence_dataloader(model=model,
                                          total_samples=total_samples,
                                          hidden_dim=hidden_dim,
                                          device=model.device)

        model.set_dataloader(data_loader)
        for _ in range(num_steps):
            model.train_batch()

        # Verify expert parallel ranks
        def get_expert_parallel_ranks():
            from deepspeed.utils import groups
            ep_group = list(groups._EXPERT_PARALLEL_GROUP.values())[0]
            expert_dp_group = list(groups._EXPERT_DATA_PARALLEL_GROUP.values())[0]

            # collect expert parallel group ranks
            my_ep_group_ranks = [None] * ep_group.size()
            dist.all_gather_object(my_ep_group_ranks, dist.get_rank(), group=ep_group)
            my_expert_dp_group_ranks = [None] * expert_dp_group.size()
            dist.all_gather_object(my_expert_dp_group_ranks,
                                   dist.get_rank(),
                                   group=expert_dp_group)

            # gather all expert parallel group ranks
            ep_group_ranks = [None] * dist.get_world_size()
            dist.all_gather_object(ep_group_ranks, my_ep_group_ranks)
            expert_dp_group_ranks = [None] * dist.get_world_size()
            dist.all_gather_object(expert_dp_group_ranks, my_expert_dp_group_ranks)

            # deduplicate
            ep_group_ranks = sorted(list(set([tuple(item) for item in ep_group_ranks])))
            expert_dp_group_ranks = sorted(
                list(set([tuple(item) for item in expert_dp_group_ranks])))

            return ep_group_ranks, expert_dp_group_ranks

        ep_group_ranks, expert_dp_group_ranks = get_expert_parallel_ranks()
        dp_group_ranks = model.mpu.topology().get_axis_comm_lists("data")
        assert dp_group_ranks == [[0, 1], [2, 3]]
        assert ep_group_ranks == [(0, 1), (2, 3)]
        assert expert_dp_group_ranks == [(0, ), (1, ), (2, ), (3, )]

    _test_moe(args=args, hidden_dim=hidden_dim)
