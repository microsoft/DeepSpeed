import torch

from torch.distributed.distributed_c10d import _pg_names, _get_default_group
import torch.distributed as dist

from ..op_builder import CommunicationBuilder

coll_comm_module = None


class CommunicationHandle:
    def __init__(self,
                 start_event: torch.cuda.Event,
                 end_event: torch.cuda.Event,
                 timing: bool) -> None:

        self.start = start_event
        self.end = end_event
        self.timing = timing

    def is_completed(self, ):
        return self.end.query()

    def wait(self, stream):
        self.end.wait(stream)

    def synchronize(self):
        self.end.synchronize()


def map_process_group(group):
    # print(f'rank {dist.get_rank(group=group)}, _pg_names {_pg_names}, _pg_group_ranks {_pg_group_ranks}')
    if group == dist.group.WORLD:
        return _get_default_group()
    else:
        return group


def inplace_allgather(output_tensors, input_tensors, group, comm_stream, timing=False):
    """"""
    global coll_comm_module
    if coll_comm_module is None:
        coll_comm_module = CommunicationBuilder().load()

    group = map_process_group(group)
    process_group_name = _pg_names[group]

    start_event = torch.cuda.Event(enable_timing=timing)
    end_event = torch.cuda.Event(enable_timing=timing)
    if timing:
        start_event.record(comm_stream)
    with torch.cuda.stream(comm_stream):
        coll_comm_module._inplace_allgather(output_tensors,
                                            input_tensors,
                                            group,
                                            process_group_name)

        end_event.record(comm_stream)

    return CommunicationHandle(start_event, end_event, timing)
