from deepspeed.runtime.pipe.schedule import (TrainSchedule, ForwardPass, BackwardPass, OptimizerStep, RecvGrad,
                                             RecvActivation,
                                             SendGrad, SendActivation, LoadMicroBatch, ReduceGrads, ReduceTiedGrads)
from pprint import pprint
from pytablewriter import MarkdownTableWriter

flatten = lambda t: [item for sublist in t for item in sublist]


def expand(steps, include_all=False):
    for c, i in enumerate(steps):
        string = ''
        for j in range(len(i)):
            if not include_all:
                cond = lambda x: (isinstance(x, ForwardPass) or isinstance(x, BackwardPass))
            else:
                cond = lambda x: x
            if not i[j]: i[j] = [None]
            if i[j] is not None:
                if cond(i[j]):
                    if string != '':
                        string += ' / '
                    string += f'{reprs[type(i[j])]}'
                    if hasattr(i[j], 'buffer_id'):
                        string += f'_{i[j].buffer_id + 1}'
        steps[c] = string if string != '' else None
    return steps


reprs = {
    ForwardPass: 'fwd',
    BackwardPass: 'bwd',
    RecvActivation: 'recv_act',
    SendActivation: 'send_act',
    RecvGrad: 'recv_grad',
    SendGrad: 'send_grad',
    LoadMicroBatch: 'load_batch',
    ReduceGrads: 'reduce_grads',
    ReduceTiedGrads: 'reduce_tied_grads',
    OptimizerStep: 'step',
}


def pipeline_visualizer(num_stages, num_microbatches, include_all=False):
    stages = {}
    for stage_id in range(num_stages):
        steps = [i for i in TrainSchedule(micro_batches=num_microbatches, stages=num_stages - 1 ,
                                          stage_id=stage_id).steps()]
        steps = expand(steps, include_all=include_all)
        stages[stage_id] = steps
    value_matrix = [v for k, v in stages.items()]
    headers = ['GPU ID'] + [str(i) for i in range(len(stages[0]))]
    value_matrix = [[f'GPU {i}'] + value_matrix[i] for i in range(len(value_matrix))]
    writer = MarkdownTableWriter(
        table_name=f"Pipe Schedule\n",
        headers=headers,
        value_matrix=value_matrix
    )
    string = writer.dumps()
    all_steps = flatten(value_matrix)
    idle_time = len([i for i in all_steps if i is None])
    print(all_steps)
    non_idle_time = len([i for i in all_steps if (i is not None and 'GPU' not in i)])
    string += f'\nNum Devices: {num_stages}\nNum Microbatches: {num_microbatches} \n' \
              f'Idle Time: {idle_time}\nNon Idle Time: {non_idle_time}'
    return string


if __name__ == "__main__":
    print(pipeline_visualizer(num_stages=8, num_microbatches=16, include_all=False))
