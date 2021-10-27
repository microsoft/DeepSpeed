import torch
import deepspeed
import deepspeed.comm as dist
import os

backend = 'nccl'

#ranks = [i for i in range(int(size))]

#print(f'rank = {rank}, size = {size}')  #, ranks = {ranks}')

#comm = create_comm(ranks)

dist.init_process_group(backend, use_deepspeed=True)

rank = dist.get_rank()
size = dist.get_world_size()
local_rank = dist.get_local_rank()

print(f"ds rank = {rank}, ds local_rank = {local_rank}, ds size = {size}")

#dist.set_backend('nccl')

mat = torch.ones(size, dtype=torch.float32).cuda(int(local_rank))

dist.all_reduce(mat)

#dist.set_backend('mpi')

#dist.all_to_all(mat, mat)

#torch.cuda.synchronize()

print(f'rank = {rank}, mat = {mat}')

#import torch.distributed as dist
#deepspeed.init_distributed()
#dist.all_reduce(mat)
#torch.cuda.synchronize()

#print(f'rank = {rank}, mat-dist = {mat}')

# Run this test using:
# Inter-node: mpirun -n 16 -npernode 8 -hostfile /job/hostfile -x NCCL_DEBUG=WARN -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x NCCL_TOPO_FILE=/opt/msft/topo.xml  python tests/comm/test.py
# Intra-node: mpirun -n 8 -x NCCL_DEBUG=WARN python test.py
