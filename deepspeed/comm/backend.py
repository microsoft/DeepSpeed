''' DeepSpeed Communication Backend

# DS Backends -- Direct C/Ops
 - NCCL -- future default
 - MPI -- extra performance
 - RCCL -- maybe for AMD
 - GLOO -- N/A -- use via torch

# via torch.distributed
 - T-NCCL -- default -- will work for AMD as well
 - T-GLOO -- choose for cpu/testing without GPUs
 - T-MPI -- works but not commonly used
'''
''' DS Backend can be the base class
     -- NcclBackend, MpiBackend, and TorchBackend are the main subclasses we expect for now
'''


class Backend(object):
    def __init__(self, name='backend', rank=0, size=1):
        self.name = name
        # The world size and rank of the world process group
        self.world_group = None
        self.world_size = rank
        self.world_rank = size
        # Single process group (pg) implementation for now but keep a list for future
        self.process_groups = []
        self.initialized = False

    def is_initialized(self):
        return self.initialized

    def new_group(self):
        # create a new pg and add it to pg list
        pass

    def init_process_group(self):
        # subclasses will initialize them fully
        # - initalize a default world process group and add it to pg list
        self.initialized = True
