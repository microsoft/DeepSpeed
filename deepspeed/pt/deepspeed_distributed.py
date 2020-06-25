import torch
from deepspeed.pt.log_utils import logger


def distributed_init(dist_backend="nccl"):
    """
    Initialize torch.distributed backend, potentially performing MPI discovery if needed

    Arguments:
        dist_backend: torch distributed backend
    """
    required_env = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    if not all(map(lambda v: v in os.environ, required_env)):
        logger.info(
            "Not using the DeepSpeed or torch.distributed launchers, attempting to detect MPI environment..."
        )
        mpi_discovery()

    if not dist.is_initialized():
        logger.info("Initializing torch distributed with backend: {}".format(
            self.dist_backend))
        torch.distributed.init_process_group(backend=self.dist_backend)


def mpi_discovery():
    """
    Discovery MPI environment and map to relevant torch.distributed state
    """
    from mpi4py import MPI
    import subprocess

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    master_addr = None
    if rank == 0:
        hostname_cmd = ["hostname -I"]
        result = subprocess.check_output(hostname_cmd, shell=True)
        master_addr = result.decode('utf-8').split()[0]
    master_addr = comm.bcast(master_addr, root=0)

    # Determine local rank by assuming hostnames are unique
    proc_name = MPI.Get_processor_name()
    all_procs = comm.allgather(proc_name)
    local_rank = sum([i == proc_name for i in all_procs[:rank]])

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = local_rank
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = TORCH_DISTRIBUTED_DEFAULT_PORT

    logger.info(
        "Discovered MPI settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}"
        .format(os.environ['RANK'],
                os.environ['LOCAL_RANK'],
                os.environ['WORLD_SIZE'],
                os.environ['MASTER_ADDR'],
                os.environ['MASTER_PORT']))
