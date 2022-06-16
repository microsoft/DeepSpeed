# Copyright 2020 The Microsoft DeepSpeed Team
"""
DeepSpeed launcher, this is similar to torch.distributed.launch but supports
additional features such as arbitrary gpu exclusion.

deepspeed.launcher.launch is intended to be run on a single worker node and
will spawn several worker sub-processes depending on how many devices/ranks
are on the worker.
"""

import sys
import subprocess
import os
import json
import base64
import time
import signal
from collections import defaultdict
from argparse import ArgumentParser, REMAINDER
from torch.distributed.launcher.api import LaunchConfig, elastic_launch
from torch.distributed.elastic.multiprocessing import Std
from ..constants import TORCH_DISTRIBUTED_DEFAULT_PORT
from ..utils import logger
PID_FILE_BASEPATH = "/tmp"
# sys.path.append('/home/t-arpanjain/work/elastic_try/using_torch/')
# from functionAgent import FunctionElasticAgent
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
import torch.distributed.elastic.rendezvous.registry as rdzv_registry
from torch.distributed.elastic.agent.server.api import WorkerSpec
from torch.distributed.elastic.multiprocessing import Std
from typing import Any, Dict, Optional, Tuple
from  torch.distributed.elastic.rendezvous.dynamic_rendezvous import (
    _BackendRendezvousStateHolder, 
    DynamicRendezvousHandler, 
    _NodeDescGenerator, 
    RendezvousSettings, 
    RendezvousTimeout,
    _RendezvousState,
    RendezvousBackend,
    create_handler,
    _get_timeout
)
from torch.distributed import  Store
from torch.distributed.elastic.rendezvous import rendezvous_handler_registry
from torch.distributed.elastic.rendezvous.registry import get_rendezvous_handler
from datetime import datetime, timedelta
from torch.distributed.elastic.events import (
    NodeState,
    construct_and_record_rdzv_event,
)
from torch.distributed.elastic.agent.server.api import RunResult, log, WorkerState
from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
from torch.distributed.elastic.metrics import prof, put_metric

def parse_args():
    parser = ArgumentParser(description="DeepSpeed distributed training launch"
                            " utility that creates multiple distributed"
                            " processes on a single node")

    # Optional arguments for the launch helper
    parser.add_argument("--node_rank",
                        type=int,
                        default=0,
                        help="The rank of the node for multi-node distributed "
                        "training")
    parser.add_argument("--master_addr",
                        default="127.0.0.1",
                        type=str,
                        help="Master node (rank 0)'s address, should be either"
                        " the IP address or the hostname of node 0, for"
                        " single node multi-proc training, the"
                        " --master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port",
                        default=TORCH_DISTRIBUTED_DEFAULT_PORT,
                        type=int,
                        help="Master node (rank 0)'s free port that needs to "
                        "be used for communication during distributed "
                        "training")
    parser.add_argument("--world_info",
                        default="None",
                        type=str,
                        help="world info base64 encoded dictionary")

    parser.add_argument("--module",
                        action="store_true",
                        help="Change each process to interpret the launch "
                        "script as a Python module, executing with the same "
                        "behavior as 'python -m'.")

    parser.add_argument("--no_python",
                        action="store_true",
                        help="Skip prepending the training script with "
                        "'python' - just execute it directly.")

    parser.add_argument("--enable_elastic_training",
                        action="store_true",
                        help="Enable elastic training support.")

    parser.add_argument("--max_nodes",
                        type=int,
                        default=-1,
                        help="Max number of nodes in elastic training.")

    parser.add_argument("--no_local_rank",
                        action="store_true",
                        help="Do not pass local_rank as an argument when calling "
                        "the user's training script.")

    parser.add_argument("--save_pid",
                        type=int,
                        default=0,
                        help="main launching process pid, for internal pid tracking")

    # positional
    parser.add_argument("training_script",
                        type=str,
                        help="The full path to the single GPU training "
                        "program/script to be launched in parallel, "
                        "followed by all the arguments for the "
                        "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def get_config_elastic(args, num_local_procs, node_rank) -> LaunchConfig:

    config: Dict[str, str] = {'timeout': 100}
    config["store_type "] =  "file"

    config = LaunchConfig(
        min_nodes=args.nnodes,
        max_nodes=args.max_nodes,
        nproc_per_node=num_local_procs,
        run_id="123456789",
        role="default",
        rdzv_endpoint="worker-0:46728",
        rdzv_backend='c10d',
        rdzv_configs=config,
        max_restarts=100,
        monitor_interval=1,
        start_method="spawn",
        redirects=Std.from_str("0"),
        tee=Std.from_str("0"),
        log_dir="",
    )
    return config

def _remove_participant_epilogue_DS(state: _RendezvousState, settings: RendezvousSettings) -> None:
    if state.complete:
        # If we do not have any participants left, move to the next round.
        if not state.participants:
            state.complete = False

            state.round += 1
        else:
            state.restart = True
    else:
        if len(state.participants) < settings.min_nodes:
            state.deadline = None



class _DSBackendRendezvousStateHolder(_BackendRendezvousStateHolder):
    def _sanitize(self) -> None:
        state = self._state

        if hasattr(state, 'property'):
            state.restart = False

        expire_time = datetime.utcnow() - (
            self._settings.keep_alive_interval * self._settings.keep_alive_max_attempt
        )

        # Filter out the dead nodes.
        self._dead_nodes = [
            node
            for node, last_heartbeat in state.last_heartbeats.items()
            if last_heartbeat < expire_time
        ]
        for node, last_heartbeat in state.last_heartbeats.items():
            print("last node:", node, last_heartbeat)

        participant_removed = False

        for dead_node in self._dead_nodes:
            del state.last_heartbeats[dead_node]

            try:
                del state.participants[dead_node]

                participant_removed = True
            except KeyError:
                pass

            try:
                state.wait_list.remove(dead_node)
            except KeyError:
                pass

        if participant_removed:
            # Common epilogue shared with the _remove_from_participants()
            # function of _DistributedRendezvousOpExecutor.
            _remove_participant_epilogue_DS(state, self._settings)

class DynamicRendezvousHandlerDS(DynamicRendezvousHandler):
    _node_desc_generator = _NodeDescGenerator()

    @classmethod
    def from_backend(
        cls,
        run_id: str,
        store: Store,
        backend: RendezvousBackend,
        min_nodes: int,
        max_nodes: int,
        timeout: Optional[RendezvousTimeout] = None,
    ):

        node = cls._node_desc_generator.generate()

        settings = RendezvousSettings(
            run_id,
            min_nodes,
            max_nodes,
            timeout or RendezvousTimeout(),
            keep_alive_interval=timedelta(seconds=5),
            keep_alive_max_attempt=3,
        )

        state_holder = _DSBackendRendezvousStateHolder(backend, settings)
        return cls(node, settings, backend.name, store, state_holder)


def create_ds_handler(
    store: Store, backend: RendezvousBackend, params: RendezvousParameters
) -> DynamicRendezvousHandler:
    try:
        timeout = RendezvousTimeout(
            _get_timeout(params, "join"),
            _get_timeout(params, "last_call"),
            _get_timeout(params, "close"),
        )

        return DynamicRendezvousHandlerDS.from_backend(
            params.run_id,
            store,
            backend,
            params.min_nodes,
            params.max_nodes,
            timeout,
        )
    except Exception as e:
        construct_and_record_rdzv_event(
            message=f"{type(e).__name__}: {str(e)}",
            run_id=params.run_id,
            node_state=NodeState.FAILED,
        )
        raise

def _create_ds_c10d_handler(params: RendezvousParameters):
    from torch.distributed.elastic.rendezvous.c10d_rendezvous_backend import create_backend

    backend, store = create_backend(params)

    return create_ds_handler(store, backend, params)
del rendezvous_handler_registry._registry["c10d"]
rendezvous_handler_registry.register("c10d", _create_ds_c10d_handler)

class DSElasticAgent(LocalElasticAgent):
    def _invoke_run(self, role: str = "default") -> RunResult:
        # NOTE: currently only works for a single role

        spec = self._worker_group.spec
        role = spec.role

        log.info(
            f"[{role}] starting workers for entrypoint: {spec.get_entrypoint_name()}"
        )

        self._initialize_workers(self._worker_group)
        monitor_interval = spec.monitor_interval
        rdzv_handler = spec.rdzv_handler

        while True:
            assert self._worker_group.state != WorkerState.INIT
            time.sleep(monitor_interval)
            run_result = self._monitor_workers(self._worker_group)
            state = run_result.state
            self._worker_group.state = state

            put_metric(f"workers.{role}.remaining_restarts", self._remaining_restarts)
            put_metric(f"workers.{role}.{state.name.lower()}", 1)

            if state == WorkerState.SUCCEEDED:
                log.info(
                    f"[{role}] worker group successfully finished."
                    f" Waiting {self._exit_barrier_timeout} seconds for other agents to finish."
                )
                self._exit_barrier()
                return run_result
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED} or rdzv_handler._state_holder.state.restart:
                if self._remaining_restarts > 0:
                    log.info(
                        f"[{role}] Worker group {state.name}. "
                        f"{self._remaining_restarts}/{spec.max_restarts} attempts left;"
                        f" will restart worker group"
                    )
                    self._remaining_restarts -= 1
                    rdzv_handler._state_holder.state.restart = False
                    self._restart_workers(self._worker_group)
                else:
                    self._stop_workers(self._worker_group)
                    self._worker_group.state = WorkerState.FAILED
                    self._exit_barrier()
                    return run_result
            elif state == WorkerState.HEALTHY:
                # membership changes do not count as retries
                num_nodes_waiting = rdzv_handler.num_nodes_waiting()
                group_rank = self._worker_group.group_rank
                if num_nodes_waiting > 0:
                    log.info(
                        f"[{role}] Detected {num_nodes_waiting} "
                        f"new nodes from group_rank={group_rank}; "
                        f"will restart worker group"
                    )
                    self._restart_workers(self._worker_group)
            else:
                raise Exception(f"[{role}] Worker group in {state.name} state")

def main():
    args = parse_args()
    current_env = os.environ.copy()



    for k in current_env.keys():
        if "NCCL" in k:
            logger.info(f"{args.node_rank} {k}={current_env[k]}")

    if args.world_info == "None":
        raise ValueError("world_info can not be None")
    world_info = base64.urlsafe_b64decode(args.world_info)
    world_info = json.loads(world_info)

    logger.info(f"WORLD INFO DICT: {world_info}")
    node_list = list(world_info.keys())
    args.nnodes = len(node_list)
    local_node = node_list[args.node_rank]
    local_gpu_ids = world_info[local_node]
    num_local_procs = len(local_gpu_ids)
    logger.info(
        f"nnodes={args.nnodes}, num_local_procs={num_local_procs}, node_rank={args.node_rank}"
    )

    global_rank_mapping = defaultdict(list)
    curr_global_rank = 0
    dist_world_size = 0
    for node_id in node_list:
        gids = world_info[node_id]
        dist_world_size += len(gids)
        for gid in gids:
            global_rank_mapping[node_id].append(curr_global_rank)
            curr_global_rank += 1
    logger.info(f"global_rank_mapping={global_rank_mapping}")
    logger.info(f"dist_world_size={dist_world_size}")
    current_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, local_gpu_ids))
    logger.info(f"Setting CUDA_VISIBLE_DEVICES={current_env['CUDA_VISIBLE_DEVICES']}")

    # set PyTorch distributed related environmental variables
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)
    current_env["CROSS_RANK"] = str(args.node_rank)
    current_env["CROSS_SIZE"] = str(args.nnodes)
    current_env["LOCAL_SIZE"] = str(num_local_procs)

    if args.save_pid:
        print(f"launcher pid: {os.getpid()}")

    pid_file = None
    if args.save_pid:
        launcher_pid = os.getpid()
        pid_file = os.path.join(PID_FILE_BASEPATH, f"{args.save_pid}.deepspeed")
        assert not os.path.isfile(pid_file), "pid file exists but shouldn't"
        with open(pid_file, 'w') as fd:
            fd.write(f"{launcher_pid}")

    processes = []
    cmd = []

    if not args.enable_elastic_training:
        for local_rank in range(0, num_local_procs):
            # each process's rank
            dist_rank = global_rank_mapping[local_node][local_rank]
            current_env["RANK"] = str(dist_rank)
            current_env["LOCAL_RANK"] = str(local_rank)

            # spawn the processes
            cmd = []
            if not args.no_python:
                cmd = [sys.executable, "-u"]
                if args.module:
                    cmd.append("-m")
            else:
                if args.module:
                    raise ValueError("Don't use both the '--no_python' flag"
                                    " and the '--module' flag at the same time.")
            cmd.append(args.training_script)
            # A user may not want to pass local_rank as a keyword arg so we make this optional.
            if not args.no_local_rank:
                cmd.append(f"--local_rank={local_rank}")
            cmd += args.training_script_args

            process = subprocess.Popen(cmd, env=current_env)
            processes.append(process)
    else:
        # dist_rank = global_rank_mapping[local_node][local_rank]
        # os.environ["RANK"] = str(dist_rank)
        # os.environ["LOCAL_RANK"] = str(local_rank)
        assert args.max_nodes > 0, "Max nodes should be provided in elastic training"

        os.environ["MASTER_ADDR"] = args.master_addr
        os.environ["MASTER_PORT"] = str(args.master_port)
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)

        # spawn the processes
        cmd = []
        if not args.no_python:
            cmd = [sys.executable, "-u"]
            if args.module:
                cmd.append("-m")
        else:
            if args.module:
                raise ValueError("Don't use both the '--no_python' flag"
                                " and the '--module' flag at the same time.")
        cmd.append(args.training_script)
        cmd += args.training_script_args
        elastic_config = get_config_elastic(args, num_local_procs,args.node_rank)
        cmd_args = cmd[1:]
        # cmd_args = ['MASTER_ADDR={}'.format(args.master_addr), 'MASTER_PORT={}'.format(args.master_port)] + cmd_args
        print ("CMD is:",cmd_args)
        
        
        # elastic_launch(
        #     config=elastic_config,
        #     entrypoint= cmd[0],
        # )(*cmd_args)
        rdzv_configs: Dict[str, str] = {'timeout': 100}
        rdzv_parameters = RendezvousParameters(
            backend='c10d',
            endpoint=args.master_addr+":29400",
            run_id='123456789',
            min_nodes=args.nnodes,
            max_nodes=args.max_nodes,
            **rdzv_configs
        )

        spec = WorkerSpec(
                role='trainer',
                local_world_size=num_local_procs,
                entrypoint=cmd[0],
                args=cmd[1:],
                rdzv_handler=rdzv_registry.get_rendezvous_handler(rdzv_parameters),
                max_restarts=100,
                monitor_interval=5,
                redirects=Std.from_str("0"),
                tee=Std.from_str("0"),
                master_addr=args.master_addr,
                master_port=str(args.master_port),
            )
        agent = DSElasticAgent(
            spec
        )
        agent.run()


    sig_names = {2: "SIGINT", 15: "SIGTERM"}
    last_return_code = None

    def sigkill_handler(signum, frame):
        for process in processes:
            logger.info(f"Killing subprocess {process.pid}")
            try:
                process.kill()
            except Exception:
                pass
        if last_return_code is not None:
            logger.error(f"{cmd} exits with return code = {last_return_code}")
            sys.exit(last_return_code)
        if signum in sig_names:
            logger.info(f"Main process received {sig_names[signum]}, exiting")
        if args.save_pid:
            if os.path.isfile(pid_file):
                os.remove(pid_file)
        sys.exit(1)

    # pass SIGINT/SIGTERM to children if the parent is being terminated
    signal.signal(signal.SIGINT, sigkill_handler)
    signal.signal(signal.SIGTERM, sigkill_handler)

    alive_processes = set(processes)
    while len(alive_processes):
        finished_processes = []
        for process in alive_processes:
            if process.poll() is None:
                # the process is still running
                continue
            else:
                if process.returncode != 0:
                    last_return_code = process.returncode  # for sigkill_handler
                    sigkill_handler(signal.SIGTERM, None)  # not coming back
                else:
                    # exited cleanly
                    logger.info(f"Process {process.pid} exits successfully.")
                    finished_processes.append(process)
        alive_processes = set(alive_processes) - set(finished_processes)

        time.sleep(1)


if __name__ == "__main__":
    main()
