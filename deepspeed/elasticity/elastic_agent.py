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
    _get_timeout,
    _RendezvousState
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
import time


def _remove_participant_epilogue_DS(state: _RendezvousState, settings: RendezvousSettings) -> None:
    if state.complete:
        # If we do not have any participants left, move to the next round.
        if not state.participants:
            state.complete = False

            state.round += 1
        else:
            state.complete = False

            state.round += 1
    else:
        if len(state.participants) < settings.min_nodes:
            state.deadline = None


# class _RendezvousStateDS(_RendezvousState):
#     round: int
#     complete: bool
#     deadline: Optional[datetime]
#     closed: bool
#     participants: Dict[_NodeDesc, int]
#     wait_list: Set[_NodeDesc]
#     last_heartbeats: Dict[_NodeDesc, datetime]
#     def __init__(self):
#         super().__init__()
#         self.restart = False


class _DSBackendRendezvousStateHolder(_BackendRendezvousStateHolder):
    # def __init__(
    #     self,
    #     backend: RendezvousBackend,
    #     settings: RendezvousSettings,
    #     cache_duration: int = 1,
    # ) -> None:
    #     super().__init__(backend, settings, cache_duration)
    #     self._state = _RendezvousStateDS()
    #     print(self._state.restart)

    # def state(self) -> _RendezvousStateDS:
    #     """See base class."""
    #     return self._state
    def _sanitize(self) -> None:
        state = self._state

        if not hasattr(state, 'property'):
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
        # for node, last_heartbeat in state.last_heartbeats.items():
        #     print("last node:", node, last_heartbeat)

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
# del rendezvous_handler_registry._registry["c10d"]
# rendezvous_handler_registry.register("c10d", _create_ds_c10d_handler)

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

        # if not hasattr(rdzv_handler._state_holder.state, 'property'):
        #     rdzv_handler._state_holder.state.restart = False

        participants = rdzv_handler._state_holder.state.participants

        while True:
            assert self._worker_group.state != WorkerState.INIT
            time.sleep(monitor_interval)
            run_result = self._monitor_workers(self._worker_group)
            state = run_result.state
            self._worker_group.state = state

            expire_time = datetime.utcnow() - (
                rdzv_handler._settings.keep_alive_interval * rdzv_handler._settings.keep_alive_max_attempt
            )
            _dead_nodes = [
                node
                for node, last_heartbeat in rdzv_handler._state_holder.state.last_heartbeats.items()
                if last_heartbeat < expire_time
            ]

            put_metric(f"workers.{role}.remaining_restarts", self._remaining_restarts)
            put_metric(f"workers.{role}.{state.name.lower()}", 1)

            if state == WorkerState.SUCCEEDED:
                log.info(
                    f"[{role}] worker group successfully finished."
                    f" Waiting {self._exit_barrier_timeout} seconds for other agents to finish."
                )
                self._exit_barrier()
                return run_result
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED} or len(participants)>len( rdzv_handler._state_holder.state.participants):
                if self._remaining_restarts > 0:
                    log.info(
                        f"[{role}] Worker group {state.name}. "
                        f"{self._remaining_restarts}/{spec.max_restarts} attempts left;"
                        f" will restart worker group"
                    )
                    self._remaining_restarts -= 1
                    # rdzv_handler._state_holder.state.restart = False
                    self._restart_workers(self._worker_group)
                    participants =  rdzv_handler._state_holder.state.participants
                    
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
                    participants =  rdzv_handler._state_holder.state.participants
            else:
                raise Exception(f"[{role}] Worker group in {state.name} state")
