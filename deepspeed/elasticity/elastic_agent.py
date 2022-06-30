from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta
from torch.distributed.elastic.agent.server.api import RunResult, log, WorkerState
from torch.distributed.elastic.metrics import prof, put_metric
import time


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
