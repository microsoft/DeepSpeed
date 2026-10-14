# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Opt-in process and pytest diagnostics for stalled torch-latest runs."""

from __future__ import annotations

import atexit
import faulthandler
import json
import multiprocessing.process
import multiprocessing.util
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_MAX_CAPTURE_BYTES = 256 * 1024
_MAX_REPORT_BYTES = 24 * 1024
_MAX_REPORT_FILES = 8
_MAX_REPORT_SECTION_BYTES = 1536
_MAX_REPORT_STACK_BYTES = 768
_MAX_TRACEBACK_BYTES = 64 * 1024
_PROCESS_MARKERS = {}
_RECORDED_EXITS = set()
_REGISTERED_IDENTITIES = set()
_STACK_FILE = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _diagnostics_root() -> Path | None:
    value = os.environ.get("DS_DIAGNOSTICS_DIR")
    return Path(value) if value else None


def _process_identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rpartition(")")[2].split()
        return f"proc:{pid}:{fields[19]}"
    except (IndexError, OSError):
        pass
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    started = result.stdout.strip()
    return f"ps:{pid}:{started}" if result.returncode == 0 and started else None


def _write_json_line(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _register_stack_handler() -> None:
    global _STACK_FILE
    root = _diagnostics_root()
    if root is None or not hasattr(signal, "SIGUSR1"):
        return
    try:
        stack_dir = root / "stacks"
        stack_dir.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        identity = _process_identity(pid)
        if identity is None:
            return
        stack_file = (stack_dir / f"stack-{pid}.log").open("a", encoding="utf-8")
        faulthandler.enable(file=stack_file, all_threads=True)
        faulthandler.unregister(signal.SIGUSR1)
        faulthandler.register(signal.SIGUSR1, file=stack_file, all_threads=True, chain=False)
        previous = _STACK_FILE
        _STACK_FILE = stack_file
        if previous is not None:
            previous.close()
        marker = {
            "argv": sys.argv,
            "identity": identity,
            "local_rank": os.environ.get("LOCAL_RANK"),
            "nodeid": os.environ.get("PYTEST_CURRENT_TEST"),
            "pgid": os.getpgid(pid),
            "pid": pid,
            "ppid": os.getppid(),
            "rank": os.environ.get("RANK"),
            "registered_utc": _utc_now(),
            "worker": os.environ.get("PYTEST_XDIST_WORKER", "controller"),
        }
        marker_path = stack_dir / f"registered-{pid}.json"
        temporary = stack_dir / f".registered-{pid}.tmp"
        temporary.write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(marker_path)
        if identity not in _REGISTERED_IDENTITIES:
            _REGISTERED_IDENTITIES.add(identity)
            _PROCESS_MARKERS[identity] = (root, pid, marker)
            _write_json_line(root / "process-events.jsonl", {"event": "process_start", **marker})
            atexit.register(_record_process_exit, root, pid, identity, marker)
    except (OSError, RuntimeError, ValueError):
        pass


def _record_process_exit(root: Path, pid: int, identity: str, marker: dict[str, Any]) -> None:
    if pid != os.getpid() or identity in _RECORDED_EXITS:
        return
    _RECORDED_EXITS.add(identity)
    try:
        _write_json_line(
            root / "process-events.jsonl", {
                "event": "process_end",
                "exit_utc": _utc_now(),
                "identity": identity,
                "local_rank": marker.get("local_rank"),
                "nodeid": marker.get("nodeid"),
                "pid": pid,
                "rank": marker.get("rank"),
                "worker": marker.get("worker"),
            })
    except OSError:
        pass


_ORIGINAL_PROCESS_RUN = multiprocessing.process.BaseProcess.run


def _run_process_with_lifecycle(self: multiprocessing.process.BaseProcess) -> Any:
    _register_stack_handler()
    try:
        return _ORIGINAL_PROCESS_RUN(self)
    finally:
        identity = _process_identity(os.getpid())
        registration = _PROCESS_MARKERS.get(identity)
        if registration is not None:
            root, pid, marker = registration
            _record_process_exit(root, pid, identity, marker)


def _emit_pytest_event(event: str, nodeid: str, **fields: Any) -> None:
    root = _diagnostics_root()
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if root is None or worker is None:
        return
    payload = {
        "event": event,
        "nodeid": nodeid,
        "pid": os.getpid(),
        "time_monotonic": time.monotonic(),
        "time_utc": _utc_now(),
        "worker": worker,
    }
    payload.update(fields)
    _write_json_line(root / "events" / f"events-{worker}-{os.getpid()}.jsonl", payload)


def pytest_runtest_logstart(nodeid: str, location: tuple[str, int | None, str]) -> None:
    del location
    _emit_pytest_event("node_start", nodeid)


def pytest_runtest_logreport(report: Any) -> None:
    fields = {
        "duration_seconds": getattr(report, "duration", None),
        "outcome": report.outcome,
        "phase": report.when,
    }
    if report.failed:
        traceback = getattr(report, "longreprtext", str(report.longrepr))
        fields["traceback"] = traceback.encode()[:_MAX_TRACEBACK_BYTES].decode(errors="replace")
    _emit_pytest_event("phase_outcome", report.nodeid, **fields)


def pytest_runtest_logfinish(nodeid: str, location: tuple[str, int | None, str]) -> None:
    del location
    _emit_pytest_event("node_finish", nodeid)


def _registered_processes(root: Path, process_group_id: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = []
    rejected = []
    for path in sorted((root / "stacks").glob("registered-*.json")):
        try:
            marker = json.loads(path.read_text(encoding="utf-8"))
            pid = marker["pid"]
            if not isinstance(pid, int) or pid <= 0:
                raise ValueError("invalid pid")
            if marker.get("identity") != _process_identity(pid):
                raise ValueError("stale process identity")
            if os.getpgid(pid) != process_group_id or marker.get("pgid") != process_group_id:
                raise ValueError("process group mismatch")
            eligible.append(marker)
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
            rejected.append({"marker": path.name, "reason": f"{type(exc).__name__}: {exc}"})
    return eligible, rejected


def _run_capture(argv: Sequence[str], timeout: float = 3.0) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(list(argv), check=False, capture_output=True, text=True, timeout=timeout)
        output = result.stdout + result.stderr
        truncated = len(output.encode()) > _MAX_CAPTURE_BYTES
        if truncated:
            output = output.encode()[-_MAX_CAPTURE_BYTES:].decode(errors="replace")
        return {
            "argv": list(argv),
            "duration_seconds": time.monotonic() - started,
            "output": output,
            "returncode": result.returncode,
            "timed_out": False,
            "truncated": truncated,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "argv": list(argv),
            "duration_seconds": time.monotonic() - started,
            "output": f"{type(exc).__name__}: {exc}",
            "returncode": None,
            "timed_out": isinstance(exc, subprocess.TimeoutExpired),
            "truncated": False,
        }


def _read_process_state(pid: int) -> dict[str, str]:
    state = {}
    for name in ("cmdline", "stack", "status", "syscall", "wchan"):
        try:
            data = Path(f"/proc/{pid}/{name}").read_bytes()[:_MAX_CAPTURE_BYTES]
            state[name] = data.replace(b"\0", b" ").decode(errors="replace")
        except OSError as exc:
            state[name] = f"{type(exc).__name__}: {exc}"
    return state


def capture_stall(process_group_id: int) -> dict[str, Any]:
    """Request verified stacks and save one bounded stall snapshot."""
    root = _diagnostics_root()
    if root is None:
        raise RuntimeError("DS_DIAGNOSTICS_DIR is not set")
    eligible, rejected = _registered_processes(root, process_group_id)
    signals = []
    for marker in eligible:
        pid = marker["pid"]
        try:
            os.kill(pid, signal.SIGUSR1)
            status = "sent"
        except OSError as exc:
            status = f"{type(exc).__name__}: {exc}"
        signals.append({"pid": pid, "status": status})
    time.sleep(0.5)
    snapshot = dict(
        capture_utc=_utc_now(),
        eligible=eligible,
        kernel=_run_capture(["dmesg", "--ctime", "--level=emerg,alert,crit,err,warn"]),
        nvidia_compute_processes=_run_capture([
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ]),
        nvidia_gpus=_run_capture([
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]),
        process_group_id=process_group_id,
        process_states={str(marker["pid"]): _read_process_state(marker["pid"])
                        for marker in eligible},
        processes=_run_capture(["ps", "-eo", "pid,ppid,pgid,sid,stat,wchan:32,etimes,comm,args", "--sort=pid"]),
        rejected=rejected,
        signals=signals,
    )
    output = root / "stall-snapshot.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = root / ".stall-snapshot.tmp"
    temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return {"eligible": len(eligible), "rejected": len(rejected), "snapshot": str(output)}


def _bounded_report_text(value: str, limit: int) -> str:
    encoded = value.encode()
    if len(encoded) <= limit:
        return value
    tail = encoded[-limit:].decode(errors="replace")
    return f"<truncated to last {limit} bytes>\n{tail}"


def _bounded_stack_text(value: str, limit: int) -> str:
    current_thread = value.rfind("Current thread ")
    if current_thread >= 0:
        value = value[current_thread:]
    encoded = value.encode()
    if len(encoded) <= limit:
        return value
    head = encoded[:limit].decode(errors="replace")
    return f"<truncated to first {limit} bytes>\n{head}"


def _report_stack_paths(root: Path, snapshot: dict[str, Any]) -> list[Path]:
    eligible = {}
    for marker in snapshot.get("eligible", []):
        pid = marker.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            continue
        path = root / "stacks" / f"stack-{pid}.log"
        try:
            if path.stat().st_size:
                eligible[pid] = (path, marker)
        except OSError:
            pass

    compute_pids = set()
    for line in snapshot.get("nvidia_compute_processes", {}).get("output", "").splitlines():
        try:
            compute_pids.add(int(line.split(",", 1)[0].strip()))
        except ValueError:
            pass

    gpu_candidates = [item for pid, item in eligible.items() if pid in compute_pids]
    worker_gpu_candidates = [item for item in gpu_candidates if item[1].get("worker") != "controller"]
    candidates = worker_gpu_candidates or gpu_candidates
    selected = sorted(candidates, key=lambda item: item[1]["pid"], reverse=True)
    selected_pids = {marker["pid"] for _, marker in selected}
    selected.extend(item for _, item in sorted(eligible.items(), reverse=True) if item[1]["pid"] not in selected_pids)
    return [path for path, _ in selected[:_MAX_REPORT_FILES]]


def format_stall_report(summary: dict[str, Any]) -> str:
    """Return a bounded JSON report that survives backend cleanup in controller output."""
    snapshot_path = Path(summary["snapshot"])
    root = snapshot_path.parent
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    events = sorted((root / "events").glob("*.jsonl"))[-_MAX_REPORT_FILES:]
    stacks = _report_stack_paths(root, snapshot)
    report = {
        "capture_utc":
        snapshot.get("capture_utc"),
        "eligible_count":
        len(snapshot.get("eligible", [])),
        "event_tails": {
            path.name: _bounded_report_text(path.read_text(encoding="utf-8", errors="replace"),
                                            _MAX_REPORT_SECTION_BYTES)
            for path in events
        },
        "kernel":
        _bounded_report_text(snapshot.get("kernel", {}).get("output", ""), _MAX_REPORT_SECTION_BYTES),
        "nvidia_compute_processes":
        _bounded_report_text(
            snapshot.get("nvidia_compute_processes", {}).get("output", ""), _MAX_REPORT_SECTION_BYTES),
        "nvidia_gpus":
        _bounded_report_text(snapshot.get("nvidia_gpus", {}).get("output", ""), _MAX_REPORT_SECTION_BYTES),
        "process_group_id":
        snapshot.get("process_group_id"),
        "processes":
        _bounded_report_text(snapshot.get("processes", {}).get("output", ""), _MAX_REPORT_SECTION_BYTES),
        "rejected_count":
        len(snapshot.get("rejected", [])),
        "signals":
        snapshot.get("signals", [])[-2 * _MAX_REPORT_FILES:],
        "stacks": {
            path.name: _bounded_stack_text(path.read_text(encoding="utf-8", errors="replace"), _MAX_REPORT_STACK_BYTES)
            for path in stacks
        },
    }
    encoded = json.dumps(report, ensure_ascii=True, sort_keys=True)
    while len(encoded.encode()) > _MAX_REPORT_BYTES and report["event_tails"]:
        report["event_tails"].pop(next(iter(report["event_tails"])))
        encoded = json.dumps(report, ensure_ascii=True, sort_keys=True)
    if len(encoded.encode()) > _MAX_REPORT_BYTES:
        report = {
            "capture_utc": snapshot.get("capture_utc"),
            "eligible_count": len(snapshot.get("eligible", [])),
            "report_truncated": True,
        }
        encoded = json.dumps(report, ensure_ascii=True, sort_keys=True)
    return encoded


_register_stack_handler()
if _diagnostics_root() is not None:
    multiprocessing.process.BaseProcess.run = _run_process_with_lifecycle
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_register_stack_handler)
multiprocessing.util.register_after_fork(_register_stack_handler, lambda _: _register_stack_handler())
