import json
import os
import sys
import time
import inspect
import traceback
import logging
import uuid

from contextlib import contextmanager

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Callable, ContextManager
from typing_extensions import Protocol
from threading import local

EventName = str
EventCallback = Callable[["BaseEvent"], None]


@dataclass
class BaseEvent:
    id: str
    name: EventName
    start_time: float
    app_code_name: str
    app_file_name: str
    app_line_no: int
    thread_group_name: str
    thread_no: int
    data: Dict[str, Any]

    end_time: Optional[float] = None
    elapsed_time: Optional[float] = None

    def asdict(self):
        result = asdict(self)
        result["type_name"] = self.type_name
        return result

    @property
    def type_name(self):
        return "base"


@dataclass
class PointwiseEvent(BaseEvent):
    @property
    def type_name(self):
        return "pointwise"


@dataclass
class TimespanEvent(BaseEvent):
    is_done: bool = False
    error_occurred: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_tb: Optional[str] = None
    children: List["TimespanEvent"] = field(default_factory=list)
    parent_id: Optional[str] = None

    @property
    def type_name(self):
        return "timespan"


class EventManagerProtocol(Protocol):
    def add_event_handler(self, callback: EventCallback):
        ...

    def remove_event_handler(self, callback: EventCallback):
        ...

    def timespan(self, name, *, data=None) -> ContextManager[TimespanEvent]:
        ...

    def pointwise(self, name, *, data=None):
        ...


class EventManager:
    @dataclass
    class ThreadState:
        thread_group_name: str
        thread_no: int
        stack: List[TimespanEvent] = field(default_factory=list)
        last_timespan_event: Optional[TimespanEvent] = None

    def __init__(self):
        self._callbacks = []
        self._thread_local = local()

        self._uuid = str(uuid.uuid4())
        self._last_id = 0

        self.init_current_thread("main")

    @property
    def thread_state(self) -> ThreadState:
        if not hasattr(self._thread_local, 'thread_state'):
            self.init_current_thread("<<unknown_thread(please call init_current_thread at the start of thread)>>")
        return self._thread_local.thread_state

    def init_current_thread(self, thread_group_name: str, thread_no: int = 0):
        if not hasattr(self._thread_local, 'thread_state'):
            self._thread_local.thread_state = self.ThreadState(thread_group_name=thread_group_name, thread_no=thread_no)

    def pointwise(self, name, *, data=None):
        ev = self._build_event(PointwiseEvent, name, data, 2)
        ev.end_time = ev.start_time
        self._invoke_callbacks(ev)
        return ev

    @contextmanager
    def timespan(self, name, *, data=None):
        start_perf = time.perf_counter()
        ev: TimespanEvent = self._build_event(TimespanEvent, name, data, 3)

        ts = self.thread_state
        try:
            top = ts.stack[-1]
            top.children.append(ev)
            ev.parent_id = top.id
        except IndexError:
            ts.last_timespan_event = ev
        ts.stack.append(ev)

        self._invoke_callbacks(ev)

        try:
            yield ev
        except Exception:
            exc_type, exc, exc_tb = sys.exc_info()
            ev.error_occurred = True
            ev.error_type = exc_type.__name__
            ev.error_message = str(exc)
            ev.error_tb = "".join(traceback.format_tb(exc_tb))
            raise
        finally:
            ts.stack.pop()
            ev.is_done = True
            end_time = time.time()
            end_perf = time.perf_counter()
            elapsed_time = end_perf - start_perf

            ev.end_time = end_time
            ev.elapsed_time = elapsed_time

            self._invoke_callbacks(ev)

    def add_event_handler(self, callback: EventCallback):
        self._callbacks.append(callback)

    def remove_event_handler(self, callback: EventCallback):
        self._callbacks.remove(callback)

    def _build_event(self, factory, name, data, depth):
        ts = self.thread_state
        self._last_id += 1
        id = "{}:{}".format(self._uuid, str(self._last_id))

        frame = inspect.currentframe()
        for _ in range(depth):
            if frame:
                frame = frame.f_back
        if frame:
            app_line_no = frame.f_lineno
            app_code_name = frame.f_code.co_name
            app_file_name = frame.f_code.co_filename
        else:
            app_line_no = 0
            app_code_name = ""
            app_file_name = "<unknown>"

        data = data or {}

        try:
            thread_group_name = ts.thread_group_name
            thread_no = ts.thread_no
        except AttributeError:
            thread_group_name = "<Undefined: call init_current_thread in advance>"
            thread_no = 0

        start_time = time.time()
        ev = factory(
            id=id,
            name=name,
            start_time=start_time,
            app_file_name=app_file_name,
            app_line_no=app_line_no,
            app_code_name=app_code_name,
            thread_group_name=thread_group_name,
            thread_no=thread_no,
            data=data,
        )
        return ev

    def _invoke_callbacks(self, ev: BaseEvent):
        for cb in self._callbacks:
            try:
                cb(ev)
            except Exception:
                logging.exception("Exception raised in the event callback loop")


class ChromeTraceRecorder:
    """This class records events from an event manager and generate trace which can be fed onto Chrome visualizer.

    Notes:
        - After you enable the recorder, it will receive all the following events and store them in its internal buffer.
          Without periodic flush using ``.dump_stats()``, allocated memory size will grows.

    Examples:
        >>> event_manager = EventManager()  # or any object that complies to ``EventManagerProtocol``
        >>> recorder = ChromeTraceRecorder(event_manager)
        >>> with recorder:  # Or, recorder.enable() followed by record.disable() works fine.
        >>>     ... # Do something
        >>> recorder.dump_stats("trace-for-chrome.json")
        >>> # Now open up chrome://tracing in chrome browser and click "Load" button to visualize the output file.

    For more information on format of trace file,
    see https://github.com/kwlzn/pytracing/blob/master/pytracing/pytracing.py
    """
    def __init__(self, event_manager: EventManagerProtocol):
        self.event_manager = event_manager
        self.enabled = False
        self.buffer: List[dict] = []
        self._cb = None
        self._pid = os.getpid()

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()

    def enable(self):
        if self.enabled:
            return
        self.enabled = True
        self._cb = self._on_event
        self.event_manager.add_event_handler(self._cb)

    def disable(self):
        if not self.enabled:
            return
        self.enabled = False
        self.event_manager.remove_event_handler(self._cb)
        self._cb = None

    def dump_stats(self, file_name: str):
        """Flush all dumps into json file named ``file_name``, after which it clears its internal buffer."""
        with open(file_name, "w") as file:
            file.write("[\n")
            for dict_ in self.buffer:
                json.dump(dict_, file)
                file.write(",\n")
            file.write("{}]\n")  # empty {} so the final entry doesn't end with a comma
        self.buffer.clear()

    def clear(self):
        self.buffer.clear()

    def _on_event(self, event):
        if isinstance(event, TimespanEvent):
            self._on_timespan_event(event)

    def _on_timespan_event(self, event: TimespanEvent):
        # See: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
        chrome_spec = dict(
            name=event.name,  # Event Name.
            cat=event.app_file_name,  # Event Category.
            tid="{}-{}".format(event.thread_group_name, event.thread_no),  # Thread ID.
            ph="E" if event.is_done else "B",  # "B": Begin, "E": End
            pid=os.getpid(),  # Process ID.
            ts=(event.end_time if event.is_done else event.start_time) * 1e+6,  # Timestamp in micros
            args=dict(
                function='{}:{}:{}'.format(event.app_file_name, event.app_line_no, event.app_code_name),
                data=event.data,
            )
        )
        self.buffer.append(chrome_spec)


event_manager = EventManager()

