from __future__ import annotations

import os
import signal
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator


TERMINATE_TIMEOUT_S = 5.0
KILL_TIMEOUT_S = 3.0


class Interrupted(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        self.signal_name = signal.Signals(self.signum).name
        self.exit_status = 128 + self.signum
        super().__init__(self.signal_name)


class ProcessRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: dict[int, Any] = {}
        self._stopping = False
        self.signal_number: int | None = None

    def register(self, process: Any) -> None:
        with self._lock:
            self._active[process.pid] = process
            stopping = self._stopping
        if stopping:
            self._terminate((process,))

    def unregister(self, process: Any) -> None:
        with self._lock:
            if self._active.get(getattr(process, "pid", None)) is process:
                self._active.pop(process.pid, None)

    @staticmethod
    def _wait(processes: tuple[Any, ...], timeout_s: float) -> tuple[Any, ...]:
        deadline = time.monotonic() + timeout_s
        remaining = tuple(item for item in processes if item.poll() is None)
        while remaining and time.monotonic() < deadline:
            time.sleep(0.02)
            remaining = tuple(item for item in remaining if item.poll() is None)
        return remaining

    @staticmethod
    def _signal(processes: tuple[Any, ...], signum: int) -> None:
        for process in processes:
            try:
                os.killpg(process.pid, signum)
            except (ProcessLookupError, PermissionError):
                pass

    def _terminate(self, processes: tuple[Any, ...]) -> None:
        alive = tuple(item for item in processes if item.poll() is None)
        self._signal(alive, signal.SIGTERM)
        remaining = self._wait(alive, TERMINATE_TIMEOUT_S)
        self._signal(remaining, signal.SIGKILL)
        self._wait(remaining, KILL_TIMEOUT_S)

    def terminate_all(self, signum: int | None = None) -> None:
        with self._lock:
            if signum is not None and self.signal_number is None:
                self.signal_number = int(signum)
            self._stopping = True
            processes = tuple(self._active.values())
        self._terminate(processes)


_REGISTRY_LOCK = threading.RLock()
_ACTIVE_REGISTRY: ProcessRegistry | None = None


@contextmanager
def process_registry(registry: ProcessRegistry) -> Iterator[ProcessRegistry]:
    global _ACTIVE_REGISTRY
    with _REGISTRY_LOCK:
        if _ACTIVE_REGISTRY is not None:
            raise RuntimeError("a process registry is already active")
        _ACTIVE_REGISTRY = registry
    try:
        yield registry
    finally:
        with _REGISTRY_LOCK:
            if _ACTIVE_REGISTRY is registry:
                _ACTIVE_REGISTRY = None


def register_process(process: Any) -> None:
    with _REGISTRY_LOCK:
        registry = _ACTIVE_REGISTRY
    if registry is not None:
        registry.register(process)


def unregister_process(process: Any) -> None:
    with _REGISTRY_LOCK:
        registry = _ACTIVE_REGISTRY
    if registry is not None:
        registry.unregister(process)


@contextmanager
def interruption_handlers(registry: ProcessRegistry) -> Iterator[None]:
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}

    def interrupt(signum: int, frame: Any) -> None:
        if registry.signal_number is not None:
            return
        try:
            registry.terminate_all(signum)
        finally:
            raise Interrupted(signum)

    for signum in previous:
        signal.signal(signum, interrupt)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
