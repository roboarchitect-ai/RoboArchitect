from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from roboarchitect.protocol import StrictSchemaError


__all__ = ["HarnessMemory", "memory_key", "memory_value"]


def memory_key(value: Any) -> str:
    if type(value) is not str or not value:
        raise StrictSchemaError("memory key: expected a nonempty string")
    return value


def memory_value(value: Any, path: str = "memory value", depth: int = 0) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise StrictSchemaError(f"{path}: expected a finite number")
        return value
    if isinstance(value, (list, tuple)):
        return [memory_value(item, f"{path}[]", depth + 1) for item in value]
    if isinstance(value, Mapping):
        result = {}
        for name, item in value.items():
            if type(name) is not str:
                raise StrictSchemaError(f"{path}: mapping keys must be strings")
            result[name] = memory_value(item, f"{path}.{name}", depth + 1)
        return result
    raise StrictSchemaError(f"{path}: {type(value).__name__} cannot be stored")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


class HarnessMemory:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve()
        self._lock = threading.Lock()
        self._entries: dict[str, Any] = {}
        self._writes = 0
        self._reads = 0
        self._hits = 0
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        if not self._path.is_file() or self._path.is_symlink():
            return
        value = json.loads(self._path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, Mapping)
            or set(value) != {"entries", "usage", "updated_ns"}
            or not isinstance(value["entries"], Mapping)
            or not isinstance(value["usage"], Mapping)
        ):
            raise StrictSchemaError("memory.json fields differ")
        entries = {}
        for name, item in value["entries"].items():
            entries[memory_key(name)] = memory_value(item)
        usage = value["usage"]
        for name in ("n_writes", "n_reads", "n_read_hits"):
            if type(usage.get(name)) is not int or usage[name] < 0:
                raise StrictSchemaError("memory.json usage counts differ")
        self._entries = entries
        self._writes = int(usage["n_writes"])
        self._reads = int(usage["n_reads"])
        self._hits = int(usage["n_read_hits"])

    def _payload(self, entries: Mapping[str, Any]) -> bytes:
        return json.dumps(
            {
                "entries": dict(sorted(entries.items())),
                "usage": {
                    "n_entries": len(entries),
                    "n_writes": self._writes,
                    "n_reads": self._reads,
                    "n_read_hits": self._hits,
                },
                "updated_ns": time.time_ns(),
            },
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")

    def _commit(self, entries: dict[str, Any]) -> None:
        _atomic_write(self._path, self._payload(entries))
        self._entries = entries

    def remember(self, key: Any, value: Any) -> None:
        name = memory_key(key)
        stored = memory_value(value)
        with self._lock:
            entries = dict(self._entries)
            entries[name] = stored
            self._writes += 1
            try:
                self._commit(entries)
            except BaseException:
                self._writes -= 1
                raise

    def recall(self, key: Any) -> Any:
        name = memory_key(key)
        with self._lock:
            self._reads += 1
            if name not in self._entries:
                return None
            self._hits += 1
            return memory_value(self._entries[name])

    def keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._entries))

    def forget(self, key: Any) -> bool:
        name = memory_key(key)
        with self._lock:
            if name not in self._entries:
                return False
            entries = dict(self._entries)
            entries.pop(name)
            self._commit(entries)
            return True

    def usage(self) -> dict[str, int]:
        with self._lock:
            return {
                "n_entries": len(self._entries),
                "n_writes": self._writes,
                "n_reads": self._reads,
                "n_read_hits": self._hits,
            }

    def flush(self) -> None:
        with self._lock:
            self._commit(dict(self._entries))
