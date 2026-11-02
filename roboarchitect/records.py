from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboarchitect.protocol.schema import (
    StrictSchemaError,
    boolean,
    enum,
    fields,
    integer,
    json_object,
    reject_json_constant,
    sequence,
    string,
)


EPISODE_STATES = frozenset({"complete", "error"})
MAX_BENCHMARK_METRICS = 16
_METRIC_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def name_slug(value: str, limit: int) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")[:limit]


@dataclass(frozen=True, order=True)
class EpisodeKey:
    task_id: str
    scenario_id: str
    environment_seed: int
    policy_seed: int
    horizon: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", string(self.task_id, "episode.task_id"))
        object.__setattr__(self, "scenario_id", string(self.scenario_id, "episode.scenario_id"))
        object.__setattr__(
            self, "environment_seed", integer(self.environment_seed, "episode.environment_seed", minimum=0)
        )
        object.__setattr__(self, "policy_seed", integer(self.policy_seed, "episode.policy_seed", minimum=0))
        object.__setattr__(self, "horizon", integer(self.horizon, "episode.horizon", minimum=1))

    @classmethod
    def from_mapping(cls, value: Any) -> "EpisodeKey":
        obj = fields(
            value,
            {"task_id", "scenario_id", "environment_seed", "policy_seed", "horizon"},
            path="episode",
        )
        return cls(**obj)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "scenario_id": self.scenario_id,
            "environment_seed": self.environment_seed,
            "policy_seed": self.policy_seed,
            "horizon": self.horizon,
        }

    def folder_name(self) -> str:
        task = name_slug(self.task_id, 64) or "task"
        scenario = name_slug(self.scenario_id, 48) or "scene"
        return f"{task}-{scenario}-e{self.environment_seed}-p{self.policy_seed}"


@dataclass(frozen=True)
class EpisodeList:
    name: str
    episodes: tuple[EpisodeKey, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", string(self.name, "episode list name"))
        episodes = tuple(sorted(self.episodes))
        if not episodes or any(not isinstance(item, EpisodeKey) for item in episodes):
            raise StrictSchemaError(f"{self.name}: expected a nonempty list of episodes")
        if len(set(episodes)) != len(episodes):
            raise StrictSchemaError(f"{self.name}: an episode appears twice")
        names = [item.folder_name() for item in episodes]
        if len(set(names)) != len(names):
            raise StrictSchemaError(f"{self.name}: two episodes would share one folder name")
        object.__setattr__(self, "episodes", episodes)

    @classmethod
    def from_mapping(cls, name: str, value: Any) -> "EpisodeList":
        return cls(name, tuple(EpisodeKey.from_mapping(item) for item in sequence(value, name)))


@dataclass(frozen=True)
class EpisodeFile:
    name: str
    size_bytes: int

    def __post_init__(self) -> None:
        name = string(self.name, "file.name")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
            raise StrictSchemaError("file.name: invalid filename")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "size_bytes", integer(self.size_bytes, "file.size_bytes", minimum=0))

    @classmethod
    def from_mapping(cls, value: Any) -> "EpisodeFile":
        obj = fields(value, {"name", "size_bytes"}, path="file")
        return cls(**obj)

    def to_mapping(self) -> dict[str, Any]:
        return {"name": self.name, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class EpisodeManifest:
    key: EpisodeKey
    state: str
    success: bool | None
    steps: int
    started_ns: int
    finished_ns: int
    files: tuple[EpisodeFile, ...]
    error: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.key, EpisodeKey):
            raise StrictSchemaError("episode.json: expected an EpisodeKey")
        state = enum(self.state, EPISODE_STATES, "episode.json state")
        success = None if self.success is None else boolean(self.success, "episode.json success")
        steps = integer(self.steps, "episode.json steps", minimum=0, maximum=self.key.horizon)
        started = integer(self.started_ns, "episode.json started_ns", minimum=0)
        finished = integer(self.finished_ns, "episode.json finished_ns", minimum=started)
        files = tuple(self.files)
        if any(not isinstance(item, EpisodeFile) for item in files):
            raise StrictSchemaError("episode.json files: expected EpisodeFile entries")
        if tuple(sorted(item.name for item in files)) != tuple(item.name for item in files):
            raise StrictSchemaError("episode.json files: expected sorted names")
        if len({item.name for item in files}) != len(files):
            raise StrictSchemaError("episode.json files: duplicate names")
        if state == "complete" and (success is None or self.error is not None):
            raise StrictSchemaError("episode.json: a complete episode has a success value and no error")
        if state == "complete" and "trace.jsonl" not in {item.name for item in files}:
            raise StrictSchemaError("episode.json: a complete episode has a trace.jsonl")
        if state == "error" and (success is not None or self.error is None):
            raise StrictSchemaError("episode.json: an errored episode has an error and no success value")
        error = None if self.error is None else string(self.error, "episode.json error")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "started_ns", started)
        object.__setattr__(self, "finished_ns", finished)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "error", error)

    @classmethod
    def from_mapping(cls, value: Any) -> "EpisodeManifest":
        obj = fields(
            value,
            {"key", "state", "success", "steps", "started_ns", "finished_ns", "files", "error"},
            path="episode.json",
        )
        return cls(
            key=EpisodeKey.from_mapping(obj["key"]),
            state=obj["state"],
            success=obj["success"],
            steps=obj["steps"],
            started_ns=obj["started_ns"],
            finished_ns=obj["finished_ns"],
            files=tuple(EpisodeFile.from_mapping(item) for item in sequence(obj["files"], "episode.json files")),
            error=obj["error"],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "key": self.key.to_mapping(),
            "state": self.state,
            "success": self.success,
            "steps": self.steps,
            "started_ns": self.started_ns,
            "finished_ns": self.finished_ns,
            "files": [item.to_mapping() for item in self.files],
            "error": self.error,
        }


def load_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=json_object, parse_constant=reject_json_constant)
    except StrictSchemaError:
        raise
    except Exception as exc:
        raise StrictSchemaError(f"failed to load {source}: {exc}") from exc


def validate_benchmark_metrics(value: Any, path: str = "benchmark_metrics") -> dict[str, bool | float]:
    if not isinstance(value, Mapping) or not 1 <= len(value) <= MAX_BENCHMARK_METRICS:
        raise StrictSchemaError(f"{path}: expected 1 to {MAX_BENCHMARK_METRICS} metrics")
    result: dict[str, bool | float] = {}
    for name, metric in sorted(value.items()):
        if type(name) is not str or _METRIC_NAME.fullmatch(name) is None:
            raise StrictSchemaError(f"{path}: invalid metric name")
        if type(metric) is bool:
            result[name] = metric
        elif type(metric) in (int, float) and math.isfinite(float(metric)):
            result[name] = float(metric)
        else:
            raise StrictSchemaError(f"{path}.{name}: expected a finite number or a bool")
    return result


@dataclass(frozen=True)
class BenchmarkOutcome:
    key: EpisodeKey
    metrics: Mapping[str, bool | float]

    def __post_init__(self) -> None:
        if not isinstance(self.key, EpisodeKey):
            raise StrictSchemaError("episode outcome: expected an EpisodeKey")
        if not isinstance(self.metrics, Mapping) or "success" not in self.metrics:
            raise StrictSchemaError("episode outcome: success is missing")
        checked: dict[str, bool | float] = {}
        for name, value in sorted(self.metrics.items()):
            if type(name) is not str or not name or len(name) > 64:
                raise StrictSchemaError("episode outcome: invalid metric name")
            if type(value) is bool:
                checked[name] = value
            elif type(value) in (int, float) and math.isfinite(float(value)):
                checked[name] = float(value)
            else:
                raise StrictSchemaError(f"episode outcome: metric {name!r} is not a finite number")
        if type(checked["success"]) is not bool:
            raise StrictSchemaError("episode outcome: success must be a bool")
        object.__setattr__(self, "metrics", checked)

    @classmethod
    def from_mapping(cls, value: Any) -> "BenchmarkOutcome":
        if not isinstance(value, Mapping) or set(value) != {"key", "metrics"}:
            raise StrictSchemaError("episode outcome: fields differ")
        return cls(EpisodeKey.from_mapping(value["key"]), value["metrics"])

    def to_mapping(self) -> dict[str, Any]:
        return {"key": self.key.to_mapping(), "metrics": dict(self.metrics)}


@dataclass(frozen=True)
class Score:
    metric: str
    value: float
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.metric) is not str or not self.metric:
            raise StrictSchemaError("score: metric name is missing")
        if type(self.value) not in (int, float) or not math.isfinite(float(self.value)):
            raise StrictSchemaError("score: value is not finite")
        if not isinstance(self.details, Mapping):
            raise StrictSchemaError("score: details must be a mapping")
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "details", dict(self.details))

    @classmethod
    def from_mapping(cls, value: Any) -> "Score":
        if not isinstance(value, Mapping) or set(value) != {"metric", "value", "details"}:
            raise StrictSchemaError("score: fields differ")
        return cls(value["metric"], value["value"], value["details"])

    def to_mapping(self) -> dict[str, Any]:
        return {"metric": self.metric, "value": self.value, "details": dict(self.details)}
