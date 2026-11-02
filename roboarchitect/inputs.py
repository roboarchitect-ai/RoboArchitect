from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from roboarchitect.protocol import ActionChunk, ActionSpec, Observation, RobotStateSpec
from roboarchitect.protocol.observation import OPTICAL_CONVENTIONS
from roboarchitect.protocol.schema import StrictSchemaError, boolean, enum, fields, integer, mapping, sequence, string
from roboarchitect.records import EpisodeList


CODE_ROOT = Path(__file__).resolve().parents[1]
TOOL_CAPABILITIES = ("detection", "grasp", "language", "pointing", "segmentation", "vision")
EPISODE_SETS = ("training", "held_out")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else CODE_ROOT / path).resolve()


def _entries(value: Any, name: str) -> dict[str, Path]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"paths file: {name} must be an object")
    result = {}
    for key, item in value.items():
        if item in (None, ""):
            continue
        if not isinstance(item, str) or not Path(item).is_absolute():
            raise ValueError(f"paths file: {name}.{key} must be an absolute path")
        result[key] = Path(item)
    return result


@dataclass(frozen=True)
class Paths:
    file: Path
    python: Mapping[str, Path]
    repositories: Mapping[str, Path]

    def python_for(self, name: str) -> Path:
        path = self.python.get(name)
        if path is None:
            raise FileNotFoundError(f"{self.file.name} has no python.{name}; build that environment and add its Python")
        if not path.is_file():
            raise FileNotFoundError(f"python.{name} in {self.file.name} does not exist: {path}")
        return path

    def repository(self, name: str) -> Path:
        path = self.repositories.get(name)
        if path is None:
            raise FileNotFoundError(f"{self.file.name} has no repositories.{name}; add the folder of that repository")
        if not path.is_dir():
            raise FileNotFoundError(f"repositories.{name} in {self.file.name} does not exist: {path}")
        return path


def load_paths(path: str | Path) -> Paths:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"paths file is missing: {source}; copy paths.example.json and fill it in")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) - {"python", "repositories"}:
        raise ValueError("paths file: expected only python and repositories")
    return Paths(
        file=source,
        python=_entries(value.get("python"), "python"),
        repositories=_entries(value.get("repositories"), "repositories"),
    )


@dataclass(frozen=True)
class CameraSpec:
    name: str
    frame_id: str
    optical_convention: str
    width: int
    height: int
    has_depth: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", string(self.name, "camera.name"))
        object.__setattr__(self, "frame_id", string(self.frame_id, "camera.frame_id"))
        object.__setattr__(
            self,
            "optical_convention",
            enum(self.optical_convention, OPTICAL_CONVENTIONS, "camera.optical_convention"),
        )
        object.__setattr__(self, "width", integer(self.width, "camera.width", minimum=1))
        object.__setattr__(self, "height", integer(self.height, "camera.height", minimum=1))
        object.__setattr__(self, "has_depth", boolean(self.has_depth, "camera.has_depth"))

    @classmethod
    def from_mapping(cls, value: Any) -> "CameraSpec":
        obj = fields(value, {"name", "frame_id", "optical_convention", "width", "height", "has_depth"}, path="camera")
        return cls(**obj)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "frame_id": self.frame_id,
            "optical_convention": self.optical_convention,
            "width": self.width,
            "height": self.height,
            "has_depth": self.has_depth,
        }


@dataclass(frozen=True)
class EnvironmentSetup:
    simulator: str
    suite: str
    embodiment: str
    cameras: tuple[CameraSpec, ...]
    robot_state: tuple[RobotStateSpec, ...]

    def __post_init__(self) -> None:
        simulator = string(self.simulator, "environment.simulator")
        module, separator, class_name = simulator.partition(":")
        if not separator or not module.startswith("benchmarks.") or not class_name.isidentifier():
            raise StrictSchemaError("environment.simulator: expected benchmarks.<module>:<class>")
        object.__setattr__(self, "simulator", simulator)
        object.__setattr__(self, "suite", string(self.suite, "environment.suite"))
        object.__setattr__(self, "embodiment", string(self.embodiment, "environment.embodiment"))
        cameras = tuple(self.cameras)
        if not cameras or [item.name for item in cameras] != sorted({item.name for item in cameras}):
            raise StrictSchemaError("environment.cameras: expected sorted unique names")
        robot_state = tuple(self.robot_state)
        if not robot_state or [item.name for item in robot_state] != sorted({item.name for item in robot_state}):
            raise StrictSchemaError("environment.robot_state: expected sorted unique names")
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "robot_state", robot_state)

    @property
    def benchmark(self) -> str:
        return self.simulator.partition(":")[0].removeprefix("benchmarks.")

    @classmethod
    def from_mapping(cls, value: Any) -> "EnvironmentSetup":
        obj = fields(value, {"simulator", "suite", "embodiment", "cameras", "robot_state"}, path="environment")
        return cls(
            simulator=obj["simulator"],
            suite=obj["suite"],
            embodiment=obj["embodiment"],
            cameras=tuple(CameraSpec.from_mapping(item) for item in sequence(obj["cameras"], "environment.cameras")),
            robot_state=tuple(
                RobotStateSpec.from_mapping(item) for item in sequence(obj["robot_state"], "environment.robot_state")
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "simulator": self.simulator,
            "suite": self.suite,
            "embodiment": self.embodiment,
            "cameras": [item.to_mapping() for item in self.cameras],
            "robot_state": [item.to_mapping() for item in self.robot_state],
        }


@dataclass(frozen=True)
class PolicySetup:
    server: str
    checkpoint: str
    adapter: str
    action_spec: ActionSpec
    chunk_horizon: int
    execution_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "server", string(self.server, "policy.server"))
        object.__setattr__(self, "checkpoint", string(self.checkpoint, "policy.checkpoint"))
        object.__setattr__(self, "adapter", string(self.adapter, "policy.adapter"))
        if not isinstance(self.action_spec, ActionSpec):
            raise StrictSchemaError("policy.action_spec: expected an ActionSpec")
        horizon = integer(self.chunk_horizon, "policy.chunk_horizon", minimum=1)
        object.__setattr__(self, "chunk_horizon", horizon)
        object.__setattr__(
            self, "execution_count", integer(self.execution_count, "policy.execution_count", minimum=1, maximum=horizon)
        )

    @property
    def key(self) -> str:
        return f"{self.server}:{self.checkpoint}"

    @classmethod
    def from_mapping(cls, value: Any) -> "PolicySetup":
        obj = fields(value, {"server", "checkpoint", "adapter", "action_spec", "chunk_horizon", "execution_count"}, path="policy")
        return cls(
            server=obj["server"],
            checkpoint=obj["checkpoint"],
            adapter=obj["adapter"],
            action_spec=ActionSpec.from_mapping(obj["action_spec"]),
            chunk_horizon=obj["chunk_horizon"],
            execution_count=obj["execution_count"],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "server": self.server,
            "checkpoint": self.checkpoint,
            "adapter": self.adapter,
            "action_spec": self.action_spec.to_mapping(),
            "chunk_horizon": self.chunk_horizon,
            "execution_count": self.execution_count,
        }


@dataclass(frozen=True)
class Setup:
    environment: EnvironmentSetup
    policy: PolicySetup
    tools: tuple[tuple[str, str | None], ...]

    def __post_init__(self) -> None:
        tools = tuple((str(capability), None if server is None else str(server)) for capability, server in self.tools)
        if tuple(capability for capability, _ in tools) != TOOL_CAPABILITIES:
            raise StrictSchemaError(f"setup.tools: expected one entry for each of {TOOL_CAPABILITIES}")
        object.__setattr__(self, "tools", tools)

    @property
    def available_tools(self) -> dict[str, str]:
        return {capability: server for capability, server in self.tools if server is not None}

    @classmethod
    def from_mapping(cls, value: Any) -> "Setup":
        obj = fields(value, {"environment", "policy", "tools"}, path="setup")
        tools = mapping(obj["tools"], "setup.tools")
        return cls(
            environment=EnvironmentSetup.from_mapping(obj["environment"]),
            policy=PolicySetup.from_mapping(obj["policy"]),
            tools=tuple((capability, tools.get(capability)) for capability in TOOL_CAPABILITIES),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "environment": self.environment.to_mapping(),
            "policy": self.policy.to_mapping(),
            "tools": dict(self.tools),
        }

    def validate_observation(self, observation: Observation) -> Observation:
        if not isinstance(observation, Observation):
            raise StrictSchemaError("setup.observation: expected an Observation")
        expected_cameras = {item.name: item for item in self.environment.cameras}
        if set(observation.cameras) != set(expected_cameras):
            raise StrictSchemaError("setup.observation: camera names differ")
        for name, expected in expected_cameras.items():
            actual = observation.cameras[name]
            if actual.frame_id != expected.frame_id or actual.optical_convention != expected.optical_convention:
                raise StrictSchemaError(f"setup.observation.cameras.{name}: frame convention differs")
            if actual.rgb.shape[:2] != (expected.height, expected.width):
                raise StrictSchemaError(f"setup.observation.cameras.{name}: image shape differs")
            if (actual.depth_m is not None) != expected.has_depth:
                raise StrictSchemaError(f"setup.observation.cameras.{name}: depth availability differs")
        expected_state = tuple(item.to_mapping() for item in self.environment.robot_state)
        actual_state = tuple(item.spec.to_mapping() for item in observation.proprioception.vectors)
        if actual_state != expected_state:
            raise StrictSchemaError("setup.observation: robot state specification differs")
        return observation

    def validate_action(self, chunk: ActionChunk) -> ActionChunk:
        if not isinstance(chunk, ActionChunk):
            raise StrictSchemaError("setup.action: expected an ActionChunk")
        if chunk.spec != self.policy.action_spec:
            raise StrictSchemaError("setup.action: action spec differs")
        if chunk.horizon > self.policy.chunk_horizon:
            raise StrictSchemaError("setup.action: more actions than chunk_horizon")
        if chunk.execution_count > self.policy.execution_count:
            raise StrictSchemaError("setup.action: execution_count exceeds the limit")
        return chunk


def load_setups(path: str | Path) -> tuple[dict[str, Setup], str]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    obj = fields(value, {"metric", "tools", "suites"}, path="setup file")
    suites = mapping(obj["suites"], "setup file: suites")
    if not suites:
        raise StrictSchemaError("setup file: suites must not be empty")
    result = {}
    for suite, entry in sorted(suites.items()):
        item = fields(entry, {"environment", "policy"}, path=f"setup file: suites.{suite}")
        setup = Setup.from_mapping({"environment": item["environment"], "policy": item["policy"], "tools": obj["tools"]})
        if setup.environment.suite != suite:
            raise StrictSchemaError(f"setup file: suites.{suite} names a different environment.suite")
        result[suite] = setup
    if len({setup.tools for setup in result.values()}) != 1:
        raise StrictSchemaError("setup file: every suite must offer the same tools")
    return result, string(obj["metric"], "setup file: metric")


def load_split(path: str | Path) -> tuple[dict[str, EpisodeList], dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    obj = fields(value, set(EPISODE_SETS), {"scenes"}, path="split")
    parts = {name: EpisodeList.from_mapping(name, obj[name]) for name in EPISODE_SETS}
    shared = set(parts["training"].episodes) & set(parts["held_out"].episodes)
    if shared:
        raise StrictSchemaError(f"the split puts an episode in both training and held_out: {sorted(shared)[0]}")
    return parts, dict(mapping(obj.get("scenes", {}), "split.scenes"))


@dataclass(frozen=True)
class Experiment:
    setup_file: Path
    split_file: Path
    setups: dict[str, Setup]
    metric: str
    split: dict[str, EpisodeList]
    scenes: dict[str, Any]

    @classmethod
    def load(cls, setup_file: str | Path, split_file: str | Path) -> "Experiment":
        setup_file = resolve(setup_file)
        split_file = resolve(split_file)
        setups, metric = load_setups(setup_file)
        split, scenes = load_split(split_file)
        for name in EPISODE_SETS:
            for key in split[name].episodes:
                if len(setups) > 1 and key.task_id.partition("::")[0] not in setups:
                    raise StrictSchemaError(f"episode task {key.task_id!r} does not start with one of the setup's suites")
        return cls(setup_file, split_file, setups, metric, split, scenes)

    def suite_of(self, task_id: str) -> str:
        if len(self.setups) == 1:
            return next(iter(self.setups))
        return task_id.partition("::")[0]

    def snapshot(self, paths_file: Path, gpu_ids: tuple[int, ...], episodes_per_gpu: int) -> dict[str, Any]:
        return {
            "setup_file": str(self.setup_file),
            "split_file": str(self.split_file),
            "metric": self.metric,
            "setups": {suite: setup.to_mapping() for suite, setup in sorted(self.setups.items())},
            "paths_file": str(paths_file),
            "gpu_ids": list(gpu_ids),
            "episodes_per_gpu": episodes_per_gpu,
        }
