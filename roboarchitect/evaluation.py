from __future__ import annotations

import json
import os
import shutil
import stat
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from roboarchitect.episode import EpisodeRollout, HarnessProcessPool, RolloutResult
from roboarchitect.harness import HarnessMemory
from roboarchitect.inputs import Experiment, Paths, Setup
from roboarchitect.model_servers import ModelServers
from roboarchitect.protocol import StrictSchemaError
from roboarchitect.records import (
    BenchmarkOutcome,
    EpisodeFile,
    EpisodeKey,
    EpisodeManifest,
    Score,
    canonical_json_bytes,
    load_json,
)

from benchmarks.metrics import compute_metric, episode_outcome


MAX_ERRORED_EPISODE_FRACTION = 0.5


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temporary, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _action_spec_tags(setups: Mapping[str, Setup]) -> dict[str, str]:
    specs: list[Any] = []
    for suite in setups:
        value = setups[suite].policy.action_spec.to_mapping()
        if value not in specs:
            specs.append(value)
    if len(specs) < 2:
        return {suite: "" for suite in setups}
    return {suite: f"s{specs.index(setups[suite].policy.action_spec.to_mapping())}" for suite in setups}


@dataclass(frozen=True)
class EvaluationResult:
    episodes: str
    score: Score
    outcomes: tuple[BenchmarkOutcome, ...]
    errored_episodes: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "score": self.score.to_mapping(),
            "outcomes": [item.to_mapping() for item in self.outcomes],
            "errored_episodes": self.errored_episodes,
        }

    @classmethod
    def load(cls, path: Path) -> "EvaluationResult":
        value = load_json(path)
        return cls(
            episodes=value["episodes"],
            score=Score.from_mapping(value["score"]),
            outcomes=tuple(BenchmarkOutcome.from_mapping(item) for item in value["outcomes"]),
            errored_episodes=int(value["errored_episodes"]),
        )


def _keep_harness_copy(output: Path, harness_dir: Path) -> None:
    payload = (harness_dir / "harness.py").read_bytes()
    target = output / "harness.py"
    if not target.exists():
        _atomic_write(target, payload)
        target.chmod(0o444)
        return
    if target.read_bytes() != payload:
        raise StrictSchemaError(f"{output} already holds the evaluation of a different harness.py")
    if stat.S_IMODE(target.stat().st_mode) != 0o444:
        target.chmod(0o444)


def _check_episode_folder(path: Path, key: EpisodeKey) -> EpisodeManifest:
    manifest = EpisodeManifest.from_mapping(load_json(path / "episode.json"))
    if manifest.key != key or manifest.state not in {"complete", "error"}:
        raise StrictSchemaError(f"episode folder {path} does not match its episode")
    expected = {"episode.json", *(item.name for item in manifest.files)}
    if {item.name for item in path.iterdir()} != expected:
        raise StrictSchemaError(f"episode folder {path} holds unexpected files")
    return manifest


class Evaluator:
    def __init__(
        self,
        experiment: Experiment,
        episodes: str,
        *,
        servers: ModelServers,
        paths: Paths,
        episodes_per_gpu: int,
    ) -> None:
        if episodes_per_gpu < 1:
            raise StrictSchemaError("episodes_per_gpu must be at least 1")
        self.setups = dict(experiment.setups)
        self.episodes = experiment.split[episodes]
        self.metric = experiment.metric
        self.scenes = experiment.scenes
        self.servers = servers
        self.paths = paths
        self.parallel_episodes = len(servers.gpu_ids) * int(episodes_per_gpu)
        self.suite_of = {key.folder_name(): experiment.suite_of(key.task_id) for key in self.episodes.episodes}
        self.spec_tags = _action_spec_tags(self.setups)

    def _load_manifests(self, output: Path) -> tuple[EpisodeManifest, ...]:
        by_name = {key.folder_name(): key for key in self.episodes.episodes}
        folders = {path.name: path for path in (output / "episodes").iterdir()}
        unknown = set(folders) - set(by_name)
        if unknown:
            raise StrictSchemaError(f"{output / 'episodes'} holds an episode outside this list: {sorted(unknown)[0]}")
        return tuple(_check_episode_folder(folders[name], key) for name, key in by_name.items() if name in folders)

    def _record(self, output: Path, staging: Path, key: EpisodeKey, result: RolloutResult, started_ns: int) -> None:
        manifest = EpisodeManifest(
            key=key,
            state="complete",
            success=result.success,
            steps=result.steps,
            started_ns=started_ns,
            finished_ns=time.time_ns(),
            files=tuple(EpisodeFile(name, len(payload)) for name, payload in sorted(result.files.items())),
            error=None,
        )
        folder = staging / key.folder_name()
        folder.mkdir(parents=True, exist_ok=False)
        for name, payload in sorted(result.files.items()):
            (folder / name).write_bytes(payload)
        (folder / "episode.json").write_bytes(canonical_json_bytes(manifest.to_mapping()))
        os.rename(folder, output / "episodes" / key.folder_name())

    def _record_error(self, output: Path, staging: Path, key: EpisodeKey, exc: Exception, started_ns: int) -> None:
        manifest = EpisodeManifest(
            key=key,
            state="error",
            success=None,
            steps=0,
            started_ns=started_ns,
            finished_ns=time.time_ns(),
            files=(),
            error=f"{type(exc).__name__}: {exc}",
        )
        folder = staging / key.folder_name()
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "episode.json").write_bytes(canonical_json_bytes(manifest.to_mapping()))
        os.rename(folder, output / "episodes" / key.folder_name())

    def _rollouts(self, harness_dir: Path, scratch: Path, memory: HarnessMemory, output: Path):
        plan = self.servers.plan((harness_dir / "harness.py").read_text(encoding="utf-8"))
        _atomic_write(output / "server_plan.json", canonical_json_bytes(plan.to_mapping()))
        gpu_ids = self.servers.gpu_ids
        gpu_assignments = {key.folder_name(): gpu_ids[index % len(gpu_ids)] for index, key in enumerate(self.episodes.episodes)}
        harness_pool = HarnessProcessPool(scratch / "harness_pool")
        rollouts = {
            suite: EpisodeRollout(
                setup,
                harness_dir,
                plan.tool_endpoints,
                plan.policy_endpoints.get(setup.policy.key),
                self.scenes.get(suite),
                paths=self.paths,
                runtime_root=scratch / "runtime",
                gpu_assignments=gpu_assignments,
                harness_pool=harness_pool,
                memory=memory,
                pool_tag=self.spec_tags[suite],
            )
            for suite, setup in self.setups.items()
        }
        return rollouts, harness_pool

    def evaluate(self, harness_dir: Path, output: Path) -> EvaluationResult:
        harness_dir = Path(harness_dir).resolve()
        output = Path(output).resolve()
        if (output / "result.json").is_file():
            return EvaluationResult.load(output / "result.json")
        (output / "episodes").mkdir(parents=True, exist_ok=True)
        _keep_harness_copy(output, harness_dir)
        scratch = output / ".scratch"
        shutil.rmtree(scratch, ignore_errors=True)
        staging = scratch / "episode_staging"
        memory = HarnessMemory(output / "memory.json")
        existing = self._load_manifests(output)
        for manifest in existing:
            if manifest.state == "error":
                shutil.rmtree(output / "episodes" / manifest.key.folder_name(), ignore_errors=True)
        finished = {item.key for item in existing if item.state == "complete"}
        pending = [key for key in self.episodes.episodes if key not in finished]
        rollouts: dict[str, EpisodeRollout] = {}
        harness_pool = None
        if pending:
            rollouts, harness_pool = self._rollouts(harness_dir, scratch, memory, output)

        def play(key: EpisodeKey) -> int:
            started = time.time_ns()
            try:
                result = rollouts[self.suite_of[key.folder_name()]](key)
                self._record(output, staging, key, result, started)
                return 0
            except Exception as exc:
                try:
                    self._record_error(output, staging, key, exc, started)
                except Exception:
                    return 2
                return 1

        unrecordable = 0
        try:
            if pending:
                with ThreadPoolExecutor(max_workers=self.parallel_episodes) as executor:
                    for future in as_completed([executor.submit(play, key) for key in pending]):
                        unrecordable += int(future.result() == 2)
        finally:
            if harness_pool is not None:
                harness_pool.close_all()
            try:
                memory.flush()
            except Exception:
                pass
            shutil.rmtree(scratch, ignore_errors=True)
        if unrecordable:
            raise RuntimeError(f"{unrecordable} episodes failed and their failure could not be recorded")
        manifests = self._load_manifests(output)
        if len(manifests) != len(self.episodes.episodes):
            raise RuntimeError("the evaluation is incomplete")
        errored_episodes = sum(1 for item in manifests if item.state == "error")
        if errored_episodes > MAX_ERRORED_EPISODE_FRACTION * len(manifests):
            raise RuntimeError(f"{errored_episodes} of {len(manifests)} episodes stopped with an error; the setup or the harness is broken, read evaluation/episodes/*/episode.json")
        outcomes = tuple(
            episode_outcome(output / "episodes" / item.key.folder_name(), item, self.metric) for item in manifests
        )
        result = EvaluationResult(
            episodes=self.episodes.name,
            score=compute_metric(self.metric, outcomes),
            outcomes=tuple(sorted(outcomes, key=lambda item: item.key)),
            errored_episodes=errored_episodes,
        )
        _atomic_write(output / "result.json", canonical_json_bytes(result.to_mapping()))
        return result
