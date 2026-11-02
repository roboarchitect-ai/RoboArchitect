from __future__ import annotations

import io
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from roboarchitect.harness import HarnessMemory, HarnessProcess, HarnessProcessConfig, StepEvent, ToolEndpoint
from roboarchitect.inputs import CODE_ROOT, Paths, Setup
from roboarchitect.messages import read_frame, write_frame
from roboarchitect.model_servers import child_environment
from roboarchitect.processes import register_process, unregister_process
from roboarchitect.protocol import ActionChunk, Observation, StrictSchemaError
from roboarchitect.records import EpisodeKey, validate_benchmark_metrics


VIDEO_FRAME_RATE = 20
VIDEO_CRF = "14"
VIDEO_PIXEL_FORMAT = "yuv444p"
VIDEO_PRESET = "veryfast"
SIMULATOR_START_TIMEOUT_S = 600.0
SIMULATOR_CALL_TIMEOUT_S = 600.0


class SimulatorProcessError(RuntimeError):
    pass


class SimulatorProcess:
    def __init__(
        self,
        setup: Setup,
        episode: EpisodeKey,
        scenes: Any,
        *,
        paths: Paths,
        gpu_id: int,
        runtime_dir: Path,
    ) -> None:
        if type(gpu_id) is not int or gpu_id < 0:
            raise StrictSchemaError("simulator: gpu_id must be a nonnegative integer")
        self.setup = setup
        self.episode = episode
        self.scenes = scenes
        self.paths = paths
        self.gpu_id = gpu_id
        self.runtime_dir = Path(runtime_dir).resolve()
        self.process: subprocess.Popen[bytes] | None = None
        self.judged_at_horizon = False
        self._stderr: Any = None
        self._sequence = 0

    def _rpc(self, operation: str, payload: Mapping[str, Any], timeout_s: float = SIMULATOR_CALL_TIMEOUT_S) -> Any:
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise SimulatorProcessError("the simulator process is not started")
        if process.poll() is not None:
            raise SimulatorProcessError(f"the simulator process exited with code {process.returncode}")
        self._sequence += 1
        sequence = self._sequence
        write_frame(process.stdin.fileno(), {"sequence": sequence, "operation": operation, "payload": dict(payload)})
        response = read_frame(process.stdout.fileno(), timeout_s)
        if not isinstance(response, Mapping) or set(response) != {"sequence", "ok", "result", "error"}:
            raise SimulatorProcessError("the simulator returned an invalid envelope")
        if response["sequence"] != sequence:
            raise SimulatorProcessError("the simulator answered out of order")
        if response["ok"] is not True:
            raise SimulatorProcessError(str(response["error"]))
        return response["result"]

    def start(self) -> None:
        if self.process is not None:
            raise SimulatorProcessError("the simulator process is already started")
        benchmark = self.setup.environment.benchmark
        python = self.paths.python_for(benchmark)
        (self.runtime_dir / "tmp").mkdir(parents=True, exist_ok=False)
        environment = child_environment(
            os.environ,
            {
                "CONDA_PREFIX": str(python.parent.parent),
                "PATH": os.pathsep.join((str(python.parent), "/usr/bin", "/bin")),
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(CODE_ROOT),
                "TMPDIR": str(self.runtime_dir / "tmp"),
            },
        )
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            self._stderr = (self.runtime_dir / "simulator.stderr.log").open("ab", buffering=0)
            self.process = subprocess.Popen(
                [str(python), "-m", "benchmarks.worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                cwd=CODE_ROOT,
                env=environment,
                start_new_session=True,
            )
            register_process(self.process)
            payload = {
                "setup": self.setup.to_mapping(),
                "episode": self.episode.to_mapping(),
                "scenes": self.scenes,
                "render_gpu_id": self.gpu_id,
            }
            result = self._rpc("initialize", payload, SIMULATOR_START_TIMEOUT_S)
            if not isinstance(result, Mapping) or set(result) != {"ready", "judged_at_horizon"} or result["ready"] is not True:
                raise SimulatorProcessError("the simulator returned an invalid handshake")
            self.judged_at_horizon = result["judged_at_horizon"] is True
        except Exception:
            self.close(force=True)
            raise

    def reset(self) -> None:
        if self._rpc("reset", {}) != {"reset": True}:
            raise SimulatorProcessError("the simulator returned an invalid reset response")

    def observe(self) -> Observation:
        return Observation.from_mapping(self._rpc("observe", {}))

    def apply(self, action: ActionChunk) -> None:
        if self._rpc("apply", {"action": action.to_mapping()}) != {"applied": True}:
            raise SimulatorProcessError("the simulator returned an invalid apply response")

    def succeeded(self) -> bool:
        result = self._rpc("succeeded", {})
        if not isinstance(result, Mapping) or set(result) != {"success"} or type(result["success"]) is not bool:
            raise SimulatorProcessError("the simulator returned an invalid success response")
        return result["success"]

    def benchmark_metrics(self) -> dict[str, bool | float] | None:
        result = self._rpc("benchmark_metrics", {})
        if not isinstance(result, Mapping) or set(result) != {"available", "metrics"}:
            raise SimulatorProcessError("the simulator returned an invalid metrics response")
        if not result["available"]:
            return None
        return validate_benchmark_metrics(result["metrics"], "simulator.benchmark_metrics")

    def close(self, force: bool = False) -> None:
        process = self.process
        try:
            if process is not None:
                graceful_close = False
                if not force and process.poll() is None:
                    try:
                        if self._rpc("close", {}, 5.0) != {"closed": True}:
                            raise SimulatorProcessError("the simulator returned an invalid close response")
                        graceful_close = True
                    except Exception:
                        force = True
                if graceful_close:
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        pass
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=5.0)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=5.0)
                if process.stdin is not None:
                    process.stdin.close()
                if process.stdout is not None:
                    process.stdout.close()
                unregister_process(process)
                self.process = None
        finally:
            if self._stderr is not None:
                self._stderr.close()
                self._stderr = None


@dataclass(frozen=True)
class StepRecord:
    observation: Observation
    action: ActionChunk | None
    events: tuple[StepEvent, ...]


@dataclass(frozen=True)
class RolloutResult:
    success: bool
    steps: int
    files: Mapping[str, bytes]


def _round_list(values, places: int = 5) -> list:
    return [round(float(item), places) for item in values]


def _state_of(observation: Observation) -> dict:
    return {
        vector.spec.name: {
            "components": list(vector.spec.component_names),
            "values": _round_list(vector.values),
        }
        for vector in observation.proprioception.vectors
    }


def _depth_of(observation: Observation) -> dict:
    summary = {}
    for name in sorted(observation.cameras):
        camera = observation.cameras[name]
        if camera.depth_m is None or camera.depth_valid is None:
            continue
        depth = np.asarray(camera.depth_m, dtype=np.float64)
        valid = np.asarray(camera.depth_valid, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
        usable = depth[valid]
        summary[name] = {
            "valid_fraction": round(float(valid.mean()), 4),
            "min_m": None if not usable.size else round(float(usable.min()), 4),
            "median_m": None if not usable.size else round(float(np.median(usable)), 4),
            "max_m": None if not usable.size else round(float(usable.max()), 4),
        }
    return summary


def trace_bytes(
    steps: list[StepRecord],
    key: EpisodeKey,
    success: bool,
    termination: str,
    pictures: Mapping[str, list[str]],
    picture_error: str | None,
) -> bytes:
    first = steps[0].observation
    lines = [
        json.dumps(
            {
                "kind": "episode_trace",
                "picture_error": picture_error,
                "episode_id": key.folder_name(),
                "task_id": key.task_id,
                "success": bool(success),
                "termination": termination,
                "n_action_steps": sum(1 for step in steps if step.action is not None),
                "cameras": {
                    name: {
                        "width": int(first.cameras[name].rgb.shape[1]),
                        "height": int(first.cameras[name].rgb.shape[0]),
                        "has_depth": first.cameras[name].depth_m is not None,
                        "pictures": pictures.get(name, []),
                    }
                    for name in sorted(first.cameras)
                },
                "robot_state": [vector.spec.name for vector in first.proprioception.vectors],
            },
            sort_keys=True,
        )
    ]
    for step in steps:
        action = None
        if step.action is not None:
            action = {
                "values": [[round(float(x), 5) for x in row] for row in step.action.values.tolist()],
                "channels": list(step.action.spec.channel_names),
                "execution_count": step.action.execution_count,
            }
        lines.append(
            json.dumps(
                {
                    "step": step.observation.step_index,
                    "instruction": step.observation.instruction,
                    "action": action,
                    "state": _state_of(step.observation),
                    "depth": _depth_of(step.observation),
                    "events": [
                        {
                            "type": event.event_type,
                            "status": event.status,
                            "detail": event.detail,
                            "capability": event.capability,
                            "result": None if event.result is None else dict(event.result),
                        }
                        for event in step.events
                    ],
                },
                sort_keys=True,
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def encode_video(frames, width: int, height: int) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "camera.mp4"
        command = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-framerate", str(VIDEO_FRAME_RATE),
            "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", VIDEO_PRESET, "-threads", "1",
            "-pix_fmt", VIDEO_PIXEL_FORMAT, "-crf", VIDEO_CRF,
            str(target),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for frame in frames:
                process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            process.stdin.close()
            error = process.stderr.read()
            if process.wait(timeout=300) != 0:
                raise RuntimeError(error.decode("utf-8", "replace").strip()[:300] or "ffmpeg failed")
        except BaseException:
            process.kill()
            process.wait(timeout=30)
            raise
        finally:
            process.stderr.close()
        return target.read_bytes()


def _png(rgb) -> bytes:
    from PIL import Image

    output = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG", compress_level=9)
    return output.getvalue()


def camera_files(steps: list[StepRecord]) -> tuple[dict[str, bytes], dict[str, list[str]], str | None]:
    first = steps[0].observation
    if not first.cameras:
        return {}, {}, None
    try:
        files: dict[str, bytes] = {}
        pictures: dict[str, list[str]] = {}
        for name in sorted(first.cameras):
            height, width = first.cameras[name].rgb.shape[:2]
            filename = f"camera-{name}.mp4"
            files[filename] = encode_video((step.observation.cameras[name].rgb for step in steps), width, height)
            pictures[name] = [filename]
        return files, pictures, None
    except Exception as exc:
        files = {}
        for position in sorted({0, len(steps) - 1}):
            observation = steps[position].observation
            camera = sorted(observation.cameras)[0]
            files[f"frame-{observation.step_index:08d}.png"] = _png(observation.cameras[camera].rgb)
        return files, {sorted(first.cameras)[0]: sorted(files)}, f"{type(exc).__name__}: {exc}"


class HarnessProcessPool:
    def __init__(self, pool_root: Path) -> None:
        self.pool_root = Path(pool_root)
        self._local = threading.local()
        self._all: list[HarnessProcess] = []
        self._lock = threading.Lock()

    def isolation_dir(self, key: str) -> Path:
        return self.pool_root / f"t{threading.get_ident()}_{key}"

    def acquire(self, key: str, make_process) -> HarnessProcess:
        pool = getattr(self._local, "pool", None)
        if pool is None:
            pool = {}
            self._local.pool = pool
        process = pool.get(key)
        if process is None:
            process = make_process()
            process.start()
            pool[key] = process
            with self._lock:
                self._all.append(process)
        return process

    def evict(self, key: str, process: HarnessProcess) -> None:
        pool = getattr(self._local, "pool", None)
        if pool is not None and pool.get(key) is process:
            del pool[key]
        with self._lock:
            if process in self._all:
                self._all.remove(process)
        try:
            process.close(force=True)
        except Exception:
            pass

    def close_all(self) -> None:
        with self._lock:
            processes = list(self._all)
            self._all.clear()
        for process in processes:
            try:
                process.close(force=True)
            except Exception:
                pass
        shutil.rmtree(self.pool_root, ignore_errors=True)


class EpisodeRollout:
    def __init__(
        self,
        setup: Setup,
        harness_dir: Path,
        tool_endpoints: Mapping[str, ToolEndpoint],
        policy_endpoints: Mapping[int, ToolEndpoint] | None,
        scenes: Any,
        *,
        paths: Paths,
        runtime_root: Path,
        gpu_assignments: Mapping[str, int],
        harness_pool: HarnessProcessPool,
        memory: HarnessMemory,
        pool_tag: str,
    ) -> None:
        self.setup = setup
        self.harness_path = Path(harness_dir).resolve() / "harness.py"
        self.tool_endpoints = dict(tool_endpoints)
        self.policy_endpoints = None if policy_endpoints is None else dict(policy_endpoints)
        self.scenes = scenes
        self.paths = paths
        self.runtime_root = Path(runtime_root).resolve()
        self.gpu_assignments = dict(gpu_assignments)
        self.harness_pool = harness_pool
        self.memory = memory
        self.pool_tag = pool_tag

    def _harness_process(self, endpoints, isolation_dir: Path) -> HarnessProcess:
        return HarnessProcess(
            HarnessProcessConfig(
                harness_path=self.harness_path,
                endpoints=endpoints,
                harness_python=self.paths.python_for("harness"),
                isolation_dir=isolation_dir,
                action_spec=self.setup.policy.action_spec,
                max_horizon=self.setup.policy.chunk_horizon,
                max_execution_count=self.setup.policy.execution_count,
                stderr_path=isolation_dir / "stderr.log",
                memory=self.memory,
            )
        )

    def __call__(self, key: EpisodeKey) -> RolloutResult:
        session_id = key.folder_name()
        runtime_dir = self.runtime_root / session_id
        shutil.rmtree(runtime_dir, ignore_errors=True)
        runtime_dir.mkdir(parents=True, exist_ok=False)
        steps: list[StepRecord] = []
        success = False
        benchmark_metrics: dict[str, bool | float] | None = None
        gpu_id = self.gpu_assignments[session_id]
        endpoints = dict(self.tool_endpoints)
        pool_key = "no-policy"
        if self.policy_endpoints is not None:
            endpoints["vla"] = self.policy_endpoints[gpu_id]
            pool_key = f"gpu{gpu_id}"
        if self.pool_tag:
            pool_key = f"{pool_key}-{self.pool_tag}"
        simulator = SimulatorProcess(
            self.setup, key, self.scenes, paths=self.paths, gpu_id=gpu_id, runtime_dir=runtime_dir / "simulator"
        )
        isolation_dir = self.harness_pool.isolation_dir(pool_key)
        harness = self.harness_pool.acquire(pool_key, lambda: self._harness_process(endpoints, isolation_dir))
        episode_ok = False
        full_horizon = False
        try:
            simulator.start()
            full_horizon = simulator.judged_at_horizon
            simulator.reset()
            policy_reset = {
                "policy_seed": key.policy_seed,
                "adapter": self.setup.policy.adapter,
                "task": key.task_id.removeprefix(f"{self.setup.environment.suite}::"),
            }
            harness.reset(session_id, policy_reset)
            for _ in range(key.horizon):
                observation = self.setup.validate_observation(simulator.observe())
                if observation.episode_id != session_id:
                    raise StrictSchemaError("the simulator returned an observation of another episode")
                if not full_horizon and simulator.succeeded():
                    success = True
                    steps.append(StepRecord(observation, None, ()))
                    break
                result = harness.act(observation, session_id, f"{session_id}#step{observation.step_index:06d}")
                action = self.setup.validate_action(result.action)
                steps.append(StepRecord(observation, action, result.events))
                simulator.apply(action)
                if not full_horizon and simulator.succeeded():
                    success = True
                    break
            if full_horizon:
                success = simulator.succeeded()
            benchmark_metrics = simulator.benchmark_metrics()
            episode_ok = True
        finally:
            if episode_ok:
                try:
                    harness.end_session(session_id)
                except Exception:
                    self.harness_pool.evict(pool_key, harness)
            else:
                self.harness_pool.evict(pool_key, harness)
            simulator.close(force=not episode_ok)
        if not steps:
            raise RuntimeError("the episode produced no steps")
        if benchmark_metrics is not None and "success" in benchmark_metrics and benchmark_metrics["success"] is not success:
            raise RuntimeError("the benchmark's own success differs from the success check")
        termination = "horizon" if full_horizon else "success" if success else "horizon"
        pictures, picture_index, picture_error = camera_files(steps)
        files = {"trace.jsonl": trace_bytes(steps, key, success, termination, picture_index, picture_error)}
        files.update(pictures)
        if benchmark_metrics is not None:
            files["benchmark_metrics.json"] = (json.dumps(benchmark_metrics, sort_keys=True, indent=2) + "\n").encode()
        shutil.rmtree(runtime_dir, ignore_errors=True)
        return RolloutResult(success=success, steps=sum(item.action is not None for item in steps), files=files)
