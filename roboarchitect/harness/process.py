from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from roboarchitect.inputs import CODE_ROOT
from roboarchitect.messages import read_frame, write_frame
from roboarchitect.processes import register_process, unregister_process
from roboarchitect.protocol import ActionSpec, Observation, StrictSchemaError

from .api import StepEvent, StepRequest, StepResult
from .memory import HarnessMemory
from .tools import ToolEndpoint, Toolbox


START_TIMEOUT_S = 3600.0
STEP_TIMEOUT_S = 3600.0
CLOSE_TIMEOUT_S = 2.0
SIMULATOR_MODULES = ("libero", "mujoco", "robocasa", "robosuite", "sapien", "simpler_env")


class HarnessProcessError(RuntimeError):
    def __init__(self, message: str, events: tuple[StepEvent, ...] = ()) -> None:
        super().__init__(message)
        self.events = events


def harness_import_path(isolation_dir: Path) -> Path:
    path = isolation_dir / "path"
    link = path / "roboarchitect"
    if not link.is_symlink():
        path.mkdir(parents=True, exist_ok=True)
        link.symlink_to(CODE_ROOT / "roboarchitect", target_is_directory=True)
    return path


def harness_environment(harness_python: Path, isolation_dir: Path, harness_dir: Path) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": str(isolation_dir / "home"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PATH": os.pathsep.join((str(harness_python.parent), "/usr/bin", "/bin")),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join((str(harness_import_path(isolation_dir)), str(harness_dir))),
        "TMPDIR": str(isolation_dir / "tmp"),
        "XDG_CACHE_HOME": str(isolation_dir / "cache"),
    }


def check_harness_python(harness_python: Path, isolation_dir: Path, harness_dir: Path) -> None:
    probe = f"import importlib.util,sys;found=[name for name in {SIMULATOR_MODULES!r} if importlib.util.find_spec(name) is not None];print(','.join(found));sys.exit(3 if found else 0)"
    result = subprocess.run(
        [str(harness_python), "-I", "-c", probe],
        env=harness_environment(harness_python, isolation_dir, harness_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        found = result.stdout.strip() or result.stderr.strip() or "probe failed"
        raise HarnessProcessError(f"the harness Python must not see any simulator package, but it has: {found}")


@dataclass(frozen=True)
class HarnessProcessConfig:
    harness_path: Path
    endpoints: Mapping[str, ToolEndpoint]
    harness_python: Path
    isolation_dir: Path
    action_spec: ActionSpec
    max_horizon: int
    max_execution_count: int
    stderr_path: Path
    memory: HarnessMemory

    def __post_init__(self) -> None:
        path = Path(self.harness_path).resolve()
        if not path.is_file() or path.is_symlink():
            raise StrictSchemaError(f"the harness must be a regular file: {path}")
        harness_python = Path(self.harness_python).resolve()
        if not harness_python.is_file():
            raise StrictSchemaError(f"the harness Python is missing: {harness_python}")
        isolation_dir = Path(self.isolation_dir).resolve()
        if isolation_dir == path.parent or isolation_dir in path.parents:
            raise StrictSchemaError("the harness process folder must not contain the harness file")
        object.__setattr__(self, "harness_path", path)
        object.__setattr__(self, "endpoints", dict(self.endpoints))
        object.__setattr__(self, "harness_python", harness_python)
        object.__setattr__(self, "isolation_dir", isolation_dir)
        object.__setattr__(self, "stderr_path", Path(self.stderr_path).resolve())


class HarnessProcess:
    def __init__(self, config: HarnessProcessConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_file: Any = None
        self._sequence = 0
        self._sessions: set[str] = set()
        self._toolbox: Toolbox | None = None

    def start(self) -> None:
        if self._process is not None:
            raise HarnessProcessError("the harness process is already started")
        check_harness_python(self.config.harness_python, self.config.isolation_dir, self.config.harness_path.parent)
        try:
            self._toolbox = Toolbox(self.config.endpoints, self.config.memory)
        except Exception as exc:
            raise HarnessProcessError(f"a model server did not answer as expected: {exc}") from exc
        self.config.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_file = self.config.stderr_path.open("ab", buffering=0)
        command = [
            str(self.config.harness_python),
            "-m",
            "roboarchitect.harness.worker",
            "--harness",
            str(self.config.harness_path),
        ]
        environment = harness_environment(
            self.config.harness_python, self.config.isolation_dir, self.config.harness_path.parent
        )
        self.config.isolation_dir.mkdir(parents=True, exist_ok=True)
        for name in ("home", "cache", "tmp"):
            (self.config.isolation_dir / name).mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            cwd=self.config.isolation_dir,
            env=environment,
            start_new_session=True,
        )
        register_process(self._process)
        try:
            response = self._rpc(
                "initialize",
                {"capabilities": self._toolbox.relay_declaration()},
                timeout_s=START_TIMEOUT_S,
            )
            if response != {"ready": True}:
                raise HarnessProcessError("the harness process returned an invalid handshake")
        except Exception:
            self.close(force=True)
            raise

    def _rpc(
        self,
        operation: str,
        payload: Mapping[str, Any],
        timeout_s: float = STEP_TIMEOUT_S,
        relay_context: tuple[str, str] | None = None,
    ) -> Any:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise HarnessProcessError("the harness process is not started")
        if process.poll() is not None:
            raise HarnessProcessError(f"the harness process exited with code {process.returncode}")
        self._sequence += 1
        sequence = self._sequence
        deadline = time.monotonic() + timeout_s
        try:
            write_frame(process.stdin.fileno(), {"sequence": sequence, "operation": operation, "payload": dict(payload)})
            while True:
                response = read_frame(process.stdout.fileno(), max(0.0, deadline - time.monotonic()))
                if isinstance(response, Mapping) and set(response) == {
                    "type",
                    "relay_id",
                    "capability",
                    "operation",
                    "payload",
                    "session_id",
                    "request_id",
                }:
                    if response["type"] != "tool_request" or relay_context is None:
                        raise HarnessProcessError("the harness process sent a model call outside a step")
                    session_id, request_id = relay_context
                    if response["session_id"] != session_id or response["request_id"] != request_id:
                        raise HarnessProcessError("a model call from the harness gives a step other than the current one")
                    toolbox = self._toolbox
                    if toolbox is None:
                        raise HarnessProcessError("the model servers are not connected")
                    try:
                        relay_result = toolbox.dispatch_relay(
                            response["capability"],
                            response["operation"],
                            response["payload"],
                            session_id,
                            request_id,
                        )
                        relay_response = {
                            "type": "tool_response",
                            "relay_id": response["relay_id"],
                            "ok": True,
                            "result": relay_result,
                            "error": None,
                        }
                    except Exception as exc:
                        relay_response = {
                            "type": "tool_response",
                            "relay_id": response["relay_id"],
                            "ok": False,
                            "result": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    write_frame(process.stdin.fileno(), relay_response)
                    continue
                break
        except Exception as exc:
            raise HarnessProcessError(f"talking to the harness process failed: {exc}") from exc
        if not isinstance(response, Mapping) or set(response) != {"sequence", "ok", "result", "error"}:
            raise HarnessProcessError("the harness process returned an invalid envelope")
        if response["sequence"] != sequence:
            raise HarnessProcessError("the harness process answered out of order")
        if response["ok"] is not True:
            error = response["error"]
            if isinstance(error, Mapping) and set(error) == {"message", "events"}:
                events = tuple(StepEvent.from_mapping(event) for event in error["events"])
                raise HarnessProcessError(str(error["message"]), events)
            raise HarnessProcessError(str(error))
        return response["result"]

    def act(self, observation: Observation, session_id: str, request_id: str) -> StepResult:
        if not isinstance(observation, Observation):
            raise StrictSchemaError("harness process: expected an Observation")
        if session_id not in self._sessions:
            raise HarnessProcessError("an episode must be reset with a policy seed before act")
        request = StepRequest(
            request_id=request_id,
            session_id=session_id,
            observation=observation,
            action_spec=self.config.action_spec,
        )
        toolbox = self._toolbox
        if toolbox is None:
            raise HarnessProcessError("the model servers are not connected")
        toolbox.begin_step(observation.step_index, request.request_id, session_id)
        try:
            result = StepResult.from_mapping(
                self._rpc("act", request.to_mapping(), relay_context=(session_id, request.request_id))
            )
        finally:
            toolbox.finish_step()
        if result.action.request_id != request.request_id or result.action.session_id != request.session_id:
            raise HarnessProcessError("the harness returned an action for another step")
        if result.action.start_step != observation.step_index:
            raise HarnessProcessError("the harness returned an action whose start_step is not this step")
        if result.action.spec != self.config.action_spec:
            raise HarnessProcessError("the harness returned an action with a different action spec")
        if result.action.horizon > self.config.max_horizon:
            raise HarnessProcessError("the harness returned more actions than this setup allows")
        if result.action.execution_count > self.config.max_execution_count:
            raise HarnessProcessError("the harness asked to execute more actions than this setup allows")
        return result

    def reset(self, session_id: str, policy_reset: Mapping[str, Any]) -> None:
        if type(session_id) is not str or not session_id:
            raise StrictSchemaError("harness process: session_id must be a nonempty string")
        toolbox = self._toolbox
        if toolbox is None:
            raise HarnessProcessError("the model servers are not connected")
        toolbox.reset_policy(session_id, policy_reset)
        try:
            result = self._rpc("reset", {"session_id": session_id})
        except BaseException:
            try:
                toolbox.close_session(session_id)
            finally:
                raise
        if result != {"reset": True}:
            raise HarnessProcessError("the harness process returned an invalid reset response")
        self._sessions.add(session_id)

    def end_session(self, session_id: str) -> None:
        if session_id not in self._sessions:
            return
        try:
            result = self._rpc("end_session", {"session_id": session_id})
            if result != {"ended": True}:
                raise HarnessProcessError("the harness process returned an invalid end_session response")
        finally:
            self._sessions.discard(session_id)
            toolbox = self._toolbox
            if toolbox is not None:
                toolbox.close_session(session_id)

    def close(self, force: bool = False) -> None:
        process = self._process
        if process is None:
            return
        if not force and process.poll() is None:
            try:
                self._rpc("close", {}, timeout_s=CLOSE_TIMEOUT_S)
            except HarnessProcessError:
                force = True
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=30.0)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None
        unregister_process(process)
        self._process = None
        toolbox = self._toolbox
        self._toolbox = None
        if toolbox is not None:
            for session_id in sorted(self._sessions):
                try:
                    toolbox.close_session(session_id)
                except Exception:
                    if not force:
                        raise
        self._sessions.clear()
