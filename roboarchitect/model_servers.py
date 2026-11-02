from __future__ import annotations

import ast
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Mapping

from roboarchitect.harness import ToolEndpoint
from roboarchitect.inputs import CODE_ROOT, TOOL_CAPABILITIES, Paths, PolicySetup, Setup
from roboarchitect.messages import ModelClient, ServerCallError, ServerIdentity
from roboarchitect.processes import register_process, unregister_process
from roboarchitect.protocol import StrictSchemaError

from auxiliary.common import MODELS


STARTUP_TIMEOUT_S = 10800.0
SHUTDOWN_TIMEOUT_S = 20.0
TOOL_CALL_TIMEOUT_S = 300.0
POLICY_CALL_TIMEOUT_S = 600.0
GRASP_GRIPPER = "franka"
POLICY_CAPABILITY = "vla"
SERVED_CAPABILITIES = TOOL_CAPABILITIES + (POLICY_CAPABILITY,)
POLICY_REPOSITORIES = {"xvla": "xvla", "rlinf_pi05": "rlinf"}
PARENT_ONLY_VARIABLES = {
    "CONDA_DEFAULT_ENV",
    "CONDA_EXE",
    "CONDA_PREFIX",
    "CONDA_PROMPT_MODIFIER",
    "CONDA_PYTHON_EXE",
    "CONDA_SHLVL",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONHOME",
    "PYTHONPATH",
    "TRANSFORMERS_CACHE",
    "VIRTUAL_ENV",
    "_CE_CONDA",
    "_CE_M",
}
_METHOD_CAPABILITY = {
    "detect": "detection",
    "vla": "vla",
    "grasp": "grasp",
    "language": "language",
    "point": "pointing",
    "segment": "segmentation",
    "vision": "vision",
}
_TYPE_CAPABILITY = {
    "DetectionRequest": "detection",
    "DetectionResult": "detection",
    "Detection": "detection",
    "GraspRequest": "grasp",
    "GraspResult": "grasp",
    "GraspCandidate": "grasp",
    "LanguageRequest": "language",
    "PointingRequest": "pointing",
    "PointingResult": "pointing",
    "SegmentationRequest": "segmentation",
    "SegmentationResult": "segmentation",
    "VisionRequest": "vision",
    "VLARequest": "vla",
}
_UNKNOWN_CALLS = frozenset({"getattr", "eval", "exec", "globals", "vars", "__import__"})


def child_environment(base: Mapping[str, str], overlay: Mapping[str, str]) -> dict[str, str]:
    environment = {key: value for key, value in base.items() if key not in PARENT_ONLY_VARIABLES}
    environment.update(overlay)
    return environment


def _config_literal_nodes(tree: ast.AST) -> set[int]:
    marked: set[int] = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if not any(isinstance(item, ast.Name) and item.id == "HARNESS_CONFIG" for item in targets):
            continue
        value = getattr(node, "value", None)
        if value is None:
            continue
        for child in ast.walk(value):
            marked.add(id(child))
    return marked


def declared_capabilities(source: str) -> frozenset[str] | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else []
        if not any(isinstance(item, ast.Name) and item.id == "HARNESS_CONFIG" for item in targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return None
        if not isinstance(value, dict):
            return None
        names: set[str] = set()
        for key in ("required_capabilities", "optional_capabilities"):
            entry = value.get(key, ())
            if not isinstance(entry, (list, tuple)):
                return None
            names.update(str(item) for item in entry)
        return frozenset(names)
    return None


def required_capabilities(source: str) -> frozenset[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return frozenset(SERVED_CAPABILITIES)
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else []
        if not any(isinstance(item, ast.Name) and item.id == "HARNESS_CONFIG" for item in targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return frozenset(SERVED_CAPABILITIES)
        entry = value.get("required_capabilities", ()) if isinstance(value, dict) else ()
        if not isinstance(entry, (list, tuple)):
            return frozenset(SERVED_CAPABILITIES)
        return frozenset(str(item) for item in entry) & frozenset(SERVED_CAPABILITIES)
    return frozenset()


def referenced_capabilities(source: str) -> frozenset[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return frozenset(SERVED_CAPABILITIES)
    config_nodes = _config_literal_nodes(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            name = function.id if isinstance(function, ast.Name) else getattr(function, "attr", None)
            if name in _UNKNOWN_CALLS:
                return frozenset(SERVED_CAPABILITIES)
        if isinstance(node, ast.Attribute):
            capability = _METHOD_CAPABILITY.get(node.attr)
            if capability is not None:
                found.add(capability)
            capability = _TYPE_CAPABILITY.get(node.attr)
            if capability is not None:
                found.add(capability)
        if isinstance(node, ast.Name):
            capability = _TYPE_CAPABILITY.get(node.id)
            if capability is not None:
                found.add(capability)
        if isinstance(node, ast.Constant) and type(node.value) is str and id(node) not in config_nodes:
            if node.value in _METHOD_CAPABILITY.values() or node.value in _METHOD_CAPABILITY:
                found.add(_METHOD_CAPABILITY.get(node.value, node.value))
    return frozenset(found) & frozenset(SERVED_CAPABILITIES)


def capabilities_for_source(source: str) -> frozenset[str]:
    declared = declared_capabilities(source)
    wanted = referenced_capabilities(source) | required_capabilities(source)
    if declared is None:
        return frozenset(SERVED_CAPABILITIES)
    return frozenset(wanted & declared & frozenset(SERVED_CAPABILITIES))


def folder_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


class ServerStartupError(RuntimeError):
    pass


class ServerProcess:
    def __init__(
        self,
        command: tuple[str, ...],
        environment: Mapping[str, str],
        identity: ServerIdentity,
        log_dir: Path,
    ) -> None:
        self.command = tuple(command)
        self.environment = child_environment(os.environ, environment)
        self.identity = identity
        self.log_dir = Path(log_dir)
        self.process: subprocess.Popen[bytes] | None = None
        self._logs: list[IO[bytes]] = []
        self._scratch: Path | None = None
        self._deadline: float | None = None

    def launch(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stdout = open(self.log_dir / f"{self.identity.replica_id}.stdout.log", "ab", buffering=0)
        stderr = open(self.log_dir / f"{self.identity.replica_id}.stderr.log", "ab", buffering=0)
        self._logs = [stdout, stderr]
        self._scratch = Path(tempfile.mkdtemp(prefix="roboarchitect-server-"))
        try:
            self.process = subprocess.Popen(
                (*self.command, "--port-file", str(self._scratch / "port")),
                cwd=CODE_ROOT,
                env={**self.environment, "HF_MODULES_CACHE": str(self._scratch / "modules")},
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            register_process(self.process)
            self._deadline = time.monotonic() + STARTUP_TIMEOUT_S
        except BaseException:
            self.stop()
            raise

    def wait_ready(self) -> ModelClient:
        process = self.process
        if process is None or self._deadline is None or self._scratch is None:
            raise RuntimeError("the server has not been launched")
        port_file = self._scratch / "port"
        while True:
            if process.poll() is not None:
                raise ServerStartupError(f"the server exited with code {process.returncode}; see {self.log_dir}")
            if port_file.is_file():
                client = ModelClient(f"http://127.0.0.1:{port_file.read_text(encoding='utf-8')}", self.identity, timeout=2.0)
                try:
                    self.identity.validate_exact(client.identity())
                    return client
                except (ServerCallError, OSError):
                    pass
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                raise ServerStartupError(f"the server did not start in time; see {self.log_dir}")
            time.sleep(min(0.25, remaining))

    def stop(self) -> None:
        process = self.process
        self.process = None
        self._deadline = None
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=SHUTDOWN_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=SHUTDOWN_TIMEOUT_S)
            unregister_process(process)
        for stream in self._logs:
            stream.close()
        self._logs = []
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None


@dataclass(frozen=True)
class ServerPlan:
    capabilities: tuple[str, ...]
    tool_endpoints: Mapping[str, ToolEndpoint]
    policy_endpoints: Mapping[str, Mapping[int, ToolEndpoint]]

    def to_mapping(self) -> dict[str, object]:
        return {
            "capabilities": list(self.capabilities),
            "tools": {capability: endpoint.expected_identity.service_name for capability, endpoint in sorted(self.tool_endpoints.items())},
            "policy_gpus": {policy: sorted(gpus) for policy, gpus in sorted(self.policy_endpoints.items())},
        }


class ModelServers:
    def __init__(self, setups: Mapping[str, Setup], *, paths: Paths, gpu_ids: tuple[int, ...], log_root: Path) -> None:
        if not setups or not gpu_ids:
            raise StrictSchemaError("model servers need at least one setup and one GPU")
        self.paths = paths
        self.gpu_ids = tuple(int(item) for item in gpu_ids)
        self.log_root = Path(log_root)
        first = next(iter(setups.values()))
        self.tools = dict(first.available_tools)
        if "grasp" in self.tools:
            if GRASP_GRIPPER not in first.environment.embodiment.lower():
                raise StrictSchemaError("the grasp model only serves the Franka gripper")
            if not any(camera.has_depth for camera in first.environment.cameras):
                raise StrictSchemaError("the grasp model needs a camera with depth")
        self.policies: dict[str, PolicySetup] = {setup.policy.key: setup.policy for setup in setups.values()}
        self._policy_servers: dict[tuple[str, int], ModelClient] = {}
        self._tool_servers: dict[str, ModelClient] = {}
        self._processes: dict[tuple[str, object], ServerProcess] = {}

    def _environment(self, gpu_id: int, python: Path) -> dict[str, str]:
        return {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(CODE_ROOT),
            "CONDA_PREFIX": str(python.parent.parent),
            "PATH": f"{python.parent}:/usr/bin:/bin",
        }

    def _tool_process(self, capability: str) -> ServerProcess:
        name = self.tools[capability]
        model = MODELS[name]
        gpu_id = self.gpu_ids[1] if model.on_second_gpu and len(self.gpu_ids) > 1 else self.gpu_ids[0]
        python = self.paths.python_for(model.python)
        identity = ServerIdentity(name, model.capability, model.model_id, False, f"gpu{gpu_id}", (gpu_id,))
        command = (
            str(python), "-m", "auxiliary.server",
            "--model", name,
            "--identity", json.dumps(identity.to_mapping()),
        )
        return ServerProcess(command, self._environment(gpu_id, python), identity, self.log_root / name)

    def _policy_process(self, policy: PolicySetup, gpu_id: int) -> ServerProcess:
        python = self.paths.python_for(policy.server)
        identity = ServerIdentity(policy.key, "policy", policy.checkpoint, True, f"gpu{gpu_id}", (gpu_id,))
        command = [
            str(python), "-m", "policies.server",
            "--policy", policy.server,
            "--checkpoint", policy.checkpoint,
            "--identity", json.dumps(identity.to_mapping()),
        ]
        if policy.server in POLICY_REPOSITORIES:
            command += ["--repository", str(self.paths.repository(POLICY_REPOSITORIES[policy.server]))]
        environment = self._environment(gpu_id, python)
        environment.update({"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
        return ServerProcess(tuple(command), environment, identity, self.log_root / folder_name(policy.key))

    def _start(self, processes: Mapping[object, ServerProcess]) -> dict[object, ModelClient]:
        launched: list[ServerProcess] = []
        try:
            for process in processes.values():
                process.launch()
                launched.append(process)
            return {key: process.wait_ready() for key, process in processes.items()}
        except BaseException:
            for process in launched:
                process.stop()
            raise

    def _use_tools(self, wanted: frozenset[str]) -> None:
        for capability in sorted(set(self._tool_servers) - wanted):
            self._tool_servers.pop(capability)
            self._processes.pop(("tool", capability)).stop()
        processes = {capability: self._tool_process(capability) for capability in sorted(wanted - set(self._tool_servers))}
        self._tool_servers.update(self._start(processes))
        self._processes.update({("tool", capability): process for capability, process in processes.items()})

    def _use_policies(self, wanted: bool) -> None:
        if not wanted:
            for key in list(self._policy_servers):
                self._policy_servers.pop(key)
                self._processes.pop(("policy", key)).stop()
        elif not self._policy_servers:
            processes = {
                (key, gpu_id): self._policy_process(policy, gpu_id)
                for key, policy in sorted(self.policies.items())
                for gpu_id in self.gpu_ids
            }
            self._policy_servers.update(self._start(processes))
            self._processes.update({("policy", key): process for key, process in processes.items()})

    def plan(self, harness_source: str) -> ServerPlan:
        capabilities = capabilities_for_source(harness_source) & (frozenset(self.tools) | {POLICY_CAPABILITY})
        tools = frozenset(capabilities - {POLICY_CAPABILITY})
        if POLICY_CAPABILITY in capabilities:
            self._use_tools(tools)
            self._use_policies(True)
        else:
            self._use_policies(False)
            self._use_tools(tools)
        tool_endpoints = {
            capability: ToolEndpoint(client.base_url, client.validate_identity(), False, timeout_s=TOOL_CALL_TIMEOUT_S)
            for capability, client in sorted(self._tool_servers.items())
        }
        policy_endpoints: dict[str, dict[int, ToolEndpoint]] = {}
        for (key, gpu_id), client in sorted(self._policy_servers.items()):
            policy_endpoints.setdefault(key, {})[gpu_id] = ToolEndpoint(
                client.base_url, client.validate_identity(), True, timeout_s=POLICY_CALL_TIMEOUT_S
            )
        return ServerPlan(tuple(sorted(capabilities)), tool_endpoints, policy_endpoints)

    def stop(self) -> None:
        for process in self._processes.values():
            process.stop()
        self._processes.clear()
        self._tool_servers.clear()
        self._policy_servers.clear()

    def __enter__(self) -> "ModelServers":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
