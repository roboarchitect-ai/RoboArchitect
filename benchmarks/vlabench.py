from __future__ import annotations

import importlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np

from roboarchitect.inputs import Setup
from roboarchitect.protocol import (
    CameraObservation,
    ActionChunk,
    Observation,
    RobotProprioception,
    RobotStateVector,
    StrictSchemaError,
)
from roboarchitect.records import EpisodeKey

from .common import dm_control_camera_3d, egl_device_index


BASE_ENV_STEP_S = 0.1
MAX_SUBSTEPS = 10
TOLERANCE = 1e-2
TRACK_FILES = {
    "track_1": "track_1_in_distribution.json",
    "track_2": "track_2_cross_category.json",
    "track_3": "track_3_common_sense.json",
    "track_4": "track_4_semantic_instruction.json",
}


def parse_vlabench_scenario(value: str) -> tuple[str, int]:
    match = re.fullmatch(r"(track_[1-4])_config_(\d{3})", value)
    if match is None:
        raise StrictSchemaError("VLABench scenario_id must be track_N_config_NNN")
    return match.group(1), int(match.group(2))


class VLABenchSimulator:
    def __init__(self, setup: Setup, episode: EpisodeKey, *, scenes: Any, render_gpu_id: int) -> None:
        if len(setup.policy.action_spec.channel_names) != 8:
            raise StrictSchemaError("VLABench takes an 8-number action")
        if setup.policy.chunk_horizon != 1 or setup.policy.execution_count != 1:
            raise StrictSchemaError("VLABench worker requires one target per policy response")
        if type(render_gpu_id) is not int or render_gpu_id < 0:
            raise StrictSchemaError("render_gpu_id must be a nonnegative int")
        self._task_id = episode.task_id.removeprefix(f"{setup.environment.suite}::")
        self._episode = episode
        self._setup = setup
        self._scenes = scenes
        self._render_gpu_id = render_gpu_id
        self._env: Any = None
        self._instruction = ""
        self._step = 0
        self._base_steps = 0
        self._success = False
        self._closed = False

    def _episode_config(self, package: Path) -> dict[str, Any]:
        track, index = parse_vlabench_scenario(self._episode.scenario_id)
        table = json.loads((package / "configs" / "evaluation" / "tracks" / TRACK_FILES[track]).read_text(encoding="utf-8"))
        if self._task_id not in table or index >= len(table[self._task_id]):
            raise StrictSchemaError("VLABench task is absent from its track")
        return table[self._task_id][index]

    def reset(self) -> None:
        if self._closed or self._env is not None:
            raise RuntimeError("VLABench worker reset is single-use")
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device_index(self._render_gpu_id))
        importlib.import_module("VLABench.tasks")
        importlib.import_module("VLABench.robots")
        envs = importlib.import_module("VLABench.envs")
        package = Path(os.environ["VLABENCH_ROOT"]).resolve()
        np.random.seed(self._episode.environment_seed)
        random.seed(self._episode.environment_seed)
        config = self._episode_config(package)
        self._env = envs.load_env(
            self._task_id,
            episode_config=config,
            random_init=False,
            eval=False,
            run_mode="eval",
        )
        self._env.reset()
        self._instruction = self._env.task.get_instruction()
        if not isinstance(self._instruction, str) or not self._instruction.strip():
            raise RuntimeError("VLABench episode instruction is absent")
        self._step = 0
        self._base_steps = 0
        self._success = False

    def observe(self) -> Observation:
        if self._env is None or self._closed:
            raise RuntimeError("VLABench worker is not active")
        raw = self._env.get_observation(require_pcd=False)
        images = np.asarray(raw["rgb"], dtype=np.uint8)
        if images.ndim != 4 or images.shape[0] < 4 or images.shape[1:] != (480, 480, 3):
            raise RuntimeError(f"VLABench multiview RGB has invalid shape {images.shape}")
        view_index = {"main": 0, "front": 2, "wrist": images.shape[0] - 1}
        camera_specs = {item.name: item for item in self._setup.environment.cameras}
        if set(camera_specs) != set(view_index):
            raise StrictSchemaError("VLABench setup cameras must be main, front and wrist")
        cameras = {}
        for name, index in view_index.items():
            spec = camera_specs[name]
            depth_m = depth_valid = intrinsics = camera_to_world = None
            if spec.has_depth:
                depth_m, depth_valid, intrinsics, camera_to_world = dm_control_camera_3d(
                    np.asarray(raw["depth"])[index],
                    np.asarray(raw["instrinsic"])[index],
                    np.asarray(raw["extrinsic"])[index],
                    height=spec.height,
                    width=spec.width,
                )
            cameras[name] = CameraObservation(
                frame_id=spec.frame_id,
                optical_convention=spec.optical_convention,
                rgb=np.ascontiguousarray(images[index]),
                depth_m=depth_m,
                depth_valid=depth_valid,
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
            )
        ee_state = np.asarray(raw["ee_state"], dtype=np.float32)
        if ee_state.shape != (8,) or not np.isfinite(ee_state).all():
            raise RuntimeError(f"VLABench end-effector state has invalid shape {ee_state.shape}")
        state_values = {"eef_pose": ee_state[:7], "gripper_position": ee_state[7:]}
        vectors = tuple(
            RobotStateVector(spec, np.ascontiguousarray(state_values[spec.name]))
            for spec in self._setup.environment.robot_state
        )
        return Observation(
            episode_id=self._episode.folder_name(),
            step_index=self._step,
            timestamp_ns=round(self._base_steps * BASE_ENV_STEP_S * 1_000_000_000),
            instruction=self._instruction,
            cameras=cameras,
            proprioception=RobotProprioception(vectors),
        )

    def apply(self, action: ActionChunk) -> None:
        if self._env is None or self._closed:
            raise RuntimeError("VLABench worker is not active")
        if not isinstance(action, ActionChunk) or action.spec != self._setup.policy.action_spec:
            raise StrictSchemaError("VLABench action spec mismatch")
        if action.execution_count != 1 or action.horizon != 1 or action.start_step != self._step:
            raise StrictSchemaError("VLABench worker requires one target at the current step")
        native = np.asarray(action.executable_values()[0], dtype=np.float64)
        if native.shape != (8,) or not np.isfinite(native).all():
            raise StrictSchemaError("VLABench native action must be finite width 8")
        if not np.isclose(native[6], native[7]) or not any(np.isclose(native[6], value) for value in (0.0, 0.04)):
            raise StrictSchemaError("VLABench finger targets must be equal and either 0 or 0.04 meter")
        utils = importlib.import_module("VLABench.utils.utils")
        quaternion = utils.euler_to_quaternion(*native[3:6])
        _, qpos = self._env.robot.get_qpos_from_ee_pos(
            physics=self._env.physics,
            pos=native[:3],
            quat=quaternion,
        )
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (7,) or not np.isfinite(qpos).all():
            raise RuntimeError("VLABench inverse kinematics returned an invalid target")
        command = np.concatenate((qpos, native[6:8]))
        for _ in range(MAX_SUBSTEPS):
            timestep = self._env.step(command)
            self._base_steps += 1
            if timestep.last():
                self._success = True
                break
            current = np.asarray(self._env.task.robot.get_qpos(self._env.physics)).reshape(-1)
            difference = current - command[:7]
            if np.max(difference) < TOLERANCE and np.min(difference) > -TOLERANCE:
                break
        self._step += 1

    def succeeded(self) -> bool:
        if self._env is None or self._closed:
            raise RuntimeError("VLABench worker is not active")
        return self._success

    def benchmark_metrics(self) -> dict[str, float | bool]:
        if self._env is None or self._closed:
            raise RuntimeError("VLABench worker is not active")
        return {
            "success": self._success,
            "intention_score": float(self._env.get_intention_score(threshold=0.1)),
            "progress_score": float(self._env.get_task_progress()),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        environment = self._env
        self._env = None
        if environment is not None:
            environment.close()


class VLABenchPoolSimulator(VLABenchSimulator):
    def _episode_config(self, package: Path) -> dict[str, Any]:
        try:
            return self._scenes[self._task_id][self._episode.scenario_id]
        except (KeyError, TypeError) as exc:
            raise StrictSchemaError(f"the split has no VLABench scene {self._task_id}/{self._episode.scenario_id}") from exc
