from __future__ import annotations

import os
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
from roboarchitect.protocol.rotations import matrix_to_quaternion_xyzw
from roboarchitect.records import EpisodeKey

from .common import egl_device_index, robosuite_camera_3d, use_libero_defaults


_SUITE = re.compile(r"libero_pro_(spatial|object|goal|10)_(lan|object|swap|task)")
_LANGUAGE = re.compile(r"^\s*\(:language\s+(.+?)\s*\)\s*$", re.MULTILINE)


def upstream_suite(value: str) -> str:
    match = _SUITE.fullmatch(value)
    if match is None:
        raise StrictSchemaError(f"unsupported LIBERO-Pro suite {value!r}")
    return f"libero_{match.group(1)}_{match.group(2)}"


def split_task_id(value: str) -> tuple[str, str]:
    suite, separator, task_slug = value.partition("::")
    if not separator or not task_slug:
        raise StrictSchemaError("LIBERO-Pro task id must be <suite>::<task>")
    upstream_suite(suite)
    return suite, task_slug


def parse_bddl_language(path: str | Path) -> str:
    source = Path(path)
    matches = _LANGUAGE.findall(source.read_text(encoding="utf-8"))
    if len(matches) != 1:
        raise StrictSchemaError(f"LIBERO-Pro BDDL must contain one language field: {source}")
    language = " ".join(matches[0].split())
    if not language:
        raise StrictSchemaError("LIBERO-Pro BDDL language is empty")
    return language


class LiberoProSimulator:
    SETTLE_STEPS = 10

    def __init__(self, setup: Setup, episode: EpisodeKey, *, scenes: Any, render_gpu_id: int) -> None:
        suite = setup.environment.suite
        task_suite, task_slug = split_task_id(episode.task_id)
        if task_suite != suite:
            raise StrictSchemaError("LIBERO-Pro episode task and setup suite differ")
        self._use_delta = setup.policy.action_spec.channel_semantics[0] == "delta"
        if setup.policy.chunk_horizon != 1 or setup.policy.execution_count != 1:
            raise StrictSchemaError("LIBERO-Pro worker requires one-action policy responses")
        if type(render_gpu_id) is not int or render_gpu_id < 0:
            raise StrictSchemaError("render_gpu_id must be a nonnegative int")
        self._action_spec = setup.policy.action_spec
        self._setup = setup
        self._episode = episode
        self._task_slug = task_slug
        self._render_gpu_id = render_gpu_id
        self._wants_depth = any(item.has_depth for item in setup.environment.cameras)
        self._env: Any = None
        self._observation: dict[str, Any] | None = None
        self._instruction: str | None = None
        self._step = 0
        self._success = False
        self._closed = False

    def reset(self) -> None:
        if self._closed or self._env is not None:
            raise RuntimeError("LIBERO-Pro worker reset is single-use")
        egl_device = egl_device_index(self._render_gpu_id)
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device)
        use_libero_defaults()
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[upstream_suite(self._setup.environment.suite)]()
        task_names = suite.get_task_names()
        if self._task_slug not in task_names:
            raise StrictSchemaError("episode task is absent from the LIBERO-Pro suite")
        task_index = task_names.index(self._task_slug)
        task = suite.get_task(task_index)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        self._instruction = parse_bddl_language(bddl)
        cameras = {item.name: item for item in self._setup.environment.cameras}
        if set(cameras) != {"main", "wrist"} or {(item.width, item.height) for item in cameras.values()} != {(256, 256)}:
            raise StrictSchemaError("LIBERO-Pro setup cameras must be main and wrist at 256x256")
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=256,
            camera_widths=256,
            camera_depths=self._wants_depth,
            render_gpu_device_id=egl_device,
        )
        self._env.seed(self._episode.environment_seed)
        self._env.reset()
        match = re.fullmatch(r"init_state_(\d{2})", self._episode.scenario_id)
        if match is None:
            raise StrictSchemaError("LIBERO-Pro scenario_id must be init_state_NN")
        state_index = int(match.group(1))
        states = suite.get_task_init_states(task_index)
        if state_index >= len(states):
            raise StrictSchemaError("LIBERO-Pro initial-state index is unavailable")
        self._observation = self._env.set_init_state(states[state_index])
        settle = np.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0), dtype=np.float32)
        for _ in range(self.SETTLE_STEPS):
            self._observation, _, _, _ = self._env.step(settle)
        for robot in self._env.env.robots:
            robot.controller.use_delta = self._use_delta
        self._step = 0
        self._success = False

    def _eef_pose(self, raw: dict[str, Any]) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray(raw["robot0_eef_pos"], dtype=np.float32),
                np.asarray(raw["robot0_eef_quat"], dtype=np.float32),
            )
        )

    def observe(self) -> Observation:
        if self._env is None or self._observation is None or self._instruction is None or self._closed:
            raise RuntimeError("LIBERO-Pro worker is not active")
        raw = self._observation
        camera_specs = {item.name: item for item in self._setup.environment.cameras}
        images = {
            "main": np.ascontiguousarray(raw["agentview_image"], dtype=np.uint8),
            "wrist": np.ascontiguousarray(raw["robot0_eye_in_hand_image"], dtype=np.uint8),
        }
        cameras = {}
        for name, image in images.items():
            spec = camera_specs[name]
            depth_m = depth_valid = intrinsics = camera_to_world = None
            if spec.has_depth:
                depth_m, depth_valid, intrinsics, camera_to_world = robosuite_camera_3d(
                    self._env.sim,
                    spec.frame_id,
                    raw[f"{spec.frame_id}_depth"],
                    height=spec.height,
                    width=spec.width,
                )
            cameras[name] = CameraObservation(
                frame_id=spec.frame_id,
                optical_convention=spec.optical_convention,
                rgb=image,
                depth_m=depth_m,
                depth_valid=depth_valid,
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
            )
        state_values = {
            "eef_pose": np.ascontiguousarray(self._eef_pose(raw), dtype=np.float32),
            "gripper_position": np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32),
        }
        vectors = []
        for spec in self._setup.environment.robot_state:
            if spec.name not in state_values:
                raise StrictSchemaError(f"LIBERO-Pro worker cannot provide robot state {spec.name!r}")
            vectors.append(RobotStateVector(spec, np.ascontiguousarray(state_values[spec.name])))
        return Observation(
            episode_id=self._episode.folder_name(),
            step_index=self._step,
            timestamp_ns=self._step * 50_000_000,
            instruction=self._instruction,
            cameras=cameras,
            proprioception=RobotProprioception(tuple(vectors)),
        )

    def apply(self, action: ActionChunk) -> None:
        if self._env is None or self._closed:
            raise RuntimeError("LIBERO-Pro worker is not active")
        if not isinstance(action, ActionChunk) or action.spec != self._action_spec:
            raise StrictSchemaError("LIBERO-Pro action spec mismatch")
        if action.execution_count != 1 or action.horizon != 1 or action.start_step != self._step:
            raise StrictSchemaError("LIBERO-Pro worker requires one action at the current step")
        native = action.executable_values()[0]
        if native.shape != (7,):
            raise StrictSchemaError("LIBERO-Pro native action must have width 7")
        self._observation, _, _, _ = self._env.step(native)
        self._success = self._success or bool(self._env.check_success())
        self._step += 1

    def succeeded(self) -> bool:
        if self._env is None or self._closed:
            raise RuntimeError("LIBERO-Pro worker is not active")
        return self._success

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._env is not None:
            self._env.close()
            self._env = None
        self._observation = None


class MolmoAct2LiberoProSimulator(LiberoProSimulator):
    SETTLE_STEPS = 50


class XVLALiberoProSimulator(LiberoProSimulator):
    def _eef_pose(self, raw: dict[str, Any]) -> np.ndarray:
        controller = self._env.env.robots[0].controller
        return np.concatenate(
            (
                np.asarray(controller.ee_pos, dtype=np.float32),
                np.asarray(matrix_to_quaternion_xyzw(controller.ee_ori_mat), dtype=np.float32),
            )
        )
