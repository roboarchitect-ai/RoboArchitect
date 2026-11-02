from __future__ import annotations

import importlib
import os
import random
import re
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

from .common import egl_device_index, robosuite_camera_3d


TARGET_TASK_GROUPS = {
    "atomic_seen": (
        "CloseBlenderLid",
        "CloseFridge",
        "CloseToasterOvenDoor",
        "CoffeeSetupMug",
        "NavigateKitchen",
        "OpenCabinet",
        "OpenDrawer",
        "OpenStandMixerHead",
        "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove",
        "PickPlaceDrawerToCounter",
        "PickPlaceSinkToCounter",
        "PickPlaceToasterToCounter",
        "SlideDishwasherRack",
        "TurnOffStove",
        "TurnOnElectricKettle",
        "TurnOnMicrowave",
        "TurnOnSinkFaucet",
    ),
    "composite_seen": (
        "DeliverStraw",
        "GetToastedBread",
        "KettleBoiling",
        "LoadDishwasher",
        "PackIdenticalLunches",
        "PreSoakPan",
        "PrepareCoffee",
        "RinseSinkBasin",
        "ScrubCuttingBoard",
        "SearingMeat",
        "SetUpCuttingStation",
        "StackBowlsCabinet",
        "SteamInMicrowave",
        "StirVegetables",
        "StoreLeftoversInBowl",
        "WashLettuce",
    ),
    "composite_unseen": (
        "ArrangeBreadBasket",
        "ArrangeTea",
        "BreadSelection",
        "CategorizeCondiments",
        "CuttingToolSelection",
        "GarnishPancake",
        "GatherTableware",
        "HeatKebabSandwich",
        "MakeIceLemonade",
        "PanTransfer",
        "PortionHotDogs",
        "RecycleBottlesByType",
        "SeparateFreezerRack",
        "WaffleReheat",
        "WashFruitColander",
        "WeighIngredients",
    ),
}
BASE_SEED = 42
SEED_STRIDE = 100_000


def episode_seeds(episode: EpisodeKey) -> tuple[int, int]:
    match = re.fullmatch(r"env_(\d)_episode_(\d{2})", episode.scenario_id)
    if match is None:
        raise StrictSchemaError("RoboCasa365 scenario must be env_N_episode_NN")
    constructor_seed = BASE_SEED + int(match.group(1))
    reset_seed = constructor_seed * SEED_STRIDE + int(match.group(2)) - 1
    if episode.environment_seed != reset_seed:
        raise StrictSchemaError("RoboCasa365 environment seed differs from its scenario")
    return constructor_seed, reset_seed


def native_action_dict(value: Any) -> dict[str, np.ndarray]:
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (12,) or not np.isfinite(action).all():
        raise StrictSchemaError("RoboCasa native action must have shape [12]")
    action = np.clip(action, -1.0, 1.0)
    return {
        "action.end_effector_position": np.ascontiguousarray(action[0:3]),
        "action.end_effector_rotation": np.ascontiguousarray(action[3:6]),
        "action.gripper_close": np.ascontiguousarray(action[6:7]),
        "action.base_motion": np.ascontiguousarray(action[7:11]),
        "action.control_mode": np.ascontiguousarray(action[11:12]),
    }


class RoboCasa365Simulator:
    def __init__(self, setup: Setup, episode: EpisodeKey, *, scenes: Any, render_gpu_id: int) -> None:
        if len(setup.policy.action_spec.channel_names) != 12:
            raise StrictSchemaError("RoboCasa365 takes a 12-number action")
        if setup.policy.chunk_horizon != 1 or setup.policy.execution_count != 1:
            raise StrictSchemaError("RoboCasa365 worker requires one-action policy responses")
        if type(render_gpu_id) is not int or render_gpu_id < 0:
            raise StrictSchemaError("render_gpu_id must be a nonnegative int")
        self._constructor_seed, self._reset_seed = episode_seeds(episode)
        self._setup = setup
        self._episode = episode
        self._render_gpu_id = render_gpu_id
        self._env: Any = None
        self._robosuite_env: Any = None
        self._observation: dict[str, Any] | None = None
        self._step = 0
        self._success = False
        self._closed = False

    def reset(self) -> None:
        if self._closed or self._env is not None:
            raise RuntimeError("RoboCasa365 worker reset is single-use")
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device_index(self._render_gpu_id))
        importlib.import_module("robocasa")
        import gymnasium as gym

        random.seed(self._constructor_seed)
        np.random.seed(self._constructor_seed)
        self._env = gym.make(
            f"robocasa/{self._episode.task_id}",
            split="target",
            seed=self._constructor_seed,
        )
        random.seed(self._reset_seed)
        np.random.seed(self._reset_seed)
        unwrapped = getattr(self._env, "unwrapped", None)
        robosuite_env = getattr(unwrapped, "env", unwrapped)
        if robosuite_env is not None:
            if hasattr(robosuite_env, "seed"):
                robosuite_env.seed = self._reset_seed
            if hasattr(robosuite_env, "rng"):
                robosuite_env.rng = np.random.default_rng(self._reset_seed)
        self._observation, _ = self._env.reset(seed=self._reset_seed)
        self._step = 0
        self._success = False

    def _simulator(self) -> Any:
        if self._robosuite_env is not None:
            return self._robosuite_env
        unwrapped = getattr(self._env, "unwrapped", self._env)
        found = getattr(unwrapped, "env", unwrapped)
        if found is None or not hasattr(found, "sim"):
            raise RuntimeError("RoboCasa365 worker cannot reach the robosuite simulator for depth")
        self._robosuite_env = found
        return found

    def _camera_3d(self, spec: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        simulator = self._simulator().sim
        rendered = simulator.render(width=spec.width, height=spec.height, camera_name=spec.frame_id, depth=True)
        if not isinstance(rendered, tuple) or len(rendered) != 2:
            raise RuntimeError("RoboCasa365 depth render returned an unexpected result")
        raw_depth = np.asarray(rendered[1], dtype=np.float64)
        if raw_depth.shape[:2] != (spec.height, spec.width):
            raise RuntimeError("RoboCasa365 depth render shape differs from the setup camera")
        return robosuite_camera_3d(simulator, spec.frame_id, raw_depth[::-1], height=spec.height, width=spec.width)

    def observe(self) -> Observation:
        if self._env is None or self._observation is None or self._closed:
            raise RuntimeError("RoboCasa365 worker is not active")
        raw = self._observation
        camera_specs = {item.name: item for item in self._setup.environment.cameras}
        cameras = {}
        for name in ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"):
            if name not in camera_specs:
                raise StrictSchemaError(f"RoboCasa365 setup camera {name!r} is absent")
            spec = camera_specs[name]
            image = np.ascontiguousarray(raw[f"video.{name}"])
            depth_m = depth_valid = intrinsics = camera_to_world = None
            if spec.has_depth:
                depth_m, depth_valid, intrinsics, camera_to_world = self._camera_3d(spec)
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
            "base_position": np.asarray(raw["state.base_position"], dtype=np.float32),
            "base_rotation": np.asarray(raw["state.base_rotation"], dtype=np.float32),
            "end_effector_position_relative": np.asarray(raw["state.end_effector_position_relative"], dtype=np.float32),
            "end_effector_rotation_relative": np.asarray(raw["state.end_effector_rotation_relative"], dtype=np.float32),
            "gripper_qpos": np.asarray(raw["state.gripper_qpos"], dtype=np.float32),
        }
        vectors = tuple(
            RobotStateVector(spec, np.ascontiguousarray(state_values[spec.name]))
            for spec in self._setup.environment.robot_state
        )
        return Observation(
            episode_id=self._episode.folder_name(),
            step_index=self._step,
            timestamp_ns=self._step * 50_000_000,
            instruction=str(raw["annotation.human.task_description"]),
            cameras=cameras,
            proprioception=RobotProprioception(vectors),
        )

    def apply(self, action: ActionChunk) -> None:
        if self._env is None or self._closed:
            raise RuntimeError("RoboCasa365 worker is not active")
        if not isinstance(action, ActionChunk) or action.spec != self._setup.policy.action_spec:
            raise StrictSchemaError("RoboCasa365 action spec mismatch")
        if action.execution_count != 1 or action.horizon != 1 or action.start_step != self._step:
            raise StrictSchemaError("RoboCasa365 worker requires one action at the current step")
        self._observation, _, _, _, info = self._env.step(native_action_dict(action.executable_values()[0]))
        self._success = self._success or bool(info["success"])
        self._step += 1

    def succeeded(self) -> bool:
        if self._env is None or self._closed:
            raise RuntimeError("RoboCasa365 worker is not active")
        return self._success

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._env is not None:
            self._env.close()
            self._env = None
        self._observation = None
