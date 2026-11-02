from __future__ import annotations

import math
import os
import re
from itertools import product
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
from roboarchitect.protocol.rotations import quaternion_xyzw_to_matrix
from roboarchitect.records import EpisodeKey

from .common import inverse_pose, maniskill_camera_3d


GOOGLE_CONTROL_MODE = "arm_pd_ee_base_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
WIDOWX_CONTROL_MODE = "arm_pd_ee_target_base_pose_gripper_pd_joint_pos"
RENDERER_DEVICE = "cuda:0"
POOL_PUT_ON_ENV = "PoolPutOnBridge-v0"
POOL_MOVE_NEAR_ENV = "MoveNearPoolObjectsInScene-v0"


def simpler_dir() -> Path:
    import simpler_env

    return Path(simpler_env.__file__).resolve().parents[1]


def register_pool_envs() -> None:
    from mani_skill2_real2sim.envs.custom_scenes.move_near_in_scene import MoveNearGoogleInSceneEnv
    from mani_skill2_real2sim.envs.custom_scenes.put_on_in_scene import PutOnBridgeInSceneEnv
    from mani_skill2_real2sim.utils.registration import REGISTERED_ENVS, register_env

    if POOL_PUT_ON_ENV not in REGISTERED_ENVS:

        class PoolPutOnBridge(PutOnBridgeInSceneEnv):
            def __init__(self, source, target, instruction, partial_target, xy_configs, quat_configs, **kwargs: Any) -> None:
                self._pool_instruction = instruction
                self._pool_partial_target = partial_target
                super().__init__(
                    source_obj_name=source,
                    target_obj_name=target,
                    xy_configs=[np.array(item) for item in xy_configs],
                    quat_configs=[np.array(item) for item in quat_configs],
                    **kwargs,
                )

            def evaluate(self, success_require_src_completely_on_target: bool = True, **kwargs: Any):
                return super().evaluate(not self._pool_partial_target, **kwargs)

            def get_language_instruction(self, **kwargs: Any) -> str:
                return self._pool_instruction

        register_env(POOL_PUT_ON_ENV, max_episode_steps=1200)(PoolPutOnBridge)

    if POOL_MOVE_NEAR_ENV not in REGISTERED_ENVS:

        class PoolMoveNear(MoveNearGoogleInSceneEnv):
            def __init__(self, triplets, object_quaternions, **kwargs: Any) -> None:
                self._pool_triplets = [tuple(item) for item in triplets]
                self._pool_quaternions = dict(object_quaternions)
                super().__init__(**kwargs)

            def _setup_obj_configs(self) -> None:
                super()._setup_obj_configs()
                self.triplets = list(self._pool_triplets)
                self.obj_init_quat_dict.update(self._pool_quaternions)

        register_env(POOL_MOVE_NEAR_ENV, max_episode_steps=80)(PoolMoveNear)


def _range(value: Any) -> list[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if not isinstance(value, list) or len(value) != 3:
        raise StrictSchemaError("Google range must be a number or [start,end,count]")
    count = int(value[2])
    if count < 1 or float(count) != float(value[2]):
        raise StrictSchemaError("Google range count must be a positive integer")
    return np.linspace(float(value[0]), float(value[1]), count).tolist()


def _euler_quaternion_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array(
        (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy),
        dtype=np.float64,
    )


def _quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def google_scenario_grid(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    rpy = config["robot_init_rot_rpy_range"]
    if not isinstance(rpy, list) or len(rpy) != 9:
        raise StrictSchemaError("Google rotation range must have nine values")
    center = np.asarray(config["robot_init_rot_quat_center"], dtype=np.float64)
    if center.shape != (4,):
        raise StrictSchemaError("Google center quaternion must have four values")
    quaternions = tuple(
        _quaternion_multiply_wxyz(_euler_quaternion_wxyz(roll, pitch, yaw), center)
        for roll, pitch, yaw in product(_range(rpy[:3]), _range(rpy[3:6]), _range(rpy[6:9]))
    )
    if config["obj_variation_mode"] == "episode":
        objects = tuple({"episode_id": index} for index in range(int(config["episode_nums"])))
    elif config["obj_variation_mode"] == "xy":
        objects = tuple(
            {"init_xy": np.array((x, y), dtype=np.float64)}
            for x, y in product(_range(config["obj_init_x_range"]), _range(config["obj_init_y_range"]))
        )
    else:
        raise StrictSchemaError("Google object variation mode is unsupported")
    return tuple(
        {
            "robot_init_options": {
                "init_xy": np.array((x, y), dtype=np.float64),
                "init_rot_quat": quaternion.copy(),
            },
            "obj_init_options": {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in obj.items()},
        }
        for x, y, quaternion, obj in product(
            _range(config["robot_init_x"]),
            _range(config["robot_init_y"]),
            quaternions,
            objects,
        )
    )


def _pose_vector(pose: Any) -> np.ndarray:
    value = np.concatenate((np.asarray(pose.p, dtype=np.float64), np.asarray(pose.q, dtype=np.float64)))
    if value.shape != (7,) or not np.isfinite(value).all():
        raise StrictSchemaError("SimplerEnv robot-link pose must be finite with width 7")
    norm = np.linalg.norm(value[3:])
    if norm < 1e-8:
        raise StrictSchemaError("SimplerEnv robot-link quaternion is invalid")
    value[3:] /= norm
    return value


def _assert_pose_copy(name: str, value: Any, actual: np.ndarray) -> None:
    observed = np.asarray(value, dtype=np.float64).copy()
    if observed.shape != (7,) or not np.isfinite(observed).all():
        raise StrictSchemaError(f"SimplerEnv {name} pose copy must be finite with width 7")
    norm = np.linalg.norm(observed[3:])
    if norm < 1e-8:
        raise StrictSchemaError(f"SimplerEnv {name} pose copy has an invalid quaternion")
    observed[3:] /= norm
    position_matches = np.allclose(observed[:3], actual[:3], rtol=0.0, atol=1e-5)
    rotation_matches = abs(float(np.dot(observed[3:], actual[3:]))) >= 1.0 - 1e-5
    if not position_matches or not rotation_matches:
        raise StrictSchemaError(f"SimplerEnv {name} pose copy differs from the actual robot link")


def _base_relative_pose(raw: dict[str, Any], environment: Any) -> np.ndarray:
    if not isinstance(raw, dict) or not isinstance(raw.get("agent"), dict) or not isinstance(raw.get("extra"), dict):
        raise StrictSchemaError("SimplerEnv observation must contain agent and extra mappings")
    unwrapped = environment.unwrapped
    base_pose = unwrapped.agent.robot.pose
    tcp_pose = unwrapped.tcp.pose
    base = _pose_vector(base_pose)
    tcp = _pose_vector(tcp_pose)
    if "base_pose" not in raw["agent"]:
        raise StrictSchemaError("SimplerEnv observation is missing the robot base pose")
    _assert_pose_copy("agent.base_pose", raw["agent"]["base_pose"], base)
    if "tcp_pose" in raw["extra"]:
        _assert_pose_copy("extra.tcp_pose", raw["extra"]["tcp_pose"], tcp)
    pose = base_pose.inv() * tcp_pose
    relative = _pose_vector(pose)
    return np.ascontiguousarray(np.concatenate((relative[:3], relative[4:], relative[3:4])), dtype=np.float32)


class _SimplerSimulator:
    CONTROL_PERIOD_NS = 200_000_000
    JUDGED_AT_HORIZON = True

    def __init__(self, setup: Setup, episode: EpisodeKey, *, scenes: Any, render_gpu_id: int) -> None:
        if len(setup.policy.action_spec.channel_names) != 7:
            raise StrictSchemaError("SimplerEnv takes a 7-number action")
        if setup.policy.chunk_horizon != 1 or setup.policy.execution_count != 1:
            raise StrictSchemaError("SimplerEnv worker requires one-action policy responses")
        if type(render_gpu_id) is not int or render_gpu_id < 0:
            raise StrictSchemaError("render_gpu_id must be a nonnegative int")
        self._setup = setup
        self._episode = episode
        self._scenes = scenes
        self._task_id = episode.task_id.removeprefix(f"{setup.environment.suite}::")
        self._render_gpu_id = render_gpu_id
        self._env: Any = None
        self._observation: dict[str, Any] | None = None
        self._step = 0
        self._success = False
        self._closed = False
        self._parse_scenario()

    def _parse_scenario(self) -> None:
        raise NotImplementedError

    def _make_and_reset(self) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError

    def _scene(self, name: str) -> dict[str, Any]:
        try:
            return self._scenes[name]
        except (KeyError, TypeError) as exc:
            raise StrictSchemaError(f"the split has no SimplerEnv scene {name!r}") from exc

    @staticmethod
    def _renderer_kwargs() -> dict[str, str]:
        return {"device": RENDERER_DEVICE}

    def reset(self) -> None:
        if self._closed or self._env is not None:
            raise RuntimeError("SimplerEnv worker reset is single-use")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._render_gpu_id)
        self._env, reset_options = self._make_and_reset()
        self._observation, _ = self._env.reset(seed=self._episode.environment_seed, options=reset_options)
        self._step = 0
        self._success = False

    def observe(self) -> Observation:
        if self._env is None or self._observation is None or self._closed:
            raise RuntimeError("SimplerEnv worker is not active")
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

        image = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(self._env, self._observation))
        camera_spec = self._setup.environment.cameras[0]
        if image.shape != (camera_spec.height, camera_spec.width, 3):
            raise StrictSchemaError(f"SimplerEnv camera shape differs: {image.shape}")
        depth_m = depth_valid = intrinsics = camera_to_world = None
        if camera_spec.has_depth:
            base_pose = self._env.unwrapped.agent.robot.pose
            base_quaternion_wxyz = np.asarray(base_pose.q, dtype=np.float64)
            world_to_base = inverse_pose(
                np.asarray(base_pose.p, dtype=np.float64),
                quaternion_xyzw_to_matrix(base_quaternion_wxyz[[1, 2, 3, 0]]),
            )
            depth_m, depth_valid, intrinsics, camera_to_world = maniskill_camera_3d(
                self._observation["image"][camera_spec.frame_id],
                self._observation["camera_param"][camera_spec.frame_id],
                height=camera_spec.height,
                width=camera_spec.width,
                world_to_reference=world_to_base,
            )
        camera = CameraObservation(
            frame_id=camera_spec.frame_id,
            optical_convention=camera_spec.optical_convention,
            rgb=image,
            depth_m=depth_m,
            depth_valid=depth_valid,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )
        values = {"eef_pose": _base_relative_pose(self._observation, self._env)}
        vectors = tuple(
            RobotStateVector(spec, values[spec.name])
            for spec in self._setup.environment.robot_state
            if spec.name in values
        )
        if len(vectors) != len(self._setup.environment.robot_state):
            raise StrictSchemaError("SimplerEnv setup requests unavailable robot state")
        try:
            instruction = self._env.unwrapped.get_language_instruction()
        except AttributeError:
            instruction = self._env.get_wrapper_attr("get_language_instruction")()
        return Observation(
            episode_id=self._episode.folder_name(),
            step_index=self._step,
            timestamp_ns=self._step * self.CONTROL_PERIOD_NS,
            instruction=str(instruction),
            cameras={"main": camera},
            proprioception=RobotProprioception(vectors),
        )

    def apply(self, action: ActionChunk) -> None:
        if self._env is None or self._closed:
            raise RuntimeError("SimplerEnv worker is not active")
        if not isinstance(action, ActionChunk) or action.spec != self._setup.policy.action_spec:
            raise StrictSchemaError("SimplerEnv action spec mismatch")
        if action.execution_count != 1 or action.horizon != 1 or action.start_step != self._step:
            raise StrictSchemaError("SimplerEnv worker requires one action at the current step")
        native = action.executable_values()[0]
        self._observation, _, terminated, _, _ = self._env.step(native)
        self._success = bool(terminated)
        self._step += 1

    def succeeded(self) -> bool:
        if self._env is None or self._closed:
            raise RuntimeError("SimplerEnv worker is not active")
        return self._success

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._env is not None:
            self._env.close()
            self._env = None
        self._observation = None


class SimplerWidowXSimulator(_SimplerSimulator):
    def _parse_scenario(self) -> None:
        match = re.fullmatch(r"episode_(\d{2})", self._episode.scenario_id)
        if match is None:
            raise StrictSchemaError("WidowX scenario must be episode_NN")
        self._object_episode = int(match.group(1))

    def _make_and_reset(self) -> tuple[Any, dict[str, Any]]:
        import simpler_env

        environment = simpler_env.make(
            self._task_id,
            control_mode=WIDOWX_CONTROL_MODE,
            max_episode_steps=self._episode.horizon,
            renderer_kwargs=self._renderer_kwargs(),
        )
        return environment, {"obj_init_options": {"episode_id": self._object_episode}}


class SimplerWidowXPoolSimulator(SimplerWidowXSimulator):
    def _make_and_reset(self) -> tuple[Any, dict[str, Any]]:
        import gymnasium as gym

        register_pool_envs()
        task = self._scene("tasks")[self._task_id]
        environment = gym.make(
            POOL_PUT_ON_ENV,
            obs_mode="rgbd",
            prepackaged_config=True,
            control_mode=WIDOWX_CONTROL_MODE,
            max_episode_steps=self._episode.horizon,
            rgb_overlay_path=self._scene("overlay").replace("{SIMPLER_DIR}", str(simpler_dir())),
            renderer_kwargs=self._renderer_kwargs(),
            source=task["source"],
            target=task["target"],
            instruction=task["instruction"],
            partial_target=task["partial_target"],
            xy_configs=self._scene("xy_configs"),
            quat_configs=self._scene("quat_configs"),
        )
        return environment, {"obj_init_options": {"episode_id": self._object_episode}}


class SimplerGoogleSimulator(_SimplerSimulator):
    CONTROL_PERIOD_NS = 333_333_333

    def _parse_scenario(self) -> None:
        match = re.fullmatch(r"([a-z0-9_]+)__grid_(\d{3})", self._episode.scenario_id)
        if match is None:
            raise StrictSchemaError("Google scenario must be <name>__grid_NNN")
        self._scenario_name = match.group(1)
        self._grid_index = int(match.group(2))

    def _make_and_reset(self) -> tuple[Any, dict[str, Any]]:
        import gymnasium as gym

        register_pool_envs()
        config = self._scene(self._scenario_name)
        grid = google_scenario_grid(config)
        if self._grid_index >= len(grid):
            raise StrictSchemaError("Google grid member is unavailable")
        kwargs = {
            "obs_mode": "rgbd",
            "max_episode_steps": self._episode.horizon,
            "robot": config["robot_name"],
            "sim_freq": 513,
            "control_freq": 3,
            "control_mode": GOOGLE_CONTROL_MODE,
            "scene_name": config["scene_name"],
            "camera_cfgs": {"add_segmentation": True},
            **config["additional_env_build_kwargs"],
            "renderer_kwargs": self._renderer_kwargs(),
        }
        for name in ("rgb_overlay_path", "rgb_overlay_cameras"):
            if name in config:
                value = config[name]
                if isinstance(value, str):
                    value = value.replace("{SIMPLER_DIR}", str(simpler_dir()))
                kwargs[name] = value
        if "rgb_overlay_path" in kwargs and "rgb_overlay_cameras" not in kwargs:
            kwargs["rgb_overlay_cameras"] = ["overhead_camera"]
        environment = gym.make(config["env_name"], **kwargs)
        return environment, grid[self._grid_index]
