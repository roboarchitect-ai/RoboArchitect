from __future__ import annotations

import json
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
from roboarchitect.records import EpisodeKey

from .common import egl_device_index, robosuite_camera_3d, use_libero_defaults


CONDITIONS = (
    "Ideal",
    "Memory_Execution",
    "Memory_Exploration",
    "Mix",
    "Observation_Mismatching",
    "Random_Disturbance",
)
CONDITIONS_IN_IDEAL_FOLDER = ("Ideal", "Observation_Mismatching", "Random_Disturbance")
SETTLE_STEPS = 15
SEGMENT_STEPS = 150
MOVABLE_OBJECTS = (
    "alphabet_soup",
    "bbq_sauce",
    "butter",
    "chocolate_pudding",
    "cookies",
    "cream_cheese",
    "ketchup",
    "macaroni_and_cheese",
    "milk",
    "orange_juice",
    "popcorn",
    "salad_dressing",
    "new_salad_dressing",
    "tomato_sauce",
    "white_bowl",
    "akita_black_bowl",
    "plate",
    "glazed_rim_porcelain_ramekin",
    "red_coffee_mug",
    "porcelain_mug",
    "white_yellow_mug",
    "chefmate_8_frypan",
    "bowl_drainer",
    "moka_pot",
    "window",
    "faucet",
    "black_book",
    "yellow_book",
    "desk_caddy",
    "wine_bottle",
)


def parse_task_id(value: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"robocerebra_public60::(Ideal|Memory_Execution|Memory_Exploration|Mix|Observation_Mismatching|Random_Disturbance)::(case(?:[1-9]|10))",
        value,
    )
    if match is None:
        raise StrictSchemaError("RoboCerebra task ID differs")
    return match.group(1), match.group(2)


def case_folder(condition: str, case: str) -> Path:
    root = os.environ.get("ROBOCEREBRA_BENCH_ROOT")
    if not root:
        raise RuntimeError("set ROBOCEREBRA_BENCH_ROOT to the RoboCerebra benchmark folder")
    folder = Path(root) / ("Ideal" if condition in CONDITIONS_IN_IDEAL_FOLDER else condition) / case
    if not folder.is_dir():
        raise RuntimeError(f"RoboCerebra case folder is absent: {folder}")
    return folder


def _parse_description(path: Path) -> tuple[list[str], list[int], str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    instruction = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("Task:")), "")
    steps = []
    starts = []
    for index, line in enumerate(lines):
        if not line.startswith("Step:"):
            continue
        steps.append(line.split(":", 1)[1].strip())
        if index + 1 >= len(lines):
            raise RuntimeError("RoboCerebra task description lacks a state interval")
        match = re.fullmatch(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", lines[index + 1])
        if match is None:
            raise RuntimeError("RoboCerebra task description interval differs")
        starts.append(int(match.group(1)))
    if not instruction or not steps or len(steps) != len(starts):
        raise RuntimeError("RoboCerebra task description differs")
    return steps, starts, instruction


def _load_goal(path: Path) -> tuple[dict[str, list[list[str]]], dict[str, list[int]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    goals: dict[str, list[list[str]]] = {}
    goal_steps: dict[str, list[int]] = {}
    for object_id, relations in value.items():
        actions = []
        steps = []
        for item in relations:
            if isinstance(item, dict) and set(item) >= {"state_pair", "task_step"}:
                action = item["state_pair"]
                step = item["task_step"]
            elif isinstance(item, list):
                action = item
                step = len(actions)
            else:
                continue
            if len(action) in {2, 3}:
                actions.append([str(action[0]).lower(), *(str(value) for value in action[1:])])
                steps.append(int(step))
        goals[str(object_id)] = actions
        goal_steps[str(object_id)] = steps
    if not goals or not any(goals.values()):
        raise RuntimeError("RoboCerebra goal metadata differs")
    return goals, goal_steps


class RoboCerebraSimulator:
    def __init__(self, setup: Setup, episode: EpisodeKey, *, scenes: Any, render_gpu_id: int) -> None:
        if setup.environment.suite != "robocerebra_public60" or setup.policy.action_spec.channel_semantics[0] != "delta":
            raise StrictSchemaError("RoboCerebra setup differs")
        if setup.policy.chunk_horizon != 1 or setup.policy.execution_count != 1:
            raise StrictSchemaError("RoboCerebra worker requires one-action policy responses")
        condition, case = parse_task_id(episode.task_id)
        if type(render_gpu_id) is not int or render_gpu_id < 0:
            raise StrictSchemaError("render_gpu_id must be a nonnegative int")
        self._setup = setup
        self._episode = episode
        self._condition = condition
        self._folder = case_folder(condition, case)
        self._render_gpu_id = render_gpu_id
        self._wants_depth = any(item.has_depth for item in setup.environment.cameras)
        self._env: Any = None
        self._raw: dict[str, Any] | None = None
        self._goal: dict[str, list[list[str]]] = {}
        self._goal_steps: dict[str, list[int]] = {}
        self._step_states: list[np.ndarray] = []
        self._steps: list[str] = []
        self._segment = 0
        self._policy_step = 0
        self._sim_step = 0
        self._completed_previous = 0
        self._agent_subtasks = 0
        self._initial_excluded = 0
        self._resume_completed = 0
        self._skip_increment = False
        self._success = False
        self._dynamic_applied = False
        self._dynamic_due: int | None = None
        self._dynamic_info: dict[str, Any] | None = None
        self._toggle_direction = -1
        self._shifted_initial = condition in {"Mix", "Observation_Mismatching"}
        self._rng = np.random.default_rng(episode.environment_seed)
        self._closed = False

    def _eef_pose(self) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray(self._raw["robot0_eef_pos"], dtype=np.float32),
                np.asarray(self._raw["robot0_eef_quat"], dtype=np.float32),
            )
        )

    def _set_anchor(self, index: int) -> None:
        self._env.sim.set_state_from_flattened(self._step_states[index])
        self._env.sim.forward()
        self._env._post_process()
        self._env._update_observables(force=True)
        self._raw = self._env._get_observations()

    def _check(self) -> tuple[int, bool]:
        _, completed, success = self._env._check_success(self._goal)
        return int(completed), bool(success)

    def _mark_prior(self, current_step: int) -> int:
        if not hasattr(self._env, "_state_progress"):
            return 0
        count = 0
        for object_id, actions in self._goal.items():
            for action_index, step in enumerate(self._goal_steps.get(object_id, ())):
                if step >= current_step:
                    continue
                if object_id in self._env._state_progress and self._env._state_progress[object_id] <= action_index:
                    self._env._state_progress[object_id] = action_index + 1
                    count += 1
        self._resume_completed += count
        return count

    def _find_y_address(self, object_name: str) -> int | None:
        for joint_name in (
            f"{object_name}_1_joint0",
            f"{object_name}_joint0",
            f"{object_name}_joint",
        ):
            if joint_name in self._env.sim.model.joint_names:
                return int(self._env.sim.model.get_joint_qpos_addr(joint_name)[0]) + 1
        return None

    def _prepare_dynamic(self, description_json: Path) -> None:
        if self._condition not in {"Mix", "Random_Disturbance"}:
            return
        value = json.loads(description_json.read_text(encoding="utf-8"))
        step_objects = {
            str(item.get("step", "")).removeprefix("Step: "): str(item.get("object", ""))
            for item in value
            if isinstance(item, dict)
        }
        related = []
        related_names = []
        for description in self._steps:
            object_name = step_objects.get(description, "")
            related_names.append(object_name)
            address = self._find_y_address(object_name) if object_name in MOVABLE_OBJECTS else None
            related.append(None if address is None else (address, float(self._env.sim.data.qpos[address])))
        unrelated = []
        for object_name in MOVABLE_OBJECTS:
            if object_name in related_names:
                continue
            address = self._find_y_address(object_name)
            if address is not None:
                unrelated.append((address, float(self._env.sim.data.qpos[address])))
        if any(item is not None for item in related) and unrelated:
            self._dynamic_info = {"related": related, "unrelated": unrelated}

    def _move_dynamic(self) -> None:
        if self._dynamic_info is None:
            self._dynamic_due = None
            return
        related = self._dynamic_info["related"][self._segment]
        if self._rng.random() < 0.5 and related is not None:
            address, base = related
        else:
            choices = self._dynamic_info["unrelated"]
            address, base = choices[int(self._rng.integers(0, len(choices)))]
        self._env.sim.data.qpos[address] = base + 0.15 * self._toggle_direction
        self._toggle_direction *= -1
        self._env.sim.forward()
        self._env._post_process()
        self._env._update_observables(force=True)
        self._raw = self._env._get_observations()
        self._dynamic_applied = True
        self._dynamic_due = None

    def _transition(self, new_segment: int) -> None:
        self._segment = new_segment
        self._set_anchor(new_segment)
        self._env.skip_pick_quat_once = True
        self._mark_prior(new_segment)
        self._skip_increment = True
        if self._condition in {"Mix", "Random_Disturbance"}:
            self._dynamic_due = self._sim_step + 10

    def reset(self) -> None:
        if self._closed or self._env is not None:
            raise RuntimeError("RoboCerebra worker reset is single-use")
        egl_device = egl_device_index(self._render_gpu_id)
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device)
        use_libero_defaults()
        import h5py
        import libero.libero.envs.bddl_utils as bddl_utils
        from libero.libero.envs import TASK_MAPPING
        from robosuite import load_controller_config

        bddl = sorted(self._folder.glob("*.bddl"))[0]
        demo = self._folder / "demo.hdf5"
        goal_path = self._folder / "goal.json"
        description = self._folder / "task_description.txt"
        description_json = self._folder / "task_description.json"
        info = bddl_utils.get_problem_info(str(bddl))
        controller = load_controller_config(default_controller="OSC_POSE")
        self._env = TASK_MAPPING[info["problem_name"]](
            bddl_file_name=str(bddl),
            robots=["Panda"],
            controller_configs=controller,
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            ignore_done=True,
            use_camera_obs=True,
            reward_shaping=True,
            camera_heights=256,
            camera_widths=256,
            camera_depths=self._wants_depth,
            control_freq=20,
        )
        with h5py.File(demo, "r") as handle:
            states = np.asarray(handle["data"]["demo_1"]["states"][()])
        self._steps, starts, _ = _parse_description(description)
        if any(index >= len(states) for index in starts):
            raise RuntimeError("RoboCerebra demonstration anchor index differs")
        self._step_states = [np.asarray(states[index]) for index in starts]
        self._goal, self._goal_steps = _load_goal(goal_path)
        self._prepare_dynamic(description_json)
        self._env.seed(self._episode.environment_seed)
        self._raw = self._env.reset()
        for robot in self._env.robots:
            robot.controller.use_delta = True
        initial_anchor = 1 if self._shifted_initial else 0
        if initial_anchor >= len(self._step_states):
            raise RuntimeError("RoboCerebra shifted initial anchor is unavailable")
        self._set_anchor(initial_anchor)
        self._env.skip_pick_quat_once = True
        if initial_anchor:
            self._check()
            for object_id in getattr(self._env, "_state_progress", {}):
                self._env._state_progress[object_id] = 0
            self._initial_excluded = self._mark_prior(initial_anchor)
            self._skip_increment = True
        settle = np.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0), dtype=np.float32)
        for _ in range(SETTLE_STEPS):
            self._raw, _, _, _ = self._env.step(settle)
        self._sim_step = SETTLE_STEPS
        self._completed_previous, self._success = self._check()

    def observe(self) -> Observation:
        if self._env is None or self._raw is None or self._closed:
            raise RuntimeError("RoboCerebra worker is not active")
        specs = {item.name: item for item in self._setup.environment.cameras}
        if set(specs) != {"main", "wrist"}:
            raise StrictSchemaError("RoboCerebra camera setup differs")
        images = {
            "main": np.ascontiguousarray(self._raw["agentview_image"], dtype=np.uint8),
            "wrist": np.ascontiguousarray(self._raw["robot0_eye_in_hand_image"], dtype=np.uint8),
        }
        cameras = {}
        for name, image in images.items():
            spec = specs[name]
            depth_m = depth_valid = intrinsics = camera_to_world = None
            if spec.has_depth:
                depth_m, depth_valid, intrinsics, camera_to_world = robosuite_camera_3d(
                    self._env.sim,
                    spec.frame_id,
                    self._raw[f"{spec.frame_id}_depth"],
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
        values = {
            "eef_pose": self._eef_pose(),
            "gripper_position": np.asarray(self._raw["robot0_gripper_qpos"], dtype=np.float32),
        }
        vectors = tuple(RobotStateVector(spec, np.ascontiguousarray(values[spec.name])) for spec in self._setup.environment.robot_state)
        return Observation(
            episode_id=self._episode.folder_name(),
            step_index=self._policy_step,
            timestamp_ns=self._policy_step * 50_000_000,
            instruction=self._steps[self._segment],
            cameras=cameras,
            proprioception=RobotProprioception(vectors),
        )

    def apply(self, action: ActionChunk) -> None:
        if self._env is None or self._closed:
            raise RuntimeError("RoboCerebra worker is not active")
        if not isinstance(action, ActionChunk) or action.spec != self._setup.policy.action_spec:
            raise StrictSchemaError("RoboCerebra action spec differs")
        if action.horizon != 1 or action.execution_count != 1 or action.start_step != self._policy_step:
            raise StrictSchemaError("RoboCerebra worker requires one current action")
        native = action.executable_values()[0]
        if native.shape != (7,):
            raise StrictSchemaError("RoboCerebra native action width differs")
        self._raw, _, _, _ = self._env.step(native)
        self._policy_step += 1
        self._sim_step += 1
        completed, self._success = self._check()
        difference = completed - self._completed_previous
        if difference > 0 and not self._skip_increment:
            self._agent_subtasks += difference
        self._completed_previous = completed
        self._skip_increment = False
        new_segment = self._sim_step // SEGMENT_STEPS
        if self._sim_step % SEGMENT_STEPS == 0 and new_segment < len(self._steps):
            if self._shifted_initial and new_segment == 1:
                self._segment = new_segment
            else:
                self._transition(new_segment)
        if self._dynamic_due is not None and self._sim_step == self._dynamic_due:
            self._move_dynamic()

    def succeeded(self) -> bool:
        if self._env is None or self._closed:
            raise RuntimeError("RoboCerebra worker is not active")
        return self._success

    def benchmark_metrics(self) -> dict[str, bool | float]:
        total = sum(len(actions) for actions in self._goal.values())
        possible = max(total - self._initial_excluded, 0)
        rate = self._agent_subtasks / possible if possible else 0.0
        return {
            "success": self._success,
            "subtask_completion": float(rate),
            "agent_subtasks": float(self._agent_subtasks),
            "possible_subtasks": float(possible),
            "resume_completed": float(self._resume_completed),
            "dynamic_applied": self._dynamic_applied,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._env is not None:
            self._env.close()
            self._env = None
        self._raw = None
