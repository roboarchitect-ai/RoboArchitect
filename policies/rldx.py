from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from roboarchitect.protocol import StrictSchemaError

from .common import Adapter, PolicyServer, action_spec, checked, deterministic_seed, state


MODEL_HORIZON = 16
EXECUTION_HORIZON = 8
HISTORY_LENGTH = 7
IMAGE_MAX_AREA = 65_536
VIDEO_DELTA_INDICES = (-6, -4, -2, 0)
CAMERA_KEYS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
STATE_KEYS = (
    ("end_effector_position_relative", 3, None),
    ("end_effector_rotation_relative", 4, "xyzw"),
    ("gripper_qpos", 2, None),
    ("base_position", 3, None),
    ("base_rotation", 4, "xyzw"),
)
ACTION_KEYS = (
    ("end_effector_position", 3),
    ("end_effector_rotation", 3),
    ("gripper_close", 1),
    ("base_motion", 4),
    ("control_mode", 1),
)


class RoboCasaAdapter(Adapter):
    action_spec = action_spec(
        (
            "right_arm_base_dx",
            "right_arm_base_dy",
            "right_arm_base_dz",
            "right_arm_base_drx",
            "right_arm_base_dry",
            "right_arm_base_drz",
            "right_gripper_close",
            "mobile_base_vx",
            "mobile_base_vy",
            "mobile_base_yaw_velocity",
            "torso_joint_delta",
            "base_control_mode",
        ),
        ("delta",) * 6 + ("binary", "velocity", "velocity", "velocity", "delta", "categorical"),
        arms=("right_arm", "mobile_base", "torso"),
        frame="per_channel_robot_base_frames",
        rotation="axis_angle",
        gripper="closed_positive",
        period_s=1 / 20,
        value_encoding="normalized_controller",
        controller_output_scale=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.05, 1.0),
    )

    def __init__(self, task: str) -> None:
        super().__init__(task)
        self.history: list[dict[str, Any]] = []


class RLDX(PolicyServer):
    ADAPTERS = {"robocasa": RoboCasaAdapter}

    def __init__(self, checkpoint: str, repository: Path | None) -> None:
        super().__init__()
        os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
        import torch
        from huggingface_hub import snapshot_download
        from rldx.data.augmentations import build_image_transformations_albumentations
        from rldx.data.embodiment_tags import EmbodimentTag
        from rldx.policy.rldx_policy import RLDXPolicy, RLDXSimPolicyWrapper

        self.torch = torch
        self.device = torch.device("cuda:0")
        policy = RLDXPolicy(
            model_path=snapshot_download(checkpoint, local_files_only=True),
            embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
            device="cuda:0",
            strict=True,
        )
        processor = policy.processor
        processor.image_max_area = IMAGE_MAX_AREA
        processor.train_image_transform, processor.eval_image_transform = build_image_transformations_albumentations(
            image_max_area=IMAGE_MAX_AREA,
            image_resize_m=processor.image_resize_m,
            random_crop_fraction=processor.random_crop_fraction,
            random_rotation_angle=processor.random_rotation_angle,
            color_jitter_params=processor.color_jitter_params,
        )
        self.wrapper = RLDXSimPolicyWrapper(policy, strict=True)

    def warm_up(self) -> None:
        batch: dict[str, Any] = {f"video.{key}": np.zeros((1, 4, 256, 256, 3), dtype=np.uint8) for key in CAMERA_KEYS}
        for name, width, _ in STATE_KEYS:
            batch[f"state.{name}"] = np.zeros((1, 1, width), dtype=np.float32)
        batch["annotation.human.task_description"] = ["open the cabinet"]
        self.predict(batch, deterministic_seed(0, 0), "warm-up", True)
        self.wrapper.reset({"session_ids": ["warm-up"]})

    def start_session(self, session) -> None:
        self.wrapper.reset({"session_ids": [session.session_id]})

    def end_session(self, session_id: str) -> None:
        self.wrapper.reset({"session_ids": [session_id]})

    def observe(self, session, request) -> None:
        observation = request.observation
        step = observation.step_index
        frame = {
            "video": {key: np.ascontiguousarray(observation.cameras[key].rgb) for key in CAMERA_KEYS},
            "state": {name: state(observation, name, width, order) for name, width, order in STATE_KEYS},
            "language": request.instruction,
            "step": step,
        }
        history = session.adapter.history
        if history and step == history[-1]["step"]:
            if not request.refresh:
                raise StrictSchemaError("policy_act: repeated step requires refresh")
            history[-1] = frame
        elif (history and step == history[-1]["step"] + 1) or (not history and step == 0):
            history.append(frame)
        else:
            raise StrictSchemaError("policy_act: RLDX observations must be contiguous from step zero")
        del history[:-HISTORY_LENGTH]

    def infer(self, session, request) -> tuple[np.ndarray, np.ndarray]:
        history = session.adapter.history
        latest = history[-1]
        indices = [max(0, len(history) - 1 + delta) for delta in VIDEO_DELTA_INDICES]
        batch: dict[str, Any] = {
            f"video.{key}": np.ascontiguousarray(np.stack([history[index]["video"][key] for index in indices])[None])
            for key in CAMERA_KEYS
        }
        for name, _, _ in STATE_KEYS:
            batch[f"state.{name}"] = np.ascontiguousarray(np.asarray(latest["state"][name], dtype=np.float32)[None, None])
        batch["annotation.human.task_description"] = [latest["language"]]
        seed = deterministic_seed(session.policy_seed, session.call_index)
        action = self.predict(batch, seed, session.session_id, session.call_index == 0)
        parts = []
        for name, width in ACTION_KEYS:
            part = np.asarray(action[f"action.{name}"], dtype=np.float32)
            parts.append(checked(part[0] if part.ndim == 3 and part.shape[0] == 1 else part, (MODEL_HORIZON, width)))
        selected = np.ascontiguousarray(np.clip(np.concatenate(parts, axis=-1)[:EXECUTION_HORIZON], -1.0, 1.0), dtype=np.float32)
        return selected, selected

    def predict(self, batch: dict[str, Any], seed: int, session_id: str, reset_memory: bool) -> Any:
        torch = self.torch
        with torch.random.fork_rng(devices=[self.device.index]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            with torch.inference_mode():
                action, _ = self.wrapper.get_action(
                    batch, options={"reset_memory": [reset_memory], "session_ids": [session_id]}
                )
        return action
