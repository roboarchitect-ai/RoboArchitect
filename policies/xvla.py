from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from roboarchitect.protocol import Observation, StrictSchemaError
from roboarchitect.protocol.rotations import (
    quaternion_xyzw_to_rotation6d,
    rotation6d_columns_to_axis_angle,
    rotation6d_to_euler_xyz,
)

from .common import Adapter, PolicyServer, action_spec, checked, deterministic_seed, state


MODEL_HORIZON = 30
DENOISE_STEPS = 10
WIDOWX_GRIPPER_THRESHOLDS = {
    "widowx_spoon_on_towel": 0.70,
    "widowx_carrot_on_plate": 0.95,
    "widowx_stack_cube": 0.91,
    "widowx_put_eggplant_in_basket": 0.80,
}
OTHER_WIDOWX_GRIPPER_THRESHOLD = 0.85
GOOGLE_RULES = {
    "google_robot_pick_coke_can": (0.25, 10),
    "google_robot_move_near": (0.25, 10),
    "google_robot_open_drawer": (0.35, 10),
    "google_robot_close_drawer": (0.35, 10),
    "google_robot_place_apple_in_closed_top_drawer": (0.28, 6),
}
SINGLE_ARM_CHANNELS = ("x", "y", "z", "rx", "ry", "rz", "gripper")
SINGLE_ARM_SEMANTICS = ("absolute",) * 6 + ("binary",)


def resample_index(context: tuple[str, ...]) -> int:
    values = [item.removeprefix("policy_resample_index=") for item in context if item.startswith("policy_resample_index=")]
    if len(values) > 1 or (values and not values[0].isdigit()):
        raise StrictSchemaError("vla.context: invalid policy_resample_index")
    return int(values[0]) if values else 0


def position(observation: Observation) -> np.ndarray:
    return state(observation, "eef_pose", 7, "xyzw")[:3]


class XVLAAdapter(Adapter):
    domain_id: int
    cameras: tuple[str, ...]
    execution_count = MODEL_HORIZON

    def images(self, observation: Observation) -> list[np.ndarray]:
        return [observation.cameras[name].rgb for name in self.cameras]

    def proprio(self, observation: Observation) -> np.ndarray:
        return self.memory

    def select(self, actions: np.ndarray) -> np.ndarray:
        return actions[: self.execution_count]

    def decode(self, actions: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LiberoAdapter(XVLAAdapter):
    action_spec = action_spec(
        SINGLE_ARM_CHANNELS, SINGLE_ARM_SEMANTICS, frame="world", rotation="axis_angle", gripper="closed_positive", period_s=1 / 20
    )
    domain_id = 3
    cameras = ("main", "wrist")

    def start(self, observation: Observation) -> None:
        rotation = quaternion_xyzw_to_rotation6d(state(observation, "eef_pose", 7, "xyzw")[3:])
        columns = np.concatenate((rotation[0:5:2], rotation[1:6:2]))
        current = np.concatenate((position(observation), columns, np.zeros(1, dtype=np.float32)))
        self.memory = np.concatenate((current, np.zeros_like(current))).astype(np.float32)

    def images(self, observation: Observation) -> list[np.ndarray]:
        main, wrist = super().images(observation)
        return [np.ascontiguousarray(main[::-1, ::-1]), wrist]

    def decode(self, actions: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (
                actions[:, :3],
                rotation6d_columns_to_axis_angle(actions[:, 3:9]),
                np.where(actions[:, 9:10] > 0.5, 1.0, -1.0),
            ),
            axis=-1,
        )

    def commit(self, action: np.ndarray) -> None:
        self.memory[:9] = action[:9]


class WidowXAdapter(XVLAAdapter):
    action_spec = action_spec(
        SINGLE_ARM_CHANNELS, SINGLE_ARM_SEMANTICS, frame="robot_base", rotation="euler_xyz", gripper="open_positive", period_s=1 / 5
    )
    domain_id = 0
    cameras = ("main",)

    def __init__(self, task: str) -> None:
        super().__init__(task)
        self.gripper_threshold = WIDOWX_GRIPPER_THRESHOLDS.get(task, OTHER_WIDOWX_GRIPPER_THRESHOLD)

    def start(self, observation: Observation) -> None:
        current = np.concatenate(
            (
                position(observation),
                np.array((1.0, 0.0, 0.0, 1.0, 0.0, 0.0), dtype=np.float32),
                np.zeros(1, dtype=np.float32),
            )
        )
        self.memory = np.concatenate((current, np.zeros_like(current))).astype(np.float32)

    def decode(self, actions: np.ndarray) -> np.ndarray:
        euler = rotation6d_to_euler_xyz(actions[:, 3:9]) + np.array((0.0, np.pi / 2, 0.0))
        return np.concatenate(
            (actions[:, :3], euler, np.where(actions[:, 9:10] < self.gripper_threshold, 1.0, -1.0)),
            axis=-1,
        )

    def commit(self, action: np.ndarray) -> None:
        self.memory[:10] = action[:10]


class GoogleAdapter(XVLAAdapter):
    action_spec = action_spec(
        SINGLE_ARM_CHANNELS, SINGLE_ARM_SEMANTICS, frame="robot_base", rotation="euler_xyz", gripper="closed_positive", period_s=1 / 3
    )
    domain_id = 1
    cameras = ("main",)

    def __init__(self, task: str) -> None:
        super().__init__(task)
        if task not in GOOGLE_RULES:
            raise StrictSchemaError(f"X-VLA has no Google robot rule for task {task!r}")
        self.gripper_threshold, self.execution_count = GOOGLE_RULES[task]

    def start(self, observation: Observation) -> None:
        self.memory = np.zeros(20, dtype=np.float32)
        self.current_xyz = position(observation).astype(np.float32).copy()

    def select(self, actions: np.ndarray) -> np.ndarray:
        self.chunk_origin = self.current_xyz.copy()
        return actions[::2][: self.execution_count]

    def decode(self, actions: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (
                actions[:, :3] + self.current_xyz,
                rotation6d_to_euler_xyz(actions[:, 3:9]),
                np.where(actions[:, 9:10] > self.gripper_threshold, 1.0, -1.0),
            ),
            axis=-1,
        )

    def commit(self, action: np.ndarray) -> None:
        self.current_xyz = self.chunk_origin + action[:3]


class VLABenchAdapter(XVLAAdapter):
    action_spec = action_spec(
        ("x", "y", "z", "rx", "ry", "rz", "left_finger", "right_finger"),
        ("absolute",) * 8,
        frame="world",
        rotation="euler_xyz",
        gripper="none",
        period_s=None,
    )
    domain_id = 8
    cameras = ("main", "front", "wrist")
    base_offset = np.array((0.0, -0.4, 0.78), dtype=np.float32)

    def proprio(self, observation: Observation) -> np.ndarray:
        pose = state(observation, "eef_pose", 7, "wxyz")
        current = np.concatenate(
            (
                pose[:3] - self.base_offset,
                quaternion_xyzw_to_rotation6d(pose[3:]),
                state(observation, "gripper_position", 1),
            )
        )
        return np.concatenate((current, np.zeros_like(current)))

    def decode(self, actions: np.ndarray) -> np.ndarray:
        width = np.where(actions[:, 9:10] <= 0.5, 0.04, 0.0)
        return np.concatenate(
            (actions[:, :3] + self.base_offset, rotation6d_to_euler_xyz(actions[:, 3:9]), width, width),
            axis=-1,
        )


class XVLA(PolicyServer):
    ADAPTERS = {"libero": LiberoAdapter, "widowx": WidowXAdapter, "google": GoogleAdapter, "vlabench": VLABenchAdapter}

    def __init__(self, checkpoint: str, repository: Path | None) -> None:
        super().__init__()
        if repository is None:
            raise FileNotFoundError("X-VLA needs the folder of the X-VLA repository (repositories.xvla in paths.json)")
        sys.path.insert(0, str(Path(repository).resolve()))
        import torch
        from models.modeling_xvla import XVLA as XVLAModel
        from models.processing_xvla import XVLAProcessor

        self.torch = torch
        self.device = torch.device("cuda:0")
        model = XVLAModel.from_pretrained(checkpoint, trust_remote_code=True, torch_dtype=torch.float32)
        self.model = model.to(self.device).to(torch.float32).eval()
        self.processor = XVLAProcessor.from_pretrained(checkpoint)

    def warm_up(self) -> None:
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        self.generate([image], "pick up the object", np.zeros(20, dtype=np.float32), 0, deterministic_seed(0, 0))

    def infer(self, session, request) -> tuple[np.ndarray, np.ndarray]:
        adapter = session.adapter
        observation = request.observation
        seed = deterministic_seed(session.policy_seed, session.call_index, resample_index(request.context))
        proprio = np.ascontiguousarray(adapter.proprio(observation), dtype=np.float32)
        actions = self.generate(adapter.images(observation), request.instruction, proprio, adapter.domain_id, seed)
        selected = np.ascontiguousarray(adapter.select(actions))
        return selected, np.ascontiguousarray(adapter.decode(selected), dtype=np.float32)

    def generate(self, images: list[np.ndarray], instruction: str, proprio: np.ndarray, domain_id: int, seed: int) -> np.ndarray:
        torch = self.torch
        inputs = self.processor(images, instruction)
        dtype = next(self.model.parameters()).dtype

        def move(value):
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            return tensor.to(self.device, dtype=dtype) if tensor.is_floating_point() else tensor.to(self.device)

        model_inputs = {name: move(value) for name, value in inputs.items()}
        model_inputs["proprio"] = move(proprio).unsqueeze(0)
        model_inputs["domain_id"] = torch.tensor([domain_id], device=self.device)
        with torch.random.fork_rng(devices=[self.device.index]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            actions = self.model.generate_actions(**model_inputs, steps=DENOISE_STEPS)
        return checked(actions.squeeze(0).float().cpu().numpy(), (MODEL_HORIZON, 20))
