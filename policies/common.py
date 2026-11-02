from __future__ import annotations

import hashlib
import threading
from typing import Any, Mapping

import numpy as np

from roboarchitect.harness.api import VLARequest
from roboarchitect.protocol import (
    ActionChunk,
    ActionSpec,
    CameraObservation,
    Observation,
    RobotProprioception,
    RobotStateSpec,
    RobotStateVector,
    StrictSchemaError,
)
from roboarchitect.protocol.schema import fields, integer, string


def action_spec(
    channels: tuple[str, ...],
    semantics: tuple[str, ...],
    *,
    frame: str,
    rotation: str,
    gripper: str,
    period_s: float | None,
    arms: tuple[str, ...] = ("arm",),
    value_encoding: str = "physical",
    controller_output_scale: tuple[float, ...] = (),
) -> ActionSpec:
    return ActionSpec(
        arm_names=arms,
        channel_names=channels,
        channel_semantics=semantics,
        coordinate_frame=frame,
        translation_unit="meter",
        rotation_representation=rotation,
        quaternion_order="none",
        gripper_convention=gripper,
        value_encoding=value_encoding,
        controller_output_scale=controller_output_scale,
        control_period_s=period_s,
    )


LIBERO_DELTA_ACTION_SPEC = action_spec(
    ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"),
    ("delta",) * 6 + ("binary",),
    frame="world",
    rotation="axis_angle",
    gripper="closed_positive",
    period_s=1 / 20,
    value_encoding="normalized_controller",
    controller_output_scale=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0),
)


def deterministic_seed(policy_seed: int, call_index: int, resample_index: int = 0) -> int:
    payload = f"{policy_seed}\0{call_index}\0{resample_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def state(observation: Observation, name: str, width: int, quaternion_order: str | None = None) -> np.ndarray:
    try:
        vector = observation.proprioception.by_name(name)
    except KeyError as exc:
        raise StrictSchemaError(f"the observation has no robot state {name!r}") from exc
    if vector.values.shape != (width,):
        raise StrictSchemaError(f"robot state {name!r} must have width {width}")
    if quaternion_order is not None and vector.spec.quaternion_order != quaternion_order:
        raise StrictSchemaError(f"robot state {name!r} must use quaternion order {quaternion_order}")
    return vector.values


def quaternion_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    w = float(np.clip(value[3], -1.0, 1.0))
    denominator = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if denominator <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    return np.ascontiguousarray(value[:3] * (2.0 * np.arccos(w) / denominator), dtype=np.float32)


def libero_state(observation: Observation) -> np.ndarray:
    pose = state(observation, "eef_pose", 7, "xyzw")
    gripper = state(observation, "gripper_position", 2)
    return np.ascontiguousarray(
        np.concatenate((pose[:3], quaternion_xyzw_to_axis_angle(pose[3:]), gripper)), dtype=np.float32
    )


def flipped(observation: Observation, camera: str) -> np.ndarray:
    return np.ascontiguousarray(observation.cameras[camera].rgb[::-1, ::-1])


def checked(actions: Any, shape: tuple[int, int]) -> np.ndarray:
    array = np.ascontiguousarray(actions, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise RuntimeError(f"the policy returned actions of shape {array.shape}; expected finite {shape}")
    return array


def libero_warm_up_request() -> VLARequest:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    cameras = {name: CameraObservation(name, "opencv_rdf", image, None, None, None, None) for name in ("main", "wrist")}
    pose = RobotStateVector(
        RobotStateSpec(
            name="eef_pose",
            quantity="end_effector_pose",
            frame_id="eef_pose",
            reference_frame="base",
            component_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
            units=("meter", "meter", "meter", "quaternion", "quaternion", "quaternion", "quaternion"),
            representation="xyz_quaternion",
            quaternion_order="xyzw",
        ),
        np.array((0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0), dtype=np.float32),
    )
    gripper = RobotStateVector(
        RobotStateSpec(
            name="gripper_position",
            quantity="gripper_position",
            frame_id="gripper_position",
            reference_frame="base",
            component_names=("finger_0", "finger_1"),
            units=("normalized", "normalized"),
            representation="vector",
            quaternion_order="none",
        ),
        np.zeros(2, dtype=np.float32),
    )
    instruction = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    observation = Observation(
        episode_id="warm-up",
        step_index=0,
        timestamp_ns=0,
        instruction=instruction,
        cameras=cameras,
        proprioception=RobotProprioception((pose, gripper)),
    )
    return VLARequest("warm-up-act", "warm-up", observation, instruction, refresh=True)


class Adapter:
    action_spec: ActionSpec

    def __init__(self, task: str) -> None:
        self.task = task

    def start(self, observation: Observation) -> None:
        pass

    def commit(self, action: np.ndarray) -> None:
        pass


class Session:
    def __init__(self, session_id: str, policy_seed: int, adapter: Adapter) -> None:
        self.session_id = session_id
        self.policy_seed = policy_seed
        self.adapter = adapter
        self.episode_id: str | None = None
        self.call_index = 0
        self.native: np.ndarray | None = None
        self.values: np.ndarray | None = None
        self.index = 0
        self.pending: np.ndarray | None = None
        self.pending_step: int | None = None
        self.last_request_id: str | None = None
        self.last_response: dict[str, Any] | None = None


class PolicyServer:
    ADAPTERS: dict[str, type[Adapter]] = {}

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()

    def warm_up(self) -> None:
        request = libero_warm_up_request()
        self.reset({"policy_seed": 0, "adapter": "libero", "task": "warm_up"}, request.session_id, "warm-up-reset")
        self.act(request.to_mapping(), request.session_id, request.request_id)
        self.close_session({}, request.session_id, "warm-up-close")

    def reset(self, payload: Any, session_id: str, request_id: str) -> dict[str, bool]:
        obj = fields(payload, {"policy_seed", "adapter", "task"}, path="policy_reset")
        name = string(obj["adapter"], "policy_reset.adapter")
        if name not in self.ADAPTERS:
            raise StrictSchemaError(f"this policy has no adapter {name!r}; it has {sorted(self.ADAPTERS)}")
        seed = integer(obj["policy_seed"], "policy_reset.policy_seed", minimum=0)
        session = Session(session_id, seed, self.ADAPTERS[name](string(obj["task"], "policy_reset.task")))
        with self.lock:
            self.start_session(session)
            self.sessions[session_id] = session
        return {"reset": True}

    def close_session(self, payload: Any, session_id: str, request_id: str) -> dict[str, bool]:
        fields(payload, set(), path="policy_close")
        with self.lock:
            self.end_session(session_id)
            self.sessions.pop(session_id, None)
        return {"closed": True}

    def act(self, payload: Any, session_id: str, request_id: str) -> Mapping[str, Any]:
        request = VLARequest.from_mapping(payload)
        if request.session_id != session_id or request.request_id != request_id:
            raise StrictSchemaError("policy_act: payload and envelope identity mismatch")
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise StrictSchemaError("policy session must be reset before act")
            if session.last_request_id == request_id:
                return session.last_response
            observation = request.observation
            if session.episode_id is None:
                session.adapter.start(observation)
                session.episode_id = observation.episode_id
            elif observation.episode_id != session.episode_id:
                raise StrictSchemaError("policy_act: the observation is from another episode")
            response = ActionChunk(
                request_id=request.request_id,
                session_id=request.session_id,
                start_step=observation.step_index,
                spec=session.adapter.action_spec,
                values=self.next_action(session, request),
                execution_count=1,
            ).to_mapping()
            session.last_request_id = request_id
            session.last_response = response
            return response

    def next_action(self, session: Session, request: VLARequest) -> np.ndarray:
        step = request.observation.step_index
        if session.pending_step is not None:
            if step == session.pending_step + 1:
                session.adapter.commit(session.pending)
            elif not request.refresh or step != session.pending_step:
                raise StrictSchemaError("policy_act: previous action is not observed as executed")
            session.pending = None
            session.pending_step = None
        self.observe(session, request)
        if request.refresh:
            session.native = None
        if session.native is None or session.index >= len(session.native):
            session.native, session.values = self.infer(session, request)
            session.index = 0
            session.call_index += 1
        index = session.index
        session.index += 1
        session.pending = session.native[index]
        session.pending_step = step
        return session.values[index : index + 1]

    def start_session(self, session: Session) -> None:
        pass

    def end_session(self, session_id: str) -> None:
        pass

    def observe(self, session: Session, request: VLARequest) -> None:
        pass

    def infer(self, session: Session, request: VLARequest) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
