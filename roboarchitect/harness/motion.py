from __future__ import annotations

from typing import Any

import numpy as np

from roboarchitect.protocol import ActionSpec


__all__ = ["MotionController", "end_effector_position", "make_controller", "supported_layouts"]


def _quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError("cannot normalize a near-zero quaternion")
    x, y, z, w = q / norm
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    y = float(np.arcsin(np.clip(-m[2, 0], -1.0, 1.0)))
    if abs(np.cos(y)) > 1e-7:
        x = float(np.arctan2(m[2, 1], m[2, 2]))
        z = float(np.arctan2(m[1, 0], m[0, 0]))
    else:
        x = 0.0
        z = float(np.arctan2(-m[0, 1], m[1, 1]))
    return np.array((x, y, z), dtype=np.float64)


def _matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    candidates = np.array(
        (1 + m[0, 0] - m[1, 1] - m[2, 2], 1 - m[0, 0] + m[1, 1] - m[2, 2], 1 - m[0, 0] - m[1, 1] + m[2, 2], 1 + trace)
    )
    index = int(np.argmax(candidates))
    q = np.zeros(4, dtype=np.float64)
    q[index] = 0.5 * np.sqrt(max(float(candidates[index]), 0.0))
    scale = 0.25 / max(float(q[index]), 1e-12)
    if index == 0:
        q[1:] = ((m[0, 1] + m[1, 0]) * scale, (m[0, 2] + m[2, 0]) * scale, (m[2, 1] - m[1, 2]) * scale)
    elif index == 1:
        q[0], q[2], q[3] = (m[0, 1] + m[1, 0]) * scale, (m[1, 2] + m[2, 1]) * scale, (m[0, 2] - m[2, 0]) * scale
    elif index == 2:
        q[0], q[1], q[3] = (m[0, 2] + m[2, 0]) * scale, (m[1, 2] + m[2, 1]) * scale, (m[1, 0] - m[0, 1]) * scale
    else:
        q[:3] = ((m[2, 1] - m[1, 2]) * scale, (m[0, 2] - m[2, 0]) * scale, (m[1, 0] - m[0, 1]) * scale)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    xyz = q[:3]
    w = float(np.clip(q[3], -1.0, 1.0))
    length = float(np.linalg.norm(xyz))
    angle = 2 * float(np.arctan2(length, w))
    factor = angle / length if length > 1e-8 else 2.0
    return np.asarray(xyz * factor, dtype=np.float64)


DELTA_EE7 = "delta_ee7"
ABSOLUTE_EE7 = "absolute_ee7"
ABSOLUTE_EE8_TWO_FINGERS = "absolute_ee8_two_fingers"
DELTA_ARM_WITH_BASE12 = "delta_arm_with_base12"

DEFAULT_MAX_STEP_M = 0.03


def supported_layouts() -> tuple[str, ...]:
    return (DELTA_EE7, ABSOLUTE_EE7, ABSOLUTE_EE8_TWO_FINGERS, DELTA_ARM_WITH_BASE12)


def end_effector_position(observation: Any) -> np.ndarray | None:
    proprioception = getattr(observation, "proprioception", None)
    if proprioception is None:
        return None
    for vector in proprioception.vectors:
        spec = vector.spec
        if spec.quantity != "end_effector_pose":
            continue
        if tuple(spec.component_names[:3]) != ("x", "y", "z"):
            continue
        return np.asarray(vector.values[:3], dtype=np.float64)
    return None


def end_effector_frame(observation: Any) -> str | None:
    proprioception = getattr(observation, "proprioception", None)
    if proprioception is None:
        return None
    for vector in proprioception.vectors:
        spec = vector.spec
        if spec.quantity == "end_effector_pose" and tuple(spec.component_names[:3]) == ("x", "y", "z"):
            return spec.reference_frame
    return None


def _end_effector_rotation_matrix(observation: Any) -> np.ndarray | None:
    proprioception = getattr(observation, "proprioception", None)
    if proprioception is None:
        return None
    for vector in proprioception.vectors:
        spec = vector.spec
        if spec.quantity != "end_effector_pose":
            continue
        values = np.asarray(vector.values, dtype=np.float64)
        if spec.representation == "xyz_quaternion" and values.shape == (7,):
            quaternion = values[3:] if spec.quaternion_order == "xyzw" else values[[4, 5, 6, 3]]
            return _quaternion_xyzw_to_matrix(quaternion)
        if spec.representation == "quaternion" and values.shape == (4,):
            quaternion = values if spec.quaternion_order == "xyzw" else values[[1, 2, 3, 0]]
            return _quaternion_xyzw_to_matrix(quaternion)
    return None


def _classify(spec: ActionSpec) -> str | None:
    channels = tuple(spec.channel_names)
    semantics = tuple(spec.channel_semantics)
    if len(channels) == 7 and semantics[:6] == ("delta",) * 6:
        return DELTA_EE7
    if len(channels) == 7 and semantics[:6] == ("absolute",) * 6:
        return ABSOLUTE_EE7
    if (
        len(channels) == 8
        and semantics == ("absolute",) * 8
        and channels[6:] == ("left_finger", "right_finger")
    ):
        return ABSOLUTE_EE8_TWO_FINGERS
    if len(channels) == 12 and channels[:7] == (
        "right_arm_base_dx",
        "right_arm_base_dy",
        "right_arm_base_dz",
        "right_arm_base_drx",
        "right_arm_base_dry",
        "right_arm_base_drz",
        "right_gripper_close",
    ):
        return DELTA_ARM_WITH_BASE12
    return None


class MotionController:
    def __init__(self, spec: ActionSpec, layout: str) -> None:
        self.spec = spec
        self.layout = layout
        self.channel_names = tuple(spec.channel_names)
        self.n_channels = len(self.channel_names)
        self.is_delta = layout in (DELTA_EE7, DELTA_ARM_WITH_BASE12)
        self.gripper_index: int | None = None
        if layout in (DELTA_EE7, ABSOLUTE_EE7):
            self.gripper_index = 6
        elif layout == DELTA_ARM_WITH_BASE12:
            self.gripper_index = 6
        if spec.value_encoding == "normalized_controller" and spec.controller_output_scale:
            self.metres_per_unit = float(spec.controller_output_scale[0])
        else:
            self.metres_per_unit = 1.0
        if not self.is_delta:
            self.metres_per_unit = 1.0
        self._last: np.ndarray | None = None
        self._gripper = -1.0 if spec.gripper_convention in ("closed_positive", "binary_closed_one") else 1.0
        self._fingers = np.array((0.04, 0.04), dtype=np.float64)

    def reset(self) -> None:
        self._last = None
        self._gripper = -1.0 if self.spec.gripper_convention in ("closed_positive", "binary_closed_one") else 1.0
        self._fingers = np.array((0.04, 0.04), dtype=np.float64)

    def note(self, action: Any) -> None:
        values = getattr(action, "values", action)
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 2:
            count = int(getattr(action, "execution_count", array.shape[0]) or array.shape[0])
            array = array[min(count, array.shape[0]) - 1]
        if array.shape != (self.n_channels,):
            raise ValueError(f"action has {array.shape} values, expected ({self.n_channels},)")
        self._last = array.copy()
        if self.gripper_index is not None:
            self._gripper = float(array[self.gripper_index])
        if self.layout == ABSOLUTE_EE8_TWO_FINGERS:
            self._fingers = array[6:8].copy()

    @property
    def last_action(self) -> np.ndarray | None:
        return None if self._last is None else self._last.copy()

    def position(self, observation: Any) -> np.ndarray | None:
        return end_effector_position(observation)

    def distance_to(self, observation: Any, target_xyz: Any) -> float | None:
        here = end_effector_position(observation)
        if here is None:
            return None
        return float(np.linalg.norm(np.asarray(target_xyz, dtype=np.float64).reshape(3) - here))

    def _gripper_value(self, closed: bool | None) -> float:
        if closed is None:
            return float(self._gripper)
        if self.spec.gripper_convention in ("closed_positive", "binary_closed_one"):
            return 1.0 if closed else -1.0
        return -1.0 if closed else 1.0

    def _base(self, reference: Any | None) -> np.ndarray:
        if reference is not None:
            values = getattr(reference, "values", reference)
            array = np.asarray(values, dtype=np.float64)
            if array.ndim == 2:
                array = array[0]
            if array.shape != (self.n_channels,):
                raise ValueError(f"reference action has {array.shape} values, expected ({self.n_channels},)")
            return array.copy()
        if self._last is not None:
            return self._last.copy()
        return np.zeros(self.n_channels, dtype=np.float64)

    def _rotation_channels(self, observation: Any, base: np.ndarray) -> np.ndarray:
        if self.is_delta:
            return np.zeros(3, dtype=np.float64)
        if self._last is not None or np.any(base[3:6]):
            return base[3:6].copy()
        matrix = _end_effector_rotation_matrix(observation)
        if matrix is None:
            raise ValueError("this setup commands an absolute pose, so a rotation is required: call note() with the policy's chunk first, or pass reference=")
        if self.spec.rotation_representation == "euler_xyz":
            return _matrix_to_euler_xyz(matrix)
        return _matrix_to_axis_angle(matrix)

    def _finish(self, values: np.ndarray) -> np.ndarray:
        result = np.ascontiguousarray(values, dtype=np.float32)
        if self.spec.value_encoding == "normalized_controller":
            result = np.clip(result, -1.0, 1.0).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("motion command produced non-finite values")
        return result

    def hold(self, observation: Any, *, closed: bool | None = None, reference: Any | None = None) -> np.ndarray:
        return self.nudge(observation, (0.0, 0.0, 0.0), closed=closed, reference=reference, max_step_m=None)

    def set_gripper(self, observation: Any, *, closed: bool, reference: Any | None = None) -> np.ndarray:
        return self.nudge(observation, (0.0, 0.0, 0.0), closed=closed, reference=reference, max_step_m=None)

    def nudge(
        self,
        observation: Any,
        delta_xyz: Any,
        *,
        closed: bool | None = None,
        reference: Any | None = None,
        max_step_m: float | None = DEFAULT_MAX_STEP_M,
    ) -> np.ndarray:
        offset = np.asarray(delta_xyz, dtype=np.float64).reshape(3)
        if max_step_m is not None and max_step_m > 0.0:
            length = float(np.linalg.norm(offset))
            if length > max_step_m:
                offset = offset * (max_step_m / length)
        base = self._base(reference)
        values = base.copy()
        values[3:6] = self._rotation_channels(observation, base)
        if self.is_delta:
            values[:3] = offset / max(self.metres_per_unit, 1e-9)
        else:
            here = end_effector_position(observation)
            if here is None:
                raise ValueError("this setup does not report an end-effector position, so nudge cannot work")
            values[:3] = here + offset
        if self.gripper_index is not None:
            values[self.gripper_index] = self._gripper_value(closed)
        if self.layout == ABSOLUTE_EE8_TWO_FINGERS:
            if closed is None:
                fingers = self._fingers
            else:
                fingers = np.array((0.0, 0.0)) if closed else np.array((0.04, 0.04))
            values[6:8] = fingers
        result = self._finish(values)
        self.note(result)
        return result

    def move_to(
        self,
        observation: Any,
        target_xyz: Any,
        *,
        closed: bool | None = None,
        reference: Any | None = None,
        max_step_m: float | None = DEFAULT_MAX_STEP_M,
        offset_xyz: Any | None = None,
    ) -> np.ndarray:
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        if offset_xyz is not None:
            target = target + np.asarray(offset_xyz, dtype=np.float64).reshape(3)
        here = end_effector_position(observation)
        if here is None:
            raise ValueError("this setup does not report an end-effector position, so move_to cannot work")
        return self.nudge(
            observation,
            target - here,
            closed=closed,
            reference=reference,
            max_step_m=max_step_m,
        )


def make_controller(spec: Any) -> MotionController | None:
    if not isinstance(spec, ActionSpec):
        spec = getattr(spec, "spec", None)
    if not isinstance(spec, ActionSpec):
        return None
    layout = _classify(spec)
    if layout is None:
        return None
    return MotionController(spec, layout)
