from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import StrictSchemaError, boolean, enum, fields, integer, number, sequence, string, string_tuple


ACTION_SEMANTICS = frozenset({"absolute", "delta", "velocity", "binary", "categorical"})
VALUE_ENCODINGS = frozenset({"physical", "normalized_controller"})
ROTATION_REPRESENTATIONS = frozenset(
    {"none", "axis_angle", "quaternion", "euler_xyz", "joint_position", "joint_delta"}
)
GRIPPER_CONVENTIONS = frozenset(
    {"none", "open_positive", "closed_positive", "binary_open_one", "binary_closed_one"}
)
QUATERNION_ORDERS = frozenset({"none", "wxyz", "xyzw"})


@dataclass(frozen=True)
class ActionSpec:
    arm_names: tuple[str, ...]
    channel_names: tuple[str, ...]
    channel_semantics: tuple[str, ...]
    coordinate_frame: str
    translation_unit: str
    rotation_representation: str
    quaternion_order: str
    gripper_convention: str
    value_encoding: str
    controller_output_scale: tuple[float, ...]
    control_period_s: float | None

    def __post_init__(self) -> None:
        arms = string_tuple(self.arm_names, "action_spec.arm_names")
        channels = string_tuple(self.channel_names, "action_spec.channel_names")
        semantics = tuple(
            enum(item, ACTION_SEMANTICS, f"action_spec.channel_semantics[{index}]")
            for index, item in enumerate(self.channel_semantics)
        )
        if len(semantics) != len(channels):
            raise StrictSchemaError("action_spec.channel_semantics: dimension mismatch")
        frame = string(self.coordinate_frame, "action_spec.coordinate_frame")
        if self.translation_unit not in {"meter", "none"}:
            raise StrictSchemaError("action_spec.translation_unit: expected meter or none")
        rotation = enum(
            self.rotation_representation,
            ROTATION_REPRESENTATIONS,
            "action_spec.rotation_representation",
        )
        quaternion_order = enum(self.quaternion_order, QUATERNION_ORDERS, "action_spec.quaternion_order")
        if rotation == "quaternion" and quaternion_order == "none":
            raise StrictSchemaError("action_spec.quaternion_order: required for quaternion")
        if rotation != "quaternion" and quaternion_order != "none":
            raise StrictSchemaError("action_spec.quaternion_order: expected none")
        gripper = enum(self.gripper_convention, GRIPPER_CONVENTIONS, "action_spec.gripper_convention")
        encoding = enum(self.value_encoding, VALUE_ENCODINGS, "action_spec.value_encoding")
        scales = tuple(
            number(item, f"action_spec.controller_output_scale[{index}]")
            for index, item in enumerate(sequence(self.controller_output_scale, "action_spec.controller_output_scale"))
        )
        if any(item <= 0.0 for item in scales):
            raise StrictSchemaError("action_spec.controller_output_scale: expected positive values")
        if encoding == "physical" and scales:
            raise StrictSchemaError("action_spec.controller_output_scale: expected empty for physical values")
        if encoding == "normalized_controller" and len(scales) != len(channels):
            raise StrictSchemaError("action_spec.controller_output_scale: dimension mismatch")
        period = (
            None
            if self.control_period_s is None
            else number(self.control_period_s, "action_spec.control_period_s", minimum=1e-6)
        )
        object.__setattr__(self, "arm_names", arms)
        object.__setattr__(self, "channel_names", channels)
        object.__setattr__(self, "channel_semantics", semantics)
        object.__setattr__(self, "coordinate_frame", frame)
        object.__setattr__(self, "rotation_representation", rotation)
        object.__setattr__(self, "quaternion_order", quaternion_order)
        object.__setattr__(self, "gripper_convention", gripper)
        object.__setattr__(self, "value_encoding", encoding)
        object.__setattr__(self, "controller_output_scale", scales)
        object.__setattr__(self, "control_period_s", period)

    @classmethod
    def from_mapping(cls, value: Any) -> "ActionSpec":
        obj = fields(
            value,
            {
                "arm_names",
                "channel_names",
                "channel_semantics",
                "coordinate_frame",
                "translation_unit",
                "rotation_representation",
                "quaternion_order",
                "gripper_convention",
                "value_encoding",
                "controller_output_scale",
                "control_period_s",
            },
            path="action_spec",
        )
        return cls(
            arm_names=string_tuple(obj["arm_names"], "action_spec.arm_names"),
            channel_names=string_tuple(obj["channel_names"], "action_spec.channel_names"),
            channel_semantics=tuple(sequence(obj["channel_semantics"], "action_spec.channel_semantics")),
            coordinate_frame=obj["coordinate_frame"],
            translation_unit=obj["translation_unit"],
            rotation_representation=obj["rotation_representation"],
            quaternion_order=obj["quaternion_order"],
            gripper_convention=obj["gripper_convention"],
            value_encoding=obj["value_encoding"],
            controller_output_scale=tuple(sequence(obj["controller_output_scale"], "action_spec.controller_output_scale")),
            control_period_s=obj["control_period_s"],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "arm_names": list(self.arm_names),
            "channel_names": list(self.channel_names),
            "channel_semantics": list(self.channel_semantics),
            "coordinate_frame": self.coordinate_frame,
            "translation_unit": self.translation_unit,
            "rotation_representation": self.rotation_representation,
            "quaternion_order": self.quaternion_order,
            "gripper_convention": self.gripper_convention,
            "value_encoding": self.value_encoding,
            "controller_output_scale": list(self.controller_output_scale),
            "control_period_s": self.control_period_s,
        }


@dataclass(frozen=True)
class ActionChunk:
    request_id: str
    session_id: str
    start_step: int
    spec: ActionSpec
    values: np.ndarray
    execution_count: int
    terminal: bool = False

    def __post_init__(self) -> None:
        request_id = string(self.request_id, "action.request_id")
        session_id = string(self.session_id, "action.session_id")
        start_step = integer(self.start_step, "action.start_step", minimum=0)
        if not isinstance(self.spec, ActionSpec):
            raise StrictSchemaError("action.spec: expected an ActionSpec")
        if not isinstance(self.values, np.ndarray):
            raise StrictSchemaError("action.values: expected numpy array")
        if self.values.dtype != np.dtype("float32"):
            raise StrictSchemaError("action.values: expected dtype float32")
        dimension = len(self.spec.channel_names)
        if self.values.ndim != 2 or self.values.shape[0] < 1 or self.values.shape[1] != dimension:
            raise StrictSchemaError("action.values: expected nonempty [horizon, dimension]")
        if not np.isfinite(self.values).all():
            raise StrictSchemaError("action.values: non-finite array")
        if self.spec.value_encoding == "normalized_controller" and (
            np.any(self.values < -1.0) or np.any(self.values > 1.0)
        ):
            raise StrictSchemaError("action.values: normalized controller values must be within [-1, 1]")
        count = integer(self.execution_count, "action.execution_count", minimum=1, maximum=self.values.shape[0])
        terminal = boolean(self.terminal, "action.terminal")
        values = np.ascontiguousarray(self.values).copy()
        values.flags.writeable = False
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "start_step", start_step)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "execution_count", count)
        object.__setattr__(self, "terminal", terminal)

    @property
    def horizon(self) -> int:
        return int(self.values.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.values.shape[1])

    @classmethod
    def from_mapping(cls, value: Any) -> "ActionChunk":
        obj = fields(
            value,
            {
                "request_id",
                "session_id",
                "start_step",
                "spec",
                "values",
                "execution_count",
                "terminal",
            },
            path="action",
        )
        return cls(
            request_id=obj["request_id"],
            session_id=obj["session_id"],
            start_step=obj["start_step"],
            spec=ActionSpec.from_mapping(obj["spec"]),
            values=obj["values"],
            execution_count=obj["execution_count"],
            terminal=obj["terminal"],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "start_step": self.start_step,
            "spec": self.spec.to_mapping(),
            "values": self.values,
            "execution_count": self.execution_count,
            "terminal": self.terminal,
        }

    def executable_values(self) -> np.ndarray:
        return self.values[: self.execution_count]
