from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .schema import StrictSchemaError, enum, fields, integer, mapping, sequence, string, string_tuple


ROBOT_QUANTITIES = frozenset(
    {
        "joint_position",
        "joint_velocity",
        "end_effector_pose",
        "end_effector_velocity",
        "gripper_position",
        "gripper_velocity",
        "base_pose",
        "base_velocity",
        "base_control_state",
    }
)
ROBOT_UNITS = frozenset(
    {
        "meter",
        "radian",
        "meter_per_second",
        "radian_per_second",
        "normalized",
        "quaternion",
        "unitless",
    }
)
ROBOT_REPRESENTATIONS = frozenset(
    {"vector", "quaternion", "xyz_quaternion", "xyz_axis_angle", "planar_xy_yaw"}
)
QUATERNION_ORDERS = frozenset({"none", "wxyz", "xyzw"})
OPTICAL_CONVENTIONS = frozenset({"opencv_rdf", "opengl_rub"})


def _array(
    value: Any,
    path: str,
    *,
    dtype: np.dtype[Any],
    ndim: int,
    shape: tuple[int | None, ...] | None = None,
    nonempty: bool = True,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise StrictSchemaError(f"{path}: expected numpy array")
    if value.dtype != dtype:
        raise StrictSchemaError(f"{path}: expected dtype {dtype}")
    if value.ndim != ndim:
        raise StrictSchemaError(f"{path}: expected {ndim} dimensions")
    if shape is not None:
        for index, expected in enumerate(shape):
            if expected is not None and value.shape[index] != expected:
                raise StrictSchemaError(f"{path}: expected shape {shape}")
    if nonempty and value.size == 0:
        raise StrictSchemaError(f"{path}: empty array")
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise StrictSchemaError(f"{path}: non-finite array")
    result = np.ascontiguousarray(value).copy()
    result.flags.writeable = False
    return result


def _optional_array(
    value: Any,
    path: str,
    *,
    dtype: np.dtype[Any],
    ndim: int,
    shape: tuple[int | None, ...] | None = None,
) -> np.ndarray | None:
    if value is None:
        return None
    return _array(value, path, dtype=dtype, ndim=ndim, shape=shape)


@dataclass(frozen=True)
class CameraObservation:
    frame_id: str
    optical_convention: str
    rgb: np.ndarray
    depth_m: np.ndarray | None
    depth_valid: np.ndarray | None
    intrinsics: np.ndarray | None
    camera_to_world: np.ndarray | None
    rgb_encoding: str = "rgb8"
    depth_unit: str = "meter"

    def __post_init__(self) -> None:
        frame_id = string(self.frame_id, "camera.frame_id")
        optical = enum(self.optical_convention, OPTICAL_CONVENTIONS, "camera.optical_convention")
        if self.rgb_encoding != "rgb8":
            raise StrictSchemaError("camera.rgb_encoding: expected rgb8")
        if self.depth_unit != "meter":
            raise StrictSchemaError("camera.depth_unit: expected meter")
        rgb = _array(self.rgb, "camera.rgb", dtype=np.dtype("uint8"), ndim=3, shape=(None, None, 3))
        depth = _optional_array(self.depth_m, "camera.depth_m", dtype=np.dtype("float32"), ndim=2)
        if depth is not None and depth.shape != rgb.shape[:2]:
            raise StrictSchemaError("camera.depth_m: shape differs from rgb")
        valid = self.depth_valid
        if valid is not None:
            if not isinstance(valid, np.ndarray) or valid.dtype not in (np.dtype("bool"), np.dtype("uint8")):
                raise StrictSchemaError("camera.depth_valid: expected bool or uint8 array")
            if valid.ndim != 2 or depth is None or valid.shape != depth.shape:
                raise StrictSchemaError("camera.depth_valid: shape differs from depth_m")
            if valid.dtype == np.dtype("uint8") and not np.isin(valid, (0, 1)).all():
                raise StrictSchemaError("camera.depth_valid: expected values 0 or 1")
            valid = np.ascontiguousarray(valid.astype(np.bool_, copy=True))
            valid.flags.writeable = False
        intrinsics = _optional_array(
            self.intrinsics, "camera.intrinsics", dtype=np.dtype("float32"), ndim=2, shape=(3, 3)
        )
        transform = _optional_array(
            self.camera_to_world, "camera.camera_to_world", dtype=np.dtype("float32"), ndim=2, shape=(4, 4)
        )
        if depth is None and any(item is not None for item in (valid, intrinsics)):
            raise StrictSchemaError("camera: depth metadata without depth_m")
        if depth is not None and any(item is None for item in (valid, intrinsics)):
            raise StrictSchemaError("camera: depth_m requires depth_valid and intrinsics")
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "optical_convention", optical)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "depth_valid", valid)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", transform)

    @classmethod
    def from_mapping(cls, value: Any) -> "CameraObservation":
        obj = fields(
            value,
            {
                "frame_id",
                "optical_convention",
                "rgb_encoding",
                "depth_unit",
                "rgb",
                "depth_m",
                "depth_valid",
                "intrinsics",
                "camera_to_world",
            },
            path="camera",
        )
        return cls(**obj)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "optical_convention": self.optical_convention,
            "rgb_encoding": self.rgb_encoding,
            "depth_unit": self.depth_unit,
            "rgb": self.rgb,
            "depth_m": self.depth_m,
            "depth_valid": self.depth_valid,
            "intrinsics": self.intrinsics,
            "camera_to_world": self.camera_to_world,
        }


@dataclass(frozen=True)
class RobotStateSpec:
    name: str
    quantity: str
    frame_id: str
    reference_frame: str
    component_names: tuple[str, ...]
    units: tuple[str, ...]
    representation: str
    quaternion_order: str

    def __post_init__(self) -> None:
        name = string(self.name, "robot_state.name")
        quantity = enum(self.quantity, ROBOT_QUANTITIES, "robot_state.quantity")
        frame_id = string(self.frame_id, "robot_state.frame_id")
        reference_frame = string(self.reference_frame, "robot_state.reference_frame")
        components = string_tuple(self.component_names, "robot_state.component_names")
        units = tuple(
            enum(unit, ROBOT_UNITS, f"robot_state.units[{index}]")
            for index, unit in enumerate(self.units)
        )
        if len(units) != len(components):
            raise StrictSchemaError("robot_state.units: dimension mismatch")
        representation = enum(self.representation, ROBOT_REPRESENTATIONS, "robot_state.representation")
        quaternion_order = enum(self.quaternion_order, QUATERNION_ORDERS, "robot_state.quaternion_order")
        if representation == "xyz_quaternion":
            if len(components) != 7 or quaternion_order == "none":
                raise StrictSchemaError("robot_state: xyz_quaternion requires 7 components and quaternion order")
            if tuple(units[:3]) != ("meter", "meter", "meter") or tuple(units[3:]) != (
                "quaternion",
                "quaternion",
                "quaternion",
                "quaternion",
            ):
                raise StrictSchemaError("robot_state.units: invalid xyz_quaternion units")
        elif representation == "quaternion":
            if len(components) != 4 or quaternion_order == "none":
                raise StrictSchemaError("robot_state: quaternion requires 4 components and quaternion order")
            if tuple(units) != ("quaternion", "quaternion", "quaternion", "quaternion"):
                raise StrictSchemaError("robot_state.units: invalid quaternion units")
        elif quaternion_order != "none":
            raise StrictSchemaError("robot_state.quaternion_order: expected none")
        if representation == "xyz_axis_angle" and len(components) != 6:
            raise StrictSchemaError("robot_state: xyz_axis_angle requires 6 components")
        if representation == "planar_xy_yaw" and len(components) != 3:
            raise StrictSchemaError("robot_state: planar_xy_yaw requires 3 components")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "reference_frame", reference_frame)
        object.__setattr__(self, "component_names", components)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "quaternion_order", quaternion_order)

    @classmethod
    def from_mapping(cls, value: Any) -> "RobotStateSpec":
        obj = fields(
            value,
            {
                "name",
                "quantity",
                "frame_id",
                "reference_frame",
                "component_names",
                "units",
                "representation",
                "quaternion_order",
            },
            path="robot_state",
        )
        return cls(
            name=obj["name"],
            quantity=obj["quantity"],
            frame_id=obj["frame_id"],
            reference_frame=obj["reference_frame"],
            component_names=string_tuple(obj["component_names"], "robot_state.component_names"),
            units=tuple(sequence(obj["units"], "robot_state.units")),
            representation=obj["representation"],
            quaternion_order=obj["quaternion_order"],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "quantity": self.quantity,
            "frame_id": self.frame_id,
            "reference_frame": self.reference_frame,
            "component_names": list(self.component_names),
            "units": list(self.units),
            "representation": self.representation,
            "quaternion_order": self.quaternion_order,
        }


@dataclass(frozen=True)
class RobotStateVector:
    spec: RobotStateSpec
    values: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.spec, RobotStateSpec):
            raise StrictSchemaError("robot_state.spec: expected RobotStateSpec")
        values = _array(self.values, "robot_state.values", dtype=np.dtype("float32"), ndim=1)
        if values.shape != (len(self.spec.component_names),):
            raise StrictSchemaError("robot_state.values: dimension mismatch")
        object.__setattr__(self, "values", values)

    @classmethod
    def from_mapping(cls, value: Any) -> "RobotStateVector":
        obj = fields(value, {"spec", "values"}, path="robot_state_vector")
        return cls(spec=RobotStateSpec.from_mapping(obj["spec"]), values=obj["values"])

    def to_mapping(self) -> dict[str, Any]:
        return {"spec": self.spec.to_mapping(), "values": self.values}


@dataclass(frozen=True)
class RobotProprioception:
    vectors: tuple[RobotStateVector, ...]

    def __post_init__(self) -> None:
        vectors = tuple(self.vectors)
        if not vectors or any(not isinstance(vector, RobotStateVector) for vector in vectors):
            raise StrictSchemaError("proprioception.vectors: expected nonempty RobotStateVector sequence")
        names = [vector.spec.name for vector in vectors]
        if len(set(names)) != len(names):
            raise StrictSchemaError("proprioception.vectors: duplicate names")
        if names != sorted(names):
            raise StrictSchemaError("proprioception.vectors: expected sorted names")
        object.__setattr__(self, "vectors", vectors)

    @classmethod
    def from_mapping(cls, value: Any) -> "RobotProprioception":
        obj = fields(value, {"vectors"}, path="proprioception")
        return cls(
            vectors=tuple(
                RobotStateVector.from_mapping(item)
                for item in sequence(obj["vectors"], "proprioception.vectors")
            )
        )

    def to_mapping(self) -> dict[str, Any]:
        return {"vectors": [vector.to_mapping() for vector in self.vectors]}

    def by_name(self, name: str) -> RobotStateVector:
        target = string(name, "name")
        for vector in self.vectors:
            if vector.spec.name == target:
                return vector
        raise KeyError(target)


@dataclass(frozen=True)
class Observation:
    episode_id: str
    step_index: int
    timestamp_ns: int
    instruction: str
    cameras: Mapping[str, CameraObservation]
    proprioception: RobotProprioception

    def __post_init__(self) -> None:
        episode_id = string(self.episode_id, "observation.episode_id")
        step_index = integer(self.step_index, "observation.step_index", minimum=0)
        timestamp_ns = integer(self.timestamp_ns, "observation.timestamp_ns", minimum=0)
        instruction = string(self.instruction, "observation.instruction")
        raw_cameras = mapping(self.cameras, "observation.cameras")
        if not raw_cameras:
            raise StrictSchemaError("observation.cameras: empty mapping")
        cameras: dict[str, CameraObservation] = {}
        for name, value in sorted(raw_cameras.items()):
            camera_name = string(name, "observation.cameras key")
            cameras[camera_name] = value if isinstance(value, CameraObservation) else CameraObservation.from_mapping(value)
        if not isinstance(self.proprioception, RobotProprioception):
            raise StrictSchemaError("observation.proprioception: expected RobotProprioception")
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "step_index", step_index)
        object.__setattr__(self, "timestamp_ns", timestamp_ns)
        object.__setattr__(self, "instruction", instruction)
        object.__setattr__(self, "cameras", MappingProxyType(cameras))

    @classmethod
    def from_mapping(cls, value: Any) -> "Observation":
        obj = fields(
            value,
            {
                "episode_id",
                "step_index",
                "timestamp_ns",
                "instruction",
                "cameras",
                "proprioception",
            },
            path="observation",
        )
        cameras = {
            string(name, "observation.cameras key"): CameraObservation.from_mapping(camera)
            for name, camera in mapping(obj["cameras"], "observation.cameras").items()
        }
        return cls(
            episode_id=obj["episode_id"],
            step_index=obj["step_index"],
            timestamp_ns=obj["timestamp_ns"],
            instruction=obj["instruction"],
            cameras=cameras,
            proprioception=RobotProprioception.from_mapping(obj["proprioception"]),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "timestamp_ns": self.timestamp_ns,
            "instruction": self.instruction,
            "cameras": {name: camera.to_mapping() for name, camera in self.cameras.items()},
            "proprioception": self.proprioception.to_mapping(),
        }
