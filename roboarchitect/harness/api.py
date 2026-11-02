from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np

from roboarchitect.protocol import (
    ActionChunk,
    ActionSpec,
    Observation,
    StrictSchemaError,
)
from roboarchitect.protocol.observation import OPTICAL_CONVENTIONS
from roboarchitect.messages import ServerIdentity


CAPABILITIES = frozenset({"language", "vision", "detection", "segmentation", "pointing", "grasp", "vla"})


class ToolError(RuntimeError):
    pass


class ToolUnavailableError(ToolError):
    pass


def _text(value: Any, path: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise StrictSchemaError(f"{path}: expected nonempty string")
    return value


def _finite(value: Any, path: str) -> float:
    if type(value) not in (int, float):
        raise StrictSchemaError(f"{path}: expected number")
    result = float(value)
    if not np.isfinite(result):
        raise StrictSchemaError(f"{path}: expected finite number")
    return result


def _rgb(value: Any, path: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise StrictSchemaError(f"{path}: expected uint8 [height, width, 3]")
    result = np.ascontiguousarray(value).copy()
    result.flags.writeable = False
    return result


def _float_array(value: Any, path: str, shape: tuple[int | None, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.ndim != len(shape):
        raise StrictSchemaError(f"{path}: expected float32 array")
    if any(expected is not None and value.shape[index] != expected for index, expected in enumerate(shape)):
        raise StrictSchemaError(f"{path}: unexpected shape")
    if not np.isfinite(value).all():
        raise StrictSchemaError(f"{path}: non-finite array")
    result = np.ascontiguousarray(value).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class StepRequest:
    request_id: str
    session_id: str
    observation: Observation
    action_spec: ActionSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text(self.request_id, "step_request.request_id"))
        object.__setattr__(self, "session_id", _text(self.session_id, "step_request.session_id"))
        if not isinstance(self.observation, Observation):
            raise StrictSchemaError("step_request.observation: expected an Observation")
        if not isinstance(self.action_spec, ActionSpec):
            raise StrictSchemaError("step_request.action_spec: expected an ActionSpec")

    @classmethod
    def from_mapping(cls, value: Any) -> "StepRequest":
        if not isinstance(value, Mapping) or set(value) != {
            "request_id",
            "session_id",
            "observation",
            "action_spec",
        }:
            raise StrictSchemaError("step_request: invalid fields")
        return cls(
            request_id=value["request_id"],
            session_id=value["session_id"],
            observation=Observation.from_mapping(value["observation"]),
            action_spec=ActionSpec.from_mapping(value["action_spec"]),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "observation": self.observation.to_mapping(),
            "action_spec": self.action_spec.to_mapping(),
        }


EVENT_RESULT_MAX_LEAVES = 4096
EVENT_RESULT_MAX_DEPTH = 5
EVENT_RESULT_MAX_TEXT = 4000


def _plain(value: Any, path: str, depth: int, budget: list[int]) -> Any:
    if depth > EVENT_RESULT_MAX_DEPTH:
        raise StrictSchemaError(f"{path}: nested too deeply")
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (bool, int):
        budget[0] -= 1
    elif type(value) is float:
        if not np.isfinite(value):
            raise StrictSchemaError(f"{path}: expected finite number")
        budget[0] -= 1
    elif type(value) is str:
        value = value[:EVENT_RESULT_MAX_TEXT]
        budget[0] -= 1
    elif isinstance(value, (list, tuple)):
        value = [_plain(item, f"{path}[]", depth + 1, budget) for item in value]
    elif isinstance(value, Mapping):
        value = {
            _text(str(name), f"{path}.key"): _plain(item, f"{path}.{name}", depth + 1, budget)
            for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    else:
        raise StrictSchemaError(f"{path}: {type(value).__name__} cannot be recorded")
    if budget[0] < 0:
        raise StrictSchemaError(f"{path}: recorded result holds more than {EVENT_RESULT_MAX_LEAVES} values")
    return value


@dataclass(frozen=True)
class StepEvent:
    step_index: int
    event_type: str
    status: str
    detail: str
    capability: str | None = None
    result: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 0:
            raise StrictSchemaError("step_event.step_index: expected a nonnegative integer")
        object.__setattr__(self, "event_type", _text(self.event_type, "step_event.event_type"))
        if self.status not in {"started", "ok", "optional_error", "infrastructure_error", "triggered", "skipped"}:
            raise StrictSchemaError("step_event.status: unsupported status")
        object.__setattr__(self, "detail", _text(self.detail, "step_event.detail"))
        if self.capability is not None and self.capability not in CAPABILITIES:
            raise StrictSchemaError("step_event.capability: unsupported capability")
        if self.result is not None:
            if not isinstance(self.result, Mapping):
                raise StrictSchemaError("step_event.result: expected a mapping or null")
            object.__setattr__(
                self, "result", _plain(self.result, "step_event.result", 0, [EVENT_RESULT_MAX_LEAVES])
            )

    @classmethod
    def from_mapping(cls, value: Any) -> "StepEvent":
        if not isinstance(value, Mapping) or set(value) != {
            "step_index",
            "event_type",
            "status",
            "detail",
            "capability",
            "result",
        }:
            raise StrictSchemaError("step_event: invalid fields")
        return cls(**value)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "event_type": self.event_type,
            "status": self.status,
            "detail": self.detail,
            "capability": self.capability,
            "result": None if self.result is None else dict(self.result),
        }


@dataclass(frozen=True)
class StepResult:
    action: ActionChunk
    events: tuple[StepEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.action, ActionChunk):
            raise StrictSchemaError("step_result.action: expected an ActionChunk")
        events = tuple(self.events)
        if any(not isinstance(event, StepEvent) for event in events):
            raise StrictSchemaError("step_result.events: expected StepEvent entries")
        object.__setattr__(self, "events", events)

    @classmethod
    def from_mapping(cls, value: Any) -> "StepResult":
        if not isinstance(value, Mapping) or set(value) != {"action", "events"}:
            raise StrictSchemaError("step_result: invalid fields")
        return cls(
            action=ActionChunk.from_mapping(value["action"]),
            events=tuple(StepEvent.from_mapping(event) for event in value["events"]),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {"action": self.action.to_mapping(), "events": [event.to_mapping() for event in self.events]}


@dataclass(frozen=True)
class LanguageRequest:
    instruction: str
    context: tuple[str, ...] = ()
    max_tokens: int = 256
    temperature: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruction", _text(self.instruction, "language.instruction"))
        object.__setattr__(self, "context", tuple(_text(x, "language.context") for x in self.context))
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 4096:
            raise StrictSchemaError("language.max_tokens: expected 1..4096")
        temperature = _finite(self.temperature, "language.temperature")
        if not 0.0 <= temperature <= 2.0:
            raise StrictSchemaError("language.temperature: expected 0..2")
        object.__setattr__(self, "temperature", temperature)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "context": list(self.context),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "LanguageRequest":
        if not isinstance(value, Mapping) or set(value) != {"instruction", "context", "max_tokens", "temperature"}:
            raise StrictSchemaError("language: invalid fields")
        return cls(
            instruction=value["instruction"],
            context=tuple(value["context"]),
            max_tokens=value["max_tokens"],
            temperature=value["temperature"],
        )


@dataclass(frozen=True)
class TextResult:
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _text(self.text, "text_result.text"))

    @classmethod
    def from_mapping(cls, value: Any) -> "TextResult":
        if not isinstance(value, Mapping) or set(value) != {"text"}:
            raise StrictSchemaError("text_result: invalid fields")
        return cls(text=value["text"])

    def to_mapping(self) -> dict[str, Any]:
        return {"text": self.text}


@dataclass(frozen=True)
class VisionRequest:
    instruction: str
    images: Mapping[str, np.ndarray]
    context: tuple[str, ...] = ()
    max_tokens: int = 256

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruction", _text(self.instruction, "vision.instruction"))
        if not isinstance(self.images, Mapping) or not self.images:
            raise StrictSchemaError("vision.images: expected nonempty mapping")
        images = {str(name): _rgb(image, f"vision.images.{name}") for name, image in sorted(self.images.items())}
        object.__setattr__(self, "images", MappingProxyType(images))
        object.__setattr__(self, "context", tuple(_text(x, "vision.context") for x in self.context))
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 4096:
            raise StrictSchemaError("vision.max_tokens: expected 1..4096")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "images": dict(self.images),
            "context": list(self.context),
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "VisionRequest":
        if not isinstance(value, Mapping) or set(value) != {"instruction", "images", "context", "max_tokens"}:
            raise StrictSchemaError("vision: invalid fields")
        return cls(
            instruction=value["instruction"],
            images=value["images"],
            context=tuple(value["context"]),
            max_tokens=value["max_tokens"],
        )


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _text(self.label, "detection.label"))
        score = _finite(self.score, "detection.score")
        if not 0.0 <= score <= 1.0:
            raise StrictSchemaError("detection.score: expected 0..1")
        box = tuple(_finite(x, "detection.box_xyxy") for x in self.box_xyxy)
        if len(box) != 4 or box[0] > box[2] or box[1] > box[3]:
            raise StrictSchemaError("detection.box_xyxy: invalid box")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "box_xyxy", box)

    @classmethod
    def from_mapping(cls, value: Any) -> "Detection":
        if not isinstance(value, Mapping) or set(value) != {"label", "score", "box_xyxy"}:
            raise StrictSchemaError("detection: invalid fields")
        return cls(label=value["label"], score=value["score"], box_xyxy=tuple(value["box_xyxy"]))

    def to_mapping(self) -> dict[str, Any]:
        return {"label": self.label, "score": self.score, "box_xyxy": list(self.box_xyxy)}


@dataclass(frozen=True)
class DetectionRequest:
    image: np.ndarray
    query: str
    threshold: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(self, "image", _rgb(self.image, "detection_request.image"))
        object.__setattr__(self, "query", _text(self.query, "detection_request.query"))
        threshold = _finite(self.threshold, "detection_request.threshold")
        if not 0.0 <= threshold <= 1.0:
            raise StrictSchemaError("detection_request.threshold: expected 0..1")
        object.__setattr__(self, "threshold", threshold)

    def to_mapping(self) -> dict[str, Any]:
        return {"image": self.image, "query": self.query, "threshold": self.threshold}

    @classmethod
    def from_mapping(cls, value: Any) -> "DetectionRequest":
        if not isinstance(value, Mapping) or set(value) != {"image", "query", "threshold"}:
            raise StrictSchemaError("detection_request: invalid fields")
        return cls(image=value["image"], query=value["query"], threshold=value["threshold"])


@dataclass(frozen=True)
class DetectionResult:
    detections: tuple[Detection, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> "DetectionResult":
        if not isinstance(value, Mapping) or set(value) != {"detections"}:
            raise StrictSchemaError("detection_result: invalid fields")
        return cls(tuple(Detection.from_mapping(item) for item in value["detections"]))

    def to_mapping(self) -> dict[str, Any]:
        return {"detections": [item.to_mapping() for item in self.detections]}


@dataclass(frozen=True)
class SegmentationRequest:
    image: np.ndarray
    boxes_xyxy: tuple[tuple[float, float, float, float], ...] = ()
    points_xy: tuple[tuple[float, float], ...] = ()
    labels: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "image", _rgb(self.image, "segmentation_request.image"))
        boxes = tuple(tuple(_finite(x, "segmentation_request.box") for x in box) for box in self.boxes_xyxy)
        points = tuple(tuple(_finite(x, "segmentation_request.point") for x in point) for point in self.points_xy)
        if any(len(box) != 4 for box in boxes) or any(len(point) != 2 for point in points):
            raise StrictSchemaError("segmentation_request: invalid prompt shape")
        labels = tuple(self.labels)
        if any(type(label) is not int or label not in (0, 1) for label in labels):
            raise StrictSchemaError("segmentation_request.labels: expected binary labels")
        if points and len(labels) != len(points):
            raise StrictSchemaError("segmentation_request.labels: point count differs")
        if not boxes and not points:
            raise StrictSchemaError("segmentation_request: prompt required")
        object.__setattr__(self, "boxes_xyxy", boxes)
        object.__setattr__(self, "points_xy", points)
        object.__setattr__(self, "labels", labels)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "boxes_xyxy": [list(x) for x in self.boxes_xyxy],
            "points_xy": [list(x) for x in self.points_xy],
            "labels": list(self.labels),
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "SegmentationRequest":
        if not isinstance(value, Mapping) or set(value) != {"image", "boxes_xyxy", "points_xy", "labels"}:
            raise StrictSchemaError("segmentation_request: invalid fields")
        return cls(
            image=value["image"],
            boxes_xyxy=tuple(tuple(x) for x in value["boxes_xyxy"]),
            points_xy=tuple(tuple(x) for x in value["points_xy"]),
            labels=tuple(value["labels"]),
        )


@dataclass(frozen=True)
class SegmentationResult:
    masks: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        masks = self.masks
        if not isinstance(masks, np.ndarray) or masks.dtype != np.bool_ or masks.ndim != 3:
            raise StrictSchemaError("segmentation_result.masks: expected bool [count, height, width]")
        masks = np.ascontiguousarray(masks).copy()
        masks.flags.writeable = False
        scores = _float_array(self.scores, "segmentation_result.scores", (masks.shape[0],))
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise StrictSchemaError("segmentation_result.scores: expected 0..1")
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "scores", scores)

    @classmethod
    def from_mapping(cls, value: Any) -> "SegmentationResult":
        if not isinstance(value, Mapping) or set(value) != {"masks", "scores"}:
            raise StrictSchemaError("segmentation_result: invalid fields")
        return cls(masks=value["masks"], scores=value["scores"])

    def to_mapping(self) -> dict[str, Any]:
        return {"masks": self.masks, "scores": self.scores}


@dataclass(frozen=True)
class PointingRequest:
    image: np.ndarray
    instruction: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "image", _rgb(self.image, "pointing_request.image"))
        object.__setattr__(self, "instruction", _text(self.instruction, "pointing_request.instruction"))

    def to_mapping(self) -> dict[str, Any]:
        return {"image": self.image, "instruction": self.instruction}

    @classmethod
    def from_mapping(cls, value: Any) -> "PointingRequest":
        if not isinstance(value, Mapping) or set(value) != {"image", "instruction"}:
            raise StrictSchemaError("pointing_request: invalid fields")
        return cls(image=value["image"], instruction=value["instruction"])


@dataclass(frozen=True)
class PointingResult:
    points_xy: tuple[tuple[float, float], ...]
    confidence: tuple[float, ...]

    def __post_init__(self) -> None:
        points = tuple(tuple(_finite(x, "pointing_result.point") for x in point) for point in self.points_xy)
        confidence = tuple(_finite(x, "pointing_result.confidence") for x in self.confidence)
        if any(len(point) != 2 for point in points) or len(points) != len(confidence):
            raise StrictSchemaError("pointing_result: dimension mismatch")
        if any(not 0.0 <= score <= 1.0 for score in confidence):
            raise StrictSchemaError("pointing_result.confidence: expected 0..1")
        object.__setattr__(self, "points_xy", points)
        object.__setattr__(self, "confidence", confidence)

    @classmethod
    def from_mapping(cls, value: Any) -> "PointingResult":
        if not isinstance(value, Mapping) or set(value) != {"points_xy", "confidence"}:
            raise StrictSchemaError("pointing_result: invalid fields")
        return cls(points_xy=tuple(tuple(x) for x in value["points_xy"]), confidence=tuple(value["confidence"]))

    def to_mapping(self) -> dict[str, Any]:
        return {"points_xy": [list(x) for x in self.points_xy], "confidence": list(self.confidence)}


@dataclass(frozen=True)
class GraspRequest:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    optical_convention: str
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        rgb = _rgb(self.rgb, "grasp_request.rgb")
        depth = _float_array(self.depth_m, "grasp_request.depth_m", rgb.shape[:2])
        intrinsics = _float_array(self.intrinsics, "grasp_request.intrinsics", (3, 3))
        transform = _float_array(self.camera_to_world, "grasp_request.camera_to_world", (4, 4))
        if self.optical_convention not in OPTICAL_CONVENTIONS:
            raise StrictSchemaError(f"grasp_request.optical_convention: expected one of {sorted(OPTICAL_CONVENTIONS)}; pass camera.optical_convention from the same camera the depth came from")
        mask = self.mask
        if mask is not None:
            if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != rgb.shape[:2]:
                raise StrictSchemaError("grasp_request.mask: expected bool [height, width]")
            mask = np.ascontiguousarray(mask).copy()
            mask.flags.writeable = False
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", transform)
        object.__setattr__(self, "mask", mask)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "rgb": self.rgb,
            "depth_m": self.depth_m,
            "intrinsics": self.intrinsics,
            "camera_to_world": self.camera_to_world,
            "optical_convention": self.optical_convention,
            "mask": self.mask,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "GraspRequest":
        if not isinstance(value, Mapping) or set(value) != {
            "rgb",
            "depth_m",
            "intrinsics",
            "camera_to_world",
            "optical_convention",
            "mask",
        }:
            raise StrictSchemaError("grasp_request: invalid fields")
        return cls(
            rgb=value["rgb"],
            depth_m=value["depth_m"],
            intrinsics=value["intrinsics"],
            camera_to_world=value["camera_to_world"],
            optical_convention=value["optical_convention"],
            mask=value["mask"],
        )


@dataclass(frozen=True)
class GraspCandidate:
    pose_world: np.ndarray
    score: float
    width_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose_world", _float_array(self.pose_world, "grasp.pose_world", (4, 4)))
        score = _finite(self.score, "grasp.score")
        width = _finite(self.width_m, "grasp.width_m")
        if not 0.0 <= score <= 1.0 or width < 0.0:
            raise StrictSchemaError("grasp: invalid score or width")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "width_m", width)

    @classmethod
    def from_mapping(cls, value: Any) -> "GraspCandidate":
        if not isinstance(value, Mapping) or set(value) != {"pose_world", "score", "width_m"}:
            raise StrictSchemaError("grasp_candidate: invalid fields")
        return cls(pose_world=value["pose_world"], score=value["score"], width_m=value["width_m"])

    def to_mapping(self) -> dict[str, Any]:
        return {"pose_world": self.pose_world, "score": self.score, "width_m": self.width_m}


@dataclass(frozen=True)
class GraspResult:
    candidates: tuple[GraspCandidate, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> "GraspResult":
        if not isinstance(value, Mapping) or set(value) != {"candidates"}:
            raise StrictSchemaError("grasp_result: invalid fields")
        return cls(tuple(GraspCandidate.from_mapping(item) for item in value["candidates"]))

    def to_mapping(self) -> dict[str, Any]:
        return {"candidates": [item.to_mapping() for item in self.candidates]}


@dataclass(frozen=True)
class VLARequest:
    request_id: str
    session_id: str
    observation: Observation
    instruction: str
    context: tuple[str, ...] = ()
    refresh: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text(self.request_id, "vla.request_id"))
        object.__setattr__(self, "session_id", _text(self.session_id, "vla.session_id"))
        if not isinstance(self.observation, Observation):
            raise StrictSchemaError("vla.observation: expected an Observation")
        object.__setattr__(self, "instruction", _text(self.instruction, "vla.instruction"))
        object.__setattr__(self, "context", tuple(_text(x, "vla.context") for x in self.context))
        if type(self.refresh) is not bool:
            raise StrictSchemaError("vla.refresh: expected bool")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "observation": self.observation.to_mapping(),
            "instruction": self.instruction,
            "context": list(self.context),
            "refresh": self.refresh,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "VLARequest":
        if not isinstance(value, Mapping) or set(value) != {
            "request_id",
            "session_id",
            "observation",
            "instruction",
            "context",
            "refresh",
        }:
            raise StrictSchemaError("vla: invalid fields")
        return cls(
            request_id=value["request_id"],
            session_id=value["session_id"],
            observation=Observation.from_mapping(value["observation"]),
            instruction=value["instruction"],
            context=tuple(value["context"]),
            refresh=value["refresh"],
        )


class Tools(Protocol):
    def has(self, capability: str) -> bool: ...

    def required(self, capability: str) -> bool: ...

    def identities(self) -> Mapping[str, ServerIdentity]: ...

    def has_memory(self) -> bool: ...

    def remember(self, key: str, value: Any) -> None: ...

    def recall(self, key: str) -> Any: ...

    def memory_keys(self) -> tuple[str, ...]: ...

    def forget(self, key: str) -> bool: ...

    def record(
        self,
        event_type: str,
        status: str,
        detail: str,
        capability: str | None = None,
        result: Mapping[str, Any] | None = None,
    ) -> None: ...

    def language(self, request: LanguageRequest) -> TextResult: ...

    def vision(self, request: VisionRequest) -> TextResult: ...

    def detect(self, request: DetectionRequest) -> DetectionResult: ...

    def segment(self, request: SegmentationRequest) -> SegmentationResult: ...

    def point(self, request: PointingRequest) -> PointingResult: ...

    def grasp(self, request: GraspRequest) -> GraspResult: ...

    def vla(self, request: VLARequest) -> ActionChunk: ...

    def reset_policy(self, session_id: str, payload: Mapping[str, Any]) -> None: ...

    def close_session(self, session_id: str) -> None: ...


class Harness(Protocol):
    def act(self, request: StepRequest, tools: Tools) -> ActionChunk: ...

    def reset(self, session_id: str) -> None: ...
