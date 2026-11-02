from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import uuid

from roboarchitect.messages import ModelClient, ServerIdentity, read_frame, write_frame
from roboarchitect.protocol import ActionChunk, StrictSchemaError

from .api import (
    CAPABILITIES,
    StepEvent,
    DetectionRequest,
    DetectionResult,
    GraspRequest,
    GraspResult,
    LanguageRequest,
    PointingRequest,
    PointingResult,
    SegmentationRequest,
    SegmentationResult,
    TextResult,
    ToolError,
    ToolUnavailableError,
    VLARequest,
    VisionRequest,
)
from .memory import HarnessMemory, memory_key, memory_value


MAX_RECORDED_ITEMS = 16
MEMORY_CHANNEL = "memory"
MEMORY_OPERATIONS = frozenset({"remember", "recall", "keys", "forget"})


def _round(value: Any, places: int = 5) -> float:
    return round(float(value), places)


def summarise_tool_result(capability: str, result: Any) -> dict[str, Any] | None:
    try:
        if capability == "detection":
            return {
                "detections": [
                    {
                        "label": item.label,
                        "score": _round(item.score, 4),
                        "box_xyxy": [_round(x, 2) for x in item.box_xyxy],
                    }
                    for item in result.detections[:MAX_RECORDED_ITEMS]
                ],
                "n_detections": len(result.detections),
            }
        if capability == "pointing":
            return {
                "points_xy": [[_round(x, 2) for x in point] for point in result.points_xy[:MAX_RECORDED_ITEMS]],
                "confidence": [_round(x, 4) for x in result.confidence[:MAX_RECORDED_ITEMS]],
                "n_points": len(result.points_xy),
            }
        if capability == "segmentation":
            masks = []
            for index in range(min(result.masks.shape[0], MAX_RECORDED_ITEMS)):
                mask = result.masks[index]
                rows, columns = np.nonzero(mask)
                masks.append(
                    {
                        "score": _round(result.scores[index], 4),
                        "area_px": int(rows.size),
                        "centroid_xy": None
                        if not rows.size
                        else [_round(columns.mean(), 2), _round(rows.mean(), 2)],
                        "box_xyxy": None
                        if not rows.size
                        else [
                            int(columns.min()),
                            int(rows.min()),
                            int(columns.max()) + 1,
                            int(rows.max()) + 1,
                        ],
                    }
                )
            return {"masks": masks, "n_masks": int(result.masks.shape[0])}
        if capability in ("language", "vision"):
            return {"text": result.text}
        if capability == "vla":
            return {
                "values": [[_round(x) for x in row] for row in result.values.tolist()],
                "channels": list(result.spec.channel_names),
                "execution_count": result.execution_count,
            }
        if capability == "grasp":
            return {
                "candidates": [
                    {
                        "score": _round(item.score, 4),
                        "width_m": _round(item.width_m, 4),
                        "position_xyz": [_round(x, 4) for x in item.pose_world[:3, 3]],
                        "approach_xyz": [_round(x, 4) for x in item.pose_world[:3, 2]],
                        "close_xyz": [_round(x, 4) for x in item.pose_world[:3, 0]],
                    }
                    for item in result.candidates[:MAX_RECORDED_ITEMS]
                ],
                "n_candidates": len(result.candidates),
            }
    except Exception:
        return None
    return None


@dataclass(frozen=True)
class ToolEndpoint:
    url: str
    expected_identity: ServerIdentity
    required: bool
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if type(self.url) is not str or not self.url.startswith(("http://", "https://")):
            raise StrictSchemaError("tool_endpoint.url: expected HTTP URL")
        if not isinstance(self.expected_identity, ServerIdentity):
            raise StrictSchemaError("tool_endpoint.expected_identity: expected ServerIdentity")
        if type(self.required) is not bool:
            raise StrictSchemaError("tool_endpoint.required: expected bool")
        if type(self.timeout_s) not in (int, float) or not 0.0 < float(self.timeout_s) <= 3600.0:
            raise StrictSchemaError("tool_endpoint.timeout_s: expected 0..3600")
        object.__setattr__(self, "timeout_s", float(self.timeout_s))


class ServerClient:
    def __init__(self, endpoint: ToolEndpoint) -> None:
        self.endpoint = endpoint
        self.client = ModelClient(
            endpoint.url,
            endpoint.expected_identity,
            timeout=endpoint.timeout_s,
        )
        self.actual_identity: ServerIdentity | None = None

    def validate(self) -> ServerIdentity:
        self.actual_identity = self.client.validate_identity()
        return self.actual_identity

    def invoke(self, operation: str, payload: Mapping[str, Any], session_id: str, request_id: str) -> Any:
        self.validate()
        return self.client.call(operation, dict(payload), session_id=session_id, request_id=request_id)


class Toolbox:
    def __init__(self, endpoints: Mapping[str, ToolEndpoint], memory: HarnessMemory) -> None:
        unknown = set(endpoints) - CAPABILITIES
        if unknown:
            raise StrictSchemaError(f"toolbox: unknown capabilities {sorted(unknown)}")
        if not isinstance(memory, HarnessMemory):
            raise StrictSchemaError("toolbox: memory must be a HarnessMemory")
        self._memory = memory
        self._clients = {name: ServerClient(endpoint) for name, endpoint in sorted(endpoints.items())}
        self._unavailable: dict[str, str] = {}
        self._events: list[StepEvent] = []
        self._pending_errors: list[tuple[str, str, str, str]] = []
        self._step_index = 0
        self._request_id = "uninitialized"
        self._session_id = "uninitialized"
        for capability, client in self._clients.items():
            expected_kind = "policy" if capability == "vla" else capability
            if client.endpoint.expected_identity.service_kind != expected_kind:
                raise StrictSchemaError(f"toolbox.{capability}: expected service_kind {expected_kind!r}, got {client.endpoint.expected_identity.service_kind!r}")
            expected_stateful = capability == "vla"
            if client.endpoint.expected_identity.stateful != expected_stateful:
                raise StrictSchemaError(f"toolbox.{capability}: stateful identity mismatch")
            try:
                client.validate()
            except Exception as exc:
                detail = f"the server did not answer as expected: {type(exc).__name__}: {exc}"
                self._unavailable[capability] = detail
                status = "infrastructure_error" if client.endpoint.required else "optional_error"
                self._pending_errors.append(("tool_identity", status, detail, capability))
                if client.endpoint.required:
                    raise ToolUnavailableError(f"required {capability}: {detail}") from exc

    def has_memory(self) -> bool:
        return True

    def remember(self, key: str, value: Any) -> None:
        self._memory.remember(key, value)

    def recall(self, key: str) -> Any:
        return self._memory.recall(key)

    def memory_keys(self) -> tuple[str, ...]:
        return self._memory.keys()

    def forget(self, key: str) -> bool:
        return self._memory.forget(key)

    def relay_declaration(self) -> dict[str, Any]:
        result = {}
        for name, client in self._clients.items():
            identity = client.actual_identity
            error = self._unavailable.get(name)
            result[name] = {
                "required": client.endpoint.required,
                "available": identity is not None and error is None,
                "identity": None if identity is None else identity.to_mapping(),
                "error": error,
            }
        return result

    def begin_step(self, step_index: int, request_id: str, session_id: str) -> None:
        if type(step_index) is not int or step_index < 0:
            raise StrictSchemaError("toolbox.step_index: expected nonnegative int")
        self._step_index = step_index
        self._request_id = request_id
        self._session_id = session_id
        self._events = []
        for event_type, status, detail, capability in self._pending_errors:
            self.record(event_type, status, detail, capability)
        self._pending_errors.clear()

    def finish_step(self) -> tuple[StepEvent, ...]:
        result = tuple(self._events)
        self._events = []
        return result

    def record(
        self,
        event_type: str,
        status: str,
        detail: str,
        capability: str | None = None,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        self._events.append(StepEvent(self._step_index, event_type, status, detail, capability, result))

    def has(self, capability: str) -> bool:
        return capability in self._clients and capability not in self._unavailable

    def required(self, capability: str) -> bool:
        client = self._clients.get(capability)
        return bool(client is not None and client.endpoint.required)

    def identities(self) -> Mapping[str, ServerIdentity]:
        identities = {
            name: client.actual_identity
            for name, client in self._clients.items()
            if client.actual_identity is not None and name not in self._unavailable
        }
        return MappingProxyType(identities)

    def _typed_call(
        self,
        capability: str,
        operation: str,
        payload: Mapping[str, Any],
        parser: Callable[[Any], Any],
    ) -> Any:
        client = self._clients.get(capability)
        if client is None:
            raise ToolUnavailableError(f"{capability}: service not configured")
        if capability in self._unavailable:
            raise ToolUnavailableError(f"{capability}: {self._unavailable[capability]}")
        self.record("tool_call", "started", operation, capability)
        try:
            result = parser(client.invoke(operation, payload, self._session_id, self._request_id))
        except Exception as exc:
            status = "infrastructure_error" if client.endpoint.required else "optional_error"
            detail = f"{operation}: {type(exc).__name__}: {exc}"
            self.record("tool_call", status, detail, capability)
            raise ToolUnavailableError(detail) from exc
        self.record("tool_call", "ok", operation, capability, summarise_tool_result(capability, result))
        return result

    def language(self, request: LanguageRequest) -> TextResult:
        return self._typed_call("language", "generate", request.to_mapping(), TextResult.from_mapping)

    def vision(self, request: VisionRequest) -> TextResult:
        return self._typed_call("vision", "describe", request.to_mapping(), TextResult.from_mapping)

    def detect(self, request: DetectionRequest) -> DetectionResult:
        return self._typed_call("detection", "detect", request.to_mapping(), DetectionResult.from_mapping)

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        return self._typed_call("segmentation", "segment", request.to_mapping(), SegmentationResult.from_mapping)

    def point(self, request: PointingRequest) -> PointingResult:
        return self._typed_call("pointing", "point", request.to_mapping(), PointingResult.from_mapping)

    def grasp(self, request: GraspRequest) -> GraspResult:
        return self._typed_call("grasp", "grasp", request.to_mapping(), GraspResult.from_mapping)

    def vla(self, request: VLARequest) -> ActionChunk:
        if request.request_id != self._request_id or request.session_id != self._session_id:
            raise ToolError("vla: the request gives a step other than the current one")
        result = self._typed_call("vla", "act", request.to_mapping(), ActionChunk.from_mapping)
        if result.request_id != request.request_id or result.session_id != request.session_id:
            self.record("tool_call", "infrastructure_error", "response identity mismatch", "vla")
            raise ToolError("vla: response identity mismatch")
        return result

    def dispatch_relay(
        self,
        capability: str,
        operation: str,
        payload: Mapping[str, Any],
        session_id: str,
        request_id: str,
    ) -> Any:
        if session_id != self._session_id or request_id != self._request_id:
            raise ToolError("a model call gives a step other than the current one")
        if capability == MEMORY_CHANNEL:
            if operation not in MEMORY_OPERATIONS:
                raise ToolError("relay requested an unsupported memory operation")
            store = self._memory
            if not isinstance(payload, Mapping):
                raise ToolError("relay memory payload differs")
            if operation == "keys":
                if set(payload):
                    raise ToolError("relay memory payload differs")
                return {"value": list(store.keys())}
            if operation == "remember":
                if set(payload) != {"key", "value"}:
                    raise ToolError("relay memory payload differs")
                store.remember(payload["key"], payload["value"])
                return {"value": None}
            if set(payload) != {"key"}:
                raise ToolError("relay memory payload differs")
            if operation == "forget":
                return {"value": store.forget(payload["key"])}
            return {"value": store.recall(payload["key"])}
        expected_operations = {
            "language": "generate",
            "vision": "describe",
            "detection": "detect",
            "segmentation": "segment",
            "pointing": "point",
            "grasp": "grasp",
            "vla": "act",
        }
        if capability not in expected_operations or operation != expected_operations[capability]:
            raise ToolError("relay requested an unsupported capability operation")
        parsers = {
            "language": (LanguageRequest.from_mapping, self.language),
            "vision": (VisionRequest.from_mapping, self.vision),
            "detection": (DetectionRequest.from_mapping, self.detect),
            "segmentation": (SegmentationRequest.from_mapping, self.segment),
            "pointing": (PointingRequest.from_mapping, self.point),
            "grasp": (GraspRequest.from_mapping, self.grasp),
            "vla": (VLARequest.from_mapping, self.vla),
        }
        parser, caller = parsers[capability]
        result = caller(parser(payload))
        return result.to_mapping()

    def reset_policy(self, session_id: str, payload: Mapping[str, Any]) -> None:
        client = self._clients.get("vla")
        if client is None:
            return
        if "vla" in self._unavailable:
            raise ToolUnavailableError("required vla policy is unavailable")
        if client.invoke("reset", dict(payload), session_id, uuid.uuid4().hex) != {"reset": True}:
            raise ToolError("vla reset: invalid response")

    def close_session(self, session_id: str) -> None:
        for client in self._clients.values():
            identity = client.actual_identity
            if identity is None or not identity.stateful:
                continue
            result = client.invoke("close_session", {}, session_id, uuid.uuid4().hex)
            if result != {"closed": True}:
                raise ToolError(f"{identity.service_name}: invalid close_session response")


class RelayedToolbox:
    def __init__(
        self,
        declarations: Mapping[str, Any],
        input_fd: int,
        output_fd: int,
    ) -> None:
        unknown = set(declarations) - CAPABILITIES
        if unknown:
            raise StrictSchemaError(f"relayed_toolbox: unknown capabilities {sorted(unknown)}")
        self._required: dict[str, bool] = {}
        self._available: dict[str, bool] = {}
        self._identities: dict[str, ServerIdentity] = {}
        self._errors: dict[str, str] = {}
        for capability, declaration in declarations.items():
            if not isinstance(declaration, Mapping) or set(declaration) != {
                "required",
                "available",
                "identity",
                "error",
            }:
                raise StrictSchemaError("relayed_toolbox: invalid declaration")
            required = declaration["required"]
            available = declaration["available"]
            error = declaration["error"]
            if type(required) is not bool or type(available) is not bool:
                raise StrictSchemaError("relayed_toolbox: invalid availability")
            if error is not None and (type(error) is not str or not error):
                raise StrictSchemaError("relayed_toolbox: invalid error")
            identity = declaration["identity"]
            if identity is not None:
                self._identities[capability] = ServerIdentity.from_mapping(identity)
            if available != (capability in self._identities and error is None):
                raise StrictSchemaError("relayed_toolbox: inconsistent availability")
            if required and not available:
                raise ToolUnavailableError(f"required {capability}: {error or 'unavailable'}")
            self._required[capability] = required
            self._available[capability] = available
            if error is not None:
                self._errors[capability] = error
        self._input_fd = input_fd
        self._output_fd = output_fd
        self._events: list[StepEvent] = []
        self._step_index = 0
        self._request_id = "uninitialized"
        self._session_id = "uninitialized"
        self._relay_id = 0

    def begin_step(self, step_index: int, request_id: str, session_id: str) -> None:
        if type(step_index) is not int or step_index < 0:
            raise StrictSchemaError("relayed_toolbox.step_index: expected nonnegative int")
        self._step_index = step_index
        self._request_id = request_id
        self._session_id = session_id
        self._events = []
        for capability, detail in sorted(self._errors.items()):
            self.record("tool_identity", "optional_error", detail, capability)

    def finish_step(self) -> tuple[StepEvent, ...]:
        result = tuple(self._events)
        self._events = []
        return result

    def record(
        self,
        event_type: str,
        status: str,
        detail: str,
        capability: str | None = None,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        self._events.append(StepEvent(self._step_index, event_type, status, detail, capability, result))

    def has(self, capability: str) -> bool:
        return self._available.get(capability, False)

    def required(self, capability: str) -> bool:
        return self._required.get(capability, False)

    def identities(self) -> Mapping[str, ServerIdentity]:
        return MappingProxyType(dict(self._identities))

    def has_memory(self) -> bool:
        return True

    def _memory_call(self, operation: str, payload: Mapping[str, Any]) -> Any:
        try:
            result = self._exchange(MEMORY_CHANNEL, operation, payload)
        except Exception as exc:
            detail = f"{operation}: {type(exc).__name__}: {exc}"
            self.record("memory", "optional_error", detail)
            raise ToolUnavailableError(detail) from exc
        if not isinstance(result, Mapping) or set(result) != {"value"}:
            raise ToolUnavailableError("memory: relay returned an invalid result")
        return result["value"]

    def remember(self, key: str, value: Any) -> None:
        self._memory_call("remember", {"key": memory_key(key), "value": memory_value(value)})

    def recall(self, key: str) -> Any:
        return self._memory_call("recall", {"key": memory_key(key)})

    def memory_keys(self) -> tuple[str, ...]:
        return tuple(self._memory_call("keys", {}))

    def forget(self, key: str) -> bool:
        return bool(self._memory_call("forget", {"key": memory_key(key)}))

    def _exchange(self, capability: str, operation: str, payload: Mapping[str, Any]) -> Any:
        self._relay_id += 1
        relay_id = self._relay_id
        write_frame(
            self._output_fd,
            {
                "type": "tool_request",
                "relay_id": relay_id,
                "capability": capability,
                "operation": operation,
                "payload": dict(payload),
                "session_id": self._session_id,
                "request_id": self._request_id,
            },
        )
        response = read_frame(self._input_fd)
        if not isinstance(response, Mapping) or set(response) != {"type", "relay_id", "ok", "result", "error"}:
            raise ToolError("relay returned an invalid envelope")
        if response["type"] != "tool_response" or response["relay_id"] != relay_id:
            raise ToolError("relay response identity mismatch")
        if response["ok"] is not True:
            raise ToolError(str(response["error"]))
        if response["error"] is not None:
            raise ToolError("relay success contained an error")
        return response["result"]

    def _typed_call(
        self,
        capability: str,
        operation: str,
        request: Any,
        parser: Callable[[Any], Any],
    ) -> Any:
        if capability not in self._required:
            raise ToolUnavailableError(f"{capability}: service not configured")
        if not self.has(capability):
            raise ToolUnavailableError(f"{capability}: {self._errors.get(capability, 'unavailable')}")
        self.record("tool_call", "started", operation, capability)
        try:
            result = parser(self._exchange(capability, operation, request.to_mapping()))
        except Exception as exc:
            status = "infrastructure_error" if self.required(capability) else "optional_error"
            detail = f"{operation}: {type(exc).__name__}: {exc}"
            self.record("tool_call", status, detail, capability)
            raise ToolUnavailableError(detail) from exc
        self.record("tool_call", "ok", operation, capability, summarise_tool_result(capability, result))
        return result

    def language(self, request: LanguageRequest) -> TextResult:
        return self._typed_call("language", "generate", request, TextResult.from_mapping)

    def vision(self, request: VisionRequest) -> TextResult:
        return self._typed_call("vision", "describe", request, TextResult.from_mapping)

    def detect(self, request: DetectionRequest) -> DetectionResult:
        return self._typed_call("detection", "detect", request, DetectionResult.from_mapping)

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        return self._typed_call("segmentation", "segment", request, SegmentationResult.from_mapping)

    def point(self, request: PointingRequest) -> PointingResult:
        return self._typed_call("pointing", "point", request, PointingResult.from_mapping)

    def grasp(self, request: GraspRequest) -> GraspResult:
        return self._typed_call("grasp", "grasp", request, GraspResult.from_mapping)

    def vla(self, request: VLARequest) -> ActionChunk:
        if request.request_id != self._request_id or request.session_id != self._session_id:
            raise ToolError("vla: the request gives a step other than the current one")
        result = self._typed_call("vla", "act", request, ActionChunk.from_mapping)
        if result.request_id != request.request_id or result.session_id != request.session_id:
            raise ToolError("vla: response identity mismatch")
        return result
