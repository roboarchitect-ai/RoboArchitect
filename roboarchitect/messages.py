from __future__ import annotations

import http.client
import logging
import os
import select
import struct
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import msgpack
import numpy as np

from roboarchitect.protocol.schema import StrictSchemaError, boolean, fields, integer, sequence, string


_ARRAY_EXT = 17
_DTYPES = frozenset(
    {
        "|b1",
        "|u1",
        "<u2",
        "<u4",
        "<u8",
        "|i1",
        "<i2",
        "<i4",
        "<i8",
        "<f2",
        "<f4",
        "<f8",
    }
)
MAX_MESSAGE_BYTES = 1 << 30
LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _native_little_array(value: np.ndarray) -> np.ndarray:
    if value.dtype.kind not in "buif" or value.dtype.fields is not None or value.dtype.subdtype is not None:
        raise StrictSchemaError("message: unsupported numpy dtype")
    dtype = value.dtype
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<")
    result = np.ascontiguousarray(value.astype(dtype, copy=False))
    if result.dtype.str not in _DTYPES:
        raise StrictSchemaError(f"message: unsupported numpy dtype {result.dtype.str}")
    return result


def _array_default(value: Any) -> msgpack.ExtType:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"unsupported type {type(value).__name__}")
    array = _native_little_array(value)
    payload = msgpack.packb(
        {"data": array.tobytes(order="C"), "dtype": array.dtype.str, "shape": list(array.shape)},
        use_bin_type=True,
        strict_types=True,
    )
    return msgpack.ExtType(_ARRAY_EXT, payload)


def _pairs(pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str:
            raise StrictSchemaError("message: expected string mapping key")
        if key in result:
            raise StrictSchemaError(f"message: duplicate key {key!r}")
        result[key] = value
    return result


def _array_hook(code: int, payload: bytes) -> Any:
    if code != _ARRAY_EXT:
        raise StrictSchemaError(f"message: unsupported extension {code}")
    try:
        obj = msgpack.unpackb(payload, raw=False, strict_map_key=True, object_pairs_hook=_pairs)
        obj = fields(obj, {"data", "dtype", "shape"}, path="array")
        dtype_name = string(obj["dtype"], "array.dtype")
        if dtype_name not in _DTYPES:
            raise StrictSchemaError(f"array.dtype: unsupported dtype {dtype_name}")
        shape = tuple(integer(x, f"array.shape[{i}]", minimum=0) for i, x in enumerate(sequence(obj["shape"], "array.shape")))
        if len(shape) > 8:
            raise StrictSchemaError("array.shape: too many dimensions")
        if type(obj["data"]) is not bytes:
            raise StrictSchemaError("array.data: expected bytes")
        dtype = np.dtype(dtype_name)
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if expected != len(obj["data"]):
            raise StrictSchemaError("array.data: size mismatch")
        result = np.frombuffer(obj["data"], dtype=dtype).reshape(shape).copy()
        result.flags.writeable = False
        return result
    except StrictSchemaError:
        raise
    except Exception as exc:
        raise StrictSchemaError(f"array: invalid extension: {exc}") from exc


def _prepare(value: Any) -> Any:
    if hasattr(value, "to_mapping") and callable(value.to_mapping):
        return _prepare(value.to_mapping())
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            if type(key) is not str:
                raise StrictSchemaError("message: expected string mapping key")
            result[key] = _prepare(value[key])
        return result
    if isinstance(value, (list, tuple)):
        return [_prepare(item) for item in value]
    if value is None or type(value) in (bool, int, float, str, bytes):
        if type(value) is float and not np.isfinite(value):
            raise StrictSchemaError("message: non-finite float")
        return value
    raise StrictSchemaError(f"message: unsupported type {type(value).__name__}")


def encode_message(value: Any) -> bytes:
    try:
        return msgpack.packb(_prepare(value), default=_array_default, use_bin_type=True, strict_types=True)
    except StrictSchemaError:
        raise
    except Exception as exc:
        raise StrictSchemaError(f"message: encoding failed: {exc}") from exc


def decode_message(payload: bytes) -> Any:
    if type(payload) is not bytes:
        raise StrictSchemaError("message: expected bytes")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise StrictSchemaError("message: payload too large")
    try:
        return msgpack.unpackb(
            payload,
            raw=False,
            strict_map_key=True,
            ext_hook=_array_hook,
            object_pairs_hook=_pairs,
        )
    except StrictSchemaError:
        raise
    except Exception as exc:
        raise StrictSchemaError(f"message: decoding failed: {exc}") from exc


def _read_exact(fd: int, count: int, timeout_s: float | None) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    poller = select.poll()
    poller.register(fd, select.POLLIN | select.POLLPRI | select.POLLHUP | select.POLLERR)
    while remaining:
        timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        ready = poller.poll(None if timeout is None else int(timeout * 1000))
        if not ready:
            raise TimeoutError("reading a message timed out")
        chunk = os.read(fd, min(remaining, 1 << 20))
        if not chunk:
            raise EOFError("the other process closed the pipe")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(fd: int, timeout_s: float | None = None) -> Any:
    header = _read_exact(fd, 8, timeout_s)
    size = struct.unpack(">Q", header)[0]
    if size > MAX_MESSAGE_BYTES:
        raise StrictSchemaError("a message exceeds the size limit")
    return decode_message(_read_exact(fd, size, timeout_s))


def write_frame(fd: int, value: Any) -> None:
    payload = encode_message(value)
    if len(payload) > MAX_MESSAGE_BYTES:
        raise StrictSchemaError("a message exceeds the size limit")
    data = memoryview(struct.pack(">Q", len(payload)) + payload)
    while data:
        written = os.write(fd, data)
        if written <= 0:
            raise EOFError("the other process closed the pipe")
        data = data[written:]


class ServerIdentityMismatch(StrictSchemaError):
    pass


@dataclass(frozen=True)
class ServerIdentity:
    service_name: str
    service_kind: str
    model_id: str
    stateful: bool
    replica_id: str
    gpu_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "service_name", string(self.service_name, "identity.service_name"))
        object.__setattr__(self, "service_kind", string(self.service_kind, "identity.service_kind"))
        object.__setattr__(self, "model_id", string(self.model_id, "identity.model_id"))
        object.__setattr__(self, "stateful", boolean(self.stateful, "identity.stateful"))
        object.__setattr__(self, "replica_id", string(self.replica_id, "identity.replica_id"))
        if isinstance(self.gpu_ids, (str, bytes)):
            raise StrictSchemaError("identity.gpu_ids: expected sequence")
        gpu_ids = tuple(integer(item, f"identity.gpu_ids[{index}]", minimum=0) for index, item in enumerate(self.gpu_ids))
        if tuple(sorted(set(gpu_ids))) != gpu_ids:
            raise StrictSchemaError("identity.gpu_ids: expected sorted unique values")
        object.__setattr__(self, "gpu_ids", gpu_ids)

    @classmethod
    def from_mapping(cls, value: Any) -> "ServerIdentity":
        obj = fields(
            value,
            {"service_name", "service_kind", "model_id", "stateful", "replica_id", "gpu_ids"},
            path="identity",
        )
        return cls(**{**obj, "gpu_ids": tuple(sequence(obj["gpu_ids"], "identity.gpu_ids"))})

    def to_mapping(self) -> dict[str, Any]:
        return {
            "service_name": self.service_name,
            "service_kind": self.service_kind,
            "model_id": self.model_id,
            "stateful": self.stateful,
            "replica_id": self.replica_id,
            "gpu_ids": list(self.gpu_ids),
        }

    def validate_exact(self, actual: "ServerIdentity") -> None:
        if not isinstance(actual, ServerIdentity):
            raise ServerIdentityMismatch("identity: expected ServerIdentity")
        expected_map = self.to_mapping()
        actual_map = actual.to_mapping()
        differences = {
            key: {"expected": expected_map[key], "actual": actual_map[key]}
            for key in expected_map
            if expected_map[key] != actual_map[key]
        }
        if differences:
            detail = ", ".join(
                f"{key}={item['actual']!r} expected {item['expected']!r}"
                for key, item in sorted(differences.items())
            )
            raise ServerIdentityMismatch(f"identity mismatch: {detail}")


class ServerCallError(RuntimeError):
    pass


class ServerProtocolError(ServerCallError):
    pass


def serialize_methods(
    methods: Mapping[str, Callable[[Any, str, str], Any]],
) -> dict[str, Callable[[Any, str, str], Any]]:
    lock = threading.Lock()

    def wrap(method: Callable[[Any, str, str], Any]) -> Callable[[Any, str, str], Any]:
        def invoke(payload: Any, session_id: str, request_id: str) -> Any:
            with lock:
                return method(payload, session_id, request_id)

        return invoke

    return {name: wrap(method) for name, method in methods.items()}


def _call_request(value: Any) -> tuple[str, str, str, Any]:
    obj = fields(value, {"method", "request_id", "session_id", "payload"}, path="call")
    return (
        string(obj["method"], "call.method"),
        string(obj["request_id"], "call.request_id"),
        string(obj["session_id"], "call.session_id"),
        obj["payload"],
    )


def _call_response(value: Any, request_id: str) -> Any:
    obj = fields(value, {"ok", "request_id", "payload", "error"}, path="response")
    if type(obj["ok"]) is not bool:
        raise ServerProtocolError("response.ok: expected bool")
    if string(obj["request_id"], "response.request_id") != request_id:
        raise ServerProtocolError("response.request_id: mismatch")
    if obj["ok"]:
        if obj["error"] is not None:
            raise ServerProtocolError("response.error: expected null")
        return obj["payload"]
    error = string(obj["error"], "response.error")
    if obj["payload"] is not None:
        raise ServerProtocolError("response.payload: expected null")
    raise ServerCallError(error)


class ModelServer:
    def __init__(
        self,
        identity: ServerIdentity,
        methods: Mapping[str, Callable[[Any, str, str], Any]],
        *,
        port_file: Path,
        max_request_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if not isinstance(identity, ServerIdentity):
            raise StrictSchemaError("server.identity: expected ServerIdentity")
        if not methods:
            raise StrictSchemaError("server.methods: empty mapping")
        checked: dict[str, Callable[[Any, str, str], Any]] = {}
        for name, method in methods.items():
            checked[string(name, "server.method name")] = method
            if not callable(method):
                raise StrictSchemaError(f"server.methods.{name}: expected callable")
        self.identity = identity
        self.methods = checked
        self.max_request_bytes = int(max_request_bytes)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send(self, status: int, value: Any) -> None:
                body = encode_message(value)
                self.send_response(status)
                self.send_header("Content-Type", "application/msgpack")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/identity":
                    self._send(200, outer.identity.to_mapping())
                else:
                    self._send(404, {"error": "not_found"})

            def do_POST(self) -> None:
                if self.path != "/call":
                    self._send(404, {"error": "not_found"})
                    return
                try:
                    raw_length = self.headers.get("Content-Length")
                    if raw_length is None:
                        raise StrictSchemaError("request: missing Content-Length")
                    length = int(raw_length)
                    if length < 0 or length > outer.max_request_bytes:
                        raise StrictSchemaError("request: invalid Content-Length")
                    content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
                    if content_type != "application/msgpack":
                        raise StrictSchemaError("request: expected application/msgpack")
                    method, request_id, session_id, payload = _call_request(decode_message(self.rfile.read(length)))
                    if method not in outer.methods:
                        self._send(
                            404,
                            {"ok": False, "request_id": request_id, "payload": None, "error": "unknown_method"},
                        )
                        return
                    result = outer.methods[method](payload, session_id, request_id)
                    self._send(200, {"ok": True, "request_id": request_id, "payload": result, "error": None})
                except (StrictSchemaError, ValueError) as exc:
                    request_id = locals().get("request_id", "invalid")
                    self._send(
                        400,
                        {"ok": False, "request_id": request_id, "payload": None, "error": str(exc)},
                    )
                except Exception as exc:
                    request_id = locals().get("request_id", "invalid")
                    logging.exception("service call failed")
                    self._send(
                        500,
                        {
                            "ok": False,
                            "request_id": request_id,
                            "payload": None,
                            "error": f"service_error:{type(exc).__name__}",
                        },
                    )

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        temporary = port_file.with_name(f".{port_file.name}.tmp")
        temporary.write_text(str(self._server.server_address[1]), encoding="utf-8")
        temporary.replace(port_file)

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class ModelClient:
    def __init__(
        self,
        base_url: str,
        expected_identity: ServerIdentity,
        *,
        timeout: float = 30.0,
        max_response_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.base_url = string(base_url, "client.base_url").rstrip("/")
        if not isinstance(expected_identity, ServerIdentity):
            raise StrictSchemaError("client.expected_identity: expected ServerIdentity")
        self.expected_identity = expected_identity
        self.timeout = float(timeout)
        self.max_response_bytes = int(max_response_bytes)
        self._validated_identity: ServerIdentity | None = None
        if self.timeout <= 0 or self.max_response_bytes < 1:
            raise StrictSchemaError("client: invalid limits")

    def _request(self, path: str, body: bytes | None = None) -> Any:
        headers = {"Accept": "application/msgpack"}
        method = "GET"
        if body is not None:
            method = "POST"
            headers["Content-Type"] = "application/msgpack"
        request = urllib.request.Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with LOCAL_OPENER.open(request, timeout=self.timeout) as response:
                content_type = response.headers.get_content_type()
                if content_type != "application/msgpack":
                    raise ServerProtocolError(f"response: unexpected content type {content_type!r}")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > self.max_response_bytes:
                    raise ServerProtocolError("response: too large")
                payload = response.read(self.max_response_bytes + 1)
                if len(payload) > self.max_response_bytes:
                    raise ServerProtocolError("response: too large")
                return decode_message(payload)
        except urllib.error.HTTPError as exc:
            payload = exc.read(self.max_response_bytes + 1)
            try:
                value = decode_message(payload)
            except Exception:
                raise ServerCallError(f"http {exc.code}") from exc
            return value
        except urllib.error.URLError as exc:
            raise ServerCallError(f"service unavailable: {exc.reason}") from exc
        except http.client.HTTPException as exc:
            raise ServerCallError(f"service unavailable: HTTP protocol error: {exc}") from exc

    def identity(self) -> ServerIdentity:
        return ServerIdentity.from_mapping(self._request("/identity"))

    def validate_identity(self) -> ServerIdentity:
        actual = self.identity()
        self.expected_identity.validate_exact(actual)
        self._validated_identity = actual
        return actual

    def call(
        self,
        method: str,
        payload: Any,
        *,
        session_id: str,
        request_id: str | None = None,
    ) -> Any:
        method = string(method, "call.method")
        session_id = string(session_id, "call.session_id")
        request_id = string(request_id or uuid.uuid4().hex, "call.request_id")
        if self._validated_identity is None:
            self.validate_identity()
        request = {
            "method": method,
            "request_id": request_id,
            "session_id": session_id,
            "payload": payload,
        }
        return _call_response(self._request("/call", encode_message(request)), request_id)
