from __future__ import annotations

import importlib
import os
import sys
import traceback
from typing import Any, Mapping

from roboarchitect.inputs import Setup
from roboarchitect.messages import read_frame, write_frame
from roboarchitect.protocol import ActionChunk, StrictSchemaError
from roboarchitect.records import EpisodeKey, validate_benchmark_metrics


def simulator_class(entrypoint: str) -> type[Any]:
    module_name, class_name = entrypoint.split(":", 1)
    found = getattr(importlib.import_module(module_name), class_name, None)
    if not isinstance(found, type):
        raise StrictSchemaError(f"environment.simulator: {entrypoint} is not a class")
    return found


def _envelope(sequence: Any, ok: bool, result: Any = None, error: str | None = None) -> dict[str, Any]:
    return {"sequence": sequence, "ok": ok, "result": result, "error": error}


def _create(payload: Any) -> Any:
    if not isinstance(payload, Mapping) or set(payload) != {"setup", "episode", "scenes", "render_gpu_id"}:
        raise StrictSchemaError("simulator initialize: invalid payload")
    setup = Setup.from_mapping(payload["setup"])
    episode = EpisodeKey.from_mapping(payload["episode"])
    simulator = simulator_class(setup.environment.simulator)
    return simulator(setup, episode, scenes=payload["scenes"], render_gpu_id=payload["render_gpu_id"])


def serve() -> int:
    protocol_out = os.dup(sys.stdout.fileno())
    sys.stdout = sys.stderr
    simulator: Any = None
    while True:
        try:
            request = read_frame(sys.stdin.fileno())
        except EOFError:
            return 0
        sequence = request.get("sequence") if isinstance(request, Mapping) else None
        try:
            if not isinstance(request, Mapping) or set(request) != {"sequence", "operation", "payload"}:
                raise StrictSchemaError("simulator request: invalid envelope")
            operation = request["operation"]
            payload = request["payload"]
            if operation == "initialize":
                if simulator is not None:
                    raise StrictSchemaError("simulator initialize: already initialized")
                simulator = _create(payload)
                result: Any = {"ready": True, "judged_at_horizon": bool(getattr(simulator, "JUDGED_AT_HORIZON", False))}
            elif operation == "reset":
                if simulator is None or payload != {}:
                    raise StrictSchemaError("simulator reset: invalid request")
                simulator.reset()
                result = {"reset": True}
            elif operation == "observe":
                if simulator is None or payload != {}:
                    raise StrictSchemaError("simulator observe: invalid request")
                result = simulator.observe().to_mapping()
            elif operation == "apply":
                if simulator is None or not isinstance(payload, Mapping) or set(payload) != {"action"}:
                    raise StrictSchemaError("simulator apply: invalid request")
                simulator.apply(ActionChunk.from_mapping(payload["action"]))
                result = {"applied": True}
            elif operation == "succeeded":
                if simulator is None or payload != {}:
                    raise StrictSchemaError("simulator succeeded: invalid request")
                result = {"success": bool(simulator.succeeded())}
            elif operation == "benchmark_metrics":
                if simulator is None or payload != {}:
                    raise StrictSchemaError("simulator benchmark_metrics: invalid request")
                method = getattr(simulator, "benchmark_metrics", None)
                result = (
                    {"available": False, "metrics": {}}
                    if not callable(method)
                    else {"available": True, "metrics": validate_benchmark_metrics(method())}
                )
            elif operation == "close":
                if payload != {}:
                    raise StrictSchemaError("simulator close: invalid request")
                if simulator is not None:
                    simulator.close()
                write_frame(protocol_out, _envelope(sequence, True, {"closed": True}))
                return 0
            else:
                raise StrictSchemaError("simulator request: unknown operation")
            write_frame(protocol_out, _envelope(sequence, True, result))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            write_frame(protocol_out, _envelope(sequence, False, error=f"{type(exc).__name__}: {exc}"))


if __name__ == "__main__":
    raise SystemExit(serve())
