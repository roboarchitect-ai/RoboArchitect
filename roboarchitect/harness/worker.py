from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

from roboarchitect.messages import read_frame, write_frame
from roboarchitect.protocol import ActionChunk, StrictSchemaError

from .api import StepRequest, StepResult
from .tools import RelayedToolbox


def _load_harness(path: Path) -> tuple[Any, frozenset[str], frozenset[str]]:
    spec = importlib.util.spec_from_file_location("editable_harness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the harness module")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
        raise
    factory = getattr(module, "create_harness", None)
    if not callable(factory):
        raise RuntimeError("the harness must define create_harness()")
    harness = factory()
    if not callable(getattr(harness, "act", None)):
        raise RuntimeError("the harness must define act(request, tools)")
    config = getattr(module, "HARNESS_CONFIG", None)
    if not isinstance(config, Mapping) or set(config) != {"required_capabilities", "optional_capabilities"}:
        raise RuntimeError("the harness must define HARNESS_CONFIG with exactly two lists: required_capabilities and optional_capabilities")
    required = frozenset(config["required_capabilities"])
    optional = frozenset(config["optional_capabilities"])
    if required & optional:
        raise RuntimeError("a capability is both required and optional in HARNESS_CONFIG")
    return harness, required, optional


def _envelope(sequence: Any, ok: bool, result: Any = None, error: str | None = None) -> dict[str, Any]:
    return {"sequence": sequence, "ok": ok, "result": result, "error": error}


def serve(harness_path: Path) -> int:
    protocol_out = os.dup(sys.stdout.fileno())
    sys.stdout = sys.stderr
    harness, required_capabilities, optional_capabilities = _load_harness(harness_path)
    toolbox: RelayedToolbox | None = None
    sessions: set[str] = set()
    while True:
        step_events = None
        try:
            request = read_frame(sys.stdin.fileno())
        except EOFError:
            return 0
        sequence = request.get("sequence") if isinstance(request, Mapping) else None
        try:
            if not isinstance(request, Mapping) or set(request) != {"sequence", "operation", "payload"}:
                raise StrictSchemaError("harness request: invalid envelope")
            operation = request["operation"]
            payload = request["payload"]
            if operation == "initialize":
                if toolbox is not None or not isinstance(payload, Mapping) or set(payload) != {"capabilities"}:
                    raise StrictSchemaError("harness initialize: invalid payload")
                capabilities = payload["capabilities"]
                if not isinstance(capabilities, Mapping):
                    raise StrictSchemaError("harness initialize: capabilities must be a mapping")
                declared = required_capabilities | optional_capabilities
                if set(capabilities) - declared or not required_capabilities <= set(capabilities):
                    raise StrictSchemaError("HARNESS_CONFIG does not match the models this setup offers")
                capabilities = {
                    capability: (
                        {**declaration, "required": capability in required_capabilities}
                        if isinstance(declaration, Mapping)
                        else declaration
                    )
                    for capability, declaration in capabilities.items()
                }
                toolbox = RelayedToolbox(capabilities, sys.stdin.fileno(), protocol_out)
                result: Any = {"ready": True}
            elif operation == "act":
                if toolbox is None:
                    raise RuntimeError("the harness process is not initialized")
                step_request = StepRequest.from_mapping(payload)
                if step_request.session_id not in sessions:
                    raise RuntimeError("an episode must be reset before act")
                toolbox.begin_step(
                    step_request.observation.step_index,
                    step_request.request_id,
                    step_request.session_id,
                )
                action = harness.act(step_request, toolbox)
                if not isinstance(action, ActionChunk):
                    raise StrictSchemaError("harness act: expected an ActionChunk")
                step_events = toolbox.finish_step()
                result = StepResult(action, step_events).to_mapping()
            elif operation == "reset":
                if toolbox is None:
                    raise RuntimeError("the harness process is not initialized")
                if not isinstance(payload, Mapping) or set(payload) != {"session_id"}:
                    raise StrictSchemaError("harness reset: invalid payload")
                reset = getattr(harness, "reset", None)
                if callable(reset):
                    reset(payload["session_id"])
                sessions.add(payload["session_id"])
                result = {"reset": True}
            elif operation == "end_session":
                if toolbox is None:
                    raise RuntimeError("the harness process is not initialized")
                if not isinstance(payload, Mapping) or set(payload) != {"session_id"}:
                    raise StrictSchemaError("harness end_session: invalid payload")
                if payload["session_id"] not in sessions:
                    raise RuntimeError("harness end_session: unknown episode")
                reset = getattr(harness, "reset", None)
                if callable(reset):
                    reset(payload["session_id"])
                sessions.discard(payload["session_id"])
                result = {"ended": True}
            elif operation == "close":
                write_frame(protocol_out, _envelope(sequence, True, {"closed": True}))
                return 0
            else:
                raise StrictSchemaError("harness request: unknown operation")
            write_frame(protocol_out, _envelope(sequence, True, result))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            if toolbox is not None and isinstance(request, Mapping) and request.get("operation") == "act" and step_events is None:
                step_events = toolbox.finish_step()
            error = {
                "message": f"{type(exc).__name__}: {exc}",
                "events": [] if step_events is None else [event.to_mapping() for event in step_events],
            }
            write_frame(protocol_out, _envelope(sequence, False, error=error))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=Path, required=True)
    args = parser.parse_args(argv)
    path = args.harness.resolve()
    if not path.is_file() or path.is_symlink():
        parser.error("--harness must be a regular file")
    return serve(path)


if __name__ == "__main__":
    raise SystemExit(main())
