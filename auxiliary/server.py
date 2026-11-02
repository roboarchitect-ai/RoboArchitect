from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from roboarchitect.messages import ModelServer, ServerIdentity, serialize_methods

from .common import MODELS


BACKENDS = {
    "qwen-language": "auxiliary.qwen:QwenLanguage",
    "qwen-vision": "auxiliary.qwen:QwenVision",
    "molmo2-vision": "auxiliary.molmo2:Molmo2",
    "molmo2-pointing": "auxiliary.molmo2:Molmo2",
    "grounding-dino": "auxiliary.grounding_dino:GroundingDino",
    "sam3": "auxiliary.sam3:Sam3",
    "graspgen": "auxiliary.graspgen:GraspGen",
}
OPERATIONS = {
    "language": "generate",
    "vision": "describe",
    "detection": "detect",
    "segmentation": "segment",
    "pointing": "point",
    "grasp": "grasp",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auxiliary.server")
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--port-file", type=Path, required=True)
    parser.add_argument("--identity", type=json.loads, required=True)
    args = parser.parse_args(argv)
    module_name, class_name = BACKENDS[args.model].split(":")
    backend = getattr(importlib.import_module(module_name), class_name)(args.model, "cuda:0")
    backend.load()
    backend.warm_up()
    operation = OPERATIONS[MODELS[args.model].capability]
    methods = serialize_methods({operation: lambda payload, session_id, request_id: backend.invoke(operation, payload)})
    server = ModelServer(ServerIdentity.from_mapping(args.identity), methods, port_file=args.port_file)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
