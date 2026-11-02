from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from roboarchitect.messages import ModelServer, ServerIdentity, serialize_methods


SERVERS = {
    "xvla": "policies.xvla:XVLA",
    "pi05": "policies.pi05:Pi05",
    "smolvla": "policies.smolvla:SmolVLA",
    "rlinf_pi05": "policies.rlinf_pi05:RLinfPi05",
    "molmoact2": "policies.molmoact2:MolmoAct2",
    "openvla": "policies.openvla:OpenVLA",
    "rldx": "policies.rldx:RLDX",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m policies.server")
    parser.add_argument("--policy", choices=sorted(SERVERS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port-file", type=Path, required=True)
    parser.add_argument("--identity", type=json.loads, required=True)
    parser.add_argument("--repository", type=Path)
    args = parser.parse_args(argv)
    module_name, class_name = SERVERS[args.policy].split(":")
    policy = getattr(importlib.import_module(module_name), class_name)(args.checkpoint, args.repository)
    policy.warm_up()
    methods = serialize_methods({"act": policy.act, "reset": policy.reset, "close_session": policy.close_session})
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
