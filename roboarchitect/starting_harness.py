from __future__ import annotations

from roboarchitect.harness import StepRequest, Tools, VLARequest
from roboarchitect.protocol import ActionChunk


HARNESS_CONFIG = {
    "required_capabilities": ("vla",),
    "optional_capabilities": ("language", "vision", "detection", "segmentation", "pointing", "grasp"),
}


class PolicyPassthrough:
    def reset(self, session_id: str) -> None:
        return None

    def act(self, request: StepRequest, tools: Tools) -> ActionChunk:
        action = tools.vla(
            VLARequest(
                request_id=request.request_id,
                session_id=request.session_id,
                observation=request.observation,
                instruction=request.observation.instruction,
                context=(),
                refresh=request.observation.step_index == 0,
            )
        )
        if action.horizon != 1 or action.execution_count != 1:
            raise RuntimeError("the policy must return exactly one action per step")
        return action


def create_harness() -> PolicyPassthrough:
    return PolicyPassthrough()
