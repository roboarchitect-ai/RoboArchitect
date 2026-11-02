from __future__ import annotations

from pathlib import Path

import numpy as np

from roboarchitect.protocol import StrictSchemaError

from .common import LIBERO_DELTA_ACTION_SPEC, Adapter, PolicyServer, checked, flipped, libero_state


MODEL_HORIZON = 10


class LiberoAdapter(Adapter):
    action_spec = LIBERO_DELTA_ACTION_SPEC


class MolmoAct2(PolicyServer):
    ADAPTERS = {"libero": LiberoAdapter}

    def __init__(self, checkpoint: str, repository: Path | None) -> None:
        super().__init__()
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = torch.device("cuda:0")
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)
        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, trust_remote_code=True, dtype=torch.float32, low_cpu_mem_usage=True
        )
        self.model = model.to(self.device).eval()

    def start_session(self, session) -> None:
        session.generator = self.torch.Generator(device=self.device).manual_seed(session.policy_seed)
        session.depth_cache = None

    def infer(self, session, request) -> tuple[np.ndarray, np.ndarray]:
        observation = request.observation
        with self.torch.inference_mode():
            output = self.model.predict_action(
                processor=self.processor,
                images=[flipped(observation, "main"), flipped(observation, "wrist")],
                task=request.instruction,
                state=libero_state(observation),
                norm_tag="libero",
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                enable_adaptive_depth=True,
                depth_cache=session.depth_cache,
                generator=session.generator,
                normalize_language=True,
                enable_cuda_graph=True,
            )
        actions = output.actions.detach().to("cpu", dtype=self.torch.float32).numpy()
        actions = checked(actions[0] if actions.ndim == 3 and actions.shape[0] == 1 else actions, (MODEL_HORIZON, 7))
        if np.any(actions < -1.0) or np.any(actions > 1.0):
            raise StrictSchemaError("MolmoAct2 returned actions outside [-1, 1]")
        session.depth_cache = output.depth_cache
        return actions, actions
