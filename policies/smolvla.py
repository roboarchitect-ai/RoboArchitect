from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import LIBERO_DELTA_ACTION_SPEC, Adapter, PolicyServer, checked, deterministic_seed, flipped, libero_state


MODEL_HORIZON = 50
EXECUTION_HORIZON = 8


class LiberoAdapter(Adapter):
    action_spec = LIBERO_DELTA_ACTION_SPEC


class SmolVLA(PolicyServer):
    ADAPTERS = {"libero": LiberoAdapter}

    def __init__(self, checkpoint: str, repository: Path | None) -> None:
        super().__init__()
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.utils import prepare_observation_for_inference
        from lerobot.processor.converters import (
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )
        from lerobot.processor.pipeline import PolicyProcessorPipeline

        self.torch = torch
        self.device = torch.device("cuda:0")
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.device = str(self.device)
        self.model = SmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=True)
        self.preprocess = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=checkpoint,
            config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": str(self.device)}},
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        self.postprocess = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=checkpoint,
            config_filename="policy_postprocessor.json",
            overrides={},
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        self.prepare = prepare_observation_for_inference

    def infer(self, session, request) -> tuple[np.ndarray, np.ndarray]:
        observation = request.observation
        frame = {
            "observation.images.image": flipped(observation, "main"),
            "observation.images.wrist_image": flipped(observation, "wrist"),
            "observation.state": libero_state(observation),
        }
        frame = self.preprocess(self.prepare(frame, self.device, task=request.instruction, robot_type=""))
        torch = self.torch
        with torch.random.fork_rng(devices=[self.device.index]):
            seed = deterministic_seed(session.policy_seed, session.call_index)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            with torch.inference_mode():
                actions = self.model.predict_action_chunk(frame)
        actions = checked(self.postprocess(actions).squeeze(0).detach().to("cpu").numpy(), (MODEL_HORIZON, 7))
        selected = actions[:EXECUTION_HORIZON].copy()
        selected[:, :6] = np.clip(selected[:, :6], -1.0, 1.0)
        selected[:, 6] = np.where(selected[:, 6] < 0.5, 1.0, -1.0)
        return selected, selected
