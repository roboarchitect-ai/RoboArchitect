from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .common import LIBERO_DELTA_ACTION_SPEC, Adapter, PolicyServer, checked, deterministic_seed, flipped, libero_state


MODEL_HORIZON = 10
EXECUTION_HORIZON = 5
DENOISE_STEPS = 10


class LiberoAdapter(Adapter):
    action_spec = LIBERO_DELTA_ACTION_SPEC


class RLinfPi05(PolicyServer):
    ADAPTERS = {"libero": LiberoAdapter}

    def __init__(self, checkpoint: str, repository: Path | None) -> None:
        super().__init__()
        if repository is None:
            raise FileNotFoundError("RLinf-pi0.5 needs the folder of the RLinf repository (repositories.rlinf in paths.json)")
        sys.path.insert(0, str(Path(repository).resolve()))
        import openpi.transforms as transforms
        import torch
        from huggingface_hub import snapshot_download
        from openpi.policies.policy import Policy
        from openpi.training import checkpoints
        from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT
        from toolkits.standalone_eval_scripts.openpi import load_pytorch

        self.torch = torch
        folder = snapshot_download(checkpoint, local_files_only=True)
        config = _CONFIGS_DICT["pi05_libero"]
        model = load_pytorch(config, os.path.join(folder, "model.safetensors"))
        model.sample_actions = torch.compile(type(model).sample_actions.__get__(model), mode="max-autotune-no-cudagraphs")
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        data = config.data.create(config.assets_dirs, config.model)
        norm_stats = checkpoints.load_norm_stats(folder, data.asset_id)
        self.policy = Policy(
            model,
            transforms=[
                transforms.InjectDefaultPrompt(None),
                *data.data_transforms.inputs,
                transforms.Normalize(norm_stats, use_quantiles=data.use_quantile_norm),
                *data.model_transforms.inputs,
            ],
            output_transforms=[
                *data.model_transforms.outputs,
                transforms.Unnormalize(norm_stats, use_quantiles=data.use_quantile_norm),
                *data.data_transforms.outputs,
            ],
            sample_kwargs={"num_steps": DENOISE_STEPS},
            metadata=config.policy_metadata,
            is_pytorch=True,
            pytorch_device="cuda:0",
        )

    def infer(self, session, request) -> tuple[np.ndarray, np.ndarray]:
        observation = request.observation
        inputs = {
            "observation/image": flipped(observation, "main"),
            "observation/wrist_image": flipped(observation, "wrist"),
            "observation/state": libero_state(observation),
            "prompt": request.instruction,
        }
        seed = deterministic_seed(session.policy_seed, session.call_index)
        noise = np.random.default_rng(seed).standard_normal((MODEL_HORIZON, 32)).astype(np.float32)
        with self.torch.inference_mode():
            actions = checked(self.policy.infer(inputs, noise=noise)["actions"], (MODEL_HORIZON, 7))
        selected = np.ascontiguousarray(np.clip(actions[:EXECUTION_HORIZON], -1.0, 1.0), dtype=np.float32)
        return selected, selected
