from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image

from roboarchitect.protocol import StrictSchemaError


@dataclass(frozen=True)
class ModelInfo:
    capability: str
    model_id: str
    python: str
    on_second_gpu: bool


MODELS = {
    "qwen-language": ModelInfo("language", "Qwen/Qwen3-30B-A3B-Instruct-2507", "language", False),
    "qwen-vision": ModelInfo("vision", "Qwen/Qwen3-VL-8B-Instruct", "vision", True),
    "molmo2-vision": ModelInfo("vision", "allenai/Molmo2-8B", "molmo2", True),
    "molmo2-pointing": ModelInfo("pointing", "allenai/Molmo2-8B", "molmo2", True),
    "grounding-dino": ModelInfo("detection", "IDEA-Research/grounding-dino-base", "grounding_dino", True),
    "sam3": ModelInfo("segmentation", "AEmotionStudio/sam3", "sam3", True),
    "graspgen": ModelInfo("grasp", "adithyamurali/GraspGenModels", "graspgen", False),
}


class Backend(ABC):
    def __init__(self, name: str, device: str) -> None:
        self.name = name
        self.model_id = MODELS[name].model_id
        self.capability = MODELS[name].capability
        self.device = device

    def load(self) -> None:
        return

    def warm_up(self) -> None:
        return

    @abstractmethod
    def invoke(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError


def pil(value: Any, path: str = "image") -> Image.Image:
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise StrictSchemaError(f"{path}: expected uint8 RGB array")
    return Image.fromarray(value, mode="RGB")
