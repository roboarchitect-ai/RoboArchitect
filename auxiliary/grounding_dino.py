from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from roboarchitect.harness import Detection, DetectionRequest, DetectionResult
from roboarchitect.protocol import StrictSchemaError

from .common import Backend, pil


class GroundingDino(Backend):
    def load(self) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device).eval()

    def invoke(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if operation != "detect":
            raise StrictSchemaError(f"{self.name}: unsupported operation {operation}")
        import torch

        request = DetectionRequest.from_mapping(payload)
        image = pil(request.image)
        query = request.query.lower().strip().rstrip(".") + "."
        inputs = self._processor(images=image, text=query, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=request.threshold,
            text_threshold=request.threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        labels = results.get("text_labels")
        if labels is None:
            labels = results.get("labels", [])
        detections = tuple(
            Detection(str(label), float(score), tuple(float(x) for x in box.tolist()))
            for box, score, label in zip(results["boxes"], results["scores"], labels, strict=True)
        )
        return DetectionResult(detections).to_mapping()

    def warm_up(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        self.invoke("detect", DetectionRequest(image, "object").to_mapping())
