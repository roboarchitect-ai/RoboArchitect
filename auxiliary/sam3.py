from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from roboarchitect.harness import SegmentationRequest, SegmentationResult
from roboarchitect.protocol import StrictSchemaError

from .common import Backend, pil


MODEL_RESOLUTION = 1008
PROMPT_ROUNDOFF_TOLERANCE_PX = 0.01


def prepare_prompts(
    request: SegmentationRequest,
) -> tuple[tuple[tuple[float, float, float, float], ...], np.ndarray | None, np.ndarray | None]:
    height, width = request.image.shape[:2]
    boxes = []
    for x0, y0, x1, y1 in request.boxes_xyxy:
        tolerance = PROMPT_ROUNDOFF_TOLERANCE_PX
        if not (-tolerance <= x0 < x1 <= width + tolerance and -tolerance <= y0 < y1 <= height + tolerance):
            raise StrictSchemaError("segmentation_request.box: expected in-bounds xyxy")
        clipped = (
            min(float(width), max(0.0, x0)),
            min(float(height), max(0.0, y0)),
            min(float(width), max(0.0, x1)),
            min(float(height), max(0.0, y1)),
        )
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            raise StrictSchemaError("segmentation_request.box: expected nondegenerate xyxy")
        boxes.append(clipped)
    for x, y in request.points_xy:
        if not (0.0 <= x < width and 0.0 <= y < height):
            raise StrictSchemaError("segmentation_request.point: expected in-bounds xy")
    points = np.asarray(request.points_xy, dtype=np.float32) if request.points_xy else None
    labels = np.asarray(request.labels, dtype=np.int32) if request.points_xy else None
    return tuple(boxes), points, labels


class Sam3(Backend):
    def load(self) -> None:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        checkpoint = Path(hf_hub_download(repo_id=self.model_id, filename="sam3.safetensors", local_files_only=True))
        source_state = load_file(checkpoint, device="cpu")
        model = build_sam3_image_model(
            device="cpu",
            checkpoint_path=None,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=True,
            compile=False,
        )
        mapped_state = {
            (key.removeprefix("detector.") if key.startswith("detector.") else "inst_interactive_predictor.model." + key.removeprefix("tracker.")): value
            for key, value in source_state.items()
        }
        missing, unexpected = model.load_state_dict(mapped_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"SAM3 state mismatch: missing={len(missing)} {sorted(missing)[:20]}, unexpected={len(unexpected)} {sorted(unexpected)[:20]}")
        del source_state, mapped_state
        self._model = model.to(self.device).eval()
        self._processor = Sam3Processor(self._model, resolution=MODEL_RESOLUTION, device=self.device)
        self._lock = threading.Lock()

    def invoke(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if operation != "segment":
            raise StrictSchemaError(f"{self.name}: unsupported operation {operation}")
        import torch

        request = SegmentationRequest.from_mapping(payload)
        prepared_boxes, points, labels = prepare_prompts(request)
        boxes = prepared_boxes or (None,)
        masks_out: list[np.ndarray] = []
        scores_out: list[float] = []
        with self._lock, torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.device.startswith("cuda"),
        ):
            state = self._processor.set_image(pil(request.image))
            for box in boxes:
                masks, scores, _ = self._model.predict_inst(
                    state,
                    point_coords=points,
                    point_labels=labels,
                    box=None if box is None else np.asarray(box, dtype=np.float32),
                    multimask_output=True,
                    return_logits=False,
                    normalize_coords=True,
                )
                scores = np.asarray(scores, dtype=np.float32)
                if masks.ndim != 3 or scores.shape != (masks.shape[0],) or not np.all(np.isfinite(scores)):
                    raise RuntimeError("SAM3 returned invalid mask scores")
                index = int(scores.argmax())
                score = float(scores[index])
                if not -1e-6 <= score <= 1.0 + 1e-6:
                    raise RuntimeError("SAM3 returned an out-of-range mask score")
                masks_out.append(np.asarray(masks[index], dtype=np.bool_))
                scores_out.append(min(1.0, max(0.0, score)))
        result = SegmentationResult(np.stack(masks_out, axis=0), np.asarray(scores_out, dtype=np.float32))
        return result.to_mapping()

    def warm_up(self) -> None:
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[32:96, 32:96] = 255
        result = SegmentationResult.from_mapping(
            self.invoke(
                "segment",
                SegmentationRequest(image, boxes_xyxy=((24.0, 24.0, 104.0, 104.0),)).to_mapping(),
            )
        )
        if result.masks.shape != (1, 128, 128):
            raise RuntimeError("SAM3 warm-up returned an invalid mask shape")
