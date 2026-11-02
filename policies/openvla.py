from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from roboarchitect.protocol import StrictSchemaError

from .common import LIBERO_DELTA_ACTION_SPEC, Adapter, PolicyServer, checked, flipped


IMAGE_SIZE = 224
CENTER_CROP_SCALE = 0.9


class LiberoAdapter(Adapter):
    action_spec = LIBERO_DELTA_ACTION_SPEC


class OpenVLA(PolicyServer):
    ADAPTERS = {"libero": LiberoAdapter}

    def __init__(self, checkpoint: str, repository: Path | None) -> None:
        super().__init__()
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
        import tensorflow
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.tf = tensorflow
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            checkpoint,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model = model.to(self.device).eval()

    def next_action(self, session, request) -> np.ndarray:
        if request.observation.step_index != session.call_index:
            raise StrictSchemaError("OpenVLA observation steps must be consecutive")
        image = Image.fromarray(self.crop(self.resize(flipped(request.observation, "main")))).convert("RGB")
        prompt = f"In: What action should the robot take to {request.instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=self.torch.bfloat16)
        with self.torch.inference_mode():
            raw = checked(np.asarray(self.model.predict_action(**inputs, do_sample=False)).reshape(1, -1), (1, 7))
        action = raw.copy()
        action[0, 6] = -1.0 if 2.0 * float(raw[0, 6]) - 1.0 > 0.0 else 1.0
        session.call_index += 1
        return action

    def resize(self, rgb: np.ndarray) -> np.ndarray:
        tf = self.tf
        encoded = tf.image.encode_jpeg(tf.convert_to_tensor(rgb))
        decoded = tf.io.decode_image(encoded, expand_animations=False, dtype=tf.uint8)
        resized = tf.image.resize(decoded, (IMAGE_SIZE, IMAGE_SIZE), method="lanczos3", antialias=True)
        return tf.cast(tf.clip_by_value(tf.round(resized), 0, 255), tf.uint8).numpy()

    def crop(self, rgb: np.ndarray) -> np.ndarray:
        tf = self.tf
        floats = tf.image.convert_image_dtype(tf.expand_dims(tf.convert_to_tensor(rgb), 0), tf.float32)
        side = float(np.sqrt(CENTER_CROP_SCALE))
        offset = (1.0 - side) / 2.0
        boxes = tf.constant([[offset, offset, offset + side, offset + side]], dtype=tf.float32)
        cropped = tf.clip_by_value(tf.image.crop_and_resize(floats, boxes, tf.range(1), (IMAGE_SIZE, IMAGE_SIZE)), 0.0, 1.0)
        return tf.image.convert_image_dtype(cropped, tf.uint8, saturate=True)[0].numpy()
