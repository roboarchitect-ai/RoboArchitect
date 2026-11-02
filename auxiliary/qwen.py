from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

from roboarchitect.harness import LanguageRequest, TextResult, VisionRequest
from roboarchitect.protocol import StrictSchemaError

from .common import Backend, pil


def _generation_options(model: Any, max_new_tokens: int, temperature: float = 0.0) -> dict[str, Any]:
    config = copy.deepcopy(model.generation_config)
    if hasattr(config, "_from_model_config"):
        config._from_model_config = False
    config.do_sample = temperature > 0
    if temperature > 0:
        config.temperature = temperature
    else:
        for name in ("temperature", "top_p", "top_k", "min_p", "typical_p"):
            if hasattr(config, name):
                setattr(config, name, None)
    return {"max_new_tokens": max_new_tokens, "generation_config": config, "use_model_defaults": False}


class QwenLanguage(Backend):
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._processor = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        ).eval()

    def invoke(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if operation != "generate":
            raise StrictSchemaError(f"{self.name}: unsupported operation {operation}")
        request = LanguageRequest.from_mapping(payload)
        prompt = "\n".join([*request.context, request.instruction])
        messages = [{"role": "user", "content": prompt}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self._processor(text, return_tensors="pt").to(self._model.device)
        output = self._model.generate(
            **inputs,
            **_generation_options(self._model, request.max_tokens, request.temperature),
        )
        generated = output[:, inputs.input_ids.shape[1] :]
        return TextResult(self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()).to_mapping()

    def warm_up(self) -> None:
        self.invoke("generate", LanguageRequest("Reply OK.", max_tokens=2).to_mapping())


class QwenVision(Backend):
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        kwargs = {"dtype": torch.bfloat16, "device_map": self.device, "trust_remote_code": True}
        try:
            self._model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs).eval()
        except (ValueError, ImportError):
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs).eval()

    def invoke(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if operation != "describe":
            raise StrictSchemaError(f"{self.name}: unsupported operation {operation}")
        request = VisionRequest.from_mapping(payload)
        images = [pil(image) for _, image in sorted(request.images.items())]
        text = "\n".join([*request.context, request.instruction])
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=images, return_tensors="pt").to(self._model.device)
        output = self._model.generate(**inputs, **_generation_options(self._model, request.max_tokens))
        input_length = inputs["input_ids"].shape[1]
        generated = output[:, input_length:]
        result = self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return TextResult(result).to_mapping()

    def warm_up(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        self.invoke("describe", VisionRequest("Reply OK.", {"warm_up": image}, max_tokens=2).to_mapping())
