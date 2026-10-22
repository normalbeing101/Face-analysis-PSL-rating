"""
Ollama VLM provider (default local backend).

Talks to a locally-running ``ollama serve`` over its REST API.  The model is
*never* hardcoded — :meth:`list_models` enumerates whatever the user pulled,
flags the vision-capable ones, and the chat call uses ``cfg.model_name``.

VRAM handling for Ollama:
  * ``num_gpu`` (layers on GPU) and ``num_ctx`` are passed in ``options`` so
    the user's VRAM budget is respected even when the model is bigger than
    the card (partial offload to CPU happens automatically by ollama).
  * ``keep_alive`` controls when the model is unloaded from VRAM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from backend.vlm.base import BaseVLMProvider, VLMError, register_provider

#: Families that accept image input in current ollama registries.  This is a
#: *capability filter*, not a model choice — the user still picks the model.
_VISION_HINTS = (
    "llava", "llama3.2-vision", "vision", "moondream", "minicpm-v",
    "qwen-vl", "qwen2-vl", "qwen2.5-vl", "bakuvla", "granite3.2-vision",
    "gemma3", "pixtral", "phi3.5", "phi4-vision", "glm-4v", "aya-vision",
    "snowflake-arctic-vision", "nvilt", "otter", "molmo",
)


@register_provider
class OllamaProvider(BaseVLMProvider):
    name = "ollama"

    # ------------------------------------------------------------------ util
    def _url(self, path: str) -> str:
        return self.cfg.base_url.rstrip("/") + path

    def _options(self) -> Dict[str, Any]:
        v = self.cfg.vram
        opts: Dict[str, Any] = {
            "temperature": self.cfg.temperature,
            "num_ctx": max(2048, v.context_length),
            "num_thread": max(1, v.cpu_offload_threads),
        }
        if v.mode == "cpu_only":
            opts["num_gpu"] = 0
        elif v.mode == "full_gpu":
            opts["num_gpu"] = -1          # ollama: -1 => all layers on GPU
        elif v.mode == "manual" and v.n_gpu_layers >= 0:
            opts["num_gpu"] = v.n_gpu_layers
        # auto mode: let ollama decide based on available VRAM
        return opts

    # ------------------------------------------------------------ capability
    def is_available(self) -> bool:
        try:
            r = requests.get(self._url("/api/tags"), timeout=4)
            return r.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _looks_vision(model_name: str) -> bool:
        n = model_name.lower()
        return any(h in n for h in _VISION_HINTS)

    def list_models(self) -> List[Dict[str, Any]]:
        try:
            r = requests.get(self._url("/api/tags"), timeout=6)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            raise VLMError(f"Cannot reach Ollama at {self.cfg.base_url}: {exc}")
        out = []
        for m in data.get("models", []):
            name = m.get("name", "")
            size_gb = round(float(m.get("size", 0)) / 1e9, 2) or None
            # trust the name heuristic; unknown families default to True so
            # fine-tuned/custom vision tags remain selectable.
            vision = self._looks_vision(name)
            out.append({"name": name, "vision": vision, "size_gb": size_gb})
        out.sort(key=lambda x: (not x["vision"], x["name"]))
        return out

    # -------------------------------------------------------------- inference
    def chat_with_image(self, prompt: str, image_png: bytes,
                        system: Optional[str] = None) -> str:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": prompt,
            "images": [self.b64(image_png)],
        })
        payload = {
            "model": self.cfg.model_name,
            "messages": messages,
            "stream": False,
            "format": "json",
            "keep_alive": self.cfg.keep_alive,
            "options": self._options(),
        }
        try:
            r = requests.post(self._url("/api/chat"), json=payload,
                              timeout=self.cfg.timeout_s)
        except requests.Timeout:
            raise VLMError(
                f"Ollama timed out after {self.cfg.timeout_s}s — lower the "
                "context length or offload fewer layers in VRAM settings.")
        except Exception as exc:
            raise VLMError(f"Ollama request failed: {exc}")
        if r.status_code != 200:
            raise VLMError(f"Ollama error {r.status_code}: {r.text[:300]}")
        reply = r.json().get("message", {}).get("content", "")
        if not reply.strip():
            raise VLMError("Ollama returned an empty completion.")
        return reply
