"""
OpenAI-compatible provider — LM Studio, llama.cpp server, vLLM, koboldcpp,
or any local/remote ``/v1/chat/completions`` endpoint that supports image_url
content parts.  Model name and base URL are fully user-configured.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from backend.vlm.base import BaseVLMProvider, VLMError, register_provider


@register_provider
class OpenAICompatProvider(BaseVLMProvider):
    name = "openai"

    def _url(self, path: str) -> str:
        base = self.cfg.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        return base + path

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    # ------------------------------------------------------------ capability
    def is_available(self) -> bool:
        try:
            r = requests.get(self._url("/models"), headers=self._headers(),
                             timeout=4)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        try:
            r = requests.get(self._url("/models"), headers=self._headers(),
                             timeout=8)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            raise VLMError(f"Cannot reach model server at {self.cfg.base_url}: {exc}")
        out = []
        for m in data.get("data", []):
            name = m.get("id", "")
            low = name.lower()
            vision = any(k in low for k in ("vision", "vl", "llava", "moondream",
                                            "minicpm", "gemma3", "pixtral",
                                            "qwen2.5-v", "phi3.5", "bakuvla"))
            out.append({"name": name, "vision": vision, "size_gb": None})
        out.sort(key=lambda x: (not x["vision"], x["name"]))
        return out

    # -------------------------------------------------------------- inference
    def chat_with_image(self, prompt: str, image_png: bytes,
                        system: Optional[str] = None) -> str:
        content: List[Dict[str, Any]] = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{self.b64(image_png)}"}},
            {"type": "text", "text": prompt},
        ]
        msgs: List[Dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": content})

        payload = {
            "model": self.cfg.model_name,
            "messages": msgs,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        try:
            r = requests.post(self._url("/chat/completions"),
                              headers=self._headers(), json=payload,
                              timeout=self.cfg.timeout_s)
        except requests.Timeout:
            raise VLMError(f"Model server timed out after {self.cfg.timeout_s}s.")
        except Exception as exc:
            raise VLMError(f"Model-server request failed: {exc}")
        if r.status_code != 200:
            raise VLMError(f"Model-server error {r.status_code}: {r.text[:300]}")
        txt = r.json()["choices"][0]["message"]["content"]
        if not txt.strip():
            raise VLMError("Model server returned an empty completion.")
        return txt
