"""
VLM provider abstraction.

Design goals (per product requirements):

* **No hardcoded model** – every field of :class:`VLMConfig` is user-editable
  through the UI / REST API and persisted to ``backend/vlm_config.json``.
* **Runs on a normal PC** – the default provider is Ollama (local server),
  with llama.cpp (in-process, explicit VRAM control) and LM Studio / any
  OpenAI-compatible local server as alternatives.
* **Vision-capable (VLM)** – providers only surface models that accept image
  input; images are sent base64-encoded alongside the prompt.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_PATH = Path(os.environ.get(
    "PSL_VLM_CONFIG",
    str(Path(__file__).resolve().parent.parent / "vlm_config.json")))


@dataclass
class VRAMSettings:
    """How much of the model lives on the GPU.

    Semantics per backend:
      * ollama   -> informational + used for auto layer math / warnings.
      * llamacpp -> ``n_gpu_layers`` computed from the budget (or manual).
      * openai   -> ignored (remote server manages its own memory).
    """

    mode: str = "auto"                 # auto | manual | cpu_only | full_gpu
    vram_budget_gb: float = 8.0        # usable VRAM for weights + context
    n_gpu_layers: int = -1             # manual override (-1 = all layers)
    context_length: int = 4096
    kv_cache_4bit: bool = False        # quantised KV cache saves VRAM
    flash_attention: bool = True       # paged/FA attention if supported
    cpu_offload_threads: int = 4       # threads for the CPU-resident part

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VLMConfig:
    """Fully user-configurable description of *which* VLM to talk to."""

    provider: str = "ollama"           # ollama | llamacpp | openai
    base_url: str = "http://localhost:11434"
    model_name: str = ""               # NEVER hardcoded – chosen by the user
    api_key: str = ""                  # only for openai-type servers
    temperature: float = 0.2
    max_tokens: int = 900
    timeout_s: int = 120
    num_parallel: int = 1              # ollama: concurrent slots (VRAM cost!)
    keep_alive: str = "5m"             # ollama: unload model after idle
    vram: VRAMSettings = field(default_factory=VRAMSettings)

    # ------------------------------------------------------------------ io
    def as_dict(self) -> Dict[str, Any]:
        d = dict(asdict(self))
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VLMConfig":
        vr = d.pop("vram", {}) or {}
        known_v = set(VRAMSettings.__dataclass_fields__)
        vr = {k: v for k, v in vr.items() if k in known_v}
        known_c = set(cls.__dataclass_fields__)
        d = {k: v for k, v in d.items() if k in known_c}
        return cls(vram=VRAMSettings(**vr), **d)

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "VLMConfig":
        if path.exists():
            try:
                return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
        return cls()  # defaults only — model stays empty until the user picks


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class VLMError(RuntimeError):
    pass


class BaseVLMProvider:
    """Interface implemented by every backend."""

    name = "base"

    def __init__(self, cfg: VLMConfig):
        self.cfg = cfg

    # -- capability probes --------------------------------------------------
    def list_models(self) -> List[Dict[str, Any]]:
        """Return [{'name':..., 'vision': bool, 'size_gb': float|None}]"""
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError

    # -- inference ------------------------------------------------------------
    def chat_with_image(self, prompt: str, image_png: bytes,
                        system: Optional[str] = None) -> str:
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def b64(image_png: bytes) -> str:
        return base64.b64encode(image_png).decode("ascii")

    @staticmethod
    def extract_json(text: str) -> Dict[str, Any]:
        """Pull the first JSON object out of an LLM reply (handles fences)."""
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        candidate = fence.group(1) if fence else None
        if candidate is None:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                candidate = text[start:end + 1]
        if candidate is None:
            raise VLMError(f"No JSON object found in VLM reply: {text[:200]!r}")
        return json.loads(candidate)


_REGISTRY: Dict[str, type] = {}


def register_provider(cls: type) -> type:
    _REGISTRY[cls.name] = cls
    return cls


def get_provider(cfg: VLMConfig) -> BaseVLMProvider:
    # import here so each module can self-register
    from backend.vlm import ollama, llamacpp, openai_compat  # noqa: F401
    key = cfg.provider.lower().strip()
    if key not in _REGISTRY:
        raise VLMError(
            f"Unknown VLM provider '{cfg.provider}'. "
            f"Choose one of: {', '.join(sorted(_REGISTRY))}"
        )
    if not cfg.model_name:
        raise VLMError(
            "No VLM model configured. Pick a vision model in Settings "
            "(nothing is hardcoded on purpose)."
        )
    return _REGISTRY[key](cfg)
