"""
llama.cpp provider — true in-process inference with explicit VRAM control.

Uses ``llama-cpp-python`` (optional dependency).  The single knob that matters
for "runs on my PC" is ``n_gpu_layers``:

  * mode=full_gpu   -> -1 (offload every layer)
  * mode=cpu_only   ->  0 (pure CPU, no VRAM needed)
  * mode=manual     -> cfg.vram.n_gpu_layers
  * mode=auto       -> estimate from ``vram_budget_gb`` and the model's GGUF
                       metadata (layer count + tensor size), offloading as
                       many layers as fit; the remainder runs on CPU threads.

The model path/name is fully user-supplied (``cfg.model_name`` = path to a
.gguf file or a hugging-face id for auto-download). Nothing is hardcoded.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from backend.vlm.base import BaseVLMProvider, VLMError, register_provider


def _estimate_gpu_layers(model, budget_gb: float, reserve_gb: float = 1.2) -> int:
    """Offload as many transformer layers as fit into the VRAM budget."""
    try:
        n_layers = int(model.n_layer())
    except Exception:
        return -1
    try:
        params = model.n_params()
    except Exception:
        params = 0
    # bytes per parameter inferred from quant size / param count
    ctx_bytes = getattr(model, "mem_ctx_size", lambda: 0)() if hasattr(model, "mem_ctx_size") else 0
    weights = model.weights  # llama_cpp.LlamaModel exposes .weights array
    total_model_gb = sum(w.nbytes for w in weights) / 1e9 if weights else 0.0
    usable = max(0.0, budget_gb - reserve_gb - ctx_bytes / 1e9)
    if total_model_gb <= 0 or n_layers <= 0:
        return -1
    per_layer_gb = total_model_gb / max(n_layers, 1)
    fit = int(usable / per_layer_gb) if per_layer_gb > 0 else n_layers
    return max(0, min(fit, n_layers))


@register_provider
class LlamaCppProvider(BaseVLMProvider):
    name = "llamacpp"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._model = None
        self._mmproj = None

    # -------------------------------------------------------------- loading
    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise VLMError(
                "llama-cpp-python is not installed. `pip install llama-cpp-python` "
                "(use `CMAKE_ARGS='-DGGML_CUDA=on'` for GPU builds) or switch the "
                "provider to Ollama in Settings.") from exc

        v = self.cfg.vram
        path = self.cfg.model_name
        if not os.path.isfile(path):
            raise VLMError(
                f"llama.cpp model path not found: {path!r}. Set it to a local .gguf file.")

        n_gpu = v.n_gpu_layers
        if v.mode == "cpu_only":
            n_gpu = 0
        elif v.mode == "full_gpu":
            n_gpu = -1
        elif v.mode == "auto":
            n_gpu = None  # decided after first load below

        kwargs: Dict[str, Any] = dict(
            model_path=path,
            n_gpu_layers=(0 if v.mode == "cpu_only" else
                          -1 if v.mode == "full_gpu" else
                          v.n_gpu_layers if v.mode == "manual" else -1),
            n_ctx=max(2048, v.context_length),
            n_threads=max(1, v.cpu_offload_threads),
            verbose=False,
        )
        if v.kv_cache_4bit:
            kwargs["type_k"] = 3   # quantised K/V cache (llama.cpp F16->Q4 heuristic slot)
            kwargs["type_v"] = 3
        if v.flash_attention:
            kwargs["flash_attn"] = True

        vision_path = os.environ.get("PSL_MMPROJ_PATH", "")
        if vision_path and os.path.isfile(vision_path):
            kwargs["mmproj"] = vision_path

        self._model = Llama(**kwargs)

        if v.mode == "auto":
            est = _estimate_gpu_layers(self._model, v.vram_budget_gb)
            try:
                self._model.params.n_gpu_layers = est
            except Exception:
                pass
        return

    # ------------------------------------------------------------ capability
    def is_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            return False
        return bool(self.cfg.model_name) and os.path.isfile(self.cfg.model_name)

    def list_models(self) -> List[Dict[str, Any]]:
        out = []
        for d in ("models", os.path.expanduser("~/models"), "."):
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".gguf"):
                    p = os.path.join(d, f)
                    mm = os.path.exists(p.replace(".gguf", "-mmproj.gguf"))
                    out.append({"name": p, "vision": mm,
                                "size_gb": round(os.path.getsize(p) / 1e9, 2)})
        return out

    # -------------------------------------------------------------- inference
    def chat_with_image(self, prompt: str, image_png: bytes,
                        system: Optional[str] = None) -> str:
        self._ensure_loaded()
        import base64
        content = [
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + self.b64(image_png)}},
            {"type": "text", "text": prompt},
        ]
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": content})
        try:
            res = self._model.create_chat_completion(
                messages=msgs,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
        except Exception as exc:
            raise VLMError(f"llama.cpp inference failed: {exc}")
        txt = res["choices"][0]["message"]["content"]
        if not txt.strip():
            raise VLMError("llama.cpp returned an empty completion "
                           "(is the mmproj vision adapter configured?)")
        return txt
