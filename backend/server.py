"""
PSL Studio backend — FastAPI server.

Run from the repo root:

    pip install -r backend/requirements.txt
    python -m backend.server            # http://127.0.0.1:8321

Endpoints
---------
GET  /                     frontend (white-themed single page)
GET  /api/presets          harmony presets for the option picker
GET  /api/vlm/config       current VLM configuration (nothing hardcoded)
PUT  /api/vlm/config       save a new configuration (+VRAM settings)
GET  /api/vlm/models       list models reported by the configured provider
POST /api/vlm/test         ping the provider with the chosen model
POST /api/analyze          image -> landmarks -> PSL math report (+VLM advice)

The math engine (``psl``) is deterministic and always runs; the VLM step is
optional per-request (``use_vlm`` flag) so the UI works even before a model
is downloaded.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from psl import analyze, PSLOptions  # noqa: E402
from psl.classification import list_harmony_presets  # noqa: E402
from backend import vision  # noqa: E402
from backend.advisor import advise  # noqa: E402
from backend.vlm.base import VLMConfig, VLMError, get_provider  # noqa: E402

app = FastAPI(title="PSL Studio", version="2.1.0")

FRONTEND_DIR = ROOT / "frontend"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class VRAMPayload(BaseModel):
    mode: str = "auto"
    vram_budget_gb: float = 8.0
    n_gpu_layers: int = -1
    context_length: int = 4096
    kv_cache_4bit: bool = False
    flash_attention: bool = True
    cpu_offload_threads: int = 4


class VLMConfigPayload(BaseModel):
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model_name: str = ""
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 900
    timeout_s: int = 120
    num_parallel: int = 1
    keep_alive: str = "5m"
    vram: VRAMPayload = Field(default_factory=VRAMPayload)


class AnalyzeRequest(BaseModel):
    image_b64: str                                   # data-url or raw base64
    preset: str = "harmony"
    band_tolerance_scale: float = 1.0
    ideal_table: Optional[str] = None
    include_symmetry: bool = True
    include_golden_checks: bool = True
    use_vlm: bool = True
    question: Optional[str] = None                   # free-form ask to the VLM


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/style.css")
def style():
    return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")


@app.get("/app.js")
def script():
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


# ---------------------------------------------------------------------------
# Presets / options
# ---------------------------------------------------------------------------

@app.get("/api/presets")
def presets():
    return {"presets": list_harmony_presets()}


# ---------------------------------------------------------------------------
# VLM configuration (fully user-driven — no hardcoded model)
# ---------------------------------------------------------------------------

@app.get("/api/vlm/config")
def get_vlm_config() -> Dict[str, Any]:
    cfg = VLMConfig.load()
    d = cfg.as_dict()
    d["api_key"] = "***set***" if d.get("api_key") else ""
    return d


@app.put("/api/vlm/config")
def put_vlm_config(payload: VLMConfigPayload) -> Dict[str, Any]:
    old = VLMConfig.load()
    d = payload.model_dump()
    if d.get("api_key") in ("", "***set***"):
        d["api_key"] = old.api_key          # keep stored key unless replaced
    try:
        cfg = VLMConfig.from_dict(d)
    except Exception as exc:
        raise HTTPException(400, f"Invalid configuration: {exc}")
    if cfg.provider not in ("ollama", "llamacpp", "openai"):
        raise HTTPException(400, "provider must be ollama | llamacpp | openai")
    if cfg.vram.mode not in ("auto", "manual", "cpu_only", "full_gpu"):
        raise HTTPException(400, "vram.mode must be auto | manual | cpu_only | full_gpu")
    cfg.save()
    return {"saved": True, "config": redact(cfg.as_dict())}


def redact(d: Dict[str, Any]) -> Dict[str, Any]:
    if d.get("api_key"):
        d = dict(d); d["api_key"] = "***set***"
    return d


@app.get("/api/vlm/models")
def vlm_models():
    cfg = VLMConfig.load()
    try:
        provider = get_provider_for_listing(cfg)
    except VLMError as exc:
        raise HTTPException(400, str(exc))
    try:
        models = provider.list_models()
    except VLMError as exc:
        raise HTTPException(502, str(exc))
    return {"provider": cfg.provider, "base_url": cfg.base_url,
            "reachable": provider.is_available(), "models": models}


def get_provider_for_listing(cfg: VLMConfig):
    """Like get_provider but tolerates an empty model_name (listing only)."""
    from backend.vlm import ollama, llamacpp, openai_compat  # noqa: F401
    from backend.vlm.base import _REGISTRY
    key = cfg.provider.lower().strip()
    if key not in _REGISTRY:
        raise VLMError(f"Unknown provider '{cfg.provider}'")
    return _REGISTRY[key](cfg)


class TestRequest(BaseModel):
    prompt: str = "Describe this image in one short sentence."


@app.post("/api/vlm/test")
def vlm_test(payload: Optional[TestRequest] = None):
    cfg = VLMConfig.load()
    provider = get_provider(cfg)             # raises helpful VLMError if unset
    if not provider.is_available():
        raise HTTPException(502,
                            f"Provider '{cfg.provider}' is not reachable at {cfg.base_url}. "
                            "Is Ollama/LM Studio running?")
    # 1x1 white png — proves the endpoint + vision pipeline without a real face
    tiny_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
    try:
        reply = provider.chat_with_image(payload.prompt if payload else
                                         "Reply with the single word: OK", tiny_png)
    except VLMError as exc:
        raise HTTPException(502, str(exc))
    return {"ok": True, "model": cfg.model_name, "reply": reply[:400]}


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------

@app.post("/api/analyze")
def analyze_image(req: AnalyzeRequest) -> Dict[str, Any]:
    # -- decode image ------------------------------------------------------
    b64 = req.image_b64
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "image_b64 is not valid base64.")

    # -- landmarks ----------------------------------------------------------
    try:
        det = vision.detect_face(img_bytes)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Landmark extraction failed: {exc}")
    if not det.found:
        return {"ok": False, "error": det.message or "No face detected."}

    def _run_analysis(lms):
        opts = PSLOptions(
            preset=req.preset,
            ideal_table=req.ideal_table,
            band_tolerance_scale=req.band_tolerance_scale,
            include_symmetry=req.include_symmetry,
            include_golden_checks=req.include_golden_checks,
        )
        return analyze(lms, opts)

    # -- deterministic PSL math --------------------------------------------
    try:
        result = _run_analysis(det.landmarks)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except ZeroDivisionError:
        # Degenerate landmark geometry (e.g. a stylised/illustrated face that
        # MediaPipe latched onto). Retry once on a smoothed copy, which is a
        # well-known trick to make detectors fit synthetic faces sanely.
        try:
            import io as _io
            from PIL import Image, ImageFilter
            im = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
            im2 = im.filter(ImageFilter.GaussianBlur(max(1, im.width // 160)))
            b2 = _io.BytesIO(); im2.save(b2, format="PNG")
            det2 = vision.detect_face(b2.getvalue())
            if det2.found:
                det = det2
                result = _run_analysis(det2.landmarks)
            else:
                raise
        except ZeroDivisionError:
            return {"ok": False,
                    "error": "The detected face geometry is degenerate "
                             "(stylised or partial face). Try a natural photo."}
    report = result.as_dict()

    overlay_b64 = base64.b64encode(det.overlay_png).decode("ascii")

    response: Dict[str, Any] = {
        "ok": True,
        "report": report,
        "detection": vision.landmarks_to_payload(det),
        "overlay_png_b64": overlay_b64,
    }

    # -- optional VLM advisory pass ----------------------------------------
    if req.use_vlm:
        cfg = VLMConfig.load()
        try:
            provider = get_provider(cfg)
            if not provider.is_available():
                raise VLMError(
                    f"Provider '{cfg.provider}' unreachable at {cfg.base_url} — "
                    "start your local server or disable AI advice.")
            response["advice"] = advise(provider, report, det.overlay_png,
                                         extra_question=req.question)
        except VLMError as exc:
            response["advice"] = None
            response["advice_error"] = str(exc)

    return response


@app.get("/api/health")
def health():
    cfg = VLMConfig.load()
    ok = False
    try:
        ok = get_provider_for_listing(cfg).is_available()
    except Exception:
        pass
    return {"engine": "psl", "version": "2.1",
            "vlm_provider": cfg.provider, "vlm_model": cfg.model_name or None,
            "vlm_reachable": ok}


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8321)


if __name__ == "__main__":
    main()
