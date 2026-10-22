"""
Face landmark extraction for the PSL backend.

Takes a raw image (bytes / path / numpy array), runs MediaPipe Face Mesh and
returns the canonical 468-point landmark set that the ``psl`` engine consumes,
together with an overlay image (face mesh drawn on the photo) for the UI.

MediaPipe is imported lazily so the web server still boots (with a clear
error at analyse-time) on machines where it is not installed.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

Point = Tuple[float, float]


@dataclass
class DetectionResult:
    found: bool
    landmarks: List[Point] = field(default_factory=list)   # normalised (x, y)
    landmarks_px: List[Tuple[int, int]] = field(default_factory=list)
    image_size: Tuple[int, int] = (0, 0)                   # (width, height)
    overlay_png: bytes = b""                               # mesh drawn on photo
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # x0,y0,x1,y1 px
    message: str = ""


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def _load_image(data):
    """Return an RGB numpy array from bytes / path / ndarray."""
    import numpy as np
    from PIL import Image

    if isinstance(data, np.ndarray):
        arr = data
        if arr.ndim == 3 and arr.shape[2] == 4:      # RGBA -> RGB
            arr = arr[:, :, :3]
        return arr

    if isinstance(data, (str, bytes)) and not isinstance(data, bytes) or \
       (isinstance(data, str)):
        img = Image.open(data)
    else:
        img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")
    return np.asarray(img)


# ---------------------------------------------------------------------------
# MediaPipe wrapper (Tasks API — works with mediapipe >= 0.10)
# ---------------------------------------------------------------------------

_MP_SINGLETON = None

#: Official model asset; downloaded automatically on first use.
_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/latest/face_landmarker.task")
_MODEL_DIR = Path(__file__).resolve().parent / "models"


def _model_path() -> Path:
    p = _MODEL_DIR / "face_landmarker.task"
    if not p.exists():
        import urllib.request
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".download")
        urllib.request.urlretrieve(_MODEL_URL, tmp)
        tmp.rename(p)
    return p


def _get_landmarker():
    global _MP_SINGLETON
    if _MP_SINGLETON is None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mtp
            from mediapipe.tasks.python import vision as mpv
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "mediapipe is not installed — run `pip install -r backend/requirements.txt`"
            ) from exc
        opts = mpv.FaceLandmarkerOptions(
            base_options=mtp.BaseOptions(model_asset_path=str(_model_path())),
            running_mode=mpv.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.4,
        )
        _MP_SINGLETON = (mp, mpv.FaceLandmarker.create_from_options(opts))
    return _MP_SINGLETON[1]


def detect_face(image_data, want_overlay: bool = True) -> DetectionResult:
    """Run MediaPipe Face Landmarker on *image_data* and package the result."""
    import numpy as np
    import mediapipe as mp_pkg

    landmarker = _get_landmarker()   # warms up singleton (may raise RuntimeError)

    rgb = _load_image(image_data)
    h, w = rgb.shape[:2]

    mp_img = mp_pkg.Image(image_format=mp_pkg.ImageFormat.SRGB,
                          data=np.ascontiguousarray(rgb))
    res = landmarker.detect(mp_img)
    if not res.face_landmarks:
        return DetectionResult(found=False, image_size=(w, h),
                               message="No face detected in the image.")

    lm_list = res.face_landmarks[0]
    n = len(lm_list)
    if n < 468:
        return DetectionResult(
            found=False, image_size=(w, h),
            message=f"Landmark count too low ({n}); expected 468+.",
        )

    norm: List[Point] = [(lm.x, lm.y) for lm in lm_list]
    # pad to 478 if refine produced exactly 468 (no iris indices)
    while len(norm) < 478:
        norm.append((float("nan"), float("nan")))

    px = [(int(x * w), int(y * h)) for (x, y) in norm]

    bbox_x0 = min(p[0] for p in px)
    bbox_y0 = min(p[1] for p in px)
    bbox_x1 = max(p[0] for p in px)
    bbox_y1 = max(p[1] for p in px)

    overlay = b""
    if want_overlay:
        overlay = _draw_overlay(rgb, px, w, h)

    return DetectionResult(
        found=True,
        landmarks=norm,
        landmarks_px=px,
        image_size=(w, h),
        overlay_png=overlay,
        face_bbox=(bbox_x0, bbox_y0, bbox_x1, bbox_y1),
    )


def _draw_overlay(rgb, px, w, h) -> bytes:
    """Draw a subtle white mesh + key-point overlay; returns PNG bytes."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(rgb).copy()
    draw = ImageDraw.Draw(img)

    def polyline(idx_ids, color, width):
        pts = [px[i] for i in idx_ids]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=width, joint="curve")

    OVAL = [101, 103, 67, 63, 105, 66, 107, 332, 280, 262, 249, 390, 373,
            374, 380, 381, 382, 384, 385, 386, 387, 388, 466, 247, 301, 299,
            297, 293, 274, 284, 291, 323, 54, 21, 16, 7, 231, 129, 113, 44,
            128, 24, 23, 130, 110, 24, 23, 130, 110, 157, 173, 133, 155, 154,
            153, 145, 144, 163, 7, 33, 132, 58, 172]
    LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61]
    NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 126, 437, 8, 2, 195, 197]
    EYE_L = [33, 160, 158, 133, 153, 144, 145, 155, 154, 153, 33]
    EYE_R = [263, 249, 390, 373, 374, 380, 381, 382, 263]
    BROW_L = [70, 63, 105, 66, 107]
    BROW_R = [300, 293, 276, 283, 282]

    WHITE = (255, 255, 255, 230)
    polyline(OVAL, WHITE, 2)
    polyline(LIPS, WHITE, 2)
    polyline(NOSE, WHITE, 2)
    polyline(EYE_L, WHITE, 2)
    polyline(EYE_R, WHITE, 2)
    polyline(BROW_L, WHITE, 2)
    polyline(BROW_R, WHITE, 2)

    # Key measurement points as dots
    KEY = [1, 2, 9, 13, 17, 168, 114, 152, 175, 33, 263, 133, 362,
           468, 473, 172, 397, 234, 454, 205, 425, 61, 291]
    for i in KEY:
        if i < len(px):
            x, y = px[i]
            r = 3
            draw.ellipse([x - r, y - r, x + r, y + r],
                         fill=(255, 255, 255), outline=(30, 30, 30), width=1)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Public helper used by the API
# ---------------------------------------------------------------------------

def landmarks_to_payload(det: DetectionResult) -> Dict[str, Any]:
    return {
        "found": det.found,
        "image_size": {"width": det.image_size[0], "height": det.image_size[1]},
        "landmark_count": len([p for p in det.landmarks if not math.isnan(p[0])]),
        "face_bbox": det.face_bbox,
        "message": det.message,
    }
