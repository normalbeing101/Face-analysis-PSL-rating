"""
Landmark helpers.

The engine works on a normalized list of 2D (or 3D) facial landmarks using the
canonical **MediaPipe Face Mesh 468-point** topology.  Only a subset of indices
is needed for PSL math; they are enumerated here so the rest of the codebase
stays readable and index-safe.

If you feed landmarks from another model, remap them to these indices first
(see ``tools/remap_landmarks.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

Point = Tuple[float, float]
Point3 = Tuple[float, float, float]

# ---------------------------------------------------------------------------
# Canonical MediaPipe Face Mesh landmark indices used by the PSL engine
# ---------------------------------------------------------------------------

#: Midline / vertical reference points
NOSE_TIP = 1
NOSE_SUPERTIPS = 4
RADIX = 168            # root of the nose (between the eyes)
GLABELLA_PROXIED = 9           # soft-tissue glabella proxy (top-most nose point)
PHILTRUM = 114
SUBNASALE = 2          # subnasale (columella midpoint base)
CHIN_TOP = 152         # soft-tissue chin pogonion proxy
CHIN_BOTTOM = 175      # gnathion (lowest chin point)
LIP_TOP_CUPID = 13     # cupid's bow peak (vermillion upper lip midline)
LIP_BOTTOM = 17        # lowest vermillion lower lip point
MOUTH_CORNER_L = 61
MOUTH_CORNER_R = 291

#: Eye landmarks
PUPIL_L = 468
PUPIL_R = 473
EYE_OUTER_L = 33
EYE_OUTER_R = 263
EYE_INNER_L = 133
EYE_INNER_R = 362
EYE_UPPER_L = 159
EYE_UPPER_R = 386
EYE_LOWER_L = 145
EYE_LOWER_R = 375
BROW_PEAK_L = 71
BROW_PEAK_R = 300
BROW_MEDIAL_L = 55
BROW_MEDIAL_R = 285
BROW_END_L = 21        # lateral tail of the left brow
BROW_END_R = 251       # lateral tail of the right brow

#: Cheek / zygomaxillary
CHEEK_L = 205          # left zygomatic region
CHEEK_R = 425          # right zygomatic region
ZYGMION_L = 234        # widest lateral face contour (left)
ZYGMION_R = 454        # widest lateral face contour (right)

#: Jaw / mandible
GONION_L = 172         # mandibular angle proxy (left)
GONION_R = 397         # mandibular angle proxy (right)
JAW_CORNER_L = 58      # gonion-adjacent jaw corner (left)
JAW_CORNER_R = 288     # gonion-adjacent jaw corner (right)
JAW_PREAURICULAR_L = 50
JAW_PREAURICULAR_R = 280
MENTOLABIAL_FOLD = 78  # mentolabial (lower lip crease) sulcus

#: Facial contour (for width measurements)
FACE_CONTOUR_LEFT = [
    172, 58, 61, 146, 78, 191, 80, 190, 189, 205, 204, 203,
    202, 201, 200, 199, 198, 207, 208, 209, 210, 211, 32,
]
FACE_CONTOUR_RIGHT = [
    397, 288, 291, 375, 324, 308, 418, 417, 416, 205 + 220,
    425, 426, 423, 240, 241, 429, 234, 139, 128, 114 + 0,
    215, 213, 214,
]

#: Iris centres (used for eye exposure / catchlight geometry)
IRIS_CENTRE_L = 468
IRIS_CENTRE_R = 473
IRIS_RADIUS_PTS_L = (469, 470, 471, 472)
IRIS_RADIUS_PTS_R = (474, 475, 476, 477)


@dataclass(frozen=True)
class LandmarkSet:
    """Thin wrapper giving semantic access to raw landmark arrays."""

    points: Sequence[Point] | Sequence[Point3]

    def __post_init__(self) -> None:
        if len(self.points) < 468:
            raise ValueError(
                f"Expected at least 468 landmarks (MediaPipe Face Mesh), got {len(self.points)}."
            )

    def p(self, idx: int) -> Point:
        """Return landmark *idx* as an (x, y) tuple."""
        pt = self.points[idx]
        return (float(pt[0]), float(pt[1]))

    def p3(self, idx: int) -> Point3:
        """Return landmark *idx* as (x, y, z); z defaults to 0.0 in 2D mode."""
        pt = self.points[idx]
        if len(pt) >= 3:
            return (float(pt[0]), float(pt[1]), float(pt[2]))
        return (float(pt[0]), float(pt[1]), 0.0)


# ---------------------------------------------------------------------------
# Primitive geometry helpers
# ---------------------------------------------------------------------------

def dist(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two points (2D or 3D)."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def midpoint(a: Sequence[float], b: Sequence[float]) -> Tuple[float, ...]:
    return tuple((x + y) / 2.0 for x, y in zip(a, b))


def angle_at_vertex(a, vertex, b) -> float:
    """Angle ABC in degrees, measured at *vertex*."""
    v1 = [a[i] - vertex[i] for i in range(len(a))]
    v2 = [b[i] - vertex[i] for i in range(len(b))]
    dot = sum(x * y for x, y in zip(v1, v2))
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos))


def line_angle_deg(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Angle of segment p1->p2 relative to the horizontal axis (-180..180)."""
    return math.degrees(math.atan2(-(p2[1] - p1[1]), p2[0] - p1[0]))


def face_midline_x(lm: LandmarkSet) -> float:
    """Best-effort vertical midline x-coordinate (average of midline points)."""
    ids = [GLABELLA_PROXIED, RADIX, NOSE_TIP, SUBNASALE, PHILTRUM,
           LIP_TOP_CUPID, MENTOLABIAL_FOLD, CHIN_TOP, CHIN_BOTTOM]
    xs = [lm.p(i)[0] for i in ids]
    return sum(xs) / len(xs)


def interp(a: Sequence[float], b: Sequence[float], t: float) -> Tuple[float, ...]:
    """Linear interpolation between two points."""
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def load_landmarks(data) -> LandmarkSet:
    """Coerce various input shapes into a :class:`LandmarkSet`.

    Accepts:
      * a list/tuple of (x, y) or (x, y, z) tuples
      * a dict with key ``"landmarks"``
      * a mediapipe ``NormalisedLandmarkList`` (objects with .x/.y/.z)
    """
    if isinstance(data, dict) and "landmarks" in data:
        data = data["landmarks"]

    pts = []
    for lm in data:
        if isinstance(lm, (tuple, list)):
            pts.append(tuple(float(v) for v in lm))
        else:  # mediapipe-style object
            z = getattr(lm, "z", 0.0)
            pts.append((float(lm.x), float(lm.y), float(z)))
    return LandmarkSet(pts)
