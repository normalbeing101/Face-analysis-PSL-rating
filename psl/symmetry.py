"""
Bilateral **symmetry** metrics.

Fluctuating asymmetry is one of the strongest biological honesty signals in
facial attractiveness research, so the PSL engine scores it explicitly and
separately from raw harmony.

All metrics are computed after *roll-correcting* the face using the
Frankfort/inter-canthal axis, so a tilted head does not fake asymmetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

from psl.landmarks import (
    BROW_END_L,
    BROW_END_R,
    BROW_MEDIAL_L,
    BROW_MEDIAL_R,
    BROW_PEAK_L,
    BROW_PEAK_R,
    CHEEK_L,
    CHEEK_R,
    EYE_INNER_L,
    EYE_INNER_R,
    EYE_LOWER_L,
    EYE_LOWER_R,
    EYE_OUTER_L,
    EYE_OUTER_R,
    EYE_UPPER_L,
    EYE_UPPER_R,
    GONION_L,
    GONION_R,
    JAW_CORNER_L,
    JAW_CORNER_R,
    MOUTH_CORNER_L,
    MOUTH_CORNER_R,
    NOSE_TIP,
    SUBNASALE,
    ZYGMION_L,
    ZYGMION_R,
    LandmarkSet,
    dist,
    line_angle_deg,
)

#: Mirror pairs (left index, right index, human label)
MIRROR_PAIRS: List[Tuple[int, int, str]] = [
    (EYE_INNER_L, EYE_INNER_R, "inner canthus height"),
    (EYE_OUTER_L, EYE_OUTER_R, "outer canthus height"),
    (EYE_UPPER_L, EYE_UPPER_R, "upper lid height"),
    (EYE_LOWER_L, EYE_LOWER_R, "lower lid height"),
    (BROW_MEDIAL_L, BROW_MEDIAL_R, "brow head position"),
    (BROW_PEAK_L, BROW_PEAK_R, "brow apex position"),
    (BROW_END_L, BROW_END_R, "brow tail position"),
    (CHEEK_L, CHEEK_R, "cheek prominence"),
    (ZYGMION_L, ZYGMION_R, "zygion width"),
    (GONION_L, GONION_R, "gonion position"),
    (JAW_CORNER_L, JAW_CORNER_R, "jaw corner"),
    (MOUTH_CORNER_L, MOUTH_CORNER_R, "mouth corner"),
]

FEATURE_ASYMMETRY_SIZES: List[Tuple[str, Sequence[int], Sequence[int]]] = [
    ("eye area", (EYE_INNER_L, EYE_UPPER_L, EYE_OUTER_L, EYE_LOWER_L),
     (EYE_INNER_R, EYE_UPPER_R, EYE_OUTER_R, EYE_LOWER_R)),
    ("mouth half", (13, 61, 17, 14), (13, 291, 17, 308)),
]


@dataclass
class SymmetryReport:
    vertical_asymmetry_pct: float      # mean |Δy| across midline / face height
    horizontal_asymmetry_pct: float    # mean |Δx offset| from centre / face width
    feature_size_asymmetry_pct: float  # mirrored polygon area difference
    rotation_compensation_deg: float   # roll angle removed before measuring
    per_pair: Dict[str, float]         # label -> % asymmetry for that pair
    overall_score: float               # 0..1 (1 = perfectly symmetric)
    classification: str

    def as_dict(self) -> Dict:
        d = asdict(self)
        d["per_pair"] = {k: round(v, 3) for k, v in self.per_pair.items()}
        d["vertical_asymmetry_pct"] = round(self.vertical_asymmetry_pct, 3)
        d["horizontal_asymmetry_pct"] = round(self.horizontal_asymmetry_pct, 3)
        d["feature_size_asymmetry_pct"] = round(self.feature_size_asymmetry_pct, 3)
        d["rotation_compensation_deg"] = round(self.rotation_compensation_deg, 3)
        d["overall_score"] = round(self.overall_score, 4)
        return d


def _polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    """Shoelace formula."""
    n = len(points)
    s = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _rotate(pt: Tuple[float, float], cx: float, cy: float, deg: float):
    r = math.radians(deg)
    dx, dy = pt[0] - cx, pt[1] - cy
    return (cx + dx * math.cos(r) - dy * math.sin(r),
            cy + dx * math.sin(r) + dy * math.cos(r))


def measure_symmetry(lm: LandmarkSet) -> SymmetryReport:
    # --- roll correction using the inter-canthal line -----------------------
    tilt = line_angle_deg(lm.p(EYE_OUTER_L), lm.p(EYE_OUTER_R))
    cx = sum(p[0] for p in lm.points[:468]) / 468
    cy = sum(p[1] for p in lm.points[:468]) / 468
    pts = {i: _rotate(lm.p(i), cx, cy, -tilt) for i in range(468)}

    face_h = abs(pts[9][1] - pts[175][1]) or 1e-9
    mid_x = (pts[SUBNASALE][0] + pts[NOSE_TIP][0] + pts[9][0]) / 3.0

    per_pair: Dict[str, float] = {}
    v_deltas, h_deltas = [], []
    for li, ri, label in MIRROR_PAIRS:
        l, r = pts[li], pts[ri]
        dy = abs(l[1] - r[1])
        dx_off = abs((l[0] - mid_x) + (r[0] - mid_x))  # mirrored x offsets
        per_pair[label] = (dy / face_h * 100.0)
        v_deltas.append(dy)
        h_deltas.append(dx_off)

    vert_pct = (sum(v_deltas) / len(v_deltas)) / face_h * 100.0
    horiz_pct = (sum(h_deltas) / len(h_deltas)) / (dist(pts[ZYGMION_L], pts[ZYGMION_R]) or 1e-9) * 100.0

    size_deltas = []
    for label, lp, rp in FEATURE_ASYMMETRY_SIZES:
        la = _polygon_area([pts[i] for i in lp])
        ra = _polygon_area([pts[i] for i in rp])
        if max(la, ra) > 0:
            size_deltas.append(abs(la - ra) / ((la + ra) / 2.0) * 100.0)
    size_pct = sum(size_deltas) / len(size_deltas) if size_deltas else 0.0

    composite = 0.45 * vert_pct + 0.30 * horiz_pct + 0.25 * size_pct
    # empirical mapping: <1.5% ≈ perfect, >8% ≈ severe
    score = max(0.0, min(1.0, 1.0 - (composite / 8.0) ** 1.35))

    if score >= 0.90:
        cls = "flawless symmetry"
    elif score >= 0.75:
        cls = "high symmetry"
    elif score >= 0.55:
        cls = "average symmetry"
    elif score >= 0.35:
        cls = "noticeable asymmetry"
    else:
        cls = "strong asymmetry"

    return SymmetryReport(
        vertical_asymmetry_pct=vert_pct,
        horizontal_asymmetry_pct=horiz_pct,
        feature_size_asymmetry_pct=size_pct,
        rotation_compensation_deg=tilt,
        per_pair=per_pair,
        overall_score=score,
        classification=cls,
    )
