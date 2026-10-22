"""
Facial **angle** measurements used by the PSL engine.

Every angle is measured from landmarks and compared against a clinically /
aesthetically derived *ideal range*.  Each measurement returns a
:class:`AngleMeasurement` containing:

* ``value``          – measured value in degrees
* ``ideal``          – (low, high) ideal range for the preset sex/mode
* ``score``          – 0..1 harmony score (1 == dead-centre of ideal)
* ``classification`` – "optimal" | "slightly below" | "slightly above" |
                       "below ideal" | "above ideal"

The classic PSL/craniofacial angle panel:

======================  =============================  ==========
Angle                    Ideal (masculine/feminine)    Reference
======================  =============================  ==========
Nasofacial (90-135°)     95-110 / 100-115             Gonzalez-Ulloa
Glabella point           28-40 / 30-45                facial convexity
Nasolabial (95-115°)     90-95 / 95-110               aesthetic lip angle
Gonial (110-130°)        120-130 / 125-135            mandibular angle
Mandibular plane         10-18 / 12-26                SN-MP style
Frankfort horizontal     0 ± 4                        head posture
Eye-palpebral fissure    tilt 0-6 / 2-8               canthal tilt
Chin-cervical (80-95°)   85-95 / 90-100               cervicomental
Philtral angle           90-110                       lip pillar
Auriculo-ocular axis     ~ horizontal ± 7             FOV alignment
======================  =============================  ==========
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

from psl.landmarks import (
    BROW_END_L,
    BROW_END_R,
    BROW_MEDIAL_L,
    BROW_MEDIAL_R,
    CHEEK_L,
    CHEEK_R,
    CHIN_BOTTOM,
    CHIN_TOP,
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
    JAW_PREAURICULAR_L,
    JAW_PREAURICULAR_R,
    LIP_BOTTOM,
    LIP_TOP_CUPID,
    MENTOLABIAL_FOLD,
    NOSE_TIP,
    RADIX,
    SUBNASALE,
    ZYGMION_L,
    ZYGMION_R,
    LandmarkSet,
    angle_at_vertex,
    dist,
    line_angle_deg,
    midpoint,
)

IdealRange = Tuple[float, float]


@dataclass
class AngleMeasurement:
    key: str
    label: str
    value: float
    ideal: IdealRange
    units: str = "deg"
    score: float = 0.0
    classification: str = ""
    notes: str = ""

    def as_dict(self) -> Dict:
        return {
            "key": self.key,
            "label": self.label,
            "value": round(self.value, 2),
            "units": self.units,
            "ideal_low": self.ideal[0],
            "ideal_high": self.ideal[1],
            "score": round(self.score, 4),
            "classification": self.classification,
            "notes": self.notes,
        }


def score_against_ideal(value: float, low: float, high: float,
                        tolerance: float | None = None) -> Tuple[float, str]:
    """Map *value* to a 0..1 harmony score against the [low, high] ideal band.

    Inside the band the score is 1.0 near the centre and tapers gently toward
    the edges; outside the band it falls off quadratically within *tolerance*
    (defaults to 60 % of the band width on each side).
    """
    if math.isnan(value):
        return 0.0, "unmeasurable"
    tol = tolerance if tolerance is not None else max((high - low) * 0.6, 1.0)
    centre = (low + high) / 2.0
    half = max((high - low) / 2.0, 1e-9)

    if low <= value <= high:
        # 1.0 at the edges, peaks slightly beyond centre → use cosine bump
        d = abs(value - centre) / half          # 0..1 inside band
        s = 1.0 - 0.15 * d                      # 0.85..1.0 inside band
        cls = "optimal"
    else:
        over = value > high
        delta = (value - high) if over else (low - value)
        s = max(0.0, 1.0 - (delta / tol) ** 1.5 * 1.0)
        if delta <= tol * 0.4:
            cls = ("slightly above ideal" if over else "slightly below ideal")
        else:
            cls = ("above ideal" if over else "below ideal")
    return s, cls


# ---------------------------------------------------------------------------
# Ideal tables per harmony preset / sex
# ---------------------------------------------------------------------------

#: Default (balanced/androgynous) ideals
IDEAL_ANGLES: Dict[str, IdealRange] = {
    "nasofacial":        (95.0, 112.0),
    "glabella_point":    (28.0, 42.0),
    "nasolabial":        (92.0, 108.0),
    "gonial_left":       (118.0, 132.0),
    "gonial_right":      (118.0, 132.0),
    "mandibular_plane":  (14.0, 26.0),
    "frankfort_pitch":   (-4.0, 4.0),
    "canthal_tilt_l":    (1.5, 7.0),
    "canthal_tilt_r":    (1.5, 7.0),
    "brow_inclination_l": (5.0, 20.0),
    "brow_inclination_r": (5.0, 20.0),
    "chin_cervical":     (88.0, 105.0),
    "mentolabial":       (105.0, 125.0),
    "philtrum_angle":    (90.0, 112.0),
    "auriculo_ocular":   (-7.0, 7.0),
    "eye_fissure_axis_l": (-6.0, 6.0),
    "eye_fissure_axis_r": (-6.0, 6.0),
    "nose_deviation":    (0.0, 4.0),
    "facial_cant":       (0.0, 2.5),
}

MASCULINE_IDEALS: Dict[str, IdealRange] = {
    **IDEAL_ANGLES,
    "nasofacial":        (93.0, 108.0),
    "nasolabial":        (90.0, 100.0),
    "gonial_left":       (120.0, 130.0),
    "gonial_right":      (120.0, 130.0),
    "mandibular_plane":  (10.0, 20.0),
    "chin_cervical":     (85.0, 100.0),
    "canthal_tilt_l":    (1.0, 6.0),
    "canthal_tilt_r":    (1.0, 6.0),
}

FEMININE_IDEALS: Dict[str, IdealRange] = {
    **IDEAL_ANGLES,
    "nasofacial":        (100.0, 115.0),
    "nasolabial":        (98.0, 112.0),
    "gonial_left":       (125.0, 138.0),
    "gonial_right":      (125.0, 138.0),
    "mandibular_plane":  (16.0, 30.0),
    "chin_cervical":     (100.0, 120.0),
    "canthal_tilt_l":    (3.0, 9.0),
    "canthal_tilt_r":    (3.0, 9.0),
}

ANGLE_IDEALS_BY_PRESET: Dict[str, Dict[str, IdealRange]] = {
    "masculine": MASCULINE_IDEALS,
    "feminine": FEMININE_IDEALS,
    "neutral": IDEAL_ANGLES,
    # alias kept for backwards compatibility
    "androgynous": IDEAL_ANGLES,
}


# ---------------------------------------------------------------------------
# Measurement functions
# ---------------------------------------------------------------------------

def _mk(key: str, label: str, value: float, ideals: Dict[str, IdealRange],
        notes: str = "") -> AngleMeasurement:
    low, high = ideals.get(key, IDEAL_ANGLES[key])
    s, cls = score_against_ideal(value, low, high)
    return AngleMeasurement(key=key, label=label, value=value,
                            ideal=(low, high), score=s,
                            classification=cls, notes=notes)


def nasofacial_angle(lm: LandmarkSet) -> float:
    """Angle between the forehead plane (glabella→radix) and the nose dorsum
    (radix→tip).  Classic Gonzalez-Uloa construction using available mesh pts."""
    glabella = lm.p(9)
    radix = lm.p(RADIX)
    tip = lm.p(NOSE_TIP)
    return angle_at_vertex(glabella, radix, tip)


def glabella_point_angle(lm: LandmarkSet) -> float:
    """Frontal facial convexity at the glabella (soft-tissue profile angle)."""
    g = lm.p(9)
    sn = lm.p(SUBNASALE)
    gn = lm.p(CHIN_BOTTOM)
    return angle_at_vertex(sn, g, gn)


def nasolabial_angle(lm: LandmarkSet) -> float:
    """Columella axis (tip→subnasale) vs upper-lip plane (subnasale→Cupid's bow),
    measured *inside* the lip — THE signature PSL lip angle: 90-95 masculine,
    95-110 feminine.  Computed as the supplement of the raw vertex angle."""
    sn = lm.p(SUBNASALE)
    tip = lm.p(NOSE_TIP)
    lip = lm.p(LIP_TOP_CUPID)
    return 180.0 - angle_at_vertex(tip, sn, lip)


def gonial_angle(lm: LandmarkSet, side: str) -> float:
    """Mandibular angle: ramus (gonion→coronoid proxy/pre-auricular) vs body
    (gonion→chin). Sharp <110 = hollowed/sigma, wide >135 = weak."""
    if side == "left":
        pre, gon, chin = lm.p(JAW_PREAURICULAR_L), lm.p(GONION_L), lm.p(CHIN_BOTTOM)
    else:
        pre, gon, chin = lm.p(JAW_PREAURICULAR_R), lm.p(GONION_R), lm.p(CHIN_BOTTOM)
    return angle_at_vertex(pre, gon, chin)


def mandibular_plane_angle(lm: LandmarkSet) -> float:
    """Inclination of the jawline (gonion → gnathion) relative to the Frankfort
    horizontal. High values = vertical grower / weak jaw."""
    fh = frankfort_line_angle(lm)
    res = []
    for gon, pre in ((GONION_L, JAW_PREAURICULAR_L), (GONION_R, JAW_PREAURICULAR_R)):
        go, gn = lm.p(gon), lm.p(CHIN_BOTTOM)
        # downward-from-horizontal slope of the jaw line (positive = slopes down)
        dx = abs(go[0] - gn[0])
        if dx < 1e-9:
            continue
        slope = math.degrees(math.atan2(max(go[1], gn[1]) - min(go[1], gn[1]), dx))
        res.append(abs(slope - abs(fh)))
    return sum(res) / len(res) if res else float("nan")


def frankfort_line_angle(lm: LandmarkSet) -> float:
    """Frankfort horizontal tilt: signed angle (deg) of the porion-proxy line
    (tragus points 50/280) vs the orbitale-proxy line (lower rims 145/375)."""
    porion = line_angle_deg(lm.p(JAW_PREAURICULAR_L), lm.p(JAW_PREAURICULAR_R))
    orbitale = line_angle_deg(lm.p(EYE_LOWER_L), lm.p(EYE_LOWER_R))
    return (porion + orbitale) / 2.0


def frankfort_pitch(lm: LandmarkSet) -> float:
    """Head pitch relative to the camera horizon (signed)."""
    return frankfort_line_angle(lm)


def canthal_tilt(lm: LandmarkSet, side: str) -> float:
    """Positive canthal tilt: outer canthus height minus inner canthus height,
    expressed as an angle across the palpebral fissure length."""
    if side == "left":
        inner, outer = lm.p(EYE_INNER_L), lm.p(EYE_OUTER_L)
    else:
        inner, outer = lm.p(EYE_INNER_R), lm.p(EYE_OUTER_R)
    dx = abs(outer[0] - inner[0])
    dy = inner[1] - outer[1]  # image y grows downward → positive = upward tilt
    if dx == 0:
        return float("nan")
    return math.degrees(math.atan2(dy, dx))


def eye_fissure_axis(lm: LandmarkSet, side: str) -> float:
    """Signed palpebral-fissure orientation (inner→outer), positive = upward."""
    if side == "left":
        p1, p2 = lm.p(EYE_INNER_L), lm.p(EYE_OUTER_L)
    else:
        p1, p2 = lm.p(EYE_INNER_R), lm.p(EYE_OUTER_R)
    dx = abs(p2[0] - p1[0])
    if dx == 0:
        return float("nan")
    return math.degrees(math.atan2(p1[1] - p2[1], dx))


def brow_inclination(lm: LandmarkSet, side: str) -> float:
    """Medial→lateral brow slope, normalised to read left-to-right.
    Positive = brow tail higher than head; negative tail = downturned brow."""
    if side == "left":
        medial, lateral = lm.p(BROW_MEDIAL_L), lm.p(BROW_END_L)
    else:
        medial, lateral = lm.p(BROW_MEDIAL_R), lm.p(BROW_END_R)
    dx = abs(lateral[0] - medial[0])
    if dx == 0:
        return float("nan")
    return math.degrees(math.atan2(medial[1] - lateral[1], dx))


def chin_cervical_angle(lm: LandmarkSet) -> float:
    """Cervicomental angle at gnathion between the chin front plane
    (mentolabial→chin) and the submental/neck line."""
    chin = lm.p(CHIN_BOTTOM)
    ment = lm.p(MENTOLABIAL_FOLD)
    go_mid_x = (lm.p(GONION_L)[0] + lm.p(GONION_R)[0]) / 2.0
    neck = ((chin[0] + go_mid_x) / 2.0, chin[1] + dist(chin, ment) * 1.6)
    return angle_at_vertex(ment, chin, neck)


def mentolabial_angle(lm: LandmarkSet) -> float:
    """Lower-lip/chin fold angle at the mentolabial sulcus (lip→fold vs fold→chin)."""
    lip = lm.p(LIP_BOTTOM)
    fold = lm.p(MENTOLABIAL_FOLD)
    chin = lm.p(CHIN_BOTTOM)
    return 180.0 - angle_at_vertex(lip, fold, chin)


def philtrum_angle(lm: LandmarkSet) -> float:
    """Columnella–lip (philtrum) pillar geometry at Cupid's bow, measured
    between the two upper-lip peaks with the bow as vertex."""
    bow = lm.p(LIP_TOP_CUPID)
    peak_l = lm.p(14)
    peak_r = lm.p(314)
    return angle_at_vertex(peak_l, bow, peak_r)


def auriculo_ocular_axis(lm: LandmarkSet) -> float:
    """Tragus proxies (pre-auricular points) vs pupil-line alignment — detects
    roll skew of the skull independent of the soft-tissue contour."""
    pa = line_angle_deg(lm.p(JAW_PREAURICULAR_L), lm.p(JAW_PREAURICULAR_R))
    upper_lids = line_angle_deg(lm.p(EYE_UPPER_L), lm.p(EYE_UPPER_R))
    return (pa + upper_lids) / 2.0


def nose_deviation(lm: LandmarkSet) -> float:
    """Deviation of the nasal axis from the facial midline (degrees)."""
    glabella = lm.p(9)
    tip = lm.p(NOSE_TIP)
    chin = lm.p(CHIN_BOTTOM)
    mid_x = (glabella[0] + chin[0]) / 2.0
    return math.degrees(math.atan2(tip[0] - mid_x, max(tip[1] - glabella[1], 1e-6)))


def facial_cant(lm: LandmarkSet) -> float:
    """Roll asymmetry between the two mouth corners relative to the interpupillary
    line (degrees)."""
    mc = line_angle_deg(lm.p(61), lm.p(291))
    ip = line_angle_deg(lm.p(EYE_UPPER_L), lm.p(EYE_UPPER_R))
    return abs(mc - ip)


# ---------------------------------------------------------------------------
# Panel runner
# ---------------------------------------------------------------------------

def measure_all_angles(lm: LandmarkSet, preset: str = "neutral") -> Dict[str, AngleMeasurement]:
    """Measure the full PSL angle panel against *preset* ideals."""
    ideals = ANGLE_IDEALS_BY_PRESET.get(preset, IDEAL_ANGLES)

    out: Dict[str, AngleMeasurement] = {}
    panel = [
        ("nasofacial", "Nasofacial angle", nasofacial_angle(lm),
         "Forehead↔nose dorsum; Gonzalez-Ulloa"),
        ("glabella_point", "Glabella point angle", glabella_point_angle(lm),
         "Frontal convexity (Ricketts-style)"),
        ("nasolabial", "Nasolabial angle", nasolabial_angle(lm),
         "Signature lip angle (90-95 m / 95-110 f)"),
        ("gonial_left", "Gonial angle (L)", gonial_angle(lm, "left"),
         "Mandibular angle; sharpness of jaw corner"),
        ("gonial_right", "Gonial angle (R)", gonial_angle(lm, "right"),
         "Mandibular angle; sharpness of jaw corner"),
        ("mandibular_plane", "Mandibular plane angle", mandibular_plane_angle(lm),
         "Jawline inclination vs Frankfort horizontal"),
        ("frankfort_pitch", "Frankfort pitch", frankfort_pitch(lm),
         "Head pitch vs camera horizon"),
        ("canthal_tilt_l", "Canthal tilt (L)", canthal_tilt(lm, "left"),
         "Positive/negative canthal tilt"),
        ("canthal_tilt_r", "Canthal tilt (R)", canthal_tilt(lm, "right"),
         "Positive/negative canthal tilt"),
        ("eye_fissure_axis_l", "Eye fissure axis (L)", eye_fissure_axis(lm, "left"),
         "Palpebral fissure orientation"),
        ("eye_fissure_axis_r", "Eye fissure axis (R)", eye_fissure_axis(lm, "right"),
         "Palpebral fissure orientation"),
        ("brow_inclination_l", "Brow inclination (L)", brow_inclination(lm, "left"),
         "Brow tail slope"),
        ("brow_inclination_r", "Brow inclination (R)", brow_inclination(lm, "right"),
         "Brow tail slope"),
        ("chin_cervical", "Chin-cervical angle", chin_cervical_angle(lm),
         "Cervicomental definition"),
        ("mentolabial", "Mentolabial angle", mentolabial_angle(lm),
         "Lower-lip / chin fold"),
        ("philtrum_angle", "Philtrum angle", philtrum_angle(lm),
         "Cupid's-bow pillar geometry"),
        ("auriculo_ocular", "Auriculo-ocular axis", auriculo_ocular_axis(lm),
         "Skull roll independent of soft tissue"),
        ("nose_deviation", "Nose deviation", nose_deviation(lm),
         "Nasal axis vs midline"),
        ("facial_cant", "Facial cant", facial_cant(lm),
         "Mouth-line roll vs inter-pupil line"),
    ]

    for key, label, value, note in panel:
        lo, hi = ideals.get(key, IDEAL_ANGLES[key])
        s, cls = score_against_ideal(value, lo, hi)
        out[key] = AngleMeasurement(key=key, label=label, value=value,
                                    ideal=(lo, hi), score=s,
                                    classification=cls, notes=note)
    return out
