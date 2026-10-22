"""
Facial **proportion & harmony ratios**.

PSL ratings are only half angles — the other half is *ratio harmony*: how each
sub-feature relates to the whole face and to the golden constant φ (1.618).

Every ratio returns a :class:`RatioMeasurement` with value, ideal target,
deviation-from-ideal percentage and a 0..1 harmony score.

Panel
-----
Vertical thirds          hairline→glabella→subnasale→gnathion ≈ 1:1:1
Horizontal fifths        five equal eye-width columns
Face width : height      bizygomatic / glabella→gnathion  ≈ 0.72 (m) / 0.70 (f)
Lower-face height ratio  subnasale→gnathion / total       ≈ 33 %
Philtrum : chin          1 : 1.5–1.6
Upper lip : lower lip    1 : 1.4–1.6
Eye width : mouth width  1 : 1.55–1.75
Inter-pupillary : bizygomatic  ≈ 0.77
Nose width : eye inner distance ≈ 0.9–1.0
Cheekbone-to-jaw taper   gonial width / bizygomatic ≈ 0.75–0.85 (m)
Iris exposure / sclera show   vertical eye opening ratios
Forehead index           forehead width / face height
Jaw angle sharpness proxy ramus length / face height
Golden-triangle overlays trios of points whose distances should approach φ
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from psl.landmarks import (
    CHIN_BOTTOM,
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
    LIP_BOTTOM,
    LIP_TOP_CUPID,
    MOUTH_CORNER_L,
    MOUTH_CORNER_R,
    NOSE_TIP,
    SUBNASALE,
    ZYGMION_L,
    ZYGMION_R,
    LandmarkSet,
    dist,
    midpoint,
)

PHI = (1 + math.sqrt(5)) / 2.0  # 1.6180339887…
IdealRange = Tuple[float, float]


@dataclass
class RatioMeasurement:
    key: str
    label: str
    value: float
    ideal: float                 # scalar target
    band: IdealRange             # acceptable [low, high]
    deviation_pct: float         # signed % away from ideal
    score: float                 # 0..1 harmony score
    classification: str
    units: str = "ratio"

    def as_dict(self) -> Dict:
        return {
            "key": self.key,
            "label": self.label,
            "value": round(self.value, 4),
            "ideal": self.ideal,
            "band_low": self.band[0],
            "band_high": self.band[1],
            "deviation_pct": round(self.deviation_pct, 2),
            "score": round(self.score, 4),
            "classification": self.classification,
            "units": self.units,
        }


def _score_ratio(value: float, ideal: float, band: IdealRange):
    if math.isnan(value) or value <= 0:
        return 0.0, "unmeasurable", float("nan")
    dev = ((value - ideal) / ideal * 100.0) if ideal else 0.0
    lo, hi = band
    if lo <= value <= hi:
        # linear taper from centre (1.0) toward edges (0.8)
        centre = (lo + hi) / 2.0
        half = max((hi - lo) / 2.0, 1e-9)
        s = 1.0 - 0.2 * min(abs(value - centre) / half, 1.0)
        cls = "harmonious"
    else:
        over = value > hi
        delta = (value - hi) if over else (lo - value)
        span = max(hi - lo, 1e-9)
        s = max(0.0, 0.8 - (delta / span) * 0.8)
        rel = "above" if over else "below"
        cls = f"slightly {rel} ideal" if delta < span * 0.35 else f"{rel} ideal"
    return s, cls, dev


# ---------------------------------------------------------------------------
# Ideal tables per preset
# ---------------------------------------------------------------------------

IDEAL_RATIOS: Dict[str, Tuple[float, IdealRange]] = {
    "vertical_thirds_upper":   (1.000, (0.90, 1.10)),
    "vertical_thirds_mid":     (1.000, (0.90, 1.10)),
    "vertical_thirds_lower":   (1.000, (0.90, 1.10)),
    "thirds_balance":          (1.000, (0.85, 1.15)),
    "horizontal_fifths_eye":   (1.000, (0.85, 1.15)),
    "face_w_h":                (0.71, (0.64, 0.78)),
    "lower_face_ratio":        (0.50, (0.43, 0.57)),
    "mid_face_ratio":          (0.50, (0.43, 0.57)),
    "philtrum_chin":           (1 / 1.55, (1 / 1.85, 1 / 1.30)),
    "lip_ratio_lower_over_upper": (1.5, (1.25, 1.8)),
    "mouth_over_eye_width":    (1.65, (1.45, 1.85)),
    "pupillary_bizygomatic":   (0.77, (0.72, 0.82)),
    "nose_w_over_inner_canthi": (0.95, (0.80, 1.10)),
    "jaw_over_zygomatic":      (0.80, (0.72, 0.88)),
    "eye_opening_index":       (0.35, (0.26, 0.44)),
    "iris_sclera_ratio":       (0.90, (0.75, 1.05)),
    "forehead_index":          (0.62, (0.54, 0.70)),
    "ramus_face_height":       (0.24, (0.19, 0.30)),
    "cheek_jaw_taper":         (0.80, (0.72, 0.88)),
    "brow_eye_distance":       (0.17, (0.12, 0.22)),
    "phi_face_deviation":      (0.0, (0.0, 6.0)),
}

MASCULINE_RATIO_OVERRIDE: Dict[str, Tuple[float, IdealRange]] = {
    "face_w_h":                (0.73, (0.66, 0.80)),
    "lower_face_ratio":        (0.52, (0.45, 0.60)),
    "lip_ratio_lower_over_upper": (1.40, (1.15, 1.65)),
    "jaw_over_zygomatic":      (0.82, (0.75, 0.90)),
    "philtrum_chin":           (1 / 1.6, (1 / 1.9, 1 / 1.35)),
}

FEMININE_RATIO_OVERRIDE: Dict[str, Tuple[float, IdealRange]] = {
    "face_w_h":                (0.69, (0.62, 0.76)),
    "lower_face_ratio":        (0.48, (0.40, 0.55)),
    "lip_ratio_lower_over_upper": (1.6, (1.35, 1.95)),
    "jaw_over_zygomatic":      (0.76, (0.68, 0.84)),
    "philtrum_chin":           (1 / 1.5, (1 / 1.8, 1 / 1.25)),
}


def ideal_ratios(preset: str = "neutral") -> Dict[str, Tuple[float, IdealRange]]:
    """Return the ideal-ratio table for a harmony preset."""
    table = dict(IDEAL_RATIOS)
    if preset == "masculine":
        table.update(MASCULINE_RATIO_OVERRIDE)
    elif preset == "feminine":
        table.update(FEMININE_RATIO_OVERRIDE)
    return table


# ---------------------------------------------------------------------------
# Raw measurements
# ---------------------------------------------------------------------------

def vertical_thirds(lm: LandmarkSet) -> Dict[str, float]:
    """Trisection check on the measured midline.

    MediaPipe exposes glabella(9) → subnasale(2) → gnathion(175); the classic
    ideal is that these two *measured* thirds split the face evenly (each ≈ 0.5
    of total height).  The forehead third above the glabella is extrapolated
    from brow-head height × 2.35 and reported as ``upper_third`` against the
    mean of all three (ideal 1.0 = equal thirds)."""
    g_y = lm.p(9)[1]
    sn_y = lm.p(SUBNASALE)[1]
    gn_y = lm.p(CHIN_BOTTOM)[1]
    brow_y = lm.p(55)[1]
    total = abs(gn_y - g_y) or 1e-9
    upper = abs(sn_y - g_y) / total      # fraction of measured height
    low = abs(gn_y - sn_y) / total       # fraction of measured height
    forehead = abs(brow_y - g_y) * 2.35
    mean = (forehead + abs(sn_y - g_y) + abs(gn_y - sn_y)) / 3.0
    return {
        "upper_third": forehead / mean if mean else float("nan"),
        "middle_third": upper * 2.0,     # normalised so perfect halves → 1.0
        "lower_third": low * 2.0,        # normalised so perfect halves → 1.0
        "balance": (min(upper, low) / max(upper, low)) if max(upper, low) else float("nan"),
    }


def horizontal_fifths(lm: LandmarkSet) -> Dict[str, float]:
    """Face width divided by 5 × palpebral width (ideally 1.0)."""
    eye_l = dist(lm.p(EYE_INNER_L), lm.p(EYE_OUTER_L))
    eye_r = dist(lm.p(EYE_INNER_R), lm.p(EYE_OUTER_R))
    eye = (eye_l + eye_r) / 2.0
    face_w = dist(lm.p(ZYGMION_L), lm.p(ZYGMION_R))
    return {
        "fifth_balance": face_w / (5 * eye) if eye else float("nan"),
        "eye_width": eye,
        "bizygomatic_width": face_w,
    }


def face_width_to_height(lm: LandmarkSet) -> float:
    """Bizygomatic width over total facial height (glabella→gnathion)."""
    h = dist(lm.p(9), lm.p(CHIN_BOTTOM))
    w = dist(lm.p(ZYGMION_L), lm.p(ZYGMION_R))
    return w / h if h else float("nan")


def lower_face_ratio(lm: LandmarkSet) -> float:
    total = dist(lm.p(9), lm.p(CHIN_BOTTOM))
    lower = dist(lm.p(SUBNASALE), lm.p(CHIN_BOTTOM))
    return lower / total if total else float("nan")


def mid_face_ratio(lm: LandmarkSet) -> float:
    """Glabella→subnasale as a fraction of glabella→gnathion (ideal ≈ 0.50)."""
    total = dist(lm.p(9), lm.p(CHIN_BOTTOM))
    mid = dist(lm.p(9), lm.p(SUBNASALE))
    return mid / total if total else float("nan")


def philtrum_to_chin(lm: LandmarkSet) -> float:
    """Philtrum length : (lower-lip→chin) length — ideal ≈ 1 : 1.5."""
    phil = dist(lm.p(SUBNASALE), lm.p(LIP_TOP_CUPID))
    chinzone = dist(lm.p(LIP_BOTTOM), lm.p(CHIN_BOTTOM))
    return phil / chinzone if chinzone else float("nan")


def lip_ratio(lm: LandmarkSet) -> float:
    """Lower lip vermillion height / upper lip vermillion height (ideal ≈1.5).
    Upper vermillion is estimated from the distance between Cupid's bow and the
    lip-union centre (landmark 0), which MediaPipe places on the wet-dry border."""
    union = lm.p(0)
    bow = lm.p(LIP_TOP_CUPID)
    bottom = lm.p(LIP_BOTTOM)
    upper = abs(union[1] - bow[1]) * 2.0
    lower = abs(bottom[1] - union[1]) * 2.0
    return lower / upper if upper else float("nan")


def mouth_to_eye_width(lm: LandmarkSet) -> float:
    mw = dist(lm.p(MOUTH_CORNER_L), lm.p(MOUTH_CORNER_R))
    eye_l = dist(lm.p(EYE_INNER_L), lm.p(EYE_OUTER_L))
    eye_r = dist(lm.p(EYE_INNER_R), lm.p(EYE_OUTER_R))
    eye = (eye_l + eye_r) / 2.0
    return mw / eye if eye else float("nan")


def pupillary_bizygomatic(lm: LandmarkSet) -> float:
    """Inter-pupillary distance (iris centres 468/473) over bizygomatic width."""
    ipd = abs(lm.p(468)[0] - lm.p(473)[0])
    bw = dist(lm.p(ZYGMION_L), lm.p(ZYGMION_R))
    return ipd / bw if bw else float("nan")


def nose_width_ratio(lm: LandmarkSet) -> float:
    alar_l, alar_r = lm.p(98), lm.p(327)
    nw = dist(alar_l, alar_r)
    ic = dist(lm.p(EYE_INNER_L), lm.p(EYE_INNER_R))
    return nw / ic if ic else float("nan")


def jaw_to_zygomatic(lm: LandmarkSet) -> float:
    jw = dist(lm.p(GONION_L), lm.p(GONION_R))
    zw = dist(lm.p(ZYGMION_L), lm.p(ZYGMION_R))
    return jw / zw if zw else float("nan")


def eye_opening_index(lm: LandmarkSet) -> float:
    """Vertical palpebral fissure height / horizontal length."""
    hl = dist(lm.p(EYE_UPPER_L), lm.p(EYE_LOWER_L))
    ll = dist(lm.p(EYE_INNER_L), lm.p(EYE_OUTER_L))
    hr = dist(lm.p(EYE_UPPER_R), lm.p(EYE_LOWER_R))
    lr = dist(lm.p(EYE_INNER_R), lm.p(EYE_OUTER_R))
    left = hl / ll if ll else float("nan")
    right = hr / lr if lr else float("nan")
    return (left + right) / 2.0


def iris_exposure(lm: LandmarkSet) -> float:
    """Iris diameter / palpebral height — <1 means superior scleral show."""
    iris_l = dist(lm.p(469), lm.p(471))
    iris_r = dist(lm.p(474), lm.p(476))
    iris = (iris_l + iris_r) / 2.0
    opening = eye_opening_index(lm)
    fiss = (dist(lm.p(EYE_INNER_L), lm.p(EYE_OUTER_L)) +
            dist(lm.p(EYE_INNER_R), lm.p(EYE_OUTER_R))) / 2.0
    height = fiss * opening if not math.isnan(opening) else float("nan")
    return iris / height if height else float("nan")


def forehead_index(lm: LandmarkSet) -> float:
    fw = dist(lm.p(21), lm.p(251))
    fh = dist(lm.p(9), lm.p(CHIN_BOTTOM))
    return fw / fh if fh else float("nan")


def ramus_proxy(lm: LandmarkSet) -> float:
    """Ramus length proxy: tragus-proxy → gonion, normalised by face height."""
    rl = dist(lm.p(50), lm.p(GONION_L))
    rr = dist(lm.p(280), lm.p(GONION_R))
    r = (rl + rr) / 2.0
    fh = dist(lm.p(9), lm.p(CHIN_BOTTOM))
    return r / fh if fh else float("nan")


def brow_eye_distance(lm: LandmarkSet) -> float:
    bl = dist(lm.p(55), lm.p(EYE_UPPER_L))
    br = dist(lm.p(285), lm.p(EYE_UPPER_R))
    fh = dist(lm.p(9), lm.p(CHIN_BOTTOM))
    return ((bl + br) / 2.0) / fh if fh else float("nan")


def golden_phi_deviations(lm: LandmarkSet) -> Dict[str, object]:
    """Mean absolute % deviation of landmark pairs that classically form
    golden triangles / golden-section divisions."""
    checks = []

    def add(name: str, big: float, small: float):
        if big and small and not math.isnan(big) and not math.isnan(small):
            phi_obs = big / small
            dev = abs(phi_obs - PHI) / PHI * 100.0
            checks.append({"pair": name, "observed_phi": phi_obs,
                           "deviation_pct": dev})

    total_h = dist(lm.p(9), lm.p(CHIN_BOTTOM))
    add("face height ÷ mid face", total_h, dist(lm.p(9), lm.p(SUBNASALE)))
    add("lower face ÷ chin complex", dist(lm.p(SUBNASALE), lm.p(CHIN_BOTTOM)),
        dist(lm.p(LIP_BOTTOM), lm.p(CHIN_BOTTOM)))
    add("bizygomatic ÷ mouth", dist(lm.p(ZYGMION_L), lm.p(ZYGMION_R)),
        dist(lm.p(MOUTH_CORNER_L), lm.p(MOUTH_CORNER_R)))
    add("mouth ÷ eye width",
        dist(lm.p(MOUTH_CORNER_L), lm.p(MOUTH_CORNER_R)),
        (dist(lm.p(EYE_INNER_L), lm.p(EYE_OUTER_L)) +
         dist(lm.p(EYE_INNER_R), lm.p(EYE_OUTER_R))) / 2.0)
    add("nose length ÷ nose width",
        dist(lm.p(9), lm.p(SUBNASALE)),
        dist(lm.p(98), lm.p(327)))
    avg = sum(c["deviation_pct"] for c in checks) / len(checks) if checks else float("nan")
    for c in checks:
        c["observed_phi"] = round(c["observed_phi"], 4)
        c["deviation_pct"] = round(c["deviation_pct"], 2)
    return {"checks": checks, "mean_deviation_pct": avg}


# ---------------------------------------------------------------------------
# Panel runner
# ---------------------------------------------------------------------------

def measure_all_ratios(lm: LandmarkSet, preset: str = "neutral") -> Dict[str, RatioMeasurement]:
    ideals = ideal_ratios(preset)

    thirds = vertical_thirds(lm)
    phi = golden_phi_deviations(lm)
    raw: Dict[str, float] = {
        "vertical_thirds_upper": thirds["upper_third"],
        "vertical_thirds_mid": thirds["middle_third"],
        "vertical_thirds_lower": thirds["lower_third"],
        "thirds_balance": thirds["balance"],
        "horizontal_fifths_eye": horizontal_fifths(lm)["fifth_balance"],
        "face_w_h": face_width_to_height(lm),
        "lower_face_ratio": lower_face_ratio(lm),
        "mid_face_ratio": mid_face_ratio(lm),
        "philtrum_chin": philtrum_to_chin(lm),
        "lip_ratio_lower_over_upper": lip_ratio(lm),
        "mouth_over_eye_width": mouth_to_eye_width(lm),
        "pupillary_bizygomatic": pupillary_bizygomatic(lm),
        "nose_w_over_inner_canthi": nose_width_ratio(lm),
        "jaw_over_zygomatic": jaw_to_zygomatic(lm),
        "eye_opening_index": eye_opening_index(lm),
        "iris_sclera_ratio": iris_exposure(lm),
        "forehead_index": forehead_index(lm),
        "ramus_face_height": ramus_proxy(lm),
        "cheek_jaw_taper": jaw_to_zygomatic(lm),
        "brow_eye_distance": brow_eye_distance(lm),
        "phi_face_deviation": phi["mean_deviation_pct"],  # type: ignore[index]
    }

    labels = {
        "vertical_thirds_upper": "Upper-third proportion",
        "vertical_thirds_mid": "Middle-third proportion",
        "vertical_thirds_lower": "Lower-third proportion",
        "thirds_balance": "Rule-of-thirds balance",
        "horizontal_fifths_eye": "Rule-of-fifths (eye unit)",
        "face_w_h": "Face width : height",
        "lower_face_ratio": "Lower-face height ratio",
        "mid_face_ratio": "Mid-face height ratio",
        "philtrum_chin": "Philtrum : chin (1:φ-ish)",
        "lip_ratio_lower_over_upper": "Lower/upper lip volume ratio",
        "mouth_over_eye_width": "Mouth width : eye width",
        "pupillary_bizygomatic": "IPD : bizygomatic width",
        "nose_w_over_inner_canthi": "Alar width : inner canthi",
        "jaw_over_zygomatic": "Bigonial : bizygomatic",
        "eye_opening_index": "Palpebral opening index",
        "iris_sclera_ratio": "Iris exposure ratio",
        "forehead_index": "Forehead width index",
        "ramus_face_height": "Ramus length : face height",
        "cheek_jaw_taper": "Zygomatic→jaw taper",
        "brow_eye_distance": "Brow-eye distance index",
        "phi_face_deviation": "Mean golden-φ deviation",
    }

    out: Dict[str, RatioMeasurement] = {}
    for key, value in raw.items():
        ideal, band = ideals.get(key, IDEAL_RATIOS[key])
        s, cls, dev = _score_ratio(value, ideal, band)
        out[key] = RatioMeasurement(
            key=key, label=labels[key], value=value, ideal=ideal, band=band,
            deviation_pct=dev, score=s, classification=cls,
        )
    return out
