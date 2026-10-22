"""
Synthetic 468-point MediaPipe-topology faces for tests, demos and docs.

The mesh is procedurally generated from an ellipse-based head model with the
key anatomical landmark families placed at their canonical indices, so all
angle/ratio math in the engine works on it exactly as it would on a real
MediaPipe detection.

Archetypes
----------
harmonious     – near-ideal thirds/fifths/angles (should rate P)
sigma          – sharp gonials, low eye opening, V-taper (S→P under sigma preset)
disharmonious  – long lower face, negative canthal tilt, wide alar base (L)
asymmetric     – harmonious but with deliberate left/right drift
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

Point = Tuple[float, float]


def _ellipse_point(cx: float, cy: float, rx: float, ry: float, t_deg: float,
                   side_sign: float = 1.0) -> Point:
    t = math.radians(t_deg)
    return (cx + side_sign * rx * math.cos(t), cy + ry * math.sin(t))


def make_face(kind: str = "harmonious", jitter_seed: int = 7) -> List[Point]:
    """Return a list of 468 (x, y) landmarks for the requested archetype."""
    import random
    rnd = random.Random(jitter_seed)

    # Archetype parameters ---------------------------------------------------
    params: Dict[str, Dict[str, float]] = {
        "harmonious": dict(
            face_h=1.00, lower_ratio=0.335, zygo_w=0.72, jaw_w=0.58,
            eye_w=0.135, eye_open=0.34, canthal=+3.5, brow_slope=+12.0,
            nose_len=0.36, alar_w=0.16, mouth_w=0.30, lip_lower=0.10,
            lip_upper=0.068, chin_h=0.16, philtrum=0.105, asym=0.0,
            face_cant=0.0, nose_dev=0.0, gonial_bias=0.0,
        ),
        "sigma": dict(
            face_h=1.00, lower_ratio=0.36, zygo_w=0.74, jaw_w=0.60,
            eye_w=0.145, eye_open=0.27, canthal=+5.5, brow_slope=+9.0,
            nose_len=0.37, alar_w=0.15, mouth_w=0.30, lip_lower=0.09,
            lip_upper=0.062, chin_h=0.18, philtrum=0.10, asym=0.0,
            face_cant=0.0, nose_dev=0.0, gonial_bias=-6.0,
        ),
        "disharmonious": dict(
            face_h=1.15, lower_ratio=0.44, zygo_w=0.78, jaw_w=0.70,
            eye_w=0.120, eye_open=0.42, canthal=-3.0, brow_slope=-4.0,
            nose_len=0.42, alar_w=0.21, mouth_w=0.36, lip_lower=0.06,
            lip_upper=0.05, chin_h=0.12, philtrum=0.16, asym=0.010,
            face_cant=2.2, nose_dev=3.5, gonial_bias=9.0,
        ),
        "asymmetric": dict(
            face_h=1.00, lower_ratio=0.34, zygo_w=0.72, jaw_w=0.59,
            eye_w=0.135, eye_open=0.34, canthal=+3.0, brow_slope=+11.0,
            nose_len=0.36, alar_w=0.16, mouth_w=0.30, lip_lower=0.10,
            lip_upper=0.066, chin_h=0.16, philtrum=0.105, asym=0.035,
            face_cant=1.4, nose_dev=1.8, gonial_bias=2.0,
        ),
    }
    if kind not in params:
        raise ValueError(f"unknown archetype '{kind}'")
    p = params[kind]

    pts: List[Point] = []  # sentinel-free construction below

    cx = 0.5
    glabella_y = 0.22
    face_h = 0.62 * p["face_h"]
    sn_y = glabella_y + face_h * (1 - p["lower_ratio"])
    chin_y = glabella_y + face_h
    mid_y = (sn_y + chin_y) / 2.0

    zygo_rx = p["zygo_w"] / 2.0
    jaw_rx = p["jaw_w"] / 2.0
    head_rx = 0.40

    assigned: Dict[int, Point] = {}

    def put(idx: int, x: float, y: float):
        assigned[idx] = (x, y)

    # --- midline ------------------------------------------------------------
    put(9, cx + p["nose_dev"] / 100.0, glabella_y)               # glabella proxy
    put(168, cx + p["nose_dev"] / 200.0, glabella_y + 0.035)     # radix
    put(1, cx + p["nose_dev"] / 40.0, sn_y - p["nose_len"] * face_h * 0.55)  # nose tip
    put(2, cx, sn_y)                                             # subnasale
    put(114, cx, sn_y + 0.02)                                    # philtrum peak
    put(13, cx, sn_y + p["philtrum"])                            # cupid's bow
    put(17, cx, sn_y + p["philtrum"] + p["lip_lower"])           # lower lip bottom
    put(78, cx, sn_y + p["philtrum"] + p["lip_lower"] + 0.035)   # mentolabial fold
    put(152, cx, chin_y - p["chin_h"] * 0.55)                    # chin top/pogonion
    put(175, cx, chin_y)                                         # gnathion

    # --- eyes ---------------------------------------------------------------
    eye_cy = glabella_y + 0.085
    eye_dx = 0.155
    for sign, tag in ((-1, "L"), (+1, "R")):
        ecx = cx + sign * eye_dx + (p["asym"] if tag == "R" else -p["asym"]) * 0.5
        ecy = eye_cy + (p["asym"] if tag == "R" else 0.0)
        ew = p["eye_w"] / 2.0
        eh = p["eye_w"] * p["eye_open"] / 2.0
        tilt = math.radians(p["canthal"])
        # canthal tilt: outer canthus sits higher than inner when positive
        inner = (ecx - sign * ew, ecy)
        outer = (ecx + sign * ew, ecy - ew * math.tan(tilt))
        upper = (ecx, ecy - eh)
        lower = (ecx, ecy + eh)
        idx_map = {
            "L": {"inner": 133, "outer": 33, "upper": 159, "lower": 145},
            "R": {"inner": 362, "outer": 263, "upper": 386, "lower": 375},
        }[tag]
        put(idx_map["inner"], *inner)
        put(idx_map["outer"], *outer)
        put(idx_map["upper"], *upper)
        put(idx_map["lower"], *lower)
        # iris ring around centre
        icx, icy = ecx, ecy
        ir = eh * 0.9
        ring = [469, 470, 471, 472] if tag == "L" else [474, 475, 476, 477]
        for k, ridx in enumerate(ring):
            ang = math.pi * k / 2.0
            put(ridx, icx + ir * math.cos(ang), icy + ir * math.sin(ang) * 0.8)
        put(468 if tag == "L" else 473, icx, icy)

    # --- brows ----------------------------------------------------------------
    brow_y = eye_cy - 0.062
    for sign, m_idx, p_idx, e_idx, tag in (
        (-1, 55, 71, 21, "L"), (+1, 285, 300, 251, "R")):
        bx = cx + sign * 0.155
        slope = math.radians(p["brow_slope"])
        dy_asym = p["asym"] * (1.0 if tag == "R" else 0.0)
        put(m_idx, bx - sign * 0.075, brow_y + 0.012 + dy_asym)
        put(p_idx, bx, brow_y - 0.006 + dy_asym)
        put(e_idx, bx + sign * 0.085, brow_y - 0.085 * math.tan(slope) * 1.0 + dy_asym)

    # --- nose alae & sides ----------------------------------------------------
    nose_tip = assigned[1]
    put(98, cx - p["alar_w"] / 2.0, sn_y - 0.008)     # alar base left
    put(327, cx + p["alar_w"] / 2.0, sn_y - 0.008)    # alar base right
    put(4, nose_tip[0], nose_tip[1] - 0.01)           # supertip
    put(11, cx - 0.012, sn_y - 0.02)                  # nostril L
    put(46, cx + 0.012, sn_y - 0.02)                  # nostril R
    put(199, cx - 0.03, sn_y - p["nose_len"] * face_h * 0.3)
    put(429, cx + 0.03, sn_y - p["nose_len"] * face_h * 0.3)

    # --- mouth -----------------------------------------------------------------
    mc_y = sn_y + p["philtrum"] + p["lip_lower"] * 0.45
    cant = math.radians(p["face_cant"])
    put(61, cx - p["mouth_w"] / 2.0, mc_y + math.sin(cant) * p["mouth_w"] / 2.0)
    put(291, cx + p["mouth_w"] / 2.0, mc_y - math.sin(cant) * p["mouth_w"] / 2.0)
    put(0, cx, mc_y)                                   # lips union centre
    put(82, cx - 0.03, mc_y + 0.008)
    put(308, cx + 0.03, mc_y + 0.008)
    put(14, cx - 0.03, assigned[13][1] + 0.004)        # upper lip L peak
    put(314, cx + 0.03, assigned[13][1] + 0.004)       # upper lip R peak

    # --- cheekbones / zygion -----------------------------------------------------
    cheek_y = eye_cy + 0.10
    put(205, cx - zygo_rx * 0.62, cheek_y)
    put(425, cx + zygo_rx * 0.62, cheek_y + p["asym"])
    put(234, cx - zygo_rx, cheek_y + 0.02)
    put(454, cx + zygo_rx, cheek_y + 0.02 - p["asym"])

    # --- jaw -----------------------------------------------------------------------
    gonial_y = sn_y + (chin_y - sn_y) * 0.42
    put(172, cx - jaw_rx * 0.98, gonial_y + p["gonial_bias"] / 400.0)
    put(397, cx + jaw_rx * 0.98, gonial_y - p["gonial_bias"] / 400.0)
    put(58, cx - jaw_rx * 0.86, gonial_y + 0.05)
    put(288, cx + jaw_rx * 0.86, gonial_y + 0.05)
    put(50, cx - head_rx * 0.92, glabella_y + 0.16)    # pre-auricular L (tragus proxy)
    put(280, cx + head_rx * 0.92, glabella_y + 0.16 - p["asym"])  # pre-auricular R

    # --- fill remaining indices along the head ellipse ------------------------------
    pts = [(0.0, 0.0)] * 478
    for idx, xy in assigned.items():
        pts[idx] = xy
    centre_y = (glabella_y + chin_y) / 2.0
    for i in range(478):
        if i in assigned:
            continue
        ang = math.radians(-90 + (i % 360))
        rx = head_rx * (0.90 + 0.10 * math.sin((i % 37) / 37 * math.pi))
        ry = face_h / 2.0 * 1.12
        x = cx + rx * math.cos(ang)
        y = centre_y + ry * math.sin(ang)
        pts[i] = (x + rnd.uniform(-0.002, 0.002), y + rnd.uniform(-0.002, 0.002))

    return pts
