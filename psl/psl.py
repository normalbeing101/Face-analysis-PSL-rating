"""
Top-level PSL analysis API.

    from psl import analyze, PSLOptions

    result = analyze(landmarks_468, PSLOptions(preset="harmony"))
    print(result.verdict.tier)        # PSLTier.PRETTY ...
    print(result.raw_values)          # every measured angle / ratio value

``PSLOptions`` exposes the full option surface (the "huge update"):

* ``preset``            – any harmony preset name/alias (see classification)
* ``ideal_table``       – override the sex ideal table used by measurements
* ``band_tolerance_scale`` – scale every ideal band at runtime
* ``include_*``         – toggle whole measurement panels on/off
* ``sigma_markers``     – manual structural markers for the sigma override
* ``strict_input``      – raise instead of clamping when landmarks look wrong
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from psl.angles import measure_all_angles
from psl.classification import (
    HarmonyPreset,
    PSLTier,
    PSLVerdict,
    classify_psl,
    resolve_preset,
)
from psl.ratios import measure_all_ratios, golden_phi_deviations
from psl.symmetry import measure_symmetry
from psl.landmarks import LandmarkSet, load_landmarks, dist


@dataclass
class PSLOptions:
    """All tunable knobs of a PSL analysis run."""

    #: harmony preset: harmony | strict | golden | sigma | pretty | neutral |
    #: masculine | feminine | celestial | disharmony | balanced-harmony (+aliases)
    preset: str = "harmony"

    #: Force a specific ideal table ("masculine"/"feminine"/"neutral");
    #: ``None`` = whatever the preset dictates.
    ideal_table: Optional[str] = None

    #: Extra global scaling of every ideal band (>1 more forgiving).
    #: Multiplied with the preset's own band_tolerance_scale.
    band_tolerance_scale: float = 1.0

    #: Panel toggles
    include_angles: bool = True
    include_ratios: bool = True
    include_symmetry: bool = True
    include_golden_checks: bool = True

    #: Manual sigma markers (0..1): gonial_sharpness, hunter_eye_ratio,
    #: cheek_hollow. When omitted and the preset supports the override they
    #: are auto-estimated from the geometry.
    sigma_markers: Optional[Dict[str, float]] = None

    #: Raise ValueError on suspicious landmark input instead of best-effort.
    strict_input: bool = False

    def resolved(self) -> "PSLOptions":
        """Validate + normalise; returns self for chaining."""
        resolve_preset(self.preset)  # raises on unknown preset
        if self.band_tolerance_scale <= 0:
            raise ValueError("band_tolerance_scale must be > 0")
        return self


@dataclass
class PSLResult:
    verdict: PSLVerdict
    angles: Dict[str, Dict[str, Any]]
    ratios: Dict[str, Dict[str, Any]]
    symmetry: Dict[str, Any]
    golden_checks: Dict[str, Any]
    raw_values: Dict[str, float]
    options: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.as_dict(),
            "angles": self.angles,
            "ratios": self.ratios,
            "symmetry": self.symmetry,
            "golden_checks": self.golden_checks,
            "raw_values": self.raw_values,
            "options": self.options,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, default=_json_default)

    def summary(self) -> str:
        v = self.verdict
        lines = [
            f"PSL TIER   : {v.sub_tier} ({v.tier.name}) — {v.tier.description}",
            f"RATING     : {v.numeric_rating:.2f} / 10  (composite {v.composite:.3f})",
            f"PRESET     : {v.preset}",
            f"PILLARS    : angles {v.angles_score:.3f} | ratios {v.ratios_score:.3f} "
            f"| symmetry {v.symmetry_score:.3f}",
        ]
        if self.symmetry:
            lines.append(f"SYMMETRY   : {self.symmetry.get('classification', 'n/a')}")
        for n in v.notes:
            lines.append(f"NOTE       : {n}")
        return "\n".join(lines)


def _json_default(o):
    if isinstance(o, PSLTier):
        return o.value
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    raise TypeError(f"not JSON serialisable: {type(o)}")


# ---------------------------------------------------------------------------
# Sigma marker auto-estimation
# ---------------------------------------------------------------------------

def _auto_sigma_markers(lm: LandmarkSet, angles, ratios) -> Dict[str, float]:
    """Estimate sigma structural markers from measured geometry."""
    # Gonial sharpness: closer to 122° (masculine ideal centre) → sharper look
    g_vals = []
    for k in ("gonial_left", "gonial_right"):
        if k in angles:
            val = angles[k].value
            if not math.isnan(val):
                g_vals.append(max(0.0, 1.0 - abs(val - 122.0) / 25.0))
    gonial_sharpness = sum(g_vals) / len(g_vals) if g_vals else 0.0

    # Hunter eyes: low vertical opening index + positive canthal tilt
    opening = ratios.get("eye_opening_index")
    tilt_l = angles.get("canthal_tilt_l")
    tilt_r = angles.get("canthal_tilt_r")
    hunter = 0.0
    if opening is not None and not math.isnan(opening.value):
        hunter += max(0.0, min(1.0, (0.34 - opening.value) / 0.12 + 0.5)) * 0.5
    tilts = [t.value for t in (tilt_l, tilt_r) if t and not math.isnan(t.value)]
    if tilts:
        mean_t = sum(tilts) / len(tilts)
        hunter += max(0.0, min(1.0, mean_t / 8.0)) * 0.5
    hunter_eye_ratio = hunter

    # Cheek hollowness: jaw/zygomatic taper well below 1 (V-taper)
    taper = ratios.get("jaw_over_zygomatic")
    cheek_hollow = 0.0
    if taper is not None and not math.isnan(taper.value):
        cheek_hollow = max(0.0, min(1.0, (0.88 - taper.value) / 0.18))

    return {
        "gonial_sharpness": round(gonial_sharpness, 4),
        "hunter_eye_ratio": round(hunter_eye_ratio, 4),
        "cheek_hollow": round(cheek_hollow, 4),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze(landmarks, options: Optional[PSLOptions] = None) -> PSLResult:
    """Run the full PSL measurement + classification pipeline.

    Parameters
    ----------
    landmarks
        468-point MediaPipe Face Mesh landmarks (list of tuples, dict with
        ``landmarks`` key, or mediapipe result object).
    options
        :class:`PSLOptions`; defaults to the ``harmony`` preset.
    """
    opts = (options or PSLOptions()).resolved()
    preset: HarmonyPreset = resolve_preset(opts.preset)
    lm = load_landmarks(landmarks)

    if opts.strict_input:
        xs = [p[0] for p in lm.points[:468]]
        ys = [p[1] for p in lm.points[:468]]
        if max(xs) - min(xs) < 1e-6 or max(ys) - min(ys) < 1e-6:
            raise ValueError("Degenerate landmark set (zero extent).")

    ideal_table = opts.ideal_table or preset.ideal_table

    angles = measure_all_angles(lm, preset=ideal_table) if opts.include_angles else {}
    ratios = measure_all_ratios(lm, preset=ideal_table) if opts.include_ratios else {}
    sym = measure_symmetry(lm) if opts.include_symmetry else None

    # Apply global band tolerance scaling on top of preset-selected ideals.
    scale = opts.band_tolerance_scale
    if scale != 1.0:
        from psl.angles import score_against_ideal
        from psl.ratios import _score_ratio
        for m in angles.values():
            c = (m.ideal[0] + m.ideal[1]) / 2.0
            half = abs(m.ideal[1] - m.ideal[0]) / 2.0 * scale
            m.ideal = (c - half, c + half)
            m.score, m.classification = score_against_ideal(m.value, *m.ideal)
        for m in ratios.values():
            c = (m.band[0] + m.band[1]) / 2.0
            half = abs(m.band[1] - m.band[0]) / 2.0 * scale
            m.band = (c - half, c + half)
            m.score, m.classification, _ = _score_ratio(m.value, m.ideal, m.band)

    golden = golden_phi_deviations(lm) if opts.include_golden_checks else {"checks": [], "mean_deviation_pct": float("nan")}

    angle_scores = {k: m.score for k, m in angles.items()}
    ratio_scores = {k: m.score for k, m in ratios.items()}
    sym_score = sym.overall_score if sym else 0.5

    markers = opts.sigma_markers
    if markers is None and preset.sigma_override:
        markers = _auto_sigma_markers(lm, angles, ratios)

    verdict = classify_psl(angle_scores, ratio_scores, sym_score,
                           preset_name=opts.preset, sigma_markers=markers)

    raw_values: Dict[str, float] = {}
    for k, m in angles.items():
        raw_values[f"angle.{k}"] = round(m.value, 3) if not math.isnan(m.value) else None  # type: ignore[assignment]
    for k, m in ratios.items():
        raw_values[f"ratio.{k}"] = round(m.value, 4) if not math.isnan(m.value) else None  # type: ignore[assignment]
    if sym:
        raw_values["symmetry.vertical_pct"] = round(sym.vertical_asymmetry_pct, 3)
        raw_values["symmetry.horizontal_pct"] = round(sym.horizontal_asymmetry_pct, 3)
        raw_values["symmetry.feature_size_pct"] = round(sym.feature_size_asymmetry_pct, 3)

    return PSLResult(
        verdict=verdict,
        angles={k: m.as_dict() for k, m in angles.items()},
        ratios={k: m.as_dict() for k, m in ratios.items()},
        symmetry=sym.as_dict() if sym else {},
        golden_checks=golden,
        raw_values=raw_values,
        options={
            "preset": preset.name,
            "preset_label": preset.label,
            "ideal_table": ideal_table,
            "band_tolerance_scale": scale,
            "sigma_markers": markers,
        },
    )
