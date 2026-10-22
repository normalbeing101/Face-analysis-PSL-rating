"""
PSL **classification** + harmony presets.

This is the "more options like harmony" layer requested in the 2.0 update:
a set of named *harmony presets* that re-weight how angles, ratios and
symmetry are combined into the final PSL (Pretty / Sigma / L) verdict, plus
the tier thresholds themselves.

Presets
-------
harmony      – default; balanced weighting, forgiving bands
strict       – tighter ideal bands, symmetry-heavy (hardmode rating)
golden       – φ-obsession mode: golden-section deviation dominates
sigma        – rewards sharp gonials, hollow cheeks, hunter eyes even when
               classic harmony is imperfect (the "model/thugline" profile)
pretty       – classic neoteny/soft-beauty weighting: big eyes, full lips,
               small nose, V-taper
neutral      – flat averaging, no feature bias (scientific mode)
masculine    – uses masculine ideal tables for every metric
feminine     – uses feminine ideal tables for every metric
celestial    – extremely strict; only near-perfect faces keep a P rating
disharmony   – inverted lens: highlights what is *off* (diagnostic report)
balanced-harmony – harmony but with per-region caps so one great region
               cannot hide a broken one

Tiers
-----
P  (Pretty)   composite ≥ 0.72
S  (Sigma)    0.55 ≤ composite < 0.72  OR sigma-profile override
L  (L)        composite < 0.55
with numeric sub-ratings on a 1–10 scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class PSLTier(str, Enum):
    PRETTY = "P"   # Pretty
    SIGMA = "S"    # Sigma
    L = "L"        # L

    @property
    def description(self) -> str:
        return {
            "P": "Pretty — facial harmony within aesthetic ideals",
            "S": "Sigma — striking structure with non-standard but compelling features",
            "L": "L — measurable disharmony vs. ideal geometry",
        }[self.value]


@dataclass(frozen=True)
class HarmonyPreset:
    """A named weighting/banding policy for the scoring engine."""

    name: str
    label: str
    description: str
    #: weights over the three pillars
    w_angles: float
    w_ratios: float
    w_symmetry: float
    #: multiplier applied to every ideal band half-width (>1 = more forgiving)
    band_tolerance_scale: float
    #: extra weight given to golden-phi deviation ratio (if key present)
    phi_weight_bonus: float
    #: sex/preset table used for ideals ("masculine"/"feminine"/"neutral")
    ideal_table: str
    #: sigma override: strong jaw+eyes can lift S→ borderline-P
    sigma_override: bool
    #: diagnostic inversion (disharmony preset)
    invert_report: bool
    #: per-region cap enabled (balanced-harmony)
    region_caps: bool

    def scaled_band(self, low: float, high: float):
        """Apply this preset's tolerance scaling around the band centre."""
        c = (low + high) / 2.0
        half = max((high - low) / 2.0, 1e-9) * self.band_tolerance_scale
        return (c - half, c + half)


HARMONY_PRESETS: Dict[str, HarmonyPreset] = {p.name: p for p in [
    HarmonyPreset(
        "harmony", "Harmony (default)",
        "Balanced angle/ratio/symmetry blend with forgiving bands.",
        w_angles=0.40, w_ratios=0.40, w_symmetry=0.20,
        band_tolerance_scale=1.00, phi_weight_bonus=0.00,
        ideal_table="neutral", sigma_override=False,
        invert_report=False, region_caps=False),
    HarmonyPreset(
        "strict", "Strict",
        "Tightened bands and symmetry-heavy; professional grading.",
        w_angles=0.40, w_ratios=0.35, w_symmetry=0.25,
        band_tolerance_scale=0.65, phi_weight_bonus=0.05,
        ideal_table="neutral", sigma_override=False,
        invert_report=False, region_caps=True),
    HarmonyPreset(
        "golden", "Golden (φ)",
        "Golden-section deviation dominates the ratio pillar.",
        w_angles=0.30, w_ratios=0.50, w_symmetry=0.20,
        band_tolerance_scale=0.90, phi_weight_bonus=0.35,
        ideal_table="neutral", sigma_override=False,
        invert_report=False, region_caps=False),
    HarmonyPreset(
        "sigma", "Sigma",
        "Rewards hunter eyes, sharp gonials and midface hollowness.",
        w_angles=0.45, w_ratios=0.35, w_symmetry=0.20,
        band_tolerance_scale=1.00, phi_weight_bonus=0.00,
        ideal_table="masculine", sigma_override=True,
        invert_report=False, region_caps=False),
    HarmonyPreset(
        "pretty", "Pretty",
        "Neoteny-friendly: large eyes, lip balance, refined nose.",
        w_angles=0.35, w_ratios=0.45, w_symmetry=0.20,
        band_tolerance_scale=1.10, phi_weight_bonus=0.05,
        ideal_table="feminine", sigma_override=False,
        invert_report=False, region_caps=False),
    HarmonyPreset(
        "neutral", "Neutral (scientific)",
        "Flat average across all metrics, no feature bias.",
        w_angles=1/3, w_ratios=1/3, w_symmetry=1/3,
        band_tolerance_scale=1.00, phi_weight_bonus=0.00,
        ideal_table="neutral", sigma_override=False,
        invert_report=False, region_caps=False),
    HarmonyPreset(
        "masculine", "Masculine ideals",
        "Harmony blend scored against masculine ideal tables.",
        w_angles=0.40, w_ratios=0.40, w_symmetry=0.20,
        band_tolerance_scale=1.00, phi_weight_bonus=0.00,
        ideal_table="masculine", sigma_override=False,
        invert_report=False, region_caps=False),
    HarmonyPreset(
        "feminine", "Feminine ideals",
        "Harmony blend scored against feminine ideal tables.",
        w_angles=0.40, w_ratios=0.40, w_symmetry=0.20,
        band_tolerance_scale=1.00, phi_weight_bonus=0.00,
        ideal_table="feminine", sigma_override=False,
        invert_report=False, region_caps=False),
    HarmonyPreset(
        "celestial", "Celestial",
        "Ultra-strict; only near-perfect geometry keeps a P.",
        w_angles=0.35, w_ratios=0.40, w_symmetry=0.25,
        band_tolerance_scale=0.45, phi_weight_bonus=0.10,
        ideal_table="neutral", sigma_override=False,
        invert_report=False, region_caps=True),
    HarmonyPreset(
        "disharmony", "Disharmony (diagnostic)",
        "Inverted lens — emphasises deviations for correction reports.",
        w_angles=0.40, w_ratios=0.40, w_symmetry=0.20,
        band_tolerance_scale=0.80, phi_weight_bonus=0.00,
        ideal_table="neutral", sigma_override=False,
        invert_report=True, region_caps=True),
    HarmonyPreset(
        "balanced-harmony", "Balanced Harmony",
        "Like harmony, but each region is capped so strengths cannot mask "
        "a broken region.",
        w_angles=0.40, w_ratios=0.40, w_symmetry=0.20,
        band_tolerance_scale=1.00, phi_weight_bonus=0.00,
        ideal_table="neutral", sigma_override=False,
        invert_report=False, region_caps=True),
]}

#: Friendly aliases so users can type e.g. ``preset="harmonious"``.
PRESET_ALIASES: Dict[str, str] = {
    "h": "harmony",
    "default": "harmony",
    "hard": "strict",
    "hardcore": "celestial",
    "phi": "golden",
    "golden_ratio": "golden",
    "thugline": "sigma",
    "neoteny": "pretty",
    "science": "neutral",
    "male": "masculine",
    "female": "feminine",
    "m": "masculine",
    "f": "feminine",
    "balanced": "balanced-harmony",
    "diag": "disharmony",
}


def resolve_preset(name: str) -> HarmonyPreset:
    """Look up a preset by canonical name or alias (case-insensitive)."""
    key = name.strip().lower().replace(" ", "-")
    key = PRESET_ALIASES.get(key, key)
    if key not in HARMONY_PRESETS:
        raise ValueError(
            f"Unknown harmony preset '{name}'. "
            f"Available: {', '.join(sorted(HARMONY_PRESETS))}"
        )
    return HARMONY_PRESETS[key]


def list_harmony_presets() -> List[Dict[str, str]]:
    """Return every available preset with its description (for CLIs/UIs)."""
    return [
        {"name": p.name, "label": p.label, "description": p.description}
        for p in HARMONY_PRESETS.values()
    ]


# ---------------------------------------------------------------------------
# Composite scoring & tiering
# ---------------------------------------------------------------------------

REGION_OF_ANGLE = {
    "nasofacial": "upper-mid", "glabella_point": "upper", "nasolabial": "lower",
    "gonial_left": "lower", "gonial_right": "lower",
    "mandibular_plane": "lower", "frankfort_pitch": "global",
    "canthal_tilt_l": "orbital", "canthal_tilt_r": "orbital",
    "eye_fissure_axis_l": "orbital", "eye_fissure_axis_r": "orbital",
    "brow_inclination_l": "orbital", "brow_inclination_r": "orbital",
    "chin_cervical": "lower", "mentolabial": "lower", "philtrum_angle": "lower",
    "auriculo_ocular": "global", "nose_deviation": "mid", "facial_cant": "global",
}

REGION_OF_RATIO = {
    "vertical_thirds_upper": "upper", "vertical_thirds_mid": "mid",
    "vertical_thirds_lower": "lower", "thirds_balance": "global",
    "horizontal_fifths_eye": "orbital", "face_w_h": "global",
    "lower_face_ratio": "lower", "mid_face_ratio": "mid",
    "philtrum_chin": "lower", "lip_ratio_lower_over_upper": "lower",
    "mouth_over_eye_width": "orbital", "pupillary_bizygomatic": "global",
    "nose_w_over_inner_canthi": "mid", "jaw_over_zygomatic": "lower",
    "eye_opening_index": "orbital", "iris_sclera_ratio": "orbital",
    "forehead_index": "upper", "ramus_face_height": "lower",
    "cheek_jaw_taper": "mid", "brow_eye_distance": "orbital",
    "phi_face_deviation": "global",
}


def _region_means(scores: Dict[str, float], mapping: Dict[str, str]) -> Dict[str, float]:
    regions: Dict[str, List[float]] = {}
    for key, val in scores.items():
        regions.setdefault(mapping.get(key, "global"), []).append(val)
    return {r: sum(v) / len(v) for r, v in regions.items()}


@dataclass
class PSLVerdict:
    tier: PSLTier
    numeric_rating: float          # 1..10
    composite: float               # 0..1
    angles_score: float
    ratios_score: float
    symmetry_score: float
    preset: str
    sub_tier: str                  # e.g. "P+", "S-", "L!"
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "tier": self.tier.value,
            "tier_description": self.tier.description,
            "sub_tier": self.sub_tier,
            "numeric_rating": round(self.numeric_rating, 2),
            "composite": round(self.composite, 4),
            "angles_score": round(self.angles_score, 4),
            "ratios_score": round(self.ratios_score, 4),
            "symmetry_score": round(self.symmetry_score, 4),
            "preset": self.preset,
            "notes": self.notes,
        }


def classify_psl(
    angle_scores: Dict[str, float],
    ratio_scores: Dict[str, float],
    symmetry_score: float,
    preset_name: str = "harmony",
    *,
    sigma_markers: Optional[Dict[str, float]] = None,
) -> PSLVerdict:
    """Combine pillar scores under a harmony preset into a PSL verdict.

    Parameters
    ----------
    angle_scores / ratio_scores : dict key → 0..1 score (see angles/ratios)
    symmetry_score              : 0..1 overall symmetry score
    preset_name                 : any HarmonyPreset name or alias
    sigma_markers               : optional dict with keys like
                                  ``gonial_sharpness``, ``hunter_eye_ratio``,
                                  ``cheek_hollow`` (0..1) enabling the
                                  sigma-override path.
    """
    preset = resolve_preset(preset_name)
    notes: List[str] = []

    a_regions = _region_means(angle_scores, REGION_OF_ANGLE)
    r_regions = _region_means(ratio_scores, REGION_OF_RATIO)

    if preset.region_caps:
        # No region may contribute above the global mean by more than +0.15,
        # and any region below 0.35 drags the whole face down hard.
        def capped(regs: Dict[str, float]) -> float:
            vals = list(regs.values())
            base = sum(vals) / len(vals)
            penalty = sum(max(0.0, 0.35 - v) * 1.8 for v in vals)
            weak_cap = min(base, sum(min(v, base + 0.15) for v in vals) / len(vals))
            return max(0.0, weak_cap - penalty)
        a = capped(a_regions)
        r = capped(r_regions)
    else:
        a = sum(a_regions.values()) / max(len(a_regions), 1)
        r = sum(r_regions.values()) / max(len(r_regions), 1)

    # golden preset: pull phi-deviation score explicitly into the ratio pillar
    if preset.phi_weight_bonus > 0 and "phi_face_deviation" in ratio_scores:
        phi_s = ratio_scores["phi_face_deviation"]
        r = (1 - preset.phi_weight_bonus) * r + preset.phi_weight_bonus * phi_s

    s = symmetry_score

    composite = preset.w_angles * a + preset.w_ratios * r + preset.w_symmetry * s

    if preset.invert_report:
        composite = 1.0 - composite
        notes.append("DISHARMONY lens active: score reflects deviation load "
                     "(higher = MORE off), read corrections from the panel.")

    # --- sigma override ------------------------------------------------------
    if preset.sigma_override and sigma_markers:
        sharp = sigma_markers.get("gonial_sharpness", 0.0)
        hunter = sigma_markers.get("hunter_eye_ratio", 0.0)
        hollow = sigma_markers.get("cheek_hollow", 0.0)
        sigma_index = 0.4 * sharp + 0.35 * hunter + 0.25 * hollow
        if sigma_index >= 0.75 and composite < 0.72:
            boost = (sigma_index - 0.6) * 0.25
            composite += boost
            notes.append(
                f"Sigma override: structural index {sigma_index:.2f} "
                f"(sharp gonials / hunter eyes / hollow cheeks) added "
                f"+{boost:.3f} composite.")

    composite = max(0.0, min(1.0, composite))

    # --- tiering -------------------------------------------------------------
    if preset.invert_report:
        # In diagnostic mode keep tier semantics normal by un-inverting for
        # the tier boundary check.
        tier_probe = 1.0 - composite
    else:
        tier_probe = composite

    if tier_probe >= 0.72:
        tier = PSLTier.PRETTY
    elif tier_probe >= 0.55:
        tier = PSLTier.SIGMA
    else:
        tier = PSLTier.L

    numeric = 1.0 + tier_probe * 9.0

    if tier is PSLTier.PRETTY:
        sub = "P+" if tier_probe >= 0.85 else "P"
    elif tier is PSLTier.SIGMA:
        sub = "S" if tier_probe >= 0.64 else "S-"
    else:
        sub = "L" if tier_probe >= 0.40 else "L!"

    return PSLVerdict(
        tier=tier,
        numeric_rating=numeric,
        composite=composite,
        angles_score=a,
        ratios_score=r,
        symmetry_score=s,
        preset=preset.name,
        sub_tier=sub,
        notes=notes,
    )
