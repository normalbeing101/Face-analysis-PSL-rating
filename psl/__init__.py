"""
Face-analysis-PSL-rating
========================

A PSL (Pretty / Sigma / L) rating engine built on mathematical facial angles,
proportions and harmony ratios.

Modules
-------
landmarks        Canonical 468-point MediaPipe-style landmark indices + helpers
angles           Facial angle measurements (nasofacial, glabella, gonial, ...)
ratios           Facial proportion & harmony ratios (phi thirds, fifths, ...)
symmetry         Bilateral symmetry metrics
classification   PSL tier classification + harmony presets (harmony, neutral,
                 disharmony, golden, celebrity archetypes, ...)
psl              Top-level ``analyze()`` API returning a full measurement report
"""

from psl.psl import analyze, PSLOptions, PSLResult
from psl.classification import (
    HarmonyPreset,
    PSLTier,
    classify_psl,
    list_harmony_presets,
)
from psl.ratios import ideal_ratios
from psl.angles import IDEAL_ANGLES

__version__ = "2.0.0"

__all__ = [
    "analyze",
    "PSLOptions",
    "PSLTier",
    "HarmonyPreset",
    "classify_psl",
    "list_harmony_presets",
    "ideal_ratios",
    "IDEAL_ANGLES",
    "__version__",
]
