"""
Advisory layer: hand the deterministic PSL measurement report to a local VLM
and get back a human-readable aesthetic assessment (strengths, weaknesses,
actionable notes).  The numbers always come from the math engine; the VLM only
*interprets* them — and can also sanity-check the photo quality/pose.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from backend.vlm.base import BaseVLMProvider, VLMError

SYSTEM_PROMPT = (
    "You are a precise, respectful facial-aesthetics analyst. You are given a "
    "photo of a face plus a deterministic anthropometric measurement report "
    "(angles in degrees, harmony ratios, symmetry percentages, and a PSL tier "
    "computed by a math engine). Interpret the measurements for a layperson. "
    "Never invent numbers; only reference values present in the report. "
    "Answer ONLY with a JSON object of this exact shape:\n"
    "{\n"
    '  "summary": "<2-3 sentence overall read>",\n'
    '  "strengths": ["<short bullet>", ...],   // max 5\n'
    '  "weaknesses": ["<short bullet>", ...],  // max 5, tactful wording\n'
    '  "harmony_notes": "<one paragraph on proportion/harmony>",\n'
    '  "photo_quality": {"pose_ok": true/false, "lighting_ok": true/false, '
    '"frontal_enough": true/false, "comment": "<short>"},\n'
    '  "confidence": <0.0-1.0>\n'
    "}"
)


def build_prompt(report: Dict[str, Any]) -> str:
    """Condense the full PSLResult into a token-cheap prompt payload."""
    v = report["verdict"]
    compact = {
        "psl": {
            "tier": v.get("sub_tier") or v.get("tier"),
            "rating_0_10": round(v.get("numeric_rating", 0.0), 2),
            "preset": v.get("preset"),
            "pillars": {
                "angles": round(v.get("angles_score", 0.0), 3),
                "ratios": round(v.get("ratios_score", 0.0), 3),
                "symmetry": round(v.get("symmetry_score", 0.0), 3),
            },
        },
        "raw_values": report.get("raw_values", {}),
        "worst_metrics": _worst(report, n=8),
        "best_metrics": _best(report, n=5),
        "symmetry": {k: report.get("symmetry", {}).get(k)
                     for k in ("classification", "vertical_asymmetry_pct",
                               "horizontal_asymmetry_pct")},
    }
    return ("Deterministic measurement report (JSON):\n\n"
            + json.dumps(compact, ensure_ascii=False)
            + "\n\nAnalyse the attached photo together with this report.")


def _scores(report: Dict[str, Any]):
    items = []
    for section in ("angles", "ratios"):
        for key, m in report.get(section, {}).items():
            if isinstance(m, dict) and "score" in m and m["score"] is not None:
                items.append((key, float(m["score"]), m.get("value"),
                              m.get("ideal") or m.get("band")))
    items.sort(key=lambda x: x[1])
    return items


def _worst(report, n=8):
    out = []
    for key, score, value, band in _scores(report)[:n]:
        out.append({"metric": key, "score": round(score, 3),
                    "measured": value, "ideal_band": band})
    return out


def _best(report, n=5):
    s = _scores(report)
    return [{"metric": k, "score": round(sc, 3)} for k, sc, _, _ in s[-n:][::-1]]


def advise(provider: BaseVLMProvider, report: Dict[str, Any],
           image_png: bytes,
           extra_question: Optional[str] = None) -> Dict[str, Any]:
    """Run the advisory pass. Raises :class:`VLMError` on any provider issue."""
    prompt = build_prompt(report)
    if extra_question:
        prompt += f"\n\nThe user additionally asks: {extra_question}\n" \
                  "Add a \"user_answer\" string field to your JSON with your reply."
    raw = provider.chat_with_image(prompt, image_png, system=SYSTEM_PROMPT)
    data = provider.extract_json(raw)
    # normalise / defend against missing keys
    result = {
        "summary": str(data.get("summary", "")).strip(),
        "strengths": [str(s) for s in data.get("strengths", [])][:5],
        "weaknesses": [str(w) for w in data.get("weaknesses", [])][:5],
        "harmony_notes": str(data.get("harmony_notes", "")).strip(),
        "photo_quality": data.get("photo_quality", {}),
        "confidence": float(data.get("confidence", 0.5) or 0.5),
        "model": {"provider": provider.cfg.provider, "name": provider.cfg.model_name},
    }
    if "user_answer" in data:
        result["user_answer"] = str(data["user_answer"])
    return result
