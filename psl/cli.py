"""
Command-line interface for the PSL engine.

    python -m psl.cli presets                       # list harmony presets
    python -m psl.cli measure face_landmarks.json   # raw angle/ratio values
    python -m psl.cli rate  face_landmarks.json --preset sigma
    python -m psl.cli demo  --preset harmony        # synthetic harmonious face
    python -m psl.cli compare face.json             # every preset side-by-side
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict

from psl.classification import list_harmony_presets
from psl.psl import PSLOptions, analyze


def _load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_presets(_: argparse.Namespace) -> int:
    print("Available harmony presets:\n")
    for p in list_harmony_presets():
        print(f"  {p['name']:<17} {p['label']:<26} — {p['description']}")
    print("\nUse any of these with:  --preset <name>")
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    opts = PSLOptions(preset=args.preset)
    result = analyze(_load(args.landmarks), opts)
    print(json.dumps({
        "angles": result.angles,
        "ratios": result.ratios,
        "symmetry": result.symmetry,
        "golden_checks": result.golden_checks,
    }, indent=2))
    return 0


def cmd_rate(args: argparse.Namespace) -> int:
    opts = PSLOptions(preset=args.preset)
    if args.no_symmetry:
        opts.include_symmetry = False
    result = analyze(_load(args.landmarks), opts)
    if args.json:
        print(result.to_json())
    else:
        print(result.summary())
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    data = _load(args.landmarks)
    names = [p["name"] for p in list_harmony_presets()]
    print(f"{'preset':<18} {'tier':<5} {'rating':<7} composite")
    print("-" * 46)
    for name in names:
        r = analyze(data, PSLOptions(preset=name))
        v = r.verdict
        print(f"{name:<18} {v.sub_tier:<5} {v.numeric_rating:<7.2f} {v.composite:.3f}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from tools.synthetic_faces import make_face

    kind = args.kind or "harmonious"
    lm = make_face(kind)
    result = analyze(lm, PSLOptions(preset=args.preset))
    print(f"[synthetic face: {kind!r}]")
    print(result.summary())
    if args.full:
        print()
        print(result.to_json())
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="psl", description="PSL facial harmony rating engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("presets", help="list all harmony presets")
    p.set_defaults(func=cmd_presets)

    p = sub.add_parser("measure", help="dump every PSL measurement value (JSON)")
    p.add_argument("landmarks", help="path to JSON landmark file (468 points)")
    p.add_argument("--preset", default="harmony")
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("rate", help="full PSL rating")
    p.add_argument("landmarks")
    p.add_argument("--preset", default="harmony")
    p.add_argument("--json", action="store_true", help="emit full JSON report")
    p.add_argument("--no-symmetry", action="store_true")
    p.set_defaults(func=cmd_rate)

    p = sub.add_parser("compare", help="rate one face under EVERY preset")
    p.add_argument("landmarks")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("demo", help="analyse a synthetic face (no camera needed)")
    p.add_argument("--preset", default="harmony")
    p.add_argument("--kind", default=None,
                   choices=["harmonious", "sigma", "disharmonious", "asymmetric"],
                   help="synthetic face archetype (default: harmonious)")
    p.add_argument("--full", action="store_true", help="include full JSON panel")
    p.set_defaults(func=cmd_demo)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
