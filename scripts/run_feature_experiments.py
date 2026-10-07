"""Experiments for: intersection handling, piece-type diversity, fixed scale and the
single-view unbounded camera axis. Writes outputs/experiments/<name>/ and a summary.

    python scripts/run_feature_experiments.py [--preset fast] [--only name1,name2]
"""
import argparse
import json
import logging
from pathlib import Path

from sculpture.config import load_config
from sculpture.evaluation.compare import run_experiment

T0, T1 = "assets/targets/view_0.png", "assets/targets/view_1.png"
SINGLE = {"hull": {"resolution": 160}}
EXPERIMENTS = {
    "two_view": ([T0, T1], {}),
    "two_view_diversity": ([T0, T1], {"diversity": {"weight": 0.3}}),
    "single_bounded_free": ([T0], SINGLE),
    "single_bounded_fixed": ([T0], {**SINGLE, "scale": {"mode": "fixed", "fixed": 0.15}}),
    "single_unbounded_free": ([T0], {**SINGLE, "bounding_volume": {"unbounded_view_axis": True}}),
    "single_unbounded_fixed": ([T0], {**SINGLE, "bounding_volume": {"unbounded_view_axis": True}, "scale": {"mode": "fixed", "fixed": 0.15}}),
}
FIELDS = ["min_view_iou", "mean_iou", "spill", "collisions", "clearance_violations", "containment_violations", "randomness", "object_count", "runtime_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--only", default="")
    ap.add_argument("--root", default="outputs/experiments")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    names = [n for n in EXPERIMENTS if not args.only or n in args.only.split(",")]
    summary = {}
    for name in names:
        targets, overrides = EXPERIMENTS[name]
        cfg = load_config(args.preset, overrides=overrides)
        out = run_experiment(cfg, "assets/models", targets, ["beam", "column_generation", "sdf_ray"], output_dir=root / name)
        summary[name] = {"overrides": overrides, "rows": out["comparison"]["rows"], "best": out["comparison"]["best_method"]}
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [f"| experiment | method | " + " | ".join(FIELDS) + " |", "|" + "---|" * (len(FIELDS) + 2)]
    for name, s in summary.items():
        for r in s["rows"]:
            if "score" not in r:
                lines.append(f"| {name} | {r['method']} | failed |")
                continue
            vals = [f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in FIELDS]
            lines.append(f"| {name} | {r['method']}{' *' if r['method'] == s['best'] else ''} | " + " | ".join(vals) + " |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
