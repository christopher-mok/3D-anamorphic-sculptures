"""A/B test configuration variants of the same code with paired seeds.

    python scripts/ab_test.py --spec ab_specs/batch1.json --out outputs/ab/batch1

Spec JSON: {"preset": "fast", "seeds": [0,1,2], "targets": [...], "models": "assets/models",
            "arms": {"beam_base": {"method": "beam", "set": [], "baseline": true},
                     "beam_overlap": {"method": "beam", "set": ["refine.w_overlap=2"], "vs": "beam_base"}}}
Each (arm, seed) runs `python -m sculpture.cli run` sequentially (no GPU contention). Results are
cached per run folder, so an interrupted batch resumes. Writes results.json and summary.md.
"""
import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_arm(spec, arm, seed, out: Path) -> dict:
    res_file = out / arm["method"] / "result.json"
    if not res_file.exists():
        cmd = [sys.executable, "-m", "sculpture.cli", "run", "--method", arm["method"], "--config", spec.get("preset", "fast"),
               "--seed", str(seed), "--models", arm.get("models", spec.get("models", "assets/models")), "--output", str(out)]
        for t in spec.get("targets", ["assets/targets/view_0.png", "assets/targets/view_1.png"]):
            cmd += ["--target", t]
        for s in spec.get("set", []) + arm.get("set", []):
            cmd += ["--set", s]
        import os

        # run THIS checkout's code (the script may live in a frozen snapshot of the repo)
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), SCULPTURE_ROOT=str(ROOT))
        subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    r = json.loads(res_file.read_text())
    m, info = r["metrics"], r["info"]
    inter = info.get("intersections", {})
    return {
        "min_view_iou": m["min_view_iou"], "mean_iou": m["mean_iou"], "collisions": m["collisions"],
        "objects": m["object_count"], "runtime_s": m["runtime_s"], "zone_min_iou": m.get("zone_min_iou"),
        "off_view_similarity": m.get("off_view_similarity"), "randomness": m.get("randomness"),
        "final_conflicts": inter.get("initial_conflicts"), "removed": inter.get("removed", 0) + info.get("removed_outside_hull", 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, arm in spec["arms"].items():
        results[name] = {}
        for seed in spec.get("seeds", [0, 1, 2]):
            results[name][seed] = run_arm(spec, arm, seed, out / f"{name}_s{seed}")
            print(f"{name} seed={seed} min_view_iou={results[name][seed]['min_view_iou']:.4f}", flush=True)
            (out / "results.json").write_text(json.dumps(results, indent=2))
    lines = ["| arm | method | n | min-view IoU (mean ± sd) | paired Δ vs baseline | wins | collisions | final-stage conflicts | objects | runtime |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, arm in spec["arms"].items():
        rs = results[name]
        iou = [r["min_view_iou"] for r in rs.values()]
        sd = st.stdev(iou) if len(iou) > 1 else 0.0
        delta, wins = "", ""
        if arm.get("vs"):
            base = results[arm["vs"]]
            d = [rs[s]["min_view_iou"] - base[s]["min_view_iou"] for s in rs if s in base]
            delta = f"{st.mean(d):+.4f}"
            wins = f"{sum(x > 0 for x in d)}/{len(d)}"
        mean = lambda k: st.mean(float(r[k] or 0) for r in rs.values())
        lines.append(f"| {name} | {arm['method']} | {len(iou)} | {st.mean(iou):.4f} ± {sd:.4f} | {delta} | {wins} | "
                     f"{mean('collisions'):.1f} | {mean('final_conflicts'):.1f} | {mean('objects'):.0f} | {mean('runtime_s'):.0f} s |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
