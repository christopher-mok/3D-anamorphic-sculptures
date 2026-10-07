"""Benchmark two code versions (e.g. a git worktree of an older commit vs. the current tree).

Runs `python -m sculpture.cli compare` for every (version, preset, seed) case, alternating
versions to balance any drift, and writes <out>/benchmark.json + benchmark.md with
per-method means.

    python scripts/benchmark_versions.py --baseline <path-to-old-checkout> --out outputs/bench \
        --fast-seeds 0 1 2 --medium-seeds 0
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TARGETS = ["assets/targets/view_0.png", "assets/targets/view_1.png"]
METRICS = ["min_view_iou", "mean_iou", "spill", "collisions", "containment_violations", "object_count", "runtime_s"]


def run(version_root: Path, preset: str, seed: int, out: Path, methods: str = "beam,column_generation,sdf_ray", sets=()) -> dict:
    env = dict(os.environ, PYTHONPATH=str(version_root / "src"), SCULPTURE_ROOT=str(version_root))
    cmd = [sys.executable, "-m", "sculpture.cli", "compare", "--config", preset, "--seed", str(seed), "--output", str(out), "--methods", methods]
    for s_ in sets:
        cmd += ["--set", s_]
    for t in TARGETS:
        cmd += ["--target", t]
    subprocess.run(cmd, cwd=version_root, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads((out / "comparison.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--current", default=str(HERE))
    ap.add_argument("--out", default="outputs/bench")
    ap.add_argument("--fast-seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--medium-seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--methods", default="beam,column_generation,sdf_ray")
    ap.add_argument("--set", action="append", default=[], help="dotted override applied to the IMPROVED version only")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    versions = {"baseline": Path(args.baseline).resolve(), "improved": Path(args.current).resolve()}
    cases = [("fast", s) for s in args.fast_seeds] + [("default", s) for s in args.medium_seeds]
    results = json.loads((out / "benchmark.json").read_text()) if (out / "benchmark.json").exists() else []
    done = {(r["version"], r["preset"], r["seed"]) for r in results}
    for preset, seed in cases:
        for vname, vroot in versions.items():
            if (vname, preset, seed) in done:
                continue
            comp = run(vroot, preset, seed, out / f"{vname}_{preset}_s{seed}", args.methods, args.set if vname == "improved" else ())
            for row in comp["rows"]:
                results.append({"version": vname, "preset": preset, "seed": seed, **{k: row.get(k) for k in ["method", *METRICS]}})
            (out / "benchmark.json").write_text(json.dumps(results, indent=2))
            print(f"done {vname} {preset} seed={seed}", flush=True)

    lines = ["| preset | method | version | n | min_view_iou (mean ± sd) | mean_iou | spill | collisions | containment viol. | objects | runtime_s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for preset in dict.fromkeys(p for p, _ in cases):
        for method in args.methods.split(","):
            for vname in versions:
                rows = [r for r in results if r["preset"] == preset and r["method"] == method and r["version"] == vname and r["min_view_iou"] is not None]
                if not rows:
                    continue
                def m(k):
                    return st.mean(float(r[k]) for r in rows)
                iou = [float(r["min_view_iou"]) for r in rows]
                sd = st.stdev(iou) if len(iou) > 1 else 0.0
                lines.append(f"| {preset} | {method} | {vname} | {len(rows)} | {st.mean(iou):.3f} ± {sd:.3f} | {m('mean_iou'):.3f} | "
                             f"{m('spill'):.4f} | {m('collisions'):.1f} | {m('containment_violations'):.1f} | {m('object_count'):.0f} | {m('runtime_s'):.0f} |")
    (out / "benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
