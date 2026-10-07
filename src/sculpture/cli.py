"""Command line interface.

  python -m sculpture.cli preprocess --models assets/models --target assets/targets/view_0.png [--target ...]
  python -m sculpture.cli run --method beam --models ... --target ... [--config fast] [--set beam.beam_width=8]
  python -m sculpture.cli compare --models ... --target ... --target ... --config configs/default.yaml
  python -m sculpture.cli make-demo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from .methods import METHOD_NAMES


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models", default="assets/models", help="folder with .obj/.ply/.stl/.glb/.gltf meshes")
    p.add_argument("--target", action="append", required=True, help="target silhouette image (repeat for a second view)")
    p.add_argument("--config", default=None, help="YAML file or preset name: fast | default | high_quality")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted config override, e.g. beam.beam_width=8")
    p.add_argument("--cameras", default=None, help="JSON file with a list of camera dicts (overrides config cameras)")
    p.add_argument("--output", default=None, help="output folder (default: outputs/run_XXX)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="resume from checkpoints in --output")
    p.add_argument("-v", "--verbose", action="store_true")


def _load(args):
    dotted = list(args.set)
    if args.seed is not None:
        dotted.append(f"seed={args.seed}")
    cfg = load_config(args.config, dotted=dotted)
    cams = json.loads(Path(args.cameras).read_text()) if args.cameras else None
    return cfg, cams


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sculpture")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_pre = sub.add_parser("preprocess", help="build hull + compatibility diagnostics only")
    _common(p_pre)
    p_run = sub.add_parser("run", help="run one optimization method")
    _common(p_run)
    p_run.add_argument("--method", required=True, choices=METHOD_NAMES)
    p_cmp = sub.add_parser("compare", help="run several methods and compare them")
    _common(p_cmp)
    p_cmp.add_argument("--methods", default=",".join(METHOD_NAMES), help="comma separated subset")
    p_demo = sub.add_parser("make-demo", help="write demo meshes and targets into assets/")
    p_demo.add_argument("--root", default="assets")
    p_demo.add_argument("--complex", action="store_true", help="also write a ~1M-triangle stress-test pool to <root>/models_complex")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    if args.cmd == "make-demo":
        from .demo import make_demo_models, make_demo_targets

        make_demo_models(Path(args.root) / "models")
        make_demo_targets(Path(args.root) / "targets")
        if args.complex:
            from .demo import make_complex_models

            make_complex_models(Path(args.root) / "models_complex")
        print(f"demo assets written to {args.root}/models and {args.root}/targets")
        return 0

    cfg, cams = _load(args)
    from .evaluation.compare import new_run_dir, run_experiment

    if args.cmd == "preprocess":
        from .context import build_context

        ctx = build_context(cfg, args.models, args.target, cams)
        out = Path(args.output) if args.output else new_run_dir(cfg.output_root)
        diag = ctx.save_preprocessing(out / "preprocessing")
        print(json.dumps({"output": str(out), "compatibility": diag["compatibility"], "timings": diag["timings"]}, indent=2))
        return 0

    methods = [args.method] if args.cmd == "run" else [m.strip() for m in args.methods.split(",") if m.strip()]
    res = run_experiment(cfg, args.models, args.target, methods, output_dir=args.output, cameras=cams, resume=args.resume)
    comp = res["comparison"]
    print(f"\nresults in {res['output_dir']}")
    hdr = f"{'method':<20}{'status':<11}{'minIoU':>8}{'meanIoU':>9}{'spill':>8}{'coll':>6}{'cviol':>6}{'rand':>6}{'objs':>6}{'time':>8}"
    print(hdr)
    for r in comp["rows"]:
        if "score" not in r:
            print(f"{r['method']:<20}{r['status']:<11}")
            continue
        print(f"{r['method']:<20}{r['status']:<11}{r['min_view_iou']:>8.4f}{r['mean_iou']:>9.4f}{r['spill']:>8.4f}"
              f"{r['collisions']:>6d}{r['containment_violations']:>6d}{r['randomness']:>6.2f}{r['object_count']:>6d}{r['runtime_s']:>8.1f}")
    print(f"best ({comp['selection_metric']}): {comp['best_method']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
