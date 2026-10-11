"""Reproduce a saved color run and report image improvement on its actual target/models."""
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch

from sculpture.config import _wrap, deep_merge
from sculpture.context import build_context
from sculpture.evaluation.compare import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/run_008")
    parser.add_argument("--seconds", type=float, default=45)
    parser.add_argument("--pieces", type=int, default=48)
    args = parser.parse_args()
    source = Path(args.source)
    config = json.loads((source / "config.json").read_text())
    diagnostics = json.loads((source / "preprocessing/diagnostics.json").read_text())
    cfg = _wrap(deep_merge(config, {
        "max_objects": args.pieces,
        "chained": {"max_runtime_s": args.seconds},
        "beam": {"candidates_per_branch": 192, "lookahead_top_k": 8,
                 "lookahead_resolution": 64, "global_refine_steps": 6, "final_refine_steps": 8},
        "targets": {"working_resolution": 128, "color": {"enabled": True, "reconstruct": True}},
        "evaluation": {"resolution": 256},
    }))
    ctx = build_context(cfg, "assets/models", diagnostics["targets"])
    output = Path("outputs") / ("color_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    results = run_experiment(cfg, "assets/models", diagnostics["targets"], ["chained"], output_dir=output, ctx=ctx)
    result = results["results"]["chained"]
    if result["status"] != "done":
        raise RuntimeError(result)
    info, metrics = result["info"], result["metrics"]
    summary = {"output": str(output), "objects": metrics["object_count"], "coverage": metrics["recall"],
               "initial_image_loss": info["initial_image_loss"], "final_image_loss": info["final_image_loss"],
               "image_terms": info["image_terms"], "stop_reason": info["stop_reason"],
               "fixed_colors_present": result["result"].assembly.colors is not None}
    print(json.dumps(summary, indent=2), flush=True)
    assert metrics["object_count"] > 0
    assert info["final_image_loss"] < info["initial_image_loss"] * 0.9
    assert torch.isfinite(result["result"].assembly.colors).all()


if __name__ == "__main__":
    main()
