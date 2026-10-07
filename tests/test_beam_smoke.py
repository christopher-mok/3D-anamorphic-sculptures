import pytest

from sculpture.evaluation.metrics import evaluate_assembly
from sculpture.methods.beam.optimizer import BeamSearchOptimizer
from sculpture.methods.beam.repair import marginal_contributions
from sculpture.scene.assembly import Assembly

from conftest import ctx_with, known_assembly, requires_cuda

SMALL = dict(max_runtime_s=25, beam_width=2, candidates_per_branch=128, utility_top_k=16, lookahead_top_k=6,
             lookahead_steps=8, lookahead_resolution=64, global_refine_every=2, global_refine_steps=15,
             repair_max_objects=2, repair_candidates=32, final_refine_steps=20, patience_rounds=3)


@requires_cuda
@pytest.mark.slow
def test_beam_two_view_improves(ctx, tmp_path):
    c = ctx_with(ctx, beam=SMALL)
    res = BeamSearchOptimizer(c, tmp_path).run()
    m = evaluate_assembly(c, res.assembly)
    assert len(res.assembly) > 0
    assert m["loss"] < 0.5  # the empty assembly has loss w_cov = 1
    assert m["min_view_iou"] > 0.6
    assert (tmp_path / "optimization.csv").exists() and (tmp_path / "checkpoint.json").exists()


@requires_cuda
@pytest.mark.slow
def test_beam_single_view_circle(circle_ctx, tmp_path):
    c = ctx_with(circle_ctx, beam=SMALL)
    res = BeamSearchOptimizer(c, tmp_path).run()
    assert evaluate_assembly(c, res.assembly)["min_view_iou"] > 0.7


@requires_cuda
def test_marginal_contributions_detect_useless_object(ctx, library):
    a = known_assembly(library)
    dup = Assembly.concat([a, a[0]])  # an exact duplicate contributes nothing
    contrib = marginal_contributions(ctx, dup, 128)
    assert contrib[3].abs() < 1e-6 and contrib[1] > 0.01


@requires_cuda
@pytest.mark.slow
def test_beam_resume(ctx, tmp_path):
    c = ctx_with(ctx, beam={**SMALL, "max_runtime_s": 8}, progress={"checkpoint_interval_s": 0.0, "preview_interval_s": 2.0})
    BeamSearchOptimizer(c, tmp_path).run()
    m = BeamSearchOptimizer(c, tmp_path, resume=True)
    assert m.load_checkpoint()
    assert m.iteration > 0 and len(m.best_assembly) > 0 and m._restored["beam"]
