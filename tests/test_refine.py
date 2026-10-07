import torch

from sculpture.evaluation.metrics import evaluate_assembly
from sculpture.refinement.continuous import refine_assembly

from conftest import known_assembly, requires_cuda


@requires_cuda
def test_fixed_structure_refinement_recovers_perturbation(ctx, library):
    gt = known_assembly(library)
    g = torch.Generator(device="cuda").manual_seed(0)
    noisy = gt.with_params(
        translation=gt.translation + 0.08 * torch.randn(gt.translation.shape, device="cuda", generator=g),
        log_scale=gt.log_scale - 0.15,
    )
    before = evaluate_assembly(ctx, noisy)
    res = refine_assembly(ctx, noisy, steps=120)
    after = evaluate_assembly(ctx, res.assembly)
    assert res.loss < res.initial_loss
    assert after["min_view_iou"] > before["min_view_iou"] + 0.05, (before["view_iou"], after["view_iou"])
    assert after["min_view_iou"] > 0.85
