import torch
import torch.nn.functional as F

from sculpture.geometry.rotation import matrix_to_rot6d
from sculpture.loss.silhouette import image_loss
from sculpture.proposals.candidate_generator import GeneratorOptions
from sculpture.proposals.silhouette_bank import FRAME
from sculpture.proposals.utility_map import approximate_utility, utility_map
from sculpture.scene.assembly import Assembly

from conftest import requires_cuda


@requires_cuda
def test_bank_orientation_transfers_to_target_camera(ctx):
    """An object oriented with R_view^T R_k at the image center looks like bank entry k."""
    bank = ctx.bank
    rod = ctx.library.by_name("rod").mesh_id
    cam = ctx.cameras[1]
    ks = torch.tensor([3, 50, 120], device="cuda")
    R = bank.world_rotation(ks, cam)
    s, z, res = 0.5, 5.0, 64
    a = Assembly(torch.full((3,), rod, device="cuda"), torch.zeros(3, 3, device="cuda"), matrix_to_rot6d(R), torch.full((3,), torch.log(torch.tensor(s)).item(), device="cuda"))
    S = ctx.render_instances(a, res)[:, 1]
    # crop the frame the bank uses: half-size = FRAME * f * s / z
    half = FRAME * cam.focal_px(res) * s / z
    lin = (torch.arange(64, device="cuda") + 0.5) / 64 * 2 - 1
    oy, ox = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([(31.5 + ox * half + 0.5) / res * 2 - 1, (31.5 + oy * half + 0.5) / res * 2 - 1], -1)
    crop = F.grid_sample(S[:, None], grid[None].expand(3, -1, -1, -1), align_corners=False)[:, 0] > 0.5
    ref = bank.masks[rod, ks] > 0.5
    iou = (crop & ref).sum((-1, -2)).float() / (crop | ref).sum((-1, -2)).float()
    assert (iou > 0.8).all(), iou


@requires_cuda
def test_bank_neighbors_are_other_meshes(ctx):
    bank = ctx.bank
    nb, iou = bank.similar_entries(torch.tensor([0, bank.K + 5], device="cuda"))
    m, _ = bank.entry(nb)
    assert (m[0] != 0).all() and (m[1] != 1).all()
    assert bank.mesh_compatibility.shape == (len(ctx.library), len(ctx.library))


@requires_cuda
def test_guided_candidates_beat_random(ctx):
    res = 64
    I = ctx.targets.soft(res)
    R0 = torch.zeros_like(I)
    W = utility_map(R0, I, 1.0, 4.0)
    gen = ctx.generator_torch(0)
    guided = ctx.generator.generate(512, W, I, gen)
    rand = ctx.generator.generate(512, W, I, gen, GeneratorOptions(explore_probability=1.0, bank_probability=0.0))
    ug = approximate_utility(W, ctx.render_instances(guided, res))
    ur = approximate_utility(W, ctx.render_instances(rand, res))
    assert ug.topk(32).values.mean() >= ur.topk(32).values.mean()
    # the best guided candidate actually reduces the image loss
    best = ug.argmax()
    S = ctx.render_instances(guided[int(best)], res)[0]
    assert image_loss(torch.maximum(R0, S), I) < image_loss(R0, I)
    # strict mode: most guided candidates have their center inside Omega
    assert (ctx.hull.query_sdf(guided.translation) <= ctx.hull.mean_voxel_size).float().mean() > 0.8
