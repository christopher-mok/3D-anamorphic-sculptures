from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from sculpture.config import _wrap
from sculpture.refinement.color import assign_instance_colors
from sculpture.scene.assembly import Assembly
from sculpture.scene.serialization import assembly_from_dict, assembly_to_dict
from sculpture.targets.target import TargetSet


def _two_color_target(res=32):
    mask = np.ones((res, res), bool)
    rgb = np.zeros((res, res, 3), np.float32)
    rgb[:, : res // 2, 0] = 1
    rgb[:, res // 2 :, 2] = 1
    return TargetSet([mask], "cpu", colors=[rgb], color_cfg={"enabled": True, "palette_size": 4})


def test_target_quantization_finds_major_color_regions():
    target = _two_color_target()
    regions = target.regions(32)[0]
    assert regions[16, 4] != regions[16, 27]
    distance = target.color_region_distance(32)[0]
    assert distance[16, 4] > distance[16, 15]


def test_color_regions_are_spatial_segments_not_only_palette_ids():
    mask = np.ones((32, 32), bool)
    rgb = np.ones((32, 32, 3), np.float32)
    rgb[4:12, 4:12] = (1, 0, 0)
    rgb[20:28, 20:28] = (1, 0, 0)
    target = TargetSet([mask], "cpu", colors=[rgb], color_cfg={"enabled": True, "palette_size": 3,
                                                                "min_segment_fraction": 0.01})
    regions = target.regions(32)[0]
    assert regions[7, 7] != regions[23, 23]


def test_whole_image_mode_uses_full_frame_for_shape_and_color(tmp_path: Path):
    from PIL import Image

    rgb = np.full((12, 20, 3), 255, np.uint8)
    rgb[3:9, 7:13] = (255, 0, 0)
    path = tmp_path / "target.png"
    Image.fromarray(rgb).save(path)

    target = TargetSet.from_files(
        [path],
        32,
        {"mask_background": False, "pad_fraction": 0, "color": {"enabled": True, "palette_size": 4}},
        device="cpu",
    )

    assert torch.all(target.base[0, 11:21] == 1)
    assert torch.all(target.base[0, :5] == 0)
    assert torch.all(target.regions(32)[target.base > 0] >= 0)
    assert torch.all(target.regions(32)[target.base == 0] == -1)
    assert target.mask_background is False
    saved = target.save_pngs(tmp_path / "saved", 32)
    restored = np.asarray(Image.open(saved[0]).convert("RGB"))
    assert np.any(restored[..., 0] > restored[..., 1] + 100)
    assert len(np.unique(restored.reshape(-1, 3), axis=0)) > 1


def test_colors_roundtrip_with_assembly_json():
    a = Assembly(torch.tensor([0]), torch.zeros(1, 3), torch.zeros(1, 6), torch.zeros(1),
                 colors=torch.tensor([[0.1, 0.2, 0.3]]))
    b = assembly_from_dict(assembly_to_dict(a), device="cpu")
    assert torch.allclose(b.colors, a.colors)


def test_visible_instances_get_automatic_or_fixed_color():
    target = _two_color_target(32)
    ids = torch.full((1, 32, 32), -1, dtype=torch.long)
    ids[0, :, :16] = 0
    ids[0, :, 16:] = 1
    renderer = SimpleNamespace(render_instance_ids=lambda *args, **kwargs: ids)
    entries = [SimpleNamespace(name="auto", filename="auto.obj"), SimpleNamespace(name="fixed", filename="fixed.obj")]
    cfg = _wrap({"targets": {"color": {"assignment_resolution": 32, "primary_view_weight": 1,
                                          "secondary_view_weight": 0.2, "fixed_models": {"fixed": "#00ff00"}}}})
    ctx = SimpleNamespace(targets=target, renderer=renderer, cameras=[object()], num_primary_views=1, num_views=1,
                          working_resolution=32, cfg=cfg, library=SimpleNamespace(entries=entries), device=torch.device("cpu"),
                          pick_lod=lambda *_: "proxy")
    a = Assembly(torch.tensor([0, 1]), torch.zeros(2, 3), torch.zeros(2, 6), torch.zeros(2))
    colored, stats = assign_instance_colors(ctx, a)
    assert colored.colors[0, 0] > 0.95 and colored.colors[0, 2] < 0.05
    assert torch.allclose(colored.colors[1], torch.tensor([0.0, 1.0, 0.0]))
    assert stats == {"assigned": 2, "automatic": 1, "fixed": 1}
