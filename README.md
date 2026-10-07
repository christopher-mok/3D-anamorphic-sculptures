# Floating Anamorphic Assembly Sculptures

This research codebase automatically designs floating sculptures in the style of Michael Murphy's view-dependent work. Each sculpture is made of many instances of meshes drawn from a model pool. Its perspective projection from one or two cameras should match one or two target silhouettes.

The question the codebase is built to study is how best to optimize this mixed discrete/continuous problem. It contains **three independent solvers**. They share the representation, preprocessing, renderer, loss and evaluation. They differ only in how they search.

| Method | Paradigm |
|---|---|
| **1. Beam Constructive** (`beam`) | Constructive discrete search that grows the sculpture one instance at a time. A cheap utility screen filters candidates, then each survivor gets a local Adam lookahead. The method also runs a beam with diversity, non-conflicting multi-add, periodic global refinement, and swap / reseed / pose-jump repair. |
| **2. Column Generation** (`column_generation`) | Global combinatorial selection of placements (columns) through a coverage MILP. LP duals become an image-space pricing map. New columns are priced with nvdiffrast gradients. Collision conflict cuts are added lazily. The binary MILP selection is followed by a raster polish. |
| **3. SDF Ray Packing** (`sdf_ray`) | Continuous 3D optimization: every foreground camera ray must hit the soft union of the model SDFs, which gives wide-basin gradients. Weak objects are reseeded and objects are added on plateaus, then a raster polish follows. |

The `README` sections below cover setup, inputs, running each method, the UI, outputs and troubleshooting. The HTTP API is documented in [docs/api.md](docs/api.md).

---

## 1. Problem and representation

* A sculpture is `X = {O_1..O_N}` with `O_i = (mesh_id, translation ∈ R³, rotation ∈ SO(3), log_scale ∈ R)`.
* **`mesh_id` is discrete**. It is never relaxed, interpolated or Gumbel-softmaxed. Translation, rotation (continuous 6D representation, Gram–Schmidt to SO(3)) and isotropic log-scale are continuous.
* Cameras are fixed **perspective** pinhole cameras that are never optimized. There are no orthographic shortcuts anywhere.
* The objective is image fidelity plus optional physical feasibility. **Piece count is never penalized**. `max_objects` is only a computational safety bound.

## 2. Environment setup

Tested with **Windows 10, Python 3.12, PyTorch 2.11 + CUDA 12.8, nvdiffrast 0.4.0, RTX 4080**. Any recent CUDA GPU, PyTorch ≥ 2.1 and Python ≥ 3.10 should work. Linux works the same way.

```bash
# 1. PyTorch with CUDA (pick the wheel matching your driver, see pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. nvdiffrast (needs a CUDA toolkit + C++ compiler the first time it JIT-compiles its plugin;
#    recent versions also ship prebuilt wheels)
pip install git+https://github.com/NVlabs/nvdiffrast

# 3. this package (+ test deps)
pip install -e ".[dev]"

# optional: Gurobi master solver (a license is required; otherwise SciPy/HiGHS is used automatically)
pip install gurobipy
```

Check the GPU path: `python -m pytest tests -q`. The suite has 39 tests and takes about 1 minute.

### Frontend

You need Node.js 18 or newer.

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173  (proxies /api and /outputs to :8000)
```

## 3. Inputs

### Models: `assets/models/`
Put `.obj .ply .stl .glb .gltf .off` files into the folder. On startup every file is processed as follows:
- Loaded, with scenes merged and faces triangulated.
- Centered at its bounding-box center and scaled to bounding-sphere radius 1. The transform is recorded.
- Decimated into a rendering proxy (≤ 8k faces) and a coarse screening proxy (≤ 1.5k faces). The original stays in high resolution for export.
- Sampled on its surface, and given a bounding box and center of mass.
- Given a canonical SDF (64³ by default), computed with a generalized winding number, or with a flood-fill occupancy plus unsigned-distance fallback for open or non-watertight meshes.
- Rendered into a 64×64 **silhouette bank** at 192 near-uniform SO(3) orientations, with a silhouette-IoU nearest-neighbour index across meshes.

Everything is cached under `cache/`.

Run `python -m sculpture.cli make-demo` to write a demo pool (cube, sphere, cylinder, cone, torus, rod, plate) and demo targets (heart, star, circle, square).

### Targets: `assets/targets/`
A target is a binary or grayscale image (a black silhouette on white, or any polarity, which is detected automatically from the border), or an image with an alpha channel. Processing:
- EXIF orientation is normalized.
- The aspect ratio is preserved by letterboxing to a square.
- The image is thresholded at `targets.threshold`. Use `targets.invert: true|false` to force the polarity.
- Pyramids and foreground/background distance transforms are derived from it.

* **One target**: pass one `--target`. Camera 0 is used.
* **Two targets**: pass `--target` twice. Camera 0 is paired with image 0 and camera 1 with image 1.

### Cameras
Cameras are defined in the config (`cameras:` list), in a JSON file passed as `--cameras cams.json`, or in the UI:

```json
[{"position": [0, 0, 5], "look_at": [0, 0, 0], "up": [0, 1, 0], "fov_y_deg": 25, "near": 0.1, "far": 20},
 {"position": [5, 0, 0], "look_at": [0, 0, 0], "up": [0, 1, 0], "fov_y_deg": 25, "near": 0.1, "far": 20}]
```

Conventions are OpenGL and three.js: +Y is up, the camera looks from `position` to `look_at`, `fov_y_deg` is the vertical field of view, and images are square. The default sculpture volume `B` is `[-1,1]³`. The default cameras are 90° apart, and B fills about 90% of each image.

## 4. Shared preprocessing (all methods)

1. **Perspective visual hull** `Ω = B ∩ ⋂_v π_v⁻¹(I_v)`. This is a voxel grid (128³ by default, 256³ in `high_quality`) built by back-projecting every voxel center into every target. No dilation is applied in STRICT mode.
2. **Hull SDF** `φ_Ω`, which is ≤ 0 inside and is queried trilinearly, with linear extrapolation outside B.
3. **Compatibility diagnostic**: `strict_coverage_upper_bound_v = |Q_v ∩ I_v| / |I_v|`, where `Q_v` is the exact projection of Ω, computed by ray-marching against the continuous hull membership. A bound below `hull.warn_coverage_bound` (0.9) produces a warning that the target pair is geometrically inconsistent under strict containment.
4. **Modes**:
   - `hull.mode: strict`: geometry must stay in Ω. This is enforced by penalties plus a hard `project_inside` step, which moves centers inside and shrinks scales to fit.
   - `relaxed`: Ω is dilated by `relaxed_dilation_voxels`, and spill is handled by the image loss.

## 5. Shared loss, constraints and evaluation

* **Silhouette loss**: `L_view = w_cov·Σ I(1−R)²/ΣI + w_neg·Σ(1−I)R²/Σ(1−I)` with `w_cov=1, w_neg=4`, averaged over views.
* **Multi-resolution**: levels 32…256 (up to 512 in `high_quality`). The weights interpolate from coarse-dominant to fine-dominant as optimization progresses. Renders and targets use identical area pooling.
* **Utility map** for proposals: `W_v(p)`.
  - The spec's first-order version `−∂L/∂R` is available as `gradient`.
  - The default is the **secant** version `L_p(R) − L_p(1)`. For binary candidate masks composited by union it is *exact*, so `ΔL = −Σ W·S`, and unlike the derivative it charges spill on empty background, where `∂R²/∂R = 0`.
* **ConstraintEvaluator** (`src/sculpture/constraints.py`): containment `mean ReLU(φ_Ω)²`, SDF collision `ReLU(−φ_j(x))²` with bounding-sphere broad phase, scale range and bounds. It exposes them as differentiable penalties, hard validity masks, `project_inside`, and combinatorial conflicts (pair lists). `constraints.hard_collisions: true` makes Beam reject intersecting candidates and makes the MILP add conflict cuts.
* **Evaluation** is identical for every method: the same renderer at `evaluation.resolution` (512).
  - Image metrics: IoU per view, mean IoU, min-view IoU, recall (coverage), spill (false positives / background), precision and loss.
  - Constraint metrics: intersecting pairs (dense winding-number reference), clearance violations (pairs closer than `collision_margin`) and containment violations (objects with max φ_Ω > `containment_tolerance`).
  - Diversity: `randomness` (normalized entropy of the piece-type counts) and `type_counts`.
  - Run metrics: object count (reported, never penalized), runtime and renderer calls.
* **Best solution** = highest `evaluation.selection_metric`. The default is `min_view_iou`, so that no view can be sacrificed. `mean_iou` or a `weighted` score can be chosen instead.

## 5b. Design options

These options apply to all methods. Set them in the UI's "Design options", with `--set`, or in YAML.

| Option | Config | Effect |
|---|---|---|
| **Fixed piece size** | `scale.mode: fixed`, `scale.fixed: 0.15` | Every piece has the same world size (bounding radius), and scale is never optimized. A piece that has to look bigger must move **toward the camera**. Candidates are placed at the depth `z = f·s/r` that gives the desired projected radius `r`. The strict-hull projection moves pieces (laterally, then along the camera ray) instead of shrinking them. |
| **Unbounded camera axis** (single view) | `bounding_volume.unbounded_view_axis: true`, `view_axis_near: 1.5`, `view_axis_far: 12` | B becomes the box around the camera frustum between those distances, so pieces can be much closer to or farther from the camera than the default cube. The hull grid becomes anisotropic, so lateral resolution is kept, and the camera's far plane is extended if needed. This is ignored, with a warning, for two views. |
| **Type variety** | `diversity.weight: 0.3`, `diversity.target: uniform` or `{cube: 2, sphere: 1, ...}` | Adds `weight · KL(type counts ‖ desired) / log M` to the objective. Beam ranks its children with it, and a **shared final swap stage** (all methods) turns pieces of over-used types into under-used ones. It tries silhouette-compatible bank orientations, locally optimized, and keeps a swap only if image loss + penalties + diversity improves and the new piece intersects nothing. The reported metric is `randomness` = normalized entropy of the type counts. |

### Intersections

Intersections are measured by a dense reference (`geometry/collisions.py::reference_intersections`):
- original-mesh surface samples, every vertex and edge midpoint;
- tested against the other piece's mesh with the exact generalized winding number.

The old fast detector (256 random samples, 0.01 penetration tolerance) missed about 90% of the real intersections: for example, 8 reported against 74 actual for a Medium Beam result. The causes were shallow penetrations, unsampled box corners, and a penalty diluted over samples.

Now:
* **Conservative boundary**: pieces closer than `constraints.collision_margin` (0.0025) count as intersecting. Collision points include all proxy vertices and edge midpoints. On every saved result, this test flagged all pairs that the reference found, for any margin ≥ 0.0025.
* **Penalty**: mean + max of `ReLU(margin − φ)²`, with `lambda_collision: 100`. Beam rejects candidates that conflict with the parent, column generation cuts conflicting pairs, and SDF ray packing uses a moderate weight while packing.
* **Shared final stage** (`refinement/feasibility.py`, `constraints.resolve_intersections: true`): separate the involved pieces, then shrink or nudge the weaker piece of each remaining pair, and remove it only if nothing works. This stage checks the margin test **and** the dense reference, so every method ends intersection-free whatever the margin.
* `metrics.collisions` is the reference count. `clearance_violations` counts pairs closer than the margin.

## 5c. Complex meshes: precise LODs, original substituted back

The optimizers never touch the full-resolution meshes. Each model gets two **precise level-of-detail approximations** (adaptive quadric decimation): the smallest mesh whose two-sided surface deviation from the original stays below a tolerance. The deviation is the 99.9th percentile of exact GPU point-to-triangle distances, in units of the bounding radius.

| LOD | Tolerance | Used for |
|---|---|---|
| `proxy` | `meshes.proxy_tolerance: 0.003` (≤ 20k faces) | losses at the working resolution, refinement, polish, collision points |
| `coarse` | `meshes.coarse_tolerance: 0.015` (≤ 3k faces) | candidate screening, silhouette bank, low-resolution lookahead |
| `original` | — | final intersection check, final evaluation render, GLB export |

* **Resolution-aware LOD** (`ctx.render(..., lod="auto")`): a render uses `coarse` when its worst-case projected error (LOD error × largest scale × focal length / nearest depth) is at most `meshes.lod_pixel_tolerance` (0.5 px), and `proxy` otherwise.
* **Clearance margins** are widened by each piece's proxy error × scale, so optimizing on proxies cannot hide an intersection of the originals.
* **Substitution at the end**: `reference_intersections` classifies dense original-mesh points against the other piece's **original** mesh in three tiers: the SDF grid; the proxy winding number for points farther than 2.5× the proxy error from its surface; and the exact winding number on the original for the rest. The final evaluation and renders use the original meshes. `metrics.proxy_min_view_iou` and `lod_substitution_delta` report the difference.
* The renderer splits scenes above `max_triangles` (1M per rasterize call) into chunks composited with max. It also halves batches automatically on rasterizer allocation failures. Unchunked, the original meshes overflowed nvdiffrast ("subtriangle count overflow" / illegal memory access).

Stress test (`python -m sculpture.cli make-demo --complex` writes a ~1M-triangle pool to `assets/models_complex`: 82k–330k triangles per model):

| | Simple pool | Complex pool |
|---|---|---|
| proxy / coarse faces | ≤ 1.5k (exact) | 8k–16k (err 0.0012–0.0028) / 500–1000 (err ≤ 0.014) |
| preprocessing (once, cached) | ~2 s | ~60 s |
| render 100 instances: proxy vs original | — | 0.017 s vs 0.13 s per call (originals ~8× slower) |
| Beam growth rounds in 30 s | 67 | 47 |
| fast min-view IoU: beam / CG / SDF ray | 0.887 / 0.765 / 0.886 | 0.865 / 0.752 / 0.888 |
| LOD substitution delta (original − proxy IoU) | 0 | ±0.0002 |
| intersections of the originals (final) | 0 | 0 |
| original-mesh intersection check, 100-piece overlap scene | — | 368 s → 16 s with the 3-tier check |

## 6. Running the methods

```bash
python -m sculpture.cli run --method beam              --target assets/targets/view_0.png --target assets/targets/view_1.png
python -m sculpture.cli run --method column_generation --target assets/targets/view_0.png --target assets/targets/view_1.png
python -m sculpture.cli run --method sdf_ray           --target assets/targets/view_0.png --target assets/targets/view_1.png

# all three + comparison.json / comparison.csv + automatic best selection
python -m sculpture.cli compare --models assets/models \
    --target assets/targets/view_0.png --target assets/targets/view_1.png --config configs/default.yaml

# useful options
--config fast|default|high_quality|path.yaml   # presets are merged over configs/default.yaml
--set beam.beam_width=8 --set hull.mode=relaxed # dotted overrides (YAML values)
--cameras cams.json   --seed 3   --output outputs/my_run   --resume
python -m sculpture.cli preprocess --target ...  # hull + compatibility diagnostics only
```

Presets:

| Preset | Runtime per method | Notes |
|---|---|---|
| `fast` | ≈ 1 min | 128 px working resolution, 96³ hull |
| `default` ("Medium") | ≈ 5 min | 256 px, 128³ hull |
| `high_quality` | up to 30 min | 512 px, 256³ hull |

**Checkpointing and resuming.** Every method writes `checkpoint.json` periodically (`progress.checkpoint_interval_s`). It holds the best-so-far assembly plus method state:
- Beam: the beam entries.
- Column generation: the column pool (`columns.npz`) and the conflict cuts.
- SDF: the current assembly and iteration. Adam moments are reset on resume.

Rerun with `--output <same dir> --resume` to continue. Runtime limits are `*.max_runtime_s`, and Beam also has `max_renderer_calls`.

**Reproducibility.** Seeds come from `seed`, and random generators are re-derived from (seed, iteration). Results are still not bit-exact, for two reasons: GPU atomics in nvdiffrast and `grid_sample` backward passes are non-deterministic, and stopping on wall-clock budgets is timing dependent. Use iteration budgets (`patience_rounds`, `pricing_rounds`, `iterations`) for tighter repeatability. All parameters are stored in `config.json` and in each `result.json`.

## 7. Method details

### Method 1: Beam constructive search (`methods/beam/`)
Start from an empty assembly. For each beam entry:
1. Compute the residual `U = I(1−R)` and the utility `W` at 64 px.
2. Generate 512 candidates:
   - Position: sampled in Ω, weighted by uncovered projections, with an exploration fraction.
   - Mesh and orientation: retrieved from the silhouette bank by matching the local utility patch.
   - Scale: from `f·s·ρ/z ≈` the inscribed radius of the uncovered region.
3. Screen by `Σ W·S`, then by the exact union loss.
4. Run a **local lookahead** that optimizes only the candidate (t, rot6d, log s) for 15 Adam steps with the parent frozen, using the image loss + containment + collision penalty. In strict mode, candidates are then projected into Ω and re-scored. Candidates that collide with the parent are rejected.
5. Form children: single adds, plus one **non-conflicting multi-add** whose dilated footprints don't overlap in any view.

The best B diverse children survive, with near-duplicate renders pruned. Every 3 rounds, and whenever a growth round fails to improve, the **best** beam entry gets a **global joint refinement** (60 steps, coarse-to-fine) and then **repairs**. Refining every entry was dropped: profiling showed it cost about 3 growth rounds while adding almost nothing beyond growth (`beam.refine_entries`, `beam.refine_on_stall`). The repairs are: leave-one-out marginal contributions, SWAP to a silhouette-compatible mesh, RESEED into the residual, and LARGE POSE JUMP. Deletion only undoes harmful objects that can't be repaired. The search stops at the target IoU, after no improvement for N rounds, at max objects, or at the time or renderer-call budget.

### Method 2: Column-generation MILP (`methods/column_generation/`)
* **Master** (`master.py`): `x_c ∈ {0,1}`, coverage `y_p ≤ Σ_{c∋p} x_c`, spill `z_p ≥ x_c`, `Σx ≤ max_objects` (safety only).
  - It maximizes `Σ a_p y_p − Σ b_p z_p` with `a_p = w_cov/(V|I_v|)` and `b_p = w_neg/(V|B_v|)`. For binary masks this equals `w_cov − L_image` **exactly**, so it optimizes the same loss as the other methods, at the MILP resolution (48 px by default).
  - Piece count is not in the objective.
* **Solvers** (`solver_base.py`): `MasterSolver` with a `ScipySolver` (HiGHS `linprog`/`milp`, tested) and a `GurobiSolver` (used automatically when `gurobipy` and a license are present, *not exercised here because no license was available*).
* **Pricing** (`pricing.py`): duals become an image map `W_dual`:
  - fg pixel with a coverage row: `u_p`;
  - fg pixel never covered: `a_p`;
  - bg pixel: the negative remaining spill price `b_p − Σ_c w_cp`.

  A column's reduced profit is `Σ W_dual·S − σ_card`. Batches of bank-initialized placements are optimized with nvdiffrast against `W_dual`, projected into Ω, binarized, and added if their profit is positive.
* **Conflicts** (`conflicts.py`): lazy. For the LP, every pair in the LP support is checked and the LP is re-solved only if a new cut is violated. For the MILP, conflicts among the selected columns, and between them and the pool, are cut until the selection is collision-free (with a final repair if the cut budget runs out).
* **Conservative column masks**: a target pixel counts as covered only if a column covers at least 70% of its area, and a background pixel counts as spilled already at 30% (`fg_coverage_threshold`, `bg_spill_threshold`). With a single 0.5 threshold, the coarse MILP grid over-estimated coverage: the MILP loss was 0.18, but the rendered loss was 0.25.
* **Final**: MILP plus a conflict-free greedy LP rounding as a second incumbent; the better one is kept. One time budget (`milp_time_share`) covers the whole MILP phase, including conflict re-solves. Then a raster polish with the shared loss.
* **Anchored re-optimization** (`max_anchor_rounds`, `anchor_patience`):
  - The polished pieces are added as columns fixed at x = 1.
  - New columns are priced against what is still uncovered, with conflicts against the anchors.
  - The MILP fills the holes, and the additions are polished.
  - Rounds repeat while they improve and time remains.

  This fixes the main weakness of a one-shot selection: rigid, coarse placements leave the target under-covered (recall 0.73), and the polish alone can't fill the gaps.
* Reporting: LP objective / loss bound (unanchored; the anchored LPs are not bounds and are reported separately as `anchored_*`), integer objective / loss, column count, pricing rounds, conflicts and MILP status.
* ⚠️ **Pricing is solved heuristically.** The LP bound is rigorous **only over the discovered columns**, not a global certificate for the continuous placement space. The log says so explicitly.

### Method 3: SDF ray packing (`methods/sdf_ray/`)
* `φ_i(x) = s_i·φ_mesh(R_iᵀ(x−t_i)/s_i)` with soft union over objects. Broad phase: pairs farther than a margin from an object's bounding sphere use the sphere distance.
* **Foreground rays** are sampled only on `ray ∩ Ω`. Rays that never meet Ω are reported as unsatisfiable.
  - `d_r = min_k φ_X(r(t_k))`.
  - `L_ray = mean τ·softplus(d_r/τ)`. This is the spec's softplus(d/τ) scaled by τ, so the gradient scale stays constant while τ anneals from 0.08 to 0.01.
  - Background rays get the mirrored term.
* Mins over objects and over ray samples use a **straight-through soft-min**: the forward value is the exact min and the backward is the soft-min gradient. A plain log-sum-exp is biased low by up to τ·log n, which made rays look "hit" when nothing intersected them.
* **Volumetric penetration** (`sdf_ray.w_overlap: 2.0`, `overlap_eps: 0.01`): `mean_q Σ_{i<j} σ(−φ_i/ε)·σ(−φ_j/ε)`. This is a Monte Carlo estimate of the soft pairwise overlap volume, computed on the foreground-ray samples inside Ω that are evaluated anyway. Unlike the surface-sample clearance term it is bounded and smooth, and it also penalizes one piece sitting fully inside another. It spreads pieces out instead of stacking them on the same rays.
* Containment `ReLU(φ_Ω)²`, SDF collisions and the scale range are included. Projected steps keep the solution feasible in the second half.
* Initialization: greedy max-utility selection from residual- and bank-guided candidates (not convex decomposition of Ω). The weakest objects (fewest owned rays) are reseeded periodically, and objects are added when ray coverage plateaus.
* **Anytime schedule**: the planned annealing cycle often ends far inside the budget (16 s of 60 s). While time remains and rays are still uncovered, the method adds pieces (collision-aware greedy initialization on the residual) and runs shorter, cooler packing cycles (`sdf_ray.anytime`, `max_extra_cycles`).
* Finally, a gentle **raster polish** (`polish_lr_scale: 0.3`, starting at fine weights) with the shared nvdiffrast multiscale loss determines the reported fidelity. The previous coarse-to-fine polish at full learning rate first moved the converged pieces away and never recovered.

## 8. UI

```bash
uvicorn sculpture.server.app:app --port 8000      # from the repo root (add --reload for development)
cd frontend && npm run dev                         # open http://localhost:5173
```

* **Setup**:
  - **Sculpture bounds**: min/max inputs, or drag the six colored face handles of the box in the 3D view (the white center handle moves the whole box). With one target and *Camera axis* mode, set the near/far distances from the camera with inputs or the yellow/pink handles on the view axis. The derived box is shown dashed. The bounds are sent as `bounding_volume` overrides.
  - Model folder: thumbnails, triangle counts and normalized dimensions.
  - Target 1 and an optional Target 2: pick from `assets/targets` or upload.
  - Camera editor: position, look-at, FOV, near and far, with live frusta and a drag gizmo.
  - Method checkboxes and a preset (Fast / Medium / High / Custom JSON overrides).
* **Live progress** over a WebSocket, with a polling fallback. Each method shows status, phase, iteration, runtime, loss, per-view IoU, object count and best min-view IoU. The current silhouettes appear next to the targets, and the 3D assembly updates periodically.
* **3D viewer** (React Three Fiber):
  - Orbit, pan and zoom; the real uploaded meshes are shown.
  - Toggles for the bounding volume, visual hull, camera frusta, per-object boxes, target images, a silhouette mode and a translucent target overlay.
  - **Target View 1/2** moves the viewer to the *exact* optimization camera so you can verify the illusion.
  - A Beam | Column Generation | SDF Ray | Best switch.
* **Comparison table** after all methods finish, with the automatically selected best highlighted and download links for `result.json`, `assembly.glb` and `comparison.json/csv`.

The HTTP/WebSocket API is documented in [docs/api.md](docs/api.md). Optimization runs in a background GPU worker thread, so requests never block.

## 9. Outputs

```
outputs/run_001/
  config.json
  preprocessing/  target_0.png target_1.png hull.npz hull_preview.obj diagnostics.json
  beam/ | column_generation/ | sdf_ray/
      result.json        # method, status, assembly, metrics, method info, cameras, full config
      metrics.json
      assembly.glb       # one node per instance (canonical mesh + world transform)
      renders/view_v.png overlay_v.png   # overlay: gray = correct, red = missed, blue = spill
      optimization.csv   # per-iteration log
      checkpoint.json    (+ columns.npz for column generation)
  comparison.json  comparison.csv
```

Assembly JSON, per object:
- `mesh_id`, `mesh_name`, `source_file`;
- `translation`, `rotation_matrix` (row-major), `rotation6d` (first two columns), `log_scale`, `scale`.

The world transform is `x_world = scale · R · x_canonical + translation`, where the canonical mesh is the source mesh centered at its bounding-box center and scaled to unit bounding radius (the normalization is stored in the cache and in `diagnostics.json`). `export_assembly(..., bake=True)` writes world-space vertices instead.

## 10. Example results (demo heart/star targets, RTX 4080)

`fast` preset, one run of `compare` (numbers vary slightly between runs):

| Method | min-view IoU | mean IoU | spill | collisions | containment viol. | objects | time |
|---|---|---|---|---|---|---|---|
| Beam | 0.912 | 0.926 | 0.24% | 3 | 0 | 92 | 62 s |
| Column generation | 0.788 | 0.849 | 0.28% | 0 | 0 | 82 | 63 s |
| SDF ray | 0.866 | 0.893 | 0.32% | 8 | 0 | 56 | 13 s |

`default` ("Medium") preset, 300 s budget per method (`compare` run; column generation re-run after the incumbent/budget fix):

| Method | min-view IoU | mean IoU | spill | collisions | containment viol. | objects | time |
|---|---|---|---|---|---|---|---|
| Beam | 0.970 | 0.974 | 0.14% | 8 | 0 | 121 | 304 s |
| Column generation | 0.927 | 0.943 | 0.24% | 0 | 1 | 164 | 315 s |
| SDF ray | 0.937 | 0.953 | 0.22% | 4 | 0 | 112 | 46 s |

The strict-containment upper bound for this pair is 0.981 and 0.983, which limits any method under strict mode. These tables were produced *before* the intersection work in section 5b. Their collision column used the old detector, which under-counted about 10×. See `outputs/experiments/summary.md` (`scripts/run_feature_experiments.py`) for runs with the current handling.

### Benchmark of the method improvements (2026-10-07)

`scripts/benchmark_versions.py` compares commit `8533862` (baseline) with the improved tree: the same heart/star targets, every run intersection-free, and alternating runs. Values are min-view IoU, with zero reference intersections in every run.

| Preset | Method | Baseline | Improved | Runtime (base → impr.) |
|---|---|---|---|---|
| fast (3 seeds, mean ± sd) | Beam | 0.889 ± 0.007 | 0.887 ± 0.014 | 64 → 66 s |
| | Column generation | 0.705 ± 0.049 | **0.765 ± 0.020** | 62 → 40 s |
| | SDF ray | 0.826 ± 0.019 | **0.853 ± 0.012** | 16 → 38 s |
| Medium (1 seed) | Beam | 0.918 | 0.913 | 311 → 313 s |
| | Column generation | 0.737 | **0.833** | 328 → 211 s |
| | SDF ray | 0.869 | **0.891** | 61 → 158 s |

* **Column generation**: conservative masks plus anchored re-optimization give the largest gain. The first anchored round typically cuts the loss by about 30% (0.155 → 0.110 on Medium).
* **SDF ray**: the anytime cycles use the budget, and the gentle polish now helps (0.109 → 0.075 loss on Medium; before, the polish changed nothing).
* **Beam**: refining only the best entry is neutral for IoU (same budget, about 50% more pieces). Beam is limited by the strict no-intersection requirement: Medium Beam was 0.970 before intersections were enforced.
* Improved column generation and SDF ray hit the `max_objects: 200` safety cap on Medium. Raising it is the next lever, since piece count is not penalized.

### Volumetric penetration loss for SDF ray packing (2026-10-07)

`scripts/benchmark_versions.py --methods sdf_ray` compares the method with `w_overlap = 0` (baseline) against `w_overlap = 2`. The weight was tuned on seeds 0–1; the benchmark uses held-out seeds.

| Preset | Version | min-view IoU | Pieces | Conflicts left for the final stage | Pieces removed there |
|---|---|---|---|---|---|
| fast (seeds 2–6) | baseline | 0.855 ± 0.019 | 120 (cap) | 4–6 | 2 (1 run) |
| fast (seeds 2–6) | + overlap | **0.886 ± 0.012** | ≈93 | 0 in 4 of 5 runs | 0 |
| Medium (seeds 0–1) | baseline | 0.897 | 198 (cap) | 14–16 | 2 per run |
| Medium (seeds 0–1) | + overlap | 0.896 | ≈94 | 4–5 | 0 |

On fast, it gives +0.031 IoU (better on 4 of 5 seeds, tied on 1). On Medium, IoU is unchanged, but it reaches that with half the pieces, about 3× fewer intersections to repair, and no removals. The final resolution stage stays as the guarantee, because shallow slivers have almost no volume.

## 11. Code map

```
src/sculpture/
  config.py, context.py (ProblemContext = shared preprocessing), constraints.py, demo.py, cli.py
  geometry/  mesh_library, mesh_preprocess, rotation (6D), transforms, sdf, collisions
  scene/     object_instance, assembly (batched tensors), serialization
  camera/    perspective_camera, projection (exact pinhole, rays)
  targets/   target loading, pyramids, distance transforms
  hull/      visual_hull, hull_sdf, compatibility
  rendering/ base (Renderer protocol), nvdiffrast_renderer (only place nvdiffrast is imported)
  loss/      silhouette, multiscale, containment, collision
  proposals/ residual, utility_map, silhouette_bank, candidate_generator
  refinement/continuous.py   fixed-structure joint refinement (Milestone-3 baseline, shared final polish)
  methods/   base (budgets, logging, checkpoints), beam/, column_generation/, sdf_ray/
  evaluation/ metrics, compare (experiment runner)
  io/        mesh_export (GLB), result_writer
  server/    app (FastAPI), jobs (worker thread), websocket, schemas
frontend/    React + TypeScript + Vite + three.js / R3F
tests/       projection, renderer gradients, hull, loss, refine, proposals, serialization,
             beam / column generation / SDF smoke tests (two-view + single-view circle), server
```

Adding a renderer means implementing `rendering/base.py::Renderer` (`render_silhouettes`, `render_instance_silhouettes`) and passing it to `build_context(..., renderer=...)`. Optimization code never touches nvdiffrast.

## 12. Not implemented / known limitations

* **Gurobi backend** is written against the same interface but untested here (no license). HiGHS through SciPy is the tested path.
* **Clique cuts** are not implemented; the MILP uses only lazy pairwise conflict cuts.
* **Fast collision test** uses collision points against canonical SDF grids (64³) with a clearance margin. It is conservative, and the dense reference (`reference_intersections`) verifies it. Extremely thin parts thinner than an SDF voxel may need a larger `meshes.sdf_resolution`.
* **Bank retrieval** ignores off-axis perspective distortion; it is used only for initialization. Swap uses the view-0 orientation.
* The SDF method resets Adam moments when it reseeds or resumes. Beam repair runs on every beam entry, which is costly for wide beams.
* There is no fabrication-specific support or suspension modelling yet. `ConstraintEvaluator` is the place to add it.

## 13. Troubleshooting

* **`nvdiffrast` import or compile errors**: install a CUDA toolkit matching your PyTorch CUDA version and a C++ compiler (MSVC Build Tools on Windows), or use a prebuilt wheel. `python -c "import nvdiffrast.torch as dr; dr.RasterizeCudaContext()"` should succeed.
* **"resolutions divisible by 8"**: the CUDA rasterizer requires it. Working and evaluation resolutions are rounded automatically; use multiples of 8 for the per-method resolutions.
* **Empty visual hull or low coverage bound**: the targets are inconsistent for these cameras (for orthogonal views sharing the vertical axis, each image row must have foreground in both or neither). Change the cameras or targets, or use `--set hull.mode=relaxed`.
* **Inverted silhouettes**: set `--set targets.invert=true` (dark is foreground) or `false`.
* **CUDA out of memory**: lower `targets.working_resolution`, `beam.candidates_per_branch`, `column_generation.pricing_batch`, `sdf_ray.rays_per_view`, or `hull.resolution`.
* **Slow first start**: mesh preprocessing, SDFs and the silhouette bank are computed once and then cached in `cache/`. Delete `cache/` after changing preprocessing code.
* **The MILP hits its time limit**: raise `column_generation.milp_time_limit_s`, lower `resolution` or `initial_columns`, or install Gurobi.
* **Frontend cannot reach the API**: the backend must run on port 8000 (the Vite proxy target) and be started from the repo root, so that `assets/` and `outputs/` resolve.
