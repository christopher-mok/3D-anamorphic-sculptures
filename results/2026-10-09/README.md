# Results — merged method, A/B-tested techniques, editing / viewing options (2026-10-09)

Targets: heart / star demo pair, two orthogonal cameras, demo model pool. Metric: **min-view IoU**
(the worse of the two views, evaluated with the original meshes at 256 px fast / 512 px Medium);
every run listed here ended with **0 intersections** (dense winding-number reference check).

## 1. Final benchmark: this version vs. the previous push (`4263c72`)

`scripts/benchmark_versions.py` (frozen code snapshots, versions alternated). Fast = 60 s per method,
3 seeds; Medium = 300 s per method, 1 seed. Full table: `final_benchmark/benchmark.md`.

| Preset | Method | `4263c72` | this version |
|---|---|---|---|
| fast | Beam | 0.883 ± 0.013 | 0.875 ± 0.012 |
| fast | Column generation | 0.821 ± 0.020 | 0.817 ± 0.027 |
| fast | SDF ray | 0.895 ± 0.008 | **0.907 ± 0.015** |
| fast | **Chained (new)** | — | **0.906 ± 0.009** |
| Medium | Beam | 0.890 | 0.895 |
| Medium | Column generation | 0.876 | **0.901** |
| Medium | SDF ray | 0.913 | 0.922 |
| Medium | **Chained (new)** | — | **0.935** (best) |

On the fast preset the Beam and column-generation differences are within run-to-run noise (seed
sd ≈ 0.01–0.03; budgets are wall-clock, so results also depend on machine load). The paired A/B
tests below isolate each change on identical code and seeds.

## 2. A/B tests of each technique (`ab_tests/`, `scripts/ab_test.py`)

Same code, only the tested option differs; paired seeds; fast preset.

| Technique | Paired Δ min-view IoU (wins) | Decision |
|---|---|---|
| Ray-meet candidate placement (uncovered rays of both views meet in 3D), SDF ray | +0.014 (3/3), batch 1 | **adopted** |
| Ray-meet placement, Beam | +0.011 (2/3) batch 1; **+0.018 (2/3)** clean re-test, seeds 3–5 | **adopted** |
| Volumetric overlap loss in column generation's polish | +0.034 (3/3) batch 1; **+0.023 (3/3)** clean re-test | **adopted** (CG only) |
| Volumetric overlap loss in Beam's (shared) refinement | −0.011 (1/3) | not used for Beam |
| Beam multi-add set chosen by exact MILP | −0.017 (0/3) | removed |
| Chained SDF → CG → Beam vs Beam | **+0.009 (3/3)** | new method 4 |
| Multi-start ×3 (concurrent, same wall time): Beam / SDF ray / CG | −0.012 (1/3) / −0.013 (0/3) / +0.004 (2/3, noise) | removed |
| Coarse-to-fine with native sizes (large models first): Beam / SDF ray | −0.052 (1/3) / −0.114 (1/3) | removed |

Batch 1 ran while headless-browser UI tests were also running on the machine (CPU contention mostly
affects column generation's CPU-bound MILP); the two techniques it supported were therefore re-tested
in a clean batch (`batch4`) on fresh seeds, which confirmed both.

## 3. Features verified end to end (`screenshots/`)

* **Chained** job through the API: stages SDF ray 0.858 → CG 0.863 → Beam 0.887 min-view IoU
  (`chained_result.png`, `chained_target_view_1.png`).
* **Lock-and-rerun**: 5 pieces of that result locked, one unlocked piece moved, Beam rerun through
  the API: all 5 locked pieces returned unchanged, 0 intersections (`lock_rerun_result.png` shows the
  edit panel: "80 pieces · 5 locked").
* Viewing zone, reveal, native sizes and lock-safety of every stage are covered by
  `tests/test_merged_features.py` (63 tests pass in total).

## 4. Files

* `final_benchmark/benchmark.{md,json}` — version-vs-version benchmark.
* `ab_tests/batch{1,2,3,4}/{spec.json,results.json,summary.md}` — every A/B arm and seed
  (batch-1 spec refers to the since-removed `beam.multi_add_solver` option).
* `renders/medium_*_overlay_{0,1}.png` — Medium results per method (gray = correct, red = missed
  target, blue = spill); `medium_chained_result.json` / `medium_chained_assembly.glb` — the best
  sculpture (chained, 0.935).
