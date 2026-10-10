| arm | method | n | min-view IoU (mean ± sd) | paired Δ vs baseline | wins | collisions | final-stage conflicts | objects | runtime |
|---|---|---|---|---|---|---|---|---|---|
| beam_no_raymeet | beam | 3 | 0.8841 ± 0.0056 |  |  | 0.0 | 5.3 | 99 | 65 s |
| beam_raymeet | beam | 3 | 0.9022 ± 0.0117 | +0.0181 | 2/3 | 0.0 | 2.7 | 107 | 66 s |
| cg_no_overlap | column_generation | 3 | 0.8067 ± 0.0035 |  |  | 0.0 | 3.0 | 119 | 36 s |
| cg_overlap | column_generation | 3 | 0.8298 ± 0.0120 | +0.0231 | 3/3 | 0.0 | 1.3 | 120 | 37 s |
