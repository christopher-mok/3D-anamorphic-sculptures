| arm | method | n | min-view IoU (mean ± sd) | paired Δ vs baseline | wins | collisions | final-stage conflicts | objects | runtime |
|---|---|---|---|---|---|---|---|---|---|
| beam_base | beam | 3 | 0.8909 ± 0.0113 |  |  | 0.0 | 3.0 | 101 | 64 s |
| beam_overlap | beam | 3 | 0.8801 ± 0.0205 | -0.0108 | 1/3 | 0.0 | 2.7 | 80 | 67 s |
| beam_milp_add | beam | 3 | 0.8744 ± 0.0147 | -0.0165 | 0/3 | 0.0 | 2.3 | 91 | 65 s |
| beam_raymeet | beam | 3 | 0.9015 ± 0.0190 | +0.0106 | 2/3 | 0.0 | 1.7 | 105 | 65 s |
| cg_base | column_generation | 3 | 0.7775 ± 0.0121 |  |  | 0.0 | 3.0 | 120 | 36 s |
| cg_overlap | column_generation | 3 | 0.8115 ± 0.0051 | +0.0340 | 3/3 | 0.0 | 1.3 | 120 | 37 s |
| sdf_base | sdf_ray | 3 | 0.8928 ± 0.0168 |  |  | 0.0 | 2.3 | 99 | 44 s |
| sdf_raymeet | sdf_ray | 3 | 0.9067 ± 0.0151 | +0.0139 | 3/3 | 0.0 | 1.0 | 100 | 44 s |
