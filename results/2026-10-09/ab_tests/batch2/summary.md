| arm | method | n | min-view IoU (mean ± sd) | paired Δ vs baseline | wins | collisions | final-stage conflicts | objects | runtime |
|---|---|---|---|---|---|---|---|---|---|
| beam_base | beam | 3 | 0.8970 ± 0.0070 |  |  | 0.0 | 3.3 | 109 | 65 s |
| beam_multistart3 | beam | 3 | 0.8852 ± 0.0120 | -0.0118 | 1/3 | 0.0 | 2.0 | 82 | 67 s |
| chained | chained | 3 | 0.9056 ± 0.0088 | +0.0086 | 3/3 | 0.0 | 11.3 | 103 | 66 s |
| sdf_base | sdf_ray | 3 | 0.9040 ± 0.0110 |  |  | 0.0 | 1.0 | 102 | 43 s |
| sdf_multistart3 | sdf_ray | 3 | 0.8908 ± 0.0073 | -0.0131 | 0/3 | 0.0 | 1.7 | 71 | 39 s |
| beam_native | beam | 3 | 0.7306 ± 0.0547 |  |  | 0.0 | 3.0 | 98 | 65 s |
| beam_native_c2f | beam | 3 | 0.6784 ± 0.0405 | -0.0523 | 1/3 | 0.0 | 3.0 | 80 | 66 s |
| sdf_native | sdf_ray | 3 | 0.7827 ± 0.0936 |  |  | 0.0 | 2.3 | 60 | 43 s |
| sdf_native_c2f | sdf_ray | 3 | 0.6692 ± 0.0397 | -0.1135 | 1/3 | 0.0 | 3.7 | 66 | 43 s |
