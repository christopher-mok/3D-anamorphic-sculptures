| preset | method | version | n | min_view_iou (mean ± sd) | mean_iou | spill | collisions | containment viol. | objects | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|
| fast | beam | baseline | 3 | 0.883 ± 0.013 | 0.901 | 0.0034 | 0.0 | 0.0 | 101 | 65 |
| fast | beam | improved | 3 | 0.875 ± 0.012 | 0.899 | 0.0029 | 0.0 | 0.0 | 102 | 65 |
| fast | column_generation | baseline | 3 | 0.821 ± 0.020 | 0.863 | 0.0027 | 0.0 | 0.0 | 120 | 38 |
| fast | column_generation | improved | 3 | 0.817 ± 0.027 | 0.863 | 0.0027 | 0.0 | 0.0 | 120 | 35 |
| fast | sdf_ray | baseline | 3 | 0.895 ± 0.008 | 0.918 | 0.0028 | 0.0 | 0.0 | 99 | 46 |
| fast | sdf_ray | improved | 3 | 0.907 ± 0.015 | 0.926 | 0.0031 | 0.0 | 0.0 | 100 | 44 |
| fast | chained | improved | 3 | 0.906 ± 0.009 | 0.924 | 0.0039 | 0.0 | 0.0 | 104 | 65 |
| default | beam | baseline | 1 | 0.890 ± 0.000 | 0.908 | 0.0034 | 0.0 | 0.0 | 174 | 313 |
| default | beam | improved | 1 | 0.895 ± 0.000 | 0.909 | 0.0041 | 0.0 | 1.0 | 149 | 307 |
| default | column_generation | baseline | 1 | 0.876 ± 0.000 | 0.915 | 0.0032 | 0.0 | 0.0 | 196 | 223 |
| default | column_generation | improved | 1 | 0.901 ± 0.000 | 0.925 | 0.0027 | 0.0 | 0.0 | 196 | 198 |
| default | sdf_ray | baseline | 1 | 0.913 ± 0.000 | 0.928 | 0.0028 | 0.0 | 0.0 | 91 | 210 |
| default | sdf_ray | improved | 1 | 0.922 ± 0.000 | 0.937 | 0.0023 | 0.0 | 0.0 | 113 | 230 |
| default | chained | improved | 1 | 0.935 ± 0.000 | 0.946 | 0.0031 | 0.0 | 0.0 | 177 | 316 |
