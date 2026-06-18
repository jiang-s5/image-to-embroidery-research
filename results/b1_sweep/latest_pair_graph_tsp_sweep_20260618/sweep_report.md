# B1 Sweep Report

Lower unified loss is better. The paired cases are evaluation cases only; this script does not add them to training.

| Method | Samples | Mean Unified Loss | Mean Exec | Mean Visual Risk | Mean Jumps | Mean Trims | Mean Off-Mask mm | Mean Visible Connectors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b2_graph_tsp_conservative_safe | 4 | 0.626582 | 0.349371 | 0.277211 | 213.500 | 26.250 | 175.534 | 35.250 |
| b2_graph_tsp_safe | 4 | 0.628021 | 0.350810 | 0.277211 | 227.500 | 26.250 | 176.192 | 40.000 |
| b1_low_connect | 4 | 0.628786 | 0.350968 | 0.277818 | 203.500 | 26.750 | 201.238 | 46.250 |
| b1_region_fill | 4 | 0.633435 | 0.355619 | 0.277817 | 211.000 | 34.250 | 750.728 | 206.000 |
| b1_conservative | 4 | 0.636772 | 0.342067 | 0.294706 | 130.750 | 25.250 | 185.828 | 37.000 |
| b1_dense_thread | 4 | 0.649885 | 0.364587 | 0.285297 | 265.750 | 42.250 | 786.770 | 221.500 |