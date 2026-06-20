# M2 DST Integration Comparison

Lower is better for `unified_loss` and `hard_score`. This compares the newly generated M2 edge-policy DST outputs against existing B1, Graph-TSP, and M1 config-regressor baselines on the same four held-out paired samples.

| Method | Samples | Mean Unified | Mean Hard | Exec | Visual Risk | Jumps | Trims | Off-mask mm | Visible | Hard Fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| m2_edge_policy_top8 | 4 | 0.610504 | 42.110343 | 0.337873 | 0.272631 | 203.500 | 20.750 | 171.912 | 32.500 | 4 |
| m1_config_regressor | 4 | 0.621028 | 41.953114 | 0.343210 | 0.277818 | 143.500 | 25.250 | 179.303 | 35.750 | 4 |
| b2_graph_tsp_conservative_safe | 4 | 0.626582 | 44.170873 | 0.349371 | 0.277211 | 213.500 | 26.250 | 175.534 | 35.250 | 4 |
| b1_conservative | 4 | 0.636772 | 42.691112 | 0.342067 | 0.294706 | 130.750 | 25.250 | 185.828 | 37.000 | 4 |

## Interpretation

- M2 top-k=8 improves mean unified loss versus B1, Graph-TSP, and M1 config on this 4-sample set.
- The improvement is not clean enough to call final quality solved: all rows remain `hard_fail`, and pair_006 still has very high visible connector/off-mask risk.
- M2 strongly changes routing where candidate edges exist: pair_006 has 185 M2 decisions / 53 overrides, pair_010 has 847 / 388, while pair_011a has no multi-candidate M2 decision opportunity.

## Files

- `m2_vs_baselines_per_sample.csv`
- `m2_vs_baselines_deltas.csv`
- `m2_vs_baselines_summary.json`