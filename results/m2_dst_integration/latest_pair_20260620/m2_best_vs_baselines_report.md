# M2 Best-vs-Baselines Comparison

Best M2 setting: `--m2-edge-policy-top-k 4`. Lower is better for `unified_loss` and `hard_score`.

| Method | Samples | Mean Unified | Mean Hard | Exec | Visual Risk | Jumps | Trims | Off-mask mm | Visible | Hard Fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| m2_edge_policy_top4 | 4 | 0.607582 | 41.652422 | 0.334952 | 0.272631 | 192.750 | 20.250 | 171.912 | 32.500 | 4 |
| m1_config_regressor | 4 | 0.621028 | 41.953114 | 0.343210 | 0.277818 | 143.500 | 25.250 | 179.303 | 35.750 | 4 |
| b2_graph_tsp_conservative_safe | 4 | 0.626582 | 44.170873 | 0.349371 | 0.277211 | 213.500 | 26.250 | 175.534 | 35.250 | 4 |
| b1_conservative | 4 | 0.636772 | 42.691112 | 0.342067 | 0.294706 | 130.750 | 25.250 | 185.828 | 37.000 | 4 |

## Judgment

- M2 top4 is better than Graph-TSP and B1 on mean unified loss, mean hard score, trim count, off-mask length, and visible connector count.
- M2 top4 is also slightly better than M1 config on mean unified loss and mean hard score, but it uses more jumps.
- This is a measurable prototype improvement, not final DST quality: every sample is still `hard_fail`, mostly because jump/trim/off-mask/visible thresholds are still exceeded on hard cases.

## Related Artifacts

- `m2_topk_sweep_report.md`
- `m2_best_vs_baselines_per_sample.csv`
- `m2_best_vs_baselines_summary.json`