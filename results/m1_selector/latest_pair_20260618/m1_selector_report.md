# M1 Planner Selector Loop

M1 is a lightweight learned planner selector. It does not generate stitches directly; it selects a planner preset from the current preset bank using pre-export image/geometry summary features plus preset parameters.

The current dataset is intentionally small, so the main validation is leave-one-out by sample. Treat this as a loop-engineering checkpoint, not a final generalization claim.

## Best Leave-One-Out Selector

- target: `hard_rank`
- alpha: `0.01`
- shrinkage-to-method-prior: `0.75`
- selection target: `hard_score`
- model artifact: `checkpoints/m1_planner_selector.json`
- training rows: `results/m1_selector/latest_pair_20260618/m1_selector_samples.jsonl`

| Metric | M1 Selected Mean |
| --- | ---: |
| unified_loss | 0.621367 |
| exec_score | 0.343550 |
| visual_risk | 0.277818 |
| jump_count | 154.000000 |
| trim_count | 24.250000 |
| jump_path_mm | 829.676300 |
| off_mask_stitch_length_mm | 176.803500 |
| visible_connector_count | 35.500000 |
| visible_connector_length_mm | 106.547600 |
| stitch_count | 3067.000000 |
| stitch_path_mm | 3021.262200 |
| hard_score | 51.927638 |
| hard_rank | 0.750000 |
| mean_rank | 1.250000 |

## Fold Decisions

| Sample | M1 Method | Oracle Method | Selected Unified | Oracle Unified | Selected Hard Score | Oracle Hard Score |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| pair_006 | b2_graph_tsp_conservative_safe | b1_conservative | 0.644702 | 0.643384 | 112.082914 | 111.986480 |
| pair_008 | b2_graph_tsp_conservative_safe | b2_graph_tsp_safe | 0.562265 | 0.562265 | 23.734511 | 23.734511 |
| pair_010 | b1_conservative | b1_conservative | 0.648072 | 0.648072 | 29.495027 | 29.495027 |
| pair_011a | b2_graph_tsp_conservative_safe | b2_graph_tsp_conservative_safe | 0.630431 | 0.630431 | 42.398102 | 42.398102 |

## Baselines

| Baseline | Unified | Hard Score | Visible Connectors | Off-Mask mm | Jumps | Trims |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle_hard_score | 0.621038 | 51.903530 | 35.750 | 178.987 | 143.250 | 25.250 |
| train_fold_best_mean_method | 0.641987 | 54.934346 | 36.750 | 184.559 | 190.250 | 27.250 |
| b1_conservative | 0.636772 | 52.691112 | 37.000 | 185.828 | 130.750 | 25.250 |
| b1_dense_thread | 0.649885 | 195.571927 | 221.500 | 786.770 | 265.750 | 42.250 |
| b1_low_connect | 0.628786 | 60.677699 | 46.250 | 201.238 | 203.500 | 26.750 |
| b1_region_fill | 0.633435 | 183.293658 | 206.000 | 750.728 | 211.000 | 34.250 |
| b2_graph_tsp_conservative_safe | 0.626582 | 54.170873 | 35.250 | 175.534 | 213.500 | 26.250 |
| b2_graph_tsp_safe | 0.628021 | 56.397792 | 40.000 | 176.191 | 227.500 | 26.250 |

## Loop Search Leaderboard

| Rank | Target | Alpha | Shrinkage | Unified | Hard Score | Visible | Off-Mask mm | Jumps |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | hard_rank | 0.01 | 0.75 | 0.621367 | 51.927638 | 35.500 | 176.804 | 154.000 |
| 2 | hard_rank | 0.1 | 0.75 | 0.621367 | 51.927638 | 35.500 | 176.804 | 154.000 |
| 3 | hard_rank | 1 | 0.50 | 0.621367 | 51.927638 | 35.500 | 176.804 | 154.000 |
| 4 | hard_rank | 1 | 0.75 | 0.621367 | 51.927638 | 35.500 | 176.804 | 154.000 |
| 5 | hard_rank | 10 | 0.25 | 0.621367 | 51.927638 | 35.500 | 176.804 | 154.000 |
| 6 | hard_rank | 10 | 0.50 | 0.621367 | 51.927638 | 35.500 | 176.804 | 154.000 |
| 7 | hard_rank | 0.1 | 0.50 | 0.621367 | 52.249542 | 36.750 | 176.045 | 154.000 |
| 8 | hard_rank | 1 | 0.25 | 0.621367 | 52.249542 | 36.750 | 176.045 | 154.000 |
| 9 | hard_rank | 10 | 0.00 | 0.621367 | 52.249542 | 36.750 | 176.045 | 154.000 |
| 10 | hard_score | 0.1 | 0.75 | 0.636772 | 52.691112 | 37.000 | 185.828 | 130.750 |
| 11 | mean_rank | 100 | 0.50 | 0.636772 | 52.691112 | 37.000 | 185.828 | 130.750 |
| 12 | mean_rank | 100 | 0.75 | 0.636772 | 52.691112 | 37.000 | 185.828 | 130.750 |

## Interpretation

- This reaches the M1 stage because planner choice is now a trained selector with leave-one-out evaluation and a saved model artifact.
- It is not M2: it does not predict individual graph edges or routes.
- The next credible improvement is to expand the holdout set and add pre-export graph statistics from `graph_tsp_trace.json` generated before final DST export.