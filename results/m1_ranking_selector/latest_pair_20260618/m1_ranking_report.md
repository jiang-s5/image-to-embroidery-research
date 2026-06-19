# M1 Ranking Planner Selector

This stage turns evaluator outputs into preference supervision.

Instead of only asking which single preset is the label, it trains a pairwise ranker:

```text
image / predicted geometry + candidate config A/B
        |
pairwise logistic ranking loss
        |
utility(candidate)
        |
choose the highest-utility planner config
```

## Requirement Check

| Requirement | Status | Evidence |
| --- | --- | --- |
| Preference learning | Done | pairwise logistic loss over evaluator-ranked candidates |
| Image-conditioned config choice | Done | candidate utility includes image geometry x config interactions |
| Leave-one-out validation | Done | every sample is held out from the ranking pairs used to train its selector |
| No post-export feature leakage at inference | Done | sample features exclude `preset.*`, `interaction.*`, and evaluator metrics |

## Best Leave-One-Out Result

- target metric: `hard_score`
- min_delta: `1e-06`
- lr: `0.005`
- weight_decay: `0.0`
- epochs: `300`
- seed: `1`
- oracle-match accuracy: `0.5`
- mean train pair accuracy: `0.98308081`
- model artifact: `checkpoints/m1_ranking_selector.json`
- ranking samples: `results/m1_ranking_selector/latest_pair_20260618/m1_ranking_samples.jsonl`
- predictions: `results/m1_ranking_selector/latest_pair_20260618/m1_ranking_predictions.csv`

| Metric | Ranking M1 LOO Mean |
| --- | ---: |
| unified_loss | 0.621812 |
| hard_score | 53.670513 |
| jump_count | 157.500000 |
| trim_count | 24.250000 |
| off_mask_stitch_length_mm | 176.956275 |
| visible_connector_count | 40.250000 |

## Fold Decisions

| Sample | Selected Method | Oracle Method | Correct | Selected Unified | Oracle Unified | Selected Hard | Oracle Hard |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| pair_006 | b2_graph_tsp_safe | b1_conservative | False | 0.646479 | 0.643384 | 117.766796 | 111.986480 |
| pair_008 | b2_graph_tsp_conservative_safe | b2_graph_tsp_conservative_safe | True | 0.562265 | 0.562265 | 23.734511 | 23.734511 |
| pair_010 | b1_conservative | b1_conservative | True | 0.648072 | 0.648072 | 29.495027 | 29.495027 |
| pair_011a | b2_graph_tsp_safe | b2_graph_tsp_conservative_safe | False | 0.630431 | 0.630431 | 43.685718 | 42.398102 |

## Baselines

| Baseline | Unified | Hard Score | Jumps | Trims | Visible Connectors | Off-Mask mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle_preference_label | 0.621038 | 51.903530 | 143.250 | 25.250 | 35.750 | 178.987 |
| fixed_b1_conservative | 0.636772 | 52.691112 | 130.750 | 25.250 | 37.000 | 185.828 |
| fixed_b1_dense_thread | 0.649885 | 195.571927 | 265.750 | 42.250 | 221.500 | 786.770 |
| fixed_b1_low_connect | 0.628786 | 60.677699 | 203.500 | 26.750 | 46.250 | 201.238 |
| fixed_b1_region_fill | 0.633435 | 183.293658 | 211.000 | 34.250 | 206.000 | 750.728 |
| fixed_b2_graph_tsp_conservative_safe | 0.626582 | 54.170873 | 213.500 | 26.250 | 35.250 | 175.534 |
| fixed_b2_graph_tsp_safe | 0.628021 | 56.397792 | 227.500 | 26.250 | 40.000 | 176.191 |

## Interpretation

- This is the first M1 variant that uses evaluator feedback as pairwise preference supervision.
- It is still a planner-selector model, not an M2 graph edge policy.
- A learned M2 would need segment/edge labels or graph traces and should be reported separately.