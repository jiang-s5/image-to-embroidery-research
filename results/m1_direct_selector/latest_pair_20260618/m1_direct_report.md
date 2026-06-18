# Strict M1 Direct Planner Selector

This is the strict M1 stage: a trainable MLP maps image/geometry summary features directly to a planner preset/config.

Unlike the earlier M0.9 scorer, strict M1 does not score every candidate preset as input. It predicts one preset from sample-level features.

## Requirement Check

| Requirement | Status | Evidence |
| --- | --- | --- |
| Learning model | Done | NumPy MLP with tanh hidden layer and softmax output |
| Training loss | Done | Cross-entropy loss saved in the model artifact |
| Image to config mapping | Done | Input features exclude `preset.*` and `interaction.*`; output is a preset plus config args |
| Closed-loop evaluation | Done | Leave-one-out by sample against sweep-derived labels |

## Best Leave-One-Out Result

- hidden_dim: `4`
- lr: `0.005`
- weight_decay: `0.0`
- epochs: `600`
- seed: `1`
- accuracy: `0.5`
- model artifact: `checkpoints/m1_direct_selector.json`
- sample dataset: `results/m1_direct_selector/latest_pair_20260618/m1_direct_samples.jsonl`

| Metric | Strict M1 LOO Mean |
| --- | ---: |
| unified_loss | 0.621038 |
| hard_score | 52.225434 |
| jump_count | 143.250000 |
| trim_count | 25.250000 |
| off_mask_stitch_length_mm | 178.228500 |
| visible_connector_count | 37.000000 |

## Fold Decisions

| Sample | Predicted Method | Label Method | Correct | Confidence | Selected Unified | Oracle Unified | Selected Hard | Oracle Hard |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pair_006 | b1_conservative | b1_conservative | True | 0.563981 | 0.643384 | 0.643384 | 111.986480 | 111.986480 |
| pair_008 | b2_graph_tsp_conservative_safe | b2_graph_tsp_safe | False | 0.822113 | 0.562265 | 0.562265 | 23.734511 | 23.734511 |
| pair_010 | b1_conservative | b1_conservative | True | 0.801789 | 0.648072 | 0.648072 | 29.495027 | 29.495027 |
| pair_011a | b2_graph_tsp_safe | b2_graph_tsp_conservative_safe | False | 0.729423 | 0.630431 | 0.630431 | 43.685718 | 42.398102 |

## Baselines

| Baseline | Unified | Hard Score | Jumps | Trims | Visible Connectors | Off-Mask mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle_label | 0.621038 | 51.903530 | 143.250 | 25.250 | 35.750 | 178.987 |
| fixed_b1_conservative | 0.636772 | 52.691112 | 130.750 | 25.250 | 37.000 | 185.828 |
| fixed_b1_dense_thread | 0.649885 | 195.571927 | 265.750 | 42.250 | 221.500 | 786.770 |
| fixed_b1_low_connect | 0.628786 | 60.677699 | 203.500 | 26.750 | 46.250 | 201.238 |
| fixed_b1_region_fill | 0.633435 | 183.293658 | 211.000 | 34.250 | 206.000 | 750.728 |
| fixed_b2_graph_tsp_conservative_safe | 0.626582 | 54.170873 | 213.500 | 26.250 | 35.250 | 175.534 |
| fixed_b2_graph_tsp_safe | 0.628021 | 56.397792 | 227.500 | 26.250 | 40.000 | 176.191 |

## Interpretation

- This is now a formal M1 prototype, not only an oracle-driven sweep scorer.
- The supervision labels are still derived from a small sweep, so the result is not a mature generalization claim.
- The next real upgrade is to expand the sweep benchmark, then train the same direct selector on Pilot-50 / Release-200.