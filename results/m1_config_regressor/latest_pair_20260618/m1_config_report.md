# M1 Config Regressor

This is the stronger M1 variant: it maps image/geometry features directly to continuous planner parameters.

It is different from preset classification because the output is a planner config vector, not only a preset id.

## Best Leave-One-Out Config Regression

- hidden_dim: `4`
- lr: `0.05`
- weight_decay: `0.01`
- epochs: `800`
- seed: `13`
- mean config MSE: `17.75667872`
- model artifact: `checkpoints/m1_config_regressor.json`
- sample dataset: `results/m1_config_regressor/latest_pair_20260618/m1_config_samples.jsonl`
- LOO predicted configs: `results/m1_config_regressor/latest_pair_20260618/m1_config_predictions.json`

## Fold Predictions

| Sample | Label Method | Nearest Preset | Config MSE |
| --- | --- | --- | ---: |
| pair_006 | b1_conservative | b1_low_connect | 9.574969 |
| pair_008 | b2_graph_tsp_safe | b1_low_connect | 29.037378 |
| pair_010 | b1_conservative | b1_low_connect | 5.585758 |
| pair_011a | b2_graph_tsp_conservative_safe | b1_low_connect | 26.828610 |

## Requirement Check

| Requirement | Status | Evidence |
| --- | --- | --- |
| Direct config generation | Done | `predicted_config_json` contains numeric planner args per sample |
| Learning model | Done | NumPy MLP regressor |
| Training loss | Done | normalized MSE over oracle config vectors |
| No sweep at inference | Done | inference is one forward pass from sample features to config vector |

The separate evaluator step runs these predicted configs through DST generation and command/visual-risk scoring.