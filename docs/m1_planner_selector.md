# M0.9 and Strict M1 Planner Selectors

This document separates two stages that are easy to confuse:

- M0.9: candidate-preset scorer, trained from sweep results, but still scores candidate presets as inputs.
- Strict M1: direct MLP selector, mapping image/geometry features to one planner preset/config.
- M1 config regressor: direct MLP regressor, mapping image/geometry features to a continuous planner config vector.
- M1 ranking selector: pairwise preference learner, mapping image/geometry + candidate config to a utility score.

Strict M1 is deliberately smaller than M2/M3. It selects a planner preset/config; it does not predict graph edges, route order, or DST commands directly.

## Why M1 Exists

The B1/B2 planner sweeps showed that no single preset is best for every image. For example, a high-density sample may prefer the conservative B1 path, while sparse or fragmented samples may benefit from Graph-TSP.

The earlier M0.9 scorer uses candidate preset parameters as input:

```text
image / predicted geometry summary
        +
candidate planner preset parameters
        |
ridge selector with interaction features
        |
best preset for this sample
```

Strict M1 removes candidate scoring at inference:

```text
image / predicted geometry summary
        |
MLP softmax selector
        |
predicted planner preset + config args
```

The strongest current M1 variant predicts numeric parameters directly:

```text
image / predicted geometry summary
        |
MLP regression model
        |
continuous planner config
(thresholds / connectivity / graph penalties / limits)
        |
DST generation + evaluator
```

The ranking variant keeps multiple candidate configs at inference, but changes the training signal from class labels or MSE into evaluator-derived preferences:

```text
image / predicted geometry + candidate config A/B
        |
pairwise logistic ranking loss
        |
utility(candidate)
        |
highest-utility planner config
```

This is useful for the "evaluator as preference signal" research direction. Current tiny-sample results do not show an improvement over the continuous config regressor.

## M0.9 Candidate Scorer Features

The default dataset builder avoids target leakage. It does not use post-export metrics such as jump count or off-mask length as input features.

Feature groups:

- preprocessing ratios: selected foreground, thread foreground, region foreground, region/thread area ratio;
- model summary: mask mean, density mean, foreground pixels, stitch-type ratios;
- preset parameters: thresholds, component limits, continuity parameters, Graph-TSP flags and costs;
- interaction features: sample complexity multiplied by planner parameters, such as density x Graph-TSP and foreground ratio x max components.

## Strict M1 Direct Features

Strict M1 uses only sample-level image/geometry features:

- preprocessing ratios;
- selected/thread/region foreground statistics;
- model mask and density means from the neutral feature method;
- stitch-type ratios;
- derived coverage and density ratios.

It explicitly excludes:

- `preset.*` features;
- candidate-preset interaction features;
- post-export metrics such as jump count or off-mask length.

## Reproduce M0.9 Loop

```powershell
python tools/run_m1_selector_loop.py --config configs/m1_selector.yaml
```

Equivalent manual steps:

```powershell
python tools/build_m1_selector_dataset.py `
  --sweep-results results/b1_sweep/latest_pair_graph_tsp_sweep_20260618/sweep_results.csv `
  --sweep-config configs/sweep_b1.yaml `
  --outputs-root outputs/b1_sweep_latest_pair_20260617 `
  --pair-root datasets/incoming_review/latest_dataset_20260615/organized/pairing_review_NOT_FOR_TRAINING_YET/paired_candidates_FOR_APPROVAL_NOT_TRAINING `
  --output-dir results/m1_selector/latest_pair_20260618

python tools/train_m1_planner_selector.py `
  --dataset-jsonl results/m1_selector/latest_pair_20260618/m1_selector_samples.jsonl `
  --output-dir results/m1_selector/latest_pair_20260618 `
  --model-output checkpoints/m1_planner_selector.json `
  --selection-target hard_score
```

Apply the trained selector to candidate rows:

```powershell
python tools/apply_m1_planner_selector.py `
  --model checkpoints/m1_planner_selector.json `
  --candidates-jsonl results/m1_selector/latest_pair_20260618/m1_selector_samples.jsonl `
  --output-csv results/m1_selector/latest_pair_20260618/m1_selector_applied.csv
```

## Reproduce Strict M1 Loop

```powershell
python tools/run_m1_direct_loop.py --config configs/m1_direct_selector.yaml
```

Equivalent manual commands:

```powershell
python tools/train_m1_direct_selector.py `
  --candidates-jsonl results/m1_selector/latest_pair_20260618/m1_selector_samples.jsonl `
  --sweep-config configs/sweep_b1.yaml `
  --feature-method b1_conservative `
  --label-target hard_score `
  --output-dir results/m1_direct_selector/latest_pair_20260618 `
  --model-output checkpoints/m1_direct_selector.json

python tools/apply_m1_direct_selector.py `
  --model checkpoints/m1_direct_selector.json `
  --samples-jsonl results/m1_direct_selector/latest_pair_20260618/m1_direct_samples.jsonl `
  --output-csv results/m1_direct_selector/latest_pair_20260618/m1_direct_applied.csv `
  --output-json results/m1_direct_selector/latest_pair_20260618/m1_direct_selected_configs.json
```

## Reproduce M1 Continuous Config Loop

```powershell
python tools/run_m1_config_loop.py --config configs/m1_config_regressor.yaml
```

Equivalent manual commands:

```powershell
python tools/train_m1_config_regressor.py `
  --candidates-jsonl results/m1_selector/latest_pair_20260618/m1_selector_samples.jsonl `
  --sweep-config configs/sweep_b1.yaml `
  --feature-method b1_conservative `
  --label-target hard_score `
  --output-dir results/m1_config_regressor/latest_pair_20260618 `
  --model-output checkpoints/m1_config_regressor.json

python tools/run_m1_config_eval.py `
  --predictions-json results/m1_config_regressor/latest_pair_20260618/m1_config_predictions.json `
  --output-dir results/m1_config_regressor/latest_pair_20260618_eval `
  --checkpoint checkpoints/best_model13_multiformat_all_vector_continuity.pt `
  --planner-config configs/relation_planner.yaml `
  --score-config configs/sweep_b1.yaml `
  --reuse
```

## Reproduce M1 Ranking Preference Loop

```powershell
python tools/run_m1_ranking_loop.py --config configs/m1_ranking_selector.yaml
```

Equivalent manual command:

```powershell
python tools/train_m1_ranking_selector.py `
  --candidates-jsonl results/m1_selector/latest_pair_20260618/m1_selector_samples.jsonl `
  --sweep-config configs/sweep_b1.yaml `
  --feature-method b1_conservative `
  --label-target hard_score `
  --output-dir results/m1_ranking_selector/latest_pair_20260618 `
  --model-output checkpoints/m1_ranking_selector.json
```

## Current M0.9 Loop Result

Current leave-one-out evaluation uses 4 paired holdout samples and 6 planner candidates per sample.

| Selector / Baseline | Unified Loss | Hard Score | Jumps | Trims | Visible Connectors | Off-Mask mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle hard-score selector | 0.621038 | 51.903530 | 143.250 | 25.250 | 35.750 | 178.987 |
| M1 learned selector | 0.621367 | 51.927638 | 154.000 | 24.250 | 35.500 | 176.804 |
| Fixed B2 Graph-TSP conservative | 0.626582 | 54.170873 | 213.500 | 26.250 | 35.250 | 175.534 |
| Fixed B1 conservative | 0.636772 | 52.691112 | 130.750 | 25.250 | 37.000 | 185.828 |

Interpretation:

- M0.9 is a learned candidate scorer with saved weights and leave-one-out evidence.
- It improves over the best fixed Graph-TSP preset on unified loss, hard score, jump count, and trim count.
- It is very close to the leave-one-out oracle on this tiny benchmark.
- The benchmark is too small to claim broad generalization. The next step is Pilot-50 or larger.

## Current Strict M1 Result

Strict M1 uses a NumPy MLP with a tanh hidden layer, softmax output, and cross-entropy loss.

| Selector / Baseline | Unified Loss | Hard Score | Jumps | Trims | Visible Connectors | Off-Mask mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle label | 0.621038 | 51.903530 | 143.250 | 25.250 | 35.750 | 178.987 |
| Strict M1 direct MLP | 0.621038 | 52.225434 | 143.250 | 25.250 | 37.000 | 178.229 |
| Fixed B2 Graph-TSP conservative | 0.626582 | 54.170873 | 213.500 | 26.250 | 35.250 | 175.534 |
| Fixed B1 conservative | 0.636772 | 52.691112 | 130.750 | 25.250 | 37.000 | 185.828 |

Strict M1 satisfies the formal stage requirement:

- learning model: MLP;
- training loss: cross-entropy;
- mapping: image/geometry features -> planner preset/config;
- no candidate preset scoring at inference.

## Current M1 Config Regressor Result

The continuous-config regressor uses a NumPy MLP with normalized-MSE loss over oracle planner config vectors. Its predicted configs are then run through real DST/PES generation and executability evaluation.

| Selector / Baseline | Unified Loss | Jumps | Trims | Visible Connectors | Off-Mask mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 continuous config regressor, DST-evaluated | 0.621028 | 143.500 | 25.250 | 35.750 | 179.303 |
| Oracle preset label | 0.621038 | 143.250 | 25.250 | 35.750 | 178.987 |
| Strict M1 direct preset MLP | 0.621038 | 143.250 | 25.250 | 37.000 | 178.229 |
| M0.9 candidate scorer | 0.621367 | 154.000 | 24.250 | 35.500 | 176.804 |
| Fixed B2 Graph-TSP conservative | 0.626582 | 213.500 | 26.250 | 35.250 | 175.534 |
| Fixed B1 conservative | 0.636772 | 130.750 | 25.250 | 37.000 | 185.828 |

This is the closest implementation to the strict definition:

```text
image -> learned model -> planner config -> DST -> evaluator
```

It converts the parameter search problem into a learned function approximation problem. The current result is still small-sample, so it should be treated as a proof-of-loop rather than a mature generalization result.

## Current M1 Ranking Preference Result

The ranking selector converts evaluator outputs into pairwise preferences. It uses a linear pairwise ranker with image-geometry x planner-config interactions.

| Selector / Baseline | Unified Loss | Hard Score | Jumps | Trims | Visible Connectors | Off-Mask mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle preference label | 0.621038 | 51.903530 | 143.250 | 25.250 | 35.750 | 178.987 |
| M1 ranking selector | 0.621812 | 53.670513 | 157.500 | 24.250 | 40.250 | 176.956 |
| M1 continuous config regressor, DST-evaluated | 0.621028 | 41.953114 | 143.500 | 25.250 | 35.750 | 179.303 |
| Fixed B2 Graph-TSP conservative | 0.626582 | 54.170873 | 213.500 | 26.250 | 35.250 | 175.534 |
| Fixed B1 conservative | 0.636772 | 52.691112 | 130.750 | 25.250 | 37.000 | 185.828 |

Interpretation:

- The preference-learning path is implemented and reproducible.
- The current ranking selector has high training-pair accuracy but worse leave-one-out hard score than the continuous config regressor.
- On only 4 paired holdout samples, this is evidence of overfit risk, not evidence that ranking has solved planner selection.
- The next valid experiment is Pilot-50 / Release-200 sweep expansion before promoting ranking as the main M1 result.

## Artifacts

- Selector model: `checkpoints/m1_planner_selector.json`
- Strict M1 direct model: `checkpoints/m1_direct_selector.json`
- M1 config regressor model: `checkpoints/m1_config_regressor.json`
- M1 ranking selector model: `checkpoints/m1_ranking_selector.json`
- Candidate dataset: `results/m1_selector/latest_pair_20260618/m1_selector_samples.jsonl`
- Strict M1 sample dataset: `results/m1_direct_selector/latest_pair_20260618/m1_direct_samples.jsonl`
- Fold decisions: `results/m1_selector/latest_pair_20260618/m1_selector_loo_decisions.csv`
- Strict M1 fold decisions: `results/m1_direct_selector/latest_pair_20260618/m1_direct_loo_decisions.csv`
- Strict M1 selected configs: `results/m1_direct_selector/latest_pair_20260618/m1_direct_selected_configs.json`
- M1 config predictions: `results/m1_config_regressor/latest_pair_20260618/m1_config_predictions.json`
- M1 config DST eval report: `results/m1_config_regressor/latest_pair_20260618_eval/m1_config_eval_report.md`
- M1 ranking report: `results/m1_ranking_selector/latest_pair_20260618/m1_ranking_report.md`
- Optional applied selector output: `results/m1_selector/latest_pair_20260618/m1_selector_applied.csv`
- Report: `results/m1_selector/latest_pair_20260618/m1_selector_report.md`
- Strict M1 report: `results/m1_direct_selector/latest_pair_20260618/m1_direct_report.md`

## What M1 Is Not

Strict M1 and M1 ranking are not graph neural networks and not reinforcement learning. They do not make learned edge-level route decisions.

The current Graph-TSP planner is a deterministic graph/path optimizer. It can produce graph traces, but it is not yet a learned M2 edge policy.

The next research stage is M2:

```text
segment / polyline graph
        |
edge feature extraction
        |
edge-level scorer or GNN
        |
route construction / TSP reranking
```
