# M0.9 and Strict M1 Planner Selectors

This document separates two stages that are easy to confuse:

- M0.9: candidate-preset scorer, trained from sweep results, but still scores candidate presets as inputs.
- Strict M1: direct MLP selector, mapping image/geometry features to one planner preset/config.

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

## Artifacts

- Selector model: `checkpoints/m1_planner_selector.json`
- Strict M1 direct model: `checkpoints/m1_direct_selector.json`
- Candidate dataset: `results/m1_selector/latest_pair_20260618/m1_selector_samples.jsonl`
- Strict M1 sample dataset: `results/m1_direct_selector/latest_pair_20260618/m1_direct_samples.jsonl`
- Fold decisions: `results/m1_selector/latest_pair_20260618/m1_selector_loo_decisions.csv`
- Strict M1 fold decisions: `results/m1_direct_selector/latest_pair_20260618/m1_direct_loo_decisions.csv`
- Strict M1 selected configs: `results/m1_direct_selector/latest_pair_20260618/m1_direct_selected_configs.json`
- Optional applied selector output: `results/m1_selector/latest_pair_20260618/m1_selector_applied.csv`
- Report: `results/m1_selector/latest_pair_20260618/m1_selector_report.md`
- Strict M1 report: `results/m1_direct_selector/latest_pair_20260618/m1_direct_report.md`

## What M1 Is Not

Strict M1 is not a graph neural network and not reinforcement learning. It does not make edge-level route decisions.

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
