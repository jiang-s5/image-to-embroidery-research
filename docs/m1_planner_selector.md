# M1 Planner Selector

M1 is the first learned planner-selection stage in this project. It is deliberately smaller than M2/M3:

- M1 selects a planner preset from the existing preset bank.
- M1 uses pre-export image/geometry summary features and candidate preset parameters.
- M1 does not predict graph edges, route order, or DST commands directly.

This makes M1 a practical bridge between fixed planner presets and future edge-level graph routing.

## Why M1 Exists

The B1/B2 planner sweeps showed that no single preset is best for every image. For example, a high-density sample may prefer the conservative B1 path, while sparse or fragmented samples may benefit from Graph-TSP. M1 turns this observation into a trainable selector:

```text
image / predicted geometry summary
        +
candidate planner preset parameters
        |
ridge selector with interaction features
        |
best preset for this sample
```

## Current Features

The default dataset builder avoids target leakage. It does not use post-export metrics such as jump count or off-mask length as input features.

Feature groups:

- preprocessing ratios: selected foreground, thread foreground, region foreground, region/thread area ratio;
- model summary: mask mean, density mean, foreground pixels, stitch-type ratios;
- preset parameters: thresholds, component limits, continuity parameters, Graph-TSP flags and costs;
- interaction features: sample complexity multiplied by planner parameters, such as density x Graph-TSP and foreground ratio x max components.

## Reproduce the Loop

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

## Current Loop Result

Current leave-one-out evaluation uses 4 paired holdout samples and 6 planner candidates per sample.

| Selector / Baseline | Unified Loss | Hard Score | Jumps | Trims | Visible Connectors | Off-Mask mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle hard-score selector | 0.621038 | 51.903530 | 143.250 | 25.250 | 35.750 | 178.987 |
| M1 learned selector | 0.621367 | 51.927638 | 154.000 | 24.250 | 35.500 | 176.804 |
| Fixed B2 Graph-TSP conservative | 0.626582 | 54.170873 | 213.500 | 26.250 | 35.250 | 175.534 |
| Fixed B1 conservative | 0.636772 | 52.691112 | 130.750 | 25.250 | 37.000 | 185.828 |

Interpretation:

- M1 is now a real learned selection loop with saved weights and leave-one-out evidence.
- It improves over the best fixed Graph-TSP preset on unified loss, hard score, jump count, and trim count.
- It is very close to the leave-one-out oracle on this tiny benchmark.
- The benchmark is too small to claim broad generalization. The next step is Pilot-50 or larger.

## Artifacts

- Selector model: `checkpoints/m1_planner_selector.json`
- Candidate dataset: `results/m1_selector/latest_pair_20260618/m1_selector_samples.jsonl`
- Fold decisions: `results/m1_selector/latest_pair_20260618/m1_selector_loo_decisions.csv`
- Optional applied selector output: `results/m1_selector/latest_pair_20260618/m1_selector_applied.csv`
- Report: `results/m1_selector/latest_pair_20260618/m1_selector_report.md`

## What M1 Is Not

M1 is not a graph neural network and not reinforcement learning. It does not make edge-level route decisions.

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
