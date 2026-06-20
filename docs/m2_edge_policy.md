# M2 Edge Policy Prototype

M2 is the local routing decision layer.

The current implementation is a learned edge-utility prototype trained from `graph_tsp_trace.json`:

```text
Graph-TSP trace
  -> current node + remaining candidate node rows
  -> pairwise logistic edge ranking
  -> next-node utility
```

## What It Learns

For every multi-node graph task, the dataset builder reconstructs local decisions:

```text
current node
remaining candidate nodes
Graph-TSP selected next node = positive
other remaining nodes = negatives
```

The model learns a utility score for each candidate edge. During final DST export, Graph-TSP can optionally restrict candidates to the deterministic top-k pool and let M2 rerank that local pool.

## Reproduce

First generate Graph-TSP traces with inference. The current paired traces are under:

```text
outputs/m2_trace_pair006/graph_tsp_trace.json
outputs/m2_trace_pair008/graph_tsp_trace.json
outputs/m2_trace_pair010/graph_tsp_trace.json
outputs/m2_trace_pair011a/graph_tsp_trace.json
```

Then run:

```powershell
python tools/run_m2_edge_policy_loop.py --config configs/m2_edge_policy.yaml
```

Manual steps:

```powershell
python tools/build_m2_edge_dataset.py `
  --trace-glob "outputs/m2_trace_pair*/graph_tsp_trace.json" `
  --output-dir results/m2_edge_policy/latest_pair_20260619 `
  --max-negatives-per-decision 24

python tools/train_m2_edge_policy.py `
  --dataset-jsonl results/m2_edge_policy/latest_pair_20260619/m2_edge_candidates.jsonl `
  --output-dir results/m2_edge_policy/latest_pair_20260619 `
  --model-output checkpoints/m2_edge_policy.json
```

## Dataset Summary

| Item | Count |
| --- | ---: |
| Trace files | 4 |
| Raw candidate rows | 319889 |
| Downsampled candidate rows | 22663 |
| Positive edge rows | 1074 |
| Local decisions | 1074 |
| Samples with decisions | 3 |

`pair011a` has only single-node tasks, so it does not contribute local edge-choice rows.

## Current Result

Leave-one-sample-out result:

| Metric | Value |
| --- | ---: |
| Decisions | 1074 |
| Accuracy | 0.478585 |
| Top-3 Accuracy | 0.780261 |
| Mean Oracle Rank | 3.000 |
| Mean Train Pair Accuracy | 0.939256 |

Per fold:

| Held Sample | Decisions | Accuracy | Top-3 Accuracy | Mean Oracle Rank |
| --- | ---: | ---: | ---: | ---: |
| pair006 | 195 | 0.779487 | 0.933333 | 1.677 |
| pair008 | 18 | 0.833333 | 0.944444 | 1.333 |
| pair010 | 861 | 0.403020 | 0.742160 | 3.334 |

## DST Integration Result

The edge utility is now wired into the Graph-TSP planner as an optional local reranker:

```text
Graph-TSP candidate pool -> M2 edge utility rerank -> route order -> DST/PES export -> eval_executability
```

Use `--m2-edge-policy checkpoints/m2_edge_policy.json` together with `--graph-tsp-planner`. The current best local pool size is `--m2-edge-policy-top-k 4`.

Current 4-sample paired-holdout comparison:

| Method | Unified Loss | Hard Score | Jumps | Trims | Visible Connectors | Off-Mask mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M2 top4 | 0.607582 | 41.652422 | 192.750 | 20.250 | 32.500 | 171.912 |
| M2 top2 | 0.608222 | 41.888061 | 196.750 | 21.750 | 32.500 | 171.912 |
| M2 top8 | 0.610504 | 42.110343 | 203.500 | 20.750 | 32.500 | 171.912 |
| M1 config regressor | 0.621028 | 41.953114 | 143.500 | 25.250 | 35.750 | 179.303 |
| Graph-TSP conservative-safe | 0.626582 | 44.170873 | 213.500 | 26.250 | 35.250 | 175.534 |
| B1 conservative | 0.636772 | 42.691112 | 130.750 | 25.250 | 37.000 | 185.828 |

Per-sample and top-k sweep artifacts are stored in:

```text
results/m2_dst_integration/latest_pair_20260620/
```

## Stage Boundary

This now completes the second M2 prototype stage:

```text
graph trace -> edge dataset -> learned edge utility -> Graph-TSP rerank -> M2-DST -> eval comparison
```

It still does not complete:

- GNN or RL-based edge policy;
- segment-level action masking;
- resolving all `hard_fail` cases;
- M1/M2 joint closed-loop training.

The next M2 step is to move from scalar edge-utility reranking to segment-level/GNN policy learning with explicit continuity, visible-connector, and off-mask constraints.
