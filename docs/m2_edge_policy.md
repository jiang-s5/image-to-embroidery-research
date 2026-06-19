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

The model learns a utility score for each candidate edge. At inference inside the prototype evaluator, the candidate with highest utility is selected.

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

## Stage Boundary

This completes the first M2 learning stage:

```text
graph trace -> edge dataset -> learned edge utility -> LOO validation
```

It does not yet complete:

- replacing Graph-TSP inside final DST generation;
- proving command-level DST improvement from M2;
- GNN or RL-based edge policy;
- M1/M2 joint closed-loop training.

The next M2 step is to integrate `checkpoints/m2_edge_policy.json` as an optional edge reranker inside the graph planner, then rerun DST/PES export and executability evaluation.
