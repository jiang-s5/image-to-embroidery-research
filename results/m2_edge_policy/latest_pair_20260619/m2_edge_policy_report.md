# M2 Edge Policy Prototype

This is the first learned M2 stage. It trains an edge-level ranker from `graph_tsp_trace.json` decisions.

The target is not a final DST replacement yet. The target is local routing imitation:

```text
current graph node + remaining candidate node
        |
M2 edge utility
        |
choose next node
```

## Best Leave-One-Sample-Out Result

- lr: `0.01`
- weight_decay: `0.0`
- epochs: `300`
- seed: `3`
- model artifact: `checkpoints/m2_edge_policy.json`
- dataset: `results/m2_edge_policy/latest_pair_20260619/m2_edge_candidates.jsonl`

| Metric | Value |
| --- | ---: |
| decision_count | 1074.000000 |
| accuracy | 0.356611 |
| top3_accuracy | 0.592179 |
| mean_oracle_rank | 5.533520 |
| mean_train_pair_accuracy | 0.974621 |

## Fold Results

| Held Sample | Decisions | Accuracy | Top-3 Accuracy | Mean Oracle Rank | Train Pair Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| pair006 | 195 | 0.738462 | 0.923077 | 1.846 | 0.990615 |
| pair008 | 18 | 0.777778 | 0.944444 | 1.500 | 0.989694 |
| pair010 | 861 | 0.261324 | 0.509872 | 6.453 | 0.943555 |

## Stage Boundary

- Done: graph trace -> edge dataset -> learned edge utility -> leave-one-sample-out validation.
- Not done: replacing Graph-TSP inside final DST generation.
- Next: integrate this utility as an optional reranker in the graph planner and compare generated DST metrics.