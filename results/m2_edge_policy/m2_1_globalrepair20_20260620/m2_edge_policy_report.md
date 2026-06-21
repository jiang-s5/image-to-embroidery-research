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
- epochs: `700`
- seed: `3`
- model artifact: `checkpoints/m2_edge_policy_m2_1_globalrepair20.json`
- dataset: `results/m2_edge_policy/m2_1_globalrepair20_20260620/m2_edge_candidates.jsonl`

| Metric | Value |
| --- | ---: |
| decision_count | 1074.000000 |
| accuracy | 0.571695 |
| top3_accuracy | 0.825885 |
| mean_oracle_rank | 2.658287 |
| mean_train_pair_accuracy | 0.935343 |

## Fold Results

| Held Sample | Decisions | Accuracy | Top-3 Accuracy | Mean Oracle Rank | Train Pair Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2_1_globalrepair20_pair_006 | 195 | 0.810256 | 0.938462 | 1.482 | 0.923583 |
| m2_1_globalrepair20_pair_008 | 18 | 0.944444 | 1.000000 | 1.056 | 0.928518 |
| m2_1_globalrepair20_pair_010 | 861 | 0.509872 | 0.796748 | 2.958 | 0.953927 |

## Stage Boundary

- Done: graph trace -> edge dataset -> learned edge utility -> leave-one-sample-out validation.
- Not done: replacing Graph-TSP inside final DST generation.
- Next: integrate this utility as an optional reranker in the graph planner and compare generated DST metrics.