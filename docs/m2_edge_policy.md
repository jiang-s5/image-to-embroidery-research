# M2 Edge Policy Prototype

M2 is the local routing decision layer.

Status note: this document records the early paired-holdout M2/M2.1/M2.2 edge-policy stage. In the current public-benchmark validation, M2.1 is now a historical baseline rather than the promoted system. The current balanced best system is M2.34; see [public_benchmark_ext33_validation.md](public_benchmark_ext33_validation.md) and [m2_34_fill_inset_notes.md](m2_34_fill_inset_notes.md).

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


## M2.1 / M2.2 Loop Result

The jump-aware decode loop has now been implemented and evaluated:

```text
Graph-TSP candidate edges
  -> hard-safe filter
  -> M2 risk-aware ranking
  -> jump-aware rerank
  -> global-mask safe-connect repair
  -> eval_executability
  -> hard-mined M2 retraining check
```

Current 4-sample paired-holdout comparison:

| Method | Unified Loss | Hard Score | Jumps | Trims | Jump Path mm | Visible Connectors | Off-Mask mm | Safe Repairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M2.1 global repair20 | 0.580058 | 37.151575 | 83.000 | 15.500 | 497.720 | 32.500 | 173.146 | 94.500 |
| M2.2 retrained policy + repair20 | 0.581322 | 37.187846 | 82.000 | 15.750 | 490.357 | 32.500 | 173.646 | 95.250 |
| M2 top4 baseline | 0.607582 | 41.652422 | 192.750 | 20.250 | 842.240 | 32.500 | 171.912 | 0.000 |

The promoted setting is **M2.1 global repair20**, not M2.2. M2.1 improves mean unified loss by 4.53% relative to M2 top4 and reduces mean jumps by 109.75, while visible connector count stays unchanged in this benchmark. The retrained M2.2 ranker learned the hard-mined route decisions, but it was slightly worse than the deterministic repair20 decode in DST-level evaluation.

Recommended inference flags:

```powershell
--m2-edge-policy checkpoints/m2_edge_policy.json `
--m2-edge-policy-top-k 4 `
--m2-hard-safe-filter `
--m2-safe-min-inside-fraction 0.96 `
--m2-jump-aware-weight 0.25 `
--m2-offmask-weight 0.5 `
--m2-visible-weight 1.0 `
--m2-trim-weight 0.25 `
--safe-connect-repair `
--safe-connect-repair-max-mm 20.0 `
--safe-connect-repair-min-inside-fraction 0.90 `
--safe-connect-repair-global-mask
```

Artifacts:

```text
results/m2_1_decoding/latest_pair_20260620/m2_1_decode_comparison_report.md
results/m2_edge_policy/m2_1_globalrepair20_20260620/m2_edge_policy_report.md
checkpoints/m2_edge_policy_m2_1_globalrepair20.json
```

Important boundary: all four samples still remain `hard_fail`. M2.1 fixes a routing/executability problem, especially excessive jumps. It does not solve upstream segmentation, mask, or visible/off-mask geometry errors.

## Stage Boundary

This now completes the second M2 prototype stage and the M2.1 decode-loop stage:

```text
graph trace -> edge dataset -> learned edge utility -> Graph-TSP rerank -> M2-DST -> eval comparison
```

It still does not complete:

- GNN or RL-based edge policy;
- segment-level action masking;
- resolving all `hard_fail` cases;
- M1/M2 joint closed-loop training.

The next M2 step is to move from scalar edge-utility reranking to segment-level/GNN policy learning with explicit continuity, visible-connector, and off-mask constraints.

## M2.1 Direction: Jump-aware Safe-connect Decoding

The current M2 top4 result is best interpreted as a visual-safety-first planner: it lowers mean unified loss, trim count, off-mask length, and visible connectors, but it still uses too many jumps and all four paired holdout samples remain `hard_fail`.

The next optimization should therefore target decoding, not model size. The recommended M2.1 direction is:

```text
M2 top4 edge utility
  -> visual-risk edge filter
  -> jump-aware reranking
  -> mask-safe connect repair
  -> DST/PES export
  -> eval_executability
```

### Why This Direction

M2 already improves visual risk, so simply making the learned model larger is unlikely to address the immediate failure mode. The current failure mode is conservative routing: uncertain transitions become jumps. M2.1 should keep the visual-risk gains while repairing jumps that can be safely stitched inside the mask.

### Proposed Decode Score

For candidate outputs or candidate transition choices, add a jump-aware second-stage score:

```text
FinalScore = UnifiedLoss
           + alpha * normalized_jump_count
           + beta  * normalized_jump_path
           + gamma * hard_fail_penalty
```

This keeps the existing command/visual objective but explicitly penalizes the current M2 weakness: excessive jump usage.

### Safe-connect vs Risky-connect

M2.1 should distinguish safe and unsafe connectors instead of treating every uncertain connection as a jump:

```text
safe_connect
risky_connect
jump
trim
```

A connector is considered `safe_connect` only when all of these are true:

```text
off_mask_fraction < threshold
visible_connector_risk < threshold
distance_mm < threshold
```

This gives the planner a way to reduce jumps without creating visible threads across blank fabric.

### Repair Pass

After M2 generates a route, run a deterministic repair pass:

```text
for each jump edge:
    test whether a connector path stays inside the active mask
    test visible connector risk and distance
    if safe:
        replace jump with stitch connector
    else:
        keep jump or trim
```

This is the lowest-risk next experiment because it does not require retraining and directly targets the observed jump-count regression.

### Target Metrics

The M2.1 experiment should be considered successful only if it improves jump behavior without giving back the visual-risk gains:

| Metric | Current M2 top4 | Target |
| --- | ---: | ---: |
| Unified Loss | 0.607582 | < 0.607 |
| Mean Jumps | 192.750 | 140-160 |
| Mean Trims | 20.250 | about 20 |
| Visible Connectors | 32.500 | <= 32 |
| Off-mask mm | 171.912 | <= 170 |
| Hard Fail Samples | 4 / 4 | <= 2 / 4 |

### Hard-fail Mining

In parallel, generate `failure_edges.json` for failed outputs. Each failure edge should record whether it contributed to:

- visible connector risk;
- off-mask stitch length;
- excessive jump count or jump path;
- illegal or near-threshold stitch length.

These failure edges can later be weighted during M2 training:

```text
hard_fail sample weight: x2-x3
visible/off-mask edge weight: x3
excessive jump edge weight: x2
```

This keeps M2.1 aligned with the paper direction: a constraint-aware learned graph planner, not a larger dense image model.
