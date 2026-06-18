# Latest Pair Graph-TSP Sweep Analysis

This sweep adds two graph/TSP-style planner candidates to the previous B1 planner set:

- `b2_graph_tsp_safe`
- `b2_graph_tsp_conservative_safe`

Both candidates use explicit polyline graph nodes and edge costs that penalize distance, trim risk, off-mask connector risk, and visible connector risk. They also enable mask-safe connectors, so a STITCH connector is only used when the connector segment stays inside the active component mask.

## Result

| Method | Mean Unified Loss | Mean Jumps | Mean Trims | Mean Off-Mask mm | Mean Visible Connectors |
| --- | ---: | ---: | ---: | ---: | ---: |
| `b2_graph_tsp_conservative_safe` | 0.626582 | 213.500 | 26.250 | 175.534 | 35.250 |
| `b2_graph_tsp_safe` | 0.628021 | 227.500 | 26.250 | 176.192 | 40.000 |
| `b1_low_connect` | 0.628786 | 203.500 | 26.750 | 201.238 | 46.250 |
| `b1_region_fill` | 0.633435 | 211.000 | 34.250 | 750.728 | 206.000 |
| `b1_conservative` | 0.636772 | 130.750 | 25.250 | 185.828 | 37.000 |
| `b1_dense_thread` | 0.649885 | 265.750 | 42.250 | 786.770 | 221.500 |

The hard-selection recommendation is now:

```text
b2_graph_tsp_conservative_safe
```

## Interpretation

Compared with `b1_conservative`, the graph-TSP conservative-safe planner improves:

- mean unified loss: `0.636772 -> 0.626582`
- off-mask stitch length: `185.828 mm -> 175.534 mm`
- visible connectors: `37.000 -> 35.250`
- stitch path length: `3080.769 mm -> 2882.453 mm`

The tradeoff is:

- jump count increases: `130.750 -> 213.500`
- trim count slightly increases: `25.250 -> 26.250`

This means graph-TSP is moving the planner in the desired visual-risk direction, but the current edge cost still over-penalizes connectors and therefore converts too many transitions into jumps.

## Research Meaning

This supports the paper direction:

```text
Geometry prediction -> segment/polyline graph -> TSP-style edge optimization -> DST/PES export
```

The result is stronger than a purely heuristic planner because the planner now has explicit graph nodes, edge costs, mask-safe constraints, and measurable command/visual tradeoffs.

## Next Optimization

1. Tune the edge cost so jump count does not rise too much.
2. Add color-layer and component-level graph nodes, not only polyline-level nodes.
3. Add a graph export JSON for node/edge visualization and future GNN training.
4. Use this graph-TSP planner as the teacher/baseline before trying GNN-assisted edge-cost prediction.
5. Keep RL out of the main paper path until the graph baseline is stable.
