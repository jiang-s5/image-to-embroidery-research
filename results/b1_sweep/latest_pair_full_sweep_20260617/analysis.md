# Latest Pair Full B1 Sweep Analysis

This sweep ran 4 paired holdout samples against 4 B1 planner presets. The paired samples are evaluation-only and were not added to training.

## Mean Ranking

| Method | Mean Unified Loss | Mean Jumps | Mean Trims | Mean Off-Mask mm | Mean Visible Connectors |
| --- | ---: | ---: | ---: | ---: | ---: |
| `b1_low_connect` | 0.628786 | 203.500 | 26.750 | 201.238 | 46.250 |
| `b1_region_fill` | 0.633435 | 211.000 | 34.250 | 750.728 | 206.000 |
| `b1_conservative` | 0.636772 | 130.750 | 25.250 | 185.828 | 37.000 |
| `b1_dense_thread` | 0.649885 | 265.750 | 42.250 | 786.770 | 221.500 |

## Hard Selection

The lowest mean unified loss is `b1_low_connect`, but it wins mainly because of one sample. Under the hard policy, `b1_conservative` is preferred because it has lower average jumps, trims, off-mask stitch length, and visible connectors.

Current hard recommendation:

```text
b1_conservative
```

## Current Limitation

All 4 holdout samples are still `hard_fail`. This means the sweep is useful for ranking planner candidates, but the current model/planner is still below the real embroidery quality target.

The next optimization should prioritize:

1. Reducing `visible_connector_count`.
2. Reducing `off_mask_stitch_length_mm`.
3. Routing between thread/region/selected preprocessing based on input type.
4. Enlarging the paired holdout set before claiming a stable improvement.
