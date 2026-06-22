# M2.30 Nearest-Row Current Best Results

Date: 2026-06-22

M2.30 adds a nearest-endpoint row-order candidate to the M2 candidate pool.

## New Candidate

```text
mask_fill_edgewalk_nearestrow_rows16_p40
```

Generator:

```text
tools/generate_mask_fill_dst.py
tools/run_mask_fill_eval.py
```

Key flags:

```text
--row-spacing-mm 1.6
--max-connect-mm 6.0
--min-connect-inside-fraction 0.95
--use-mask-path-connectors
--max-mask-path-mm 40
--row-order nearest_endpoint
```

The important change is `--row-order nearest_endpoint`. Instead of visiting fill runs in fixed scanline order, the generator chooses the next fill segment by nearest endpoint and automatically flips segment direction when useful.

## Standalone Candidate Result

| Candidate | Samples | Hard Fail | Mean Loss | Mean Jump | Mean Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old edgewalk rows16 p40 | 33 | 20 | 0.266298 | 90.696970 | 16.424242 | 1.313558 | 0.999722 | 0.720092 |
| nearest-row edgewalk rows16 p40 | 33 | 0 | 0.077217 | 7.515152 | 0.909091 | 0.412527 | 0.999689 | 0.749214 |

This shows the actual breakthrough: the same high-coverage fill family becomes executable once row order is endpoint-aware.

## Public Leave-One-Out

M2.30 was also evaluated with leave-one-out calibrated selection. For each fold, the selector config was chosen on the other 32 public benchmark samples and then applied to the held-out sample.

| Model | Hard Fail | Mean Loss | Mean Jump | Mean Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.14 previous best LOO | 0 | 0.096228 | 8.848485 | 3.484848 | 0.090794 | 0.855224 | 0.837897 |
| M2.30 calibrated LOO | 0 | 0.068519 | 6.636364 | 1.848485 | 0.105806 | 0.911108 | 0.798703 |

This is the strongest public benchmark evidence because every sample is evaluated as a held-out sample. The selected config was stable: config `83` was selected in 32 of 33 folds.

## Public Core-to-Full Holdout

| Model | Hard Fail | Mean Loss | Mean Jump | Mean Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.14 previous best | 0 | 0.111060 | 10.1875 | 3.7500 | 0.187262 | 0.826159 | 0.843477 |
| M2.30 nearest-row selector | 0 | 0.074071 | 6.9375 | 2.0625 | 0.187256 | 0.927512 | 0.801813 |

## Public All-Sample Apply

M2.30 applied to all 33 public benchmark samples:

```text
hard_fail: 0
mean_unified_loss: 0.065631
mean_jump_count: 6.333333
mean_trim_count: 1.787879
mean_off_mask_stitch_length_mm: 0.105806
mean_visible_connector_count: 0.0
mean_coverage_ratio: 0.888958
mean_stitch_precision_ratio: 0.802090
```

Selected candidates:

| Candidate | Count |
|---|---:|
| mask_fill_edgewalk_nearestrow_rows16_p40 | 15 |
| auto_evalrepair_r10_c20 | 5 |
| skeleton_c30_m20 | 3 |
| skeleton_c30_m4 | 3 |
| mask_fill_edgewalk_rows16_p40 | 2 |
| skeleton_c30_m8 | 2 |
| skeleton_c20_m4 | 1 |
| mask_fill_rows16_c6 | 1 |
| skeleton_c30_m12 | 1 |

## Incoming Review

Incoming review is evaluation-only and not used for training.

| Model | Hard Fail | Mean Loss | Mean Jump | Mean Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.14 previous best | 0 | 0.090709 | 7.5000 | 4.7500 | 0.000000 | 0.856321 | 0.792180 |
| M2.30 nearest-row selector | 0 | 0.084419 | 6.7500 | 4.7500 | 0.000000 | 0.856344 | 0.793664 |

## Decision

M2.30 is promoted as the current best.

## Research Interpretation

Earlier M2.22-M2.29 experiments showed that selector-only tuning could not solve the tradeoff between coverage and command risk. M2.30 improves the candidate generator itself. The result is stronger because it changes the feasible candidate set:

```text
old high-coverage row fill: high coverage but many jumps/trims
nearest-row fill: high coverage with low jumps/trims
```

This confirms the next research direction:

```text
Improve candidate geometry/path generation first,
then let the selector choose among better candidates.
```
