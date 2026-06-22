# M2.34 Fill-Inset Current Best Results

Date: 2026-06-22

## Promotion Summary

M2.34 is promoted over M2.30 because it improves all three validation views used for current planner selection:

1. public ext33 leave-one-out,
2. public core-to-full holdout,
3. incoming review evaluation-only holdout.

The main change is not a new neural checkpoint. It is a better candidate family in the planner selector: nearest-endpoint fill rows with a 1px/2px fill inset.

## Public ext33 Leave-One-Out

| Version | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.30 | 0 | 0.068519 | 6.636364 | 1.848485 | 0.105806 | 0.911108 | 0.798703 |
| M2.34 | 0 | 0.051174 | 5.636364 | 0.424242 | 0.082527 | 0.968881 | 0.819738 |

## Public Core-To-Full Holdout

| Version | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.30 | 0 | 0.074071 | 6.937500 | 2.062500 | 0.187256 | 0.927512 | 0.801813 |
| M2.34 | 0 | 0.051748 | 5.750000 | 0.437500 | 0.061937 | 0.951650 | 0.827483 |

## Incoming Review Evaluation-Only Holdout

| Version | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.30 | 0 | 0.084419 | 6.750000 | 4.750000 | 0.000000 | 0.856344 | 0.793664 |
| M2.34 | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.995549 | 0.781068 |

## Candidate Counts

Public ext33 LOO selected:

- `mask_fill_edgewalk_nearestrow_inset2_rows16_p40`: 15
- `mask_fill_edgewalk_nearestrow_inset1_rows16_p40`: 7
- raw nearest-row p40: 5
- auto/skeleton/standard fill candidates: 6

Public core-to-full test selected:

- `mask_fill_edgewalk_nearestrow_inset2_rows16_p40`: 8
- raw nearest-row p40: 3
- `mask_fill_edgewalk_nearestrow_inset1_rows16_p40`: 3
- auto/skeleton candidates: 2

Incoming review selected:

- `mask_fill_edgewalk_nearestrow_inset2_rows16_p40`: 2
- raw nearest-row p40: 1
- standard edgewalk fill: 1

## Interpretation

M2.30 showed that nearest-endpoint row ordering made high-coverage fill usable. M2.34 shows that high-coverage fill also needs a small boundary margin. Fixed 1px/2px fill inset reduces boundary spill and trim pressure without the jump explosion caused by hard quantized path validation.

## Next Direction

The next version should make the fill inset adaptive per component:

- use distance-transform statistics,
- use local boundary confidence,
- avoid eroding thin line/running-stitch components,
- keep the calibrated selector as a safety net.
