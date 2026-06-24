# M2.42 Precision-Aware Safe Adaptive Selector Notes

Date: 2026-06-24

M2.42 is a small selector-level refinement after M2.41. It does not add a new generator or a new stitch format. It keeps the M2.41 `safe_dt` candidate in the pool, then retunes the calibrated selector to put more pressure on stitch precision.

## Why This Was Tried

M2.41 improved unified execution quality, jump count, and off-mask length, but it lowered stitch precision. The goal of M2.42 is to keep the M2.41 executable-quality gain while recovering some precision.

## Selector Setting

Recommended public setting:

```text
--precision-weights 0.10,0.20,0.35
--min-precision 0.78
```

Candidate pool:

- fixed inset1
- fixed inset2
- raw nearest-row
- edgewalk rows16
- auto/eval repair
- skeleton variants
- safe_dt adaptive fill-inset candidate

## Results

Public ext33 leave-one-out:

| Selector | Samples | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.34 core selector | 33 | 0 | 0.051174 | 5.636364 | 0.424242 | 0.082527 | 0.967981 | 0.819674 |
| M2.41 core + safe_dt | 33 | 0 | 0.049160 | 5.484848 | 0.424242 | 0.049952 | 0.978413 | 0.799193 |
| M2.42 precision-aware safe_dt | 33 | 0 | 0.049160 | 5.484848 | 0.424242 | 0.049952 | 0.978737 | 0.801506 |

Incoming review leave-one-out:

| Selector | Samples | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.34 incoming | 4 | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.995549 | 0.781068 |
| M2.41 incoming core + safe_dt | 4 | 0 | 0.074710 | 8.250000 | 1.000000 | 0.000000 | 0.996541 | 0.753519 |
| M2.42 incoming precision-aware safe_dt | 4 | 0 | 0.074710 | 8.250000 | 1.000000 | 0.000000 | 0.996541 | 0.753519 |

## Decision

M2.42 becomes the recommended current selector configuration because it keeps M2.41's unified loss improvement and slightly improves public precision. The improvement is small, so this should be reported as a selector calibration step, not as a new model architecture.

M2.34 remains the precision-safer baseline, and the next real optimization should directly model coverage/precision Pareto tradeoffs or add a precision-aware objective to the learned selector rather than relying on manual selector grids.

## Artifacts

- `configs/best_current_model_m2_42_precision_safeadt_selector.json`
- `results/public_benchmark_v1_ext33_m2_42_precision_p078_safeadt_plus_loo/`
- `results/incoming_review_eval_v1_m2_42_precision_p078_safeadt_plus_loo/`