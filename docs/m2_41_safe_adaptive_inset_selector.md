# M2.41 Safe Adaptive Fill-Inset Selector Notes

Date: 2026-06-24

M2.41 tests a safer version of the adaptive fill-inset idea after the M2.35 negative ablation. The key change is not to promote adaptive inset as a standalone planner. Instead, it is added as one more candidate inside the calibrated M2 selector, so the selector can use it only when it helps a specific image.

## Implementation

`tools/generate_mask_fill_dst.py` now supports:

```text
--adaptive-fill-inset-policy distance|safe_dt
--adaptive-fill-inset-min-retained-ratio <float>
--adaptive-fill-inset-thin-area-px <int>
```

The existing `distance` policy is unchanged and remains the default. The new `safe_dt` policy uses per-component distance-transform statistics plus an erosion-retention guard:

1. Compute connected-component area and distance-transform values.
2. Protect very thin or small components from fill erosion.
3. Use 1px or 2px inset for wider components.
4. Reject an inset if erosion keeps too little component support.
5. Keep the existing fallback that reduces the inset if a component disappears.

This makes the policy more conservative than fixed inset2 while avoiding the crude width-only behavior tested in M2.35.

## Single-Candidate Result

The best tested `safe_dt` single candidate was:

```text
safe_dt t2/m5 retained=0.55 thin_area=64
```

Public ext33 result:

| Candidate | Samples | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.34 fixed inset2 | 33 | 0 | 0.071837 | 7.696970 | 1.000000 | 0.081876 | 0.982203 | 0.854755 |
| M2.41 safe_dt single | 33 | 0 | 0.076831 | 8.090909 | 0.939394 | 0.176148 | 0.997858 | 0.709807 |

Incoming review result:

| Candidate | Samples | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.34 fixed inset2 | 4 | 0 | 0.095803 | 10.250000 | 1.750000 | 0.000000 | - | - |
| M2.41 safe_dt single | 4 | 0 | 0.103669 | 11.750000 | 1.000000 | 0.000000 | 1.000000 | 0.708240 |

Conclusion: `safe_dt` is not a stronger standalone planner. It over-covers and lowers stitch precision.

## Selector Result

The useful result appears when `safe_dt` is added to the M2.34 core candidate pool.

Public ext33 leave-one-out selector:

| Selector | Samples | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Core M2.34 candidates | 33 | 0 | 0.051174 | 5.636364 | 0.424242 | 0.082527 | 0.967981 | 0.819674 |
| Core + safe_dt | 33 | 0 | 0.049160 | 5.484848 | 0.424242 | 0.049952 | 0.978413 | 0.799193 |

Incoming review leave-one-out selector:

| Selector | Samples | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Core M2.34 candidates | 4 | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.995549 | 0.781068 |
| Core + safe_dt | 4 | 0 | 0.074710 | 8.250000 | 1.000000 | 0.000000 | 0.996541 | 0.753519 |

Public selected candidate counts for `Core + safe_dt`:

- fixed inset1: 7
- fixed inset2: 11
- safe_dt: 6
- raw nearest-row: 4
- auto/eval repair: 1
- edgewalk rows16: 2
- skeleton variants: 2

Incoming selected candidate counts for `Core + safe_dt`:

- safe_dt: 1
- raw nearest-row: 1
- fixed inset2: 1
- edgewalk rows16: 1

## Decision

M2.41 is a real selector-level improvement under the current unified execution objective:

- public loss improves from 0.051174 to 0.049160;
- public jump count improves from 5.636364 to 5.484848;
- public off-mask length improves from 0.082527mm to 0.049952mm;
- incoming loss improves from 0.078290 to 0.074710;
- hard_fail remains 0 on both validation sets.

The tradeoff is lower stitch precision:

- public precision drops from 0.819674 to 0.799193;
- incoming precision drops from 0.781068 to 0.753519.

Therefore M2.41 should be treated as the current best for unified executable DST quality, while M2.34 remains the cleaner precision-safe baseline. The next optimization should add a precision-aware selector penalty or coverage/precision Pareto reporting instead of making `safe_dt` the default generator policy.

## Result Artifacts

Public:

- `results/public_benchmark_v1_ext33_mask_fill_edgewalk_nearestrow_safeadt_t2m5_r055_a64_rows16_p40/`
- `results/public_benchmark_v1_ext33_m2_41_core_safeadt_baseline_loo/`
- `results/public_benchmark_v1_ext33_m2_41_core_safeadt_plus_loo/`

Incoming:

- `results/incoming_review_eval_v1_mask_fill_edgewalk_nearestrow_safeadt_t2m5_r055_a64_rows16_p40/`
- `results/incoming_review_eval_v1_m2_41_core_safeadt_baseline_loo/`
- `results/incoming_review_eval_v1_m2_41_core_safeadt_plus_loo/`