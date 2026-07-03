# M2.68 Shape-Aware Profile

Date: 2026-07-03

## Summary

M2.68 adds a fourth professional production profile:

```text
shape_fill_precision
```

It targets Openclipart-style filled vector shapes, where the main failure mode is not text legibility or line-art continuity, but boundary overrun and off-mask stitching.

The current profile stack is now:

```text
safe_low_cost
text_legibility
line_art_precision
shape_fill_precision
```

This is closer to a real digitizing workflow: different artwork families get different production tradeoffs.

## What Changed

M2.68 switches one sample:

| Sample | Profile | Candidate |
|---|---|---|
| ocp_001 / flower | shape_fill_precision | public_benchmark_v1_ext33_mask_fill_edgewalk_nearestrow_safeadt_rows16_p40 |

This candidate was selected because it removes off-mask stitching while improving both strict and adaptive precision.

## ocp_001 Delta

| Metric | Delta |
|---|---:|
| off-mask reduction | +2.0983 mm |
| adaptive precision | +0.072612 |
| strict precision | +0.080757 |
| coverage | -0.009748 |
| adaptive coverage | -0.032456 |
| unified loss | -0.00766382 |
| jump count | +4 |
| trim count | +1 |
| visible connector | 0 |

This is a strong production tradeoff: boundary safety and precision improve, coverage remains high, and unified loss decreases.

## M2.67 vs M2.68

| Metric | M2.67 | M2.68 | Delta |
|---|---:|---:|---:|
| unified loss | 0.05775575 | 0.05752351 | -0.00023224 |
| jump count | 5.84848485 | 5.96969697 | +0.12121212 |
| trim count | 0.84848485 | 0.87878788 | +0.03030303 |
| off-mask length | 0.18479091 | 0.12120606 | -0.06358485 |
| visible connector | 0 | 0 | 0 |
| strict precision | 0.82837770 | 0.83082488 | +0.00244718 |
| adaptive precision | 0.96138682 | 0.96358718 | +0.00220036 |
| overfill outside adaptive band | 0.03861318 | 0.03641282 | -0.00220036 |
| low adaptive precision samples | 2 | 1 | -1 |
| component-aware risk score | 0.10509489 | 0.10228844 | -0.00280645 |
| review samples | 13 | 12 | -1 |
| hard fail | 0 | 0 | 0 |

## Decision Rule

M2.68 switches Openclipart samples to the shape profile only if:

```text
off-mask reduction >= 1.0 mm
adaptive precision gain >= 0.03
strict precision gain >= 0.03
coverage >= 0.96
adaptive coverage >= 0.90
loss increase <= 0.05
jump increase <= 6
trim increase <= 2
visible connector increase == 0
candidate is not hard_fail
```

## Candidate Pool

M2.68 scans these shape/fill candidates:

```text
public_benchmark_v1_ext33_mask_fill_edgewalk_nearestrow_inset1_rows16_p40
public_benchmark_v1_ext33_mask_fill_edgewalk_nearestrow_safeadt_rows16_p40
public_benchmark_v1_ext33_mask_fill_edgewalk_nearestrow_safeadt_t2m5_r035_a32_rows16_p40
public_benchmark_v1_ext33_mask_fill_edgewalk_nearestrow_inset2_rows16_p40
public_benchmark_v1_ext33_mask_fill_edgewalk_nearestrow_adapt_t2_m5_rows16_p40
public_benchmark_v1_ext33_mask_fill_edgewalk_nearestrow_qpath090_rows16_p40
public_benchmark_v1_ext33_mask_fill_edgewalk_nearestrow_quantvalid_rows16_p40
```

## Interpretation

M2.68 is one of the cleaner profile upgrades so far:

```text
risk score decreases
off-mask decreases
precision increases
low adaptive precision samples decrease
hard fail remains zero
```

The only cost is a small increase in jump/trim count and a tiny coverage decrease.

This makes M2.68 a stronger balanced profile than M2.67, because it improves both visual-risk and precision metrics.

## Remaining Bottlenecks

After M2.68, the main remaining adaptive-precision review sample is:

```text
txt_003 / WKU
```

The main execution-risk issue is:

```text
qd_002 / dog high_jump_excess
```

The next practical stage should therefore be either:

```text
M2.69 text WKU rescue
```

or:

```text
M2.69 jump-aware line-art refinement
```

## Artifacts

Selector:

```text
results/public_benchmark_v1_ext33_m2_68_shape_aware_profile/
```

Adaptive precision audit:

```text
results/public_benchmark_v1_ext33_m2_68_shape_aware_profile_adaptive_audit/
```

Component-aware risk audit:

```text
results/public_benchmark_v1_ext33_m2_68_shape_aware_profile_component_audit/
```

Candidate scan:

```text
results/m2_68_shape_candidate_scan/
```

Probe:

```text
results/m2_68_shape_aware_profile_probe/
```
