# M2.63 Adaptive Line Precision

Date: 2026-06-25

## Summary

M2.63 adds an adaptive line-domain precision metric. It keeps the existing strict stitch precision, but adds a second metric that allows for expected thread width around thin target strokes.

This is an evaluation upgrade, not a new DST generator.

The main finding is that M2.62's precision-rescue improvement remains visible under adaptive precision:

```text
M2.60 adaptive precision: 0.92835473
M2.62 adaptive precision: 0.94587239
delta: +0.01751766
```

Adaptive overfill also decreases:

```text
M2.60 overfill outside adaptive band: 0.07164527
M2.62 overfill outside adaptive band: 0.05412761
delta: -0.01751766
```

## Why This Matters

Strict pixel precision compares the rendered stitch footprint directly against the original binary mask. For thin line drawings, this can be too harsh because thread has physical width.

Example:

```text
thin target stroke
rendered thread footprint is wider
strict precision penalizes expected thread width
```

M2.63 therefore reports both:

- `stitch_precision_ratio`: strict precision against the original mask
- `adaptive_stitch_precision_ratio`: precision against a dilated mask that represents expected thread width

## Tool Changes

Updated:

```text
tools/eval_stitch_coverage.py
```

New metrics:

```text
adaptive_target_pixels
adaptive_overlap_pixels
adaptive_off_target_pixels
adaptive_coverage_ratio
adaptive_stitch_precision_ratio
overfill_outside_adaptive_ratio
adaptive_precision_radius_px
```

New audit tool:

```text
tools/audit_adaptive_line_precision.py
```

Primary audit:

```text
results/public_benchmark_v1_ext33_m2_63_adaptive_precision_audit_r2/
```

Probe summary:

```text
results/m2_63_adaptive_precision_probe/
```

## Main Result

Primary radius is `2 px`, matching the `line_radius_px=2` stitch rendering setting.

| Metric | M2.60 | M2.62 | Delta |
|---|---:|---:|---:|
| strict precision | 0.79361039 | 0.80950997 | +0.01589958 |
| adaptive precision | 0.92835473 | 0.94587239 | +0.01751766 |
| overfill outside adaptive band | 0.07164527 | 0.05412761 | -0.01751766 |
| low strict precision samples | 11 | 11 | 0 |
| low adaptive precision samples | 8 | 6 | -2 |

This means M2.62 does not merely improve a brittle strict metric. It also reduces stitch footprint outside the expected thread-width band.

## Radius Sensitivity

| Adaptive Radius | M2.60 Adaptive Precision | M2.62 Adaptive Precision | Delta | M2.60 Low Samples | M2.62 Low Samples |
|---:|---:|---:|---:|---:|---:|
| 1 px | 0.86298188 | 0.88171897 | +0.01873709 | 17 | 17 |
| 2 px | 0.92835473 | 0.94587239 | +0.01751766 | 8 | 6 |
| 3 px | 0.95895397 | 0.97266788 | +0.01371391 | 3 | 1 |

The direction is stable across radii: M2.62 improves adaptive precision and reduces adaptive overfill.

## Source-Level Finding

Average adaptive precision by source:

| Source | Samples | Adaptive Precision |
|---|---:|---:|
| OpenMoji | 20 | 0.9689798 |
| Openclipart | 20 | 0.9609810 |
| QuickDraw | 16 | 0.90658806 |
| Rendered text | 10 | 0.8744870 |

Rendered text remains the weakest source. This suggests the next improvement should not be another generic line-domain rescue, but a text-specific branch.

## Interpretation

M2.63 changes how we read precision:

```text
strict precision = exact binary-mask footprint
adaptive precision = footprint inside expected thread-width band
overfill outside adaptive band = true excessive spread
```

For research reporting, use both strict and adaptive precision. Strict precision catches thin-mask mismatch, while adaptive precision better reflects physical embroidery thread width.

## Recommendation

Use M2.62 as the precision-aware generation default, and use M2.63 adaptive precision as the review/evaluation lens.

Next stage:

```text
M2.64 text-specific precision rescue
```

Recommended components:

- detect text/source family
- prefer satin/outline-aware candidates for rendered text
- separate text holes/counters from line doodles
- optimize adaptive precision without losing legibility

