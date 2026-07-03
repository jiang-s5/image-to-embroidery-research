# M2.72 Line Skeleton Fidelity Profile

Date: 2026-07-03

## Summary

M2.72 adds the first production profile that treats line-art samples as centerline/stroke data instead of generic filled masks:

```text
line_skeleton_fidelity
```

The key change is to use original QuickDraw vector strokes when they are available. This is closer to how a human digitizer handles simple line drawings:

```text
follow the original stroke path
avoid inventing connectors across empty space
use jumps between separate stroke groups
```

## Why This Is Different From M2.71

M2.71 rejected route/segment candidates because they tried to reduce jumps by drawing extra connections. That was unsafe:

```text
fewer jumps
but more off-mask / hard-fail / precision regressions
```

M2.72 changes the representation instead:

```text
raw QuickDraw vector strokes
  -> nearest stroke ordering
  -> optional stroke reversal
  -> conservative in-mask connection only
  -> DST running-stitch trace
```

This avoids the earlier mistake of treating a line drawing as a filled region that should be connected through arbitrary mask routes.

## New Tools

```text
tools/run_quickdraw_vector_stroke_candidate.py
tools/audit_line_skeleton_coverage.py
tools/apply_m2_line_skeleton_profile.py
```

## Candidate

The promoted candidate source is:

```text
results/public_benchmark_v1_ext33_m2_72_qd_vector_nearest_reverse_connect8/
```

It uses:

```text
stroke_order = nearest
allow_reverse = true
connect_strokes = true
max_connect_mm = 8
min_connect_inside_fraction = 0.95
```

## Promotion Gate

A QuickDraw sample is switched only if the vector-stroke candidate:

```text
does not increase off-mask length
does not increase visible connectors
does not increase jump count
does not increase trim count
does not increase unified loss
keeps adaptive stitch precision >= 0.95
keeps adaptive skeleton coverage >= 0.95
improves skeleton coverage by >= 0.005
improves adaptive skeleton coverage by >= 0.005
```

This gate prevents the vector method from being used on samples where it looks cleaner but costs too many jumps.

## Switched Samples

M2.72 switches 3 QuickDraw samples:

| Sample | Category | Loss Delta | Jump Delta | Coverage Delta | Strict Precision Delta | Skeleton Coverage Delta | Adaptive Skeleton Coverage Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| qd_003 | fish | -0.00912909 | -1 | +0.011692 | +0.081947 | +0.016322 | +0.021081 |
| qd_004 | flower | -0.00029207 | 0 | +0.071889 | +0.074691 | +0.041797 | +0.054945 |
| qd_007 | moon | -0.00756218 | -1 | +0.011519 | +0.080449 | +0.012625 | +0.010389 |

The most important switch is `qd_007 / moon`, which was the worst risk sample after M2.71.

## M2.71 vs M2.72

| Metric | M2.71 | M2.72 | Delta |
|---|---:|---:|---:|
| unified loss | 0.05558126 | 0.05506661 | -0.00051465 |
| jump count | 5.75757576 | 5.69696970 | -0.06060606 |
| trim count | 0.84848485 | 0.84848485 | 0 |
| off-mask length | 0.12120606 | 0.12120606 | 0 |
| visible connector | 0 | 0 | 0 |
| coverage | 0.97682379 | 0.97970561 | +0.00288182 |
| strict precision | 0.82978270 | 0.83696715 | +0.00718445 |
| adaptive coverage | 0.93203370 | 0.92858318 | -0.00345052 |
| adaptive precision | 0.96332882 | 0.96633306 | +0.00300424 |
| low strict precision samples | 8 | 6 | -2 |
| low adaptive precision samples | 0 | 0 | 0 |
| review samples | 12 | 10 | -2 |
| component-aware risk score | 0.09763792 | 0.09213475 | -0.00550317 |
| worst risk score | 0.25929687 | 0.23471031 | -0.02458656 |
| hard fail | 0 | 0 | 0 |

## QuickDraw Skeleton Metrics

The line-art specific metric improves:

| Metric | M2.71 | M2.72 | Delta |
|---|---:|---:|---:|
| skeleton coverage | 0.97261538 | 0.98145837 | +0.00884299 |
| skeleton precision | 0.14839162 | 0.15470350 | +0.00631188 |
| adaptive skeleton coverage | 0.96616825 | 0.97697013 | +0.01080188 |
| adaptive skeleton precision | 0.37492287 | 0.39161612 | +0.01669325 |
| low adaptive skeleton coverage samples | 3 | 2 | -1 |

This matters because mask coverage alone is not enough for line drawings. A line-art DST should be judged against the centerline/skeleton as well as the thick mask.

## Interpretation

M2.72 is a meaningful step toward professional digitizing behavior:

```text
line drawing input
  -> use stroke structure
  -> preserve contour/centerline fidelity
  -> avoid artificial connectors
  -> still pass execution and visual-risk gates
```

It does not solve full professional embroidery realism. The system still needs explicit stitch-type generation:

```text
running stitch for line art
satin stitch for borders/text strokes
fill stitch for large colored regions
thread-density and direction variation
render-back texture metrics
```

But M2.72 fixes an important conceptual gap: line-art samples now have a line-art-specific planner path instead of being forced through the filled-mask pipeline.

## Artifacts

Config:

```text
configs/best_current_model_m2_72_line_skeleton_profile.json
```

Selection:

```text
results/public_benchmark_v1_ext33_m2_72_line_skeleton_profile/
```

QuickDraw vector candidate:

```text
results/public_benchmark_v1_ext33_m2_72_qd_vector_nearest_reverse_connect8/
```

Skeleton audit:

```text
results/public_benchmark_v1_ext33_m2_72_line_skeleton_profile_skeleton_audit/
```

Component-aware risk audit:

```text
results/public_benchmark_v1_ext33_m2_72_line_skeleton_profile_component_audit/
```

Adaptive precision audit:

```text
results/public_benchmark_v1_ext33_m2_72_line_skeleton_profile_adaptive_audit/
```

## Next Direction

The next strongest direction is a stitch-type-aware planner:

```text
line skeleton -> running stitch
wide contours / text strokes -> satin stitch
large color regions -> fill stitch with direction and density control
```

That is the next necessary step toward the user's target: output that visually resembles professional embroidery software rather than a single generic row-fill preview.
