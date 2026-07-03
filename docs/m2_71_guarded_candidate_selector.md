# M2.71 Guarded Candidate Selector

Date: 2026-07-03

## Summary

M2.71 adds a production guard on top of M2.70:

```text
m2_71_guarded_candidate_selector
```

This version does not promote a new stitch output. Instead, it formalizes a no-regression gate for candidate selection. This is important because the M2.71 route/segment experiments showed an attractive but unsafe pattern:

```text
slightly lower jump on one line-art sample
but higher off-mask length, more hard failures, or lower coverage/precision
```

The guarded selector prevents those candidates from entering the production profile.

## Why This Matters

The goal is professional embroidery-like output, not metric gaming. A candidate is not acceptable if it improves one number while making the stitch file visibly dirtier or less executable.

M2.71 therefore treats candidate selection as a governance step:

```text
candidate pool
  -> no-regression gate
  -> material improvement check
  -> promotion only if execution, visual, and coverage metrics remain safe
```

This is a response to the current failure mode:

```text
mask/route based connection can reduce a jump,
but it can also draw a connector through the wrong area.
```

## Candidate Pool

M2.71 evaluates candidates from:

```text
results/m2_70_line_art_all_candidate_scan/candidate_rows.csv
results/m2_71_quickdraw_route_probe_scan/candidate_rows.csv
```

The second scan includes the new QuickDraw route/segment probes:

```text
public_benchmark_v1_ext33_m2_71_sg_open_d1_c24_i090_r42_f32_min8
public_benchmark_v1_ext33_m2_71_sg_euler_d1_c24_i090_r42_f32_min8
public_benchmark_v1_ext33_m2_71_maskpath_d1_c24_i090_r42_f32_min8
public_benchmark_v1_ext33_m2_71_maskpath_d2_c28_i085_r48_f36_min6
```

## Guard Rules

Candidates are rejected if they introduce any of the following regressions:

```text
off-mask increase > 0
visible connector increase > 0
jump increase > 0
trim increase > 0
unified loss increase > 0
coverage drop > 0.005
strict precision drop > 0.005
adaptive coverage drop > 0.01
adaptive precision drop > 0.01
absolute coverage < 0.90
absolute adaptive coverage < 0.85
absolute adaptive precision < 0.94
candidate is hard_fail
```

A candidate must also provide a material improvement:

```text
loss reduction >= 0.0005
or jump reduction >= 1
or trim reduction >= 1
or coverage gain >= 0.01
or strict precision gain >= 0.01
or adaptive coverage gain >= 0.01
or adaptive precision gain >= 0.005
```

## Result

No candidate passed the guard:

| Metric | M2.70 | M2.71 |
|---|---:|---:|
| promoted samples | 1 from M2.69 to M2.70 | 0 new promotions |
| hard fail | 0 | 0 |
| unified loss | 0.05558126 | 0.05558126 |
| jump count | 5.75757576 | 5.75757576 |
| trim count | 0.84848485 | 0.84848485 |
| off-mask length | 0.12120606 | 0.12120606 |
| visible connector | 0 | 0 |
| coverage | 0.97682379 | 0.97682379 |
| strict precision | 0.82978270 | 0.82978270 |
| adaptive coverage | 0.93203370 | 0.93203370 |
| adaptive precision | 0.96332882 | 0.96332882 |
| low adaptive precision samples | 0 | 0 |
| low strict precision samples | 8 | 8 |
| component-aware risk score | 0.09763792 | 0.09763792 |
| worst risk sample | qd_007 / moon | qd_007 / moon |

## Rejection Reasons

The main rejection reasons are:

| Reason | Count |
|---|---:|
| jump increase | 490 |
| candidate hard_fail | 262 |
| off-mask increase | 146 |
| strict precision drop | 54 |
| absolute adaptive coverage too low | 46 |
| no material improvement | 34 |
| absolute adaptive precision too low | 31 |
| coverage drop | 22 |
| loss increase | 10 |
| trim increase | 7 |

This confirms that the newer route/segment probes are not ready for production use.

## qd_007 Interpretation

The current worst sample is:

```text
qd_007 / moon
```

Its source image is a sparse QuickDraw line drawing: an outer moon contour plus several disconnected interior marks. The correct embroidery behavior should be closer to:

```text
outer contour as one stroke group
interior holes/marks as separate short stroke groups
minimal jumps between groups
no artificial connectors across empty space
```

The failed M2.71 route experiments tried to connect components through a mask route, but that is not equivalent to real digitizing. The next work should treat the drawing as a stroke graph, not as a generic filled mask.

## Research Interpretation

M2.71 is a negative-but-useful result:

```text
The candidate pool does not contain a safe improvement over M2.70.
```

This is valuable because it prevents reward hacking. A narrow metric such as jump count could prefer a candidate that produces visible/off-mask artifacts. The guarded selector blocks that.

## Artifacts

Selector:

```text
tools/apply_m2_guarded_candidate_selector.py
```

M2.71 output:

```text
results/public_benchmark_v1_ext33_m2_71_guarded_candidate_selector/
```

Route probe scan:

```text
results/m2_71_quickdraw_route_probe_scan/
```

Adaptive precision audit:

```text
results/public_benchmark_v1_ext33_m2_71_guarded_candidate_selector_adaptive_audit/
```

Component-aware risk audit:

```text
results/public_benchmark_v1_ext33_m2_71_guarded_candidate_selector_component_audit/
```

Config:

```text
configs/best_current_model_m2_71_guarded_candidate_selector.json
```

## Next Direction

The next real improvement should not be another relaxed connector.

The most promising next step is:

```text
stroke-aware graph grouping for line art
```

Specifically:

```text
1. split line-art into contour groups and interior mark groups
2. preserve original stroke order when QuickDraw vector strokes are available
3. use graph traversal only inside a group
4. use jump/trim only between groups unless a connector is proven invisible
5. evaluate line-art samples with group-level continuity and no artificial connector metrics
```

This is the path toward the user's stated requirement: continuous, believable embroidery lines rather than disconnected or artificially connected line segments.
