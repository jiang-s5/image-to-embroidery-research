# M2.69 Text WKU Precision Profile

Date: 2026-07-03

## Summary

M2.69 adds a fifth production profile:

```text
text_wku_precision
```

The goal is narrow and deliberate: after M2.68, the remaining adaptive-precision failure was:

```text
txt_003 / WKU
```

M2.69 scans existing candidate DST outputs and switches only rendered-text samples whose adaptive stitch precision is below `0.90`, provided the candidate improves adaptive precision without increasing jump, trim, visible connector, or off-mask risk.

The profile stack is now:

```text
safe_low_cost
text_legibility
line_art_precision
shape_fill_precision
text_wku_precision
```

## What Changed

M2.69 switches one sample:

| Sample | Base | M2.69 Candidate |
|---|---|---|
| txt_003 / WKU | inset2 | public_benchmark_v1_ext33_mask_fill_edgewalk_nearestrow_inset1_rows16_p40 |

## txt_003 Delta

| Metric | Delta |
|---|---:|
| adaptive precision | +0.017257 |
| adaptive coverage | +0.022784 |
| overfill outside adaptive band | -0.017257 |
| coverage | -0.004012 |
| strict precision | -0.006480 |
| unified loss | +0.00000892 |
| jump count | 0 |
| trim count | 0 |
| off-mask length | 0 |
| visible connector | 0 |

This is a good production tradeoff because it removes the last low-adaptive-precision text sample without making the DST less executable.

## M2.68 vs M2.69

| Metric | M2.68 | M2.69 | Delta |
|---|---:|---:|---:|
| unified loss | 0.05752351 | 0.05752378 | +0.00000027 |
| jump count | 5.96969697 | 5.96969697 | 0 |
| trim count | 0.87878788 | 0.87878788 | 0 |
| off-mask length | 0.12120606 | 0.12120606 | 0 |
| visible connector | 0 | 0 | 0 |
| coverage | 0.97544161 | 0.97532003 | -0.00012158 |
| strict precision | 0.83082488 | 0.83062852 | -0.00019636 |
| adaptive coverage | 0.92970985 | 0.93040027 | +0.00069042 |
| adaptive precision | 0.96358718 | 0.96411012 | +0.00052294 |
| overfill outside adaptive band | 0.03641282 | 0.03588988 | -0.00052294 |
| low adaptive precision samples | 1 | 0 | -1 |
| low strict precision samples | 7 | 7 | 0 |
| hard fail | 0 | 0 | 0 |

## Decision Rule

M2.69 switches a text sample only if:

```text
source_name == Rendered text
current adaptive precision < 0.90
candidate adaptive precision >= 0.90
adaptive precision gain >= 0.01
strict precision >= 0.75
coverage >= 0.98
adaptive coverage >= 0.94
loss increase <= 0.01
jump increase <= 1
trim increase <= 1
visible connector increase == 0
off-mask increase == 0
candidate is not hard_fail
```

This keeps the profile conservative: it rescues WKU but does not replace the stronger `text_legibility` profile used by `txt_005 / HELLO`.

## Candidate Evidence

The selected WKU candidate:

```text
public_benchmark_v1_ext33_mask_fill_edgewalk_nearestrow_inset1_rows16_p40
```

has:

```text
quality_level = excellent
unified_loss = 0.03294437
jump_count = 4
trim_count = 0
coverage = 0.992627
strict_precision = 0.753064
adaptive_coverage = 0.975194
adaptive_precision = 0.911903
```

The M2.68 WKU baseline had:

```text
adaptive_precision = 0.894646
adaptive_coverage = 0.952410
```

So M2.69 removes the last adaptive precision failure while preserving execution safety.

## Remaining Bottlenecks

The next highest-value issue is no longer text precision. The remaining bottlenecks are:

```text
qd_002 / dog high_jump_excess
QuickDraw strict precision on line-art samples
professional satin/fill/running texture realism
```

The next practical stage should be:

```text
M2.70 jump-aware line-art profile
```

or a larger realism step:

```text
stitch-type texture renderer + satin/fill/running preview evaluation
```

## Artifacts

Candidate scanner:

```text
tools/scan_m2_sample_candidates.py
```

Selector:

```text
tools/apply_m2_text_wku_profile.py
```

Candidate scans:

```text
results/m2_69_text_wku_candidate_scan/
results/m2_69_text_all_candidate_scan/
```

M2.69 selector:

```text
results/public_benchmark_v1_ext33_m2_69_text_wku_profile/
```

Adaptive precision audit:

```text
results/public_benchmark_v1_ext33_m2_69_text_wku_profile_adaptive_audit/
```

Component-aware risk audit:

```text
results/public_benchmark_v1_ext33_m2_69_text_wku_profile_component_audit/
```
