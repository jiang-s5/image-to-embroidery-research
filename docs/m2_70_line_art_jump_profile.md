# M2.70 Line-Art Jump Profile

Date: 2026-07-03

## Summary

M2.70 adds a sixth production profile:

```text
line_art_jump_relief
```

The goal is to address the remaining execution-risk bottleneck after M2.69:

```text
qd_002 / dog high_loss | high_jump_excess
```

M2.70 is not a free improvement. It is a production tradeoff: it reduces excessive jumps and improves coverage/loss, while accepting a small strict precision drop on the affected line-art sample.

The profile stack is now:

```text
safe_low_cost
text_legibility
line_art_precision
shape_fill_precision
text_wku_precision
line_art_jump_relief
```

## What Changed

M2.70 switches one QuickDraw sample:

| Sample | Base Profile | M2.70 Profile | Candidate |
|---|---|---|---|
| qd_002 / dog | line_art_precision | line_art_jump_relief | public_benchmark_v1_ext33_mask_fill_edgewalk_nearestrow_adapt_t2_m5_rows16_p40 |

## qd_002 Delta

| Metric | M2.69 | M2.70 | Delta |
|---|---:|---:|---:|
| unified loss | 0.13667042 | 0.07256709 | -0.06410333 |
| jump count | 15 | 8 | -7 |
| trim count | 2 | 1 | -1 |
| coverage | 0.942126 | 0.991750 | +0.049624 |
| adaptive coverage | 0.831707 | 0.885610 | +0.053903 |
| adaptive precision | 0.989936 | 0.964153 | -0.025783 |
| strict precision | 0.751403 | 0.723491 | -0.027912 |
| off-mask length | 0 | 0 | 0 |
| visible connector | 0 | 0 | 0 |

This removes the high jump-excess issue for `qd_002` while keeping adaptive precision above `0.94`.

## M2.69 vs M2.70

| Metric | M2.69 | M2.70 | Delta |
|---|---:|---:|---:|
| unified loss | 0.05752378 | 0.05558126 | -0.00194252 |
| jump count | 5.96969697 | 5.75757576 | -0.21212121 |
| trim count | 0.87878788 | 0.84848485 | -0.03030303 |
| off-mask length | 0.12120606 | 0.12120606 | 0 |
| visible connector | 0 | 0 | 0 |
| coverage | 0.97532003 | 0.97682379 | +0.00150376 |
| strict precision | 0.83062852 | 0.82978270 | -0.00084582 |
| adaptive coverage | 0.93040027 | 0.93203370 | +0.00163343 |
| adaptive precision | 0.96411012 | 0.96332882 | -0.00078130 |
| low adaptive precision samples | 0 | 0 | 0 |
| low strict precision samples | 7 | 8 | +1 |
| component-aware risk score | 0.10228871 | 0.09763792 | -0.00465079 |
| worst risk score | 0.31067042 | 0.25929687 | -0.05137355 |
| hard fail | 0 | 0 | 0 |

## Decision Rule

M2.70 switches a QuickDraw sample only if:

```text
source_name == QuickDraw
current jump_count >= 12
jump reduction >= 4
candidate jump_count <= 10
candidate adaptive precision >= 0.94
candidate adaptive coverage >= 0.85
candidate coverage >= 0.94
candidate strict precision >= 0.70
adaptive precision drop <= 0.04
strict precision drop <= 0.06
loss does not increase
trim does not increase
visible connector increase == 0
off-mask increase == 0
candidate is not hard_fail
```

## Interpretation

M2.70 is closer to a real digitizing tradeoff than the earlier all-or-nothing profiles:

```text
M2.67 line_art_precision:
  cleaner line precision, but qd_002 has too many jumps

M2.70 line_art_jump_relief:
  fewer jumps and lower risk, but qd_002 line tightness is slightly weaker
```

This matters because professional embroidery software is not optimizing one metric. A digitizer often chooses between:

```text
maximum line tightness
lower jump / trim burden
cleaner machine execution
```

M2.70 explicitly exposes that tradeoff.

## Remaining Bottlenecks

After M2.70, the worst risk sample becomes:

```text
qd_007 / moon
```

with:

```text
high_loss | low_precision
```

The remaining technical gap is no longer just jump count. The next serious work should target:

```text
line-art strict precision without reintroducing high jumps
```

and then the larger realism gap:

```text
satin / fill / running stitch texture and stitch-type-aware preview metrics
```

## Artifacts

Candidate scanner:

```text
tools/scan_m2_sample_candidates.py
```

Selector:

```text
tools/apply_m2_line_art_jump_profile.py
```

Candidate scans:

```text
results/m2_70_line_art_jump_candidate_scan/
results/m2_70_line_art_all_candidate_scan/
```

M2.70 selector:

```text
results/public_benchmark_v1_ext33_m2_70_line_art_jump_profile/
```

Adaptive precision audit:

```text
results/public_benchmark_v1_ext33_m2_70_line_art_jump_profile_adaptive_audit/
```

Component-aware risk audit:

```text
results/public_benchmark_v1_ext33_m2_70_line_art_jump_profile_component_audit/
```
