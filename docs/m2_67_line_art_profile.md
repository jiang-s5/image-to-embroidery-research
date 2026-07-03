# M2.67 Line-Art Profile

Date: 2026-07-03

## Summary

M2.67 adds a third professional production profile:

```text
line_art_precision
```

It builds on M2.66, which already introduced:

```text
safe_low_cost
text_legibility
```

The purpose is to avoid treating sketch-like QuickDraw inputs as normal fill graphics. Line drawings need a running-stitch / topology-oriented profile that prioritizes line adherence and reduced overfill.

## Candidate Pool

M2.67 does not use one fixed line-art candidate. It chooses from a small candidate pool:

```text
public_benchmark_v1_ext33_m2_56_line_stroke_d0_c8_i095
public_benchmark_v1_ext33_m2_57_topology_open_d0_c12_i090_min12
public_benchmark_v1_ext33_m2_57_topology_open_d0_c12_i090_min20
public_benchmark_v1_ext33_m2_57_topology_open_d0_c20_i080_min30
```

The chosen candidate must improve both adaptive precision and strict precision while keeping visible/off-mask risk clean.

## Decision Rule

For QuickDraw samples, M2.67 switches to `line_art_precision` only when:

```text
adaptive precision gain >= 0.10
strict precision gain >= 0.03
overfill reduction >= 0.05
candidate adaptive precision >= 0.95
candidate adaptive coverage >= 0.80
loss increase <= 0.18
jump increase <= 22
trim increase <= 4
visible connector increase == 0
off-mask increase <= 0.25 mm
candidate is not hard_fail
```

## What Changed

M2.67 switches one sample:

| Sample | Profile | Candidate |
|---|---|---|
| qd_002 / dog | line_art_precision | public_benchmark_v1_ext33_m2_57_topology_open_d0_c12_i090_min12 |

The selected candidate is a topology-open line-art candidate, not the earlier higher-jump line-stroke candidate.

## qd_002 Delta

| Metric | Delta |
|---|---:|
| adaptive precision | +0.125918 |
| strict precision | +0.150804 |
| overfill reduction | +0.125918 |
| unified loss | +0.10330388 |
| jump count | +11 |
| trim count | +2 |
| visible connector | 0 |
| off-mask length | 0 |

This is a deliberate professional tradeoff: the line becomes much cleaner, but execution cost increases.

## M2.66 vs M2.67

| Metric | M2.66 | M2.67 | Delta |
|---|---:|---:|---:|
| unified loss | 0.05462533 | 0.05775575 | +0.00313042 |
| jump count | 5.51515152 | 5.84848485 | +0.33333333 |
| trim count | 0.78787879 | 0.84848485 | +0.06060606 |
| off-mask length | 0.18479091 | 0.18479091 | 0 |
| visible connector | 0 | 0 | 0 |
| strict precision | 0.82380788 | 0.82837770 | +0.00456982 |
| adaptive precision | 0.95757112 | 0.96138682 | +0.00381570 |
| overfill outside adaptive band | 0.04242888 | 0.03861318 | -0.00381570 |
| low strict precision samples | 8 | 7 | -1 |
| low adaptive precision samples | 3 | 2 | -1 |
| component-aware risk score | 0.09933872 | 0.10509489 | +0.00575617 |
| hard fail | 0 | 0 | 0 |

## Interpretation

M2.67 is a precision-profile improvement, not a free universal improvement.

It shows that a professional-style profile system can now represent three different production intentions:

```text
safe_low_cost       -> general low-cost output
text_legibility     -> preserve text shape
line_art_precision  -> preserve sketch/running-stitch structure
```

That matters more than the small aggregate metric gain, because professional embroidery software is not one global mode. It is a set of production tradeoffs.

## Remaining Bottlenecks

M2.67 still leaves:

```text
7 QuickDraw strict-precision review samples
Openclipart ocp_001 / flower
Rendered text txt_003 / WKU
```

The next technical bottleneck is not another selector threshold. It is a better continuous stroke tracer that can reduce jump count while keeping the line-art precision gain.

## Artifacts

Selector:

```text
results/public_benchmark_v1_ext33_m2_67_line_art_profile/
```

Adaptive precision audit:

```text
results/public_benchmark_v1_ext33_m2_67_line_art_profile_adaptive_audit/
```

Component-aware risk audit:

```text
results/public_benchmark_v1_ext33_m2_67_line_art_profile_component_audit/
```

Candidate scan:

```text
results/m2_67_line_art_candidate_scan/
```

Probe:

```text
results/m2_67_line_art_profile_probe/
```
