# M2.65 Text Legibility Rescue

Date: 2026-07-03

## Summary

M2.65 introduces a text-legibility production profile. It builds on M2.64, but allows a larger execution-cost budget for rendered text when the candidate strongly improves adaptive precision and keeps visual safety unchanged.

This is closer to how professional embroidery digitizing software behaves: text can justify more jumps/trims if the result preserves letter shape and avoids overfilled counters/gaps.

## What Changed

M2.64 changed two rendered-text samples:

```text
txt_002
txt_004
```

M2.65 additionally changes:

```text
txt_005 / HELLO
```

The selected candidate is:

```text
mask_fill_edgewalk_nearestrow_adapt_t2_m5_rows16_p40
```

For `txt_005`, this tradeoff is:

| Metric | Delta |
|---|---:|
| adaptive precision | +0.211287 |
| unified loss | +0.10975887 |
| jump count | +11 |
| adaptive coverage | -0.086964 |
| visible connector | +0 |
| off-mask length | +0 |

The large adaptive precision gain is why this sample belongs in a text-legibility profile rather than the safer M2.64 profile.

## M2.64 vs M2.65

| Metric | M2.64 | M2.65 | Delta |
|---|---:|---:|---:|
| unified loss | 0.05129930 | 0.05462533 | +0.00332603 |
| jump count | 5.18181818 | 5.51515152 | +0.33333334 |
| trim count | 0.69696970 | 0.78787879 | +0.09090909 |
| strict precision | 0.81689582 | 0.82380788 | +0.00691206 |
| adaptive precision | 0.95116848 | 0.95757112 | +0.00640264 |
| overfill outside adaptive band | 0.04883152 | 0.04242888 | -0.00640264 |
| low strict precision samples | 9 | 8 | -1 |
| low adaptive precision samples | 4 | 3 | -1 |
| visible connector | 0 | 0 | 0 |
| hard fail | 0 | 0 | 0 |

M2.65 improves text/precision quality, but it is not a free win: execution cost and component-aware risk score rise slightly.

## Text-Specific Result

Rendered text average delta from M2.64 to M2.65:

| Metric | Delta |
|---|---:|
| coverage ratio | -0.00291900 |
| strict precision | +0.04561960 |
| adaptive coverage | -0.01739280 |
| adaptive precision | +0.04225740 |
| overfill outside adaptive band | -0.04225740 |

This means the text branch is doing what it is supposed to do: it sacrifices a small amount of coverage and execution cost to reduce overfill and improve letter clarity.

## Gate Difference From M2.64

M2.64 used a safer profile:

```text
max_loss_slack = 0.04
max_jump_increase = 4
max_trim_increase = 2
```

M2.65 uses a legibility profile:

```text
max_loss_slack = 0.12
max_jump_increase = 12
max_trim_increase = 4
min_adaptive_precision = 0.95
min_adaptive_precision_gain = 0.02
min_adaptive_coverage = 0.88
min_coverage = 0.95
visible connector increase = 0
off-mask increase <= 0.25 mm
```

The important rule is that M2.65 only spends extra execution cost if visual safety stays clean.

## Artifacts

Selector:

```text
results/public_benchmark_v1_ext33_m2_65_text_legibility_rescue_selector/
```

Adaptive precision audit:

```text
results/public_benchmark_v1_ext33_m2_65_text_legibility_rescue_adaptive_audit/
```

Component-aware risk audit:

```text
results/public_benchmark_v1_ext33_m2_65_text_legibility_rescue_component_audit/
```

Probe:

```text
results/m2_65_text_legibility_rescue_probe/
```

## Interpretation

M2.65 is a positive professional-profile experiment, not a universal default.

Use M2.64 when the goal is:

```text
lower execution cost
lower jump/trim growth
safer general-purpose output
```

Use M2.65 when the goal is:

```text
text legibility
less overfill
clearer rendered text
professional text profile behavior
```

## Remaining Bottlenecks

M2.65 leaves three adaptive-precision review samples:

```text
qd_002 / dog
ocp_001 / flower
txt_003 / WKU
```

The next step should be a profile switcher rather than another single global selector:

```text
M2.66 profile switcher
```

Recommended profiles:

- low-cost execution profile
- text-legibility profile
- line-art precision profile
- balanced default profile

This is the direction that moves the project closer to professional digitizing software: not one fixed rule, but selectable production profiles with different tradeoffs.
