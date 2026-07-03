# M2.66 Profile Switcher

Date: 2026-07-03

## Summary

M2.66 converts the M2.64/M2.65 tradeoff into a professional-style profile switcher.

Instead of using one universal selector for every image, M2.66 chooses between:

```text
safe_low_cost     -> M2.64 text precision rescue
text_legibility   -> M2.65 text legibility rescue
```

This moves the system closer to embroidery digitizing software behavior: the production profile is part of the decision, not a hidden side effect of one global score.

## Decision Rule

M2.66 keeps the safe low-cost profile for all non-text artwork.

For rendered text, it switches to the text-legibility profile only when:

```text
adaptive precision gain >= 0.015
overfill reduction >= 0.015
candidate adaptive precision >= 0.95
candidate adaptive coverage >= 0.88
loss increase <= 0.12
jump increase <= 12
trim increase <= 4
visible connector increase == 0
off-mask increase <= 0.25 mm
```

## What Changed

M2.66 switches one sample:

| Sample | Profile | Reason |
|---|---|---|
| txt_005 / HELLO | text_legibility | large text precision gain with zero visible/off-mask increase |

The other 32 samples stay on the safe low-cost profile.

## M2.64 vs M2.66

| Metric | M2.64 | M2.66 | Delta |
|---|---:|---:|---:|
| unified loss | 0.05129930 | 0.05462533 | +0.00332603 |
| jump count | 5.18181818 | 5.51515152 | +0.33333334 |
| trim count | 0.69696970 | 0.78787879 | +0.09090909 |
| off-mask length | 0.18479091 | 0.18479091 | 0 |
| visible connector | 0 | 0 | 0 |
| strict precision | 0.81689582 | 0.82380788 | +0.00691206 |
| adaptive precision | 0.95116848 | 0.95757112 | +0.00640264 |
| overfill outside adaptive band | 0.04883152 | 0.04242888 | -0.00640264 |
| low strict precision samples | 9 | 8 | -1 |
| low adaptive precision samples | 4 | 3 | -1 |
| component-aware risk score | 0.09592804 | 0.09933872 | +0.00341068 |
| hard fail | 0 | 0 | 0 |

## HELLO Switch

For `txt_005 / HELLO`, M2.66 selects:

```text
mask_fill_edgewalk_nearestrow_adapt_t2_m5_rows16_p40
```

The sample-level delta is:

| Metric | Delta |
|---|---:|
| adaptive precision | +0.211287 |
| overfill reduction | +0.211287 |
| unified loss | +0.10975887 |
| jump count | +11 |
| trim count | +3 |
| visible connector | 0 |
| off-mask length | 0 |

This is a conscious production tradeoff: spend more execution cost to preserve text shape.

## Interpretation

M2.66 is not a large metric jump over M2.65 because M2.65 already contained the text-legibility rescue. The important improvement is architectural:

```text
single global selector -> auditable profile system
```

This matters for professional embroidery because different designs require different priorities:

- text needs legibility
- line art needs continuity and running-stitch clarity
- flat icons need safe fill and low jump
- complex graphics need balanced execution cost

M2.66 implements the first version of that production-profile layer.

## Artifacts

Selector:

```text
results/public_benchmark_v1_ext33_m2_66_profile_switcher/
```

Adaptive precision audit:

```text
results/public_benchmark_v1_ext33_m2_66_profile_switcher_adaptive_audit/
```

Component-aware risk audit:

```text
results/public_benchmark_v1_ext33_m2_66_profile_switcher_component_audit/
```

Probe:

```text
results/m2_66_profile_switcher_probe/
```

Config:

```text
configs/best_current_model_m2_66_profile_switcher.json
```

## Remaining Bottlenecks

M2.66 leaves these main problems:

```text
QuickDraw line-art strict precision is still low.
Openclipart flower remains an adaptive-precision review sample.
Only two profiles are available so far.
```

The next stage should be:

```text
M2.67 line-art profile
```

The goal is to give QuickDraw and sketch-like inputs a dedicated running-stitch / topology-preserving profile instead of forcing them through text or fill-oriented logic.
