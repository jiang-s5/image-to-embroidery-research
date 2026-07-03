# M2.64 Text Precision Rescue

Date: 2026-07-03

## Summary

M2.64 adds a conservative text-only adaptive precision rescue on top of M2.62/M2.63.

M2.63 showed that rendered text was the weakest source family under adaptive precision. M2.64 therefore does not change the global planner for all samples. It only applies a text-specific candidate switch when a replacement improves adaptive precision while keeping coverage, visible connectors, off-mask length, loss, and jump growth inside conservative limits.

This is a generation selector update for text samples, not a new stitch generator.

## Main Result

M2.64 changes 2 of 5 rendered-text samples:

```text
txt_002: m2_60 -> mask_fill_edgewalk_nearestrow_adapt_t2_m5_rows16_p40
txt_004: m2_60 -> mask_fill_edgewalk_nearestrow_adapt_t2_m5_rows16_p40
```

Overall public benchmark result:

| Metric | M2.62 | M2.64 | Delta |
|---|---:|---:|---:|
| unified loss | 0.04916026 | 0.05129930 | +0.00213930 |
| jump count | 4.96969697 | 5.18181818 | +0.21212118 |
| strict precision | 0.80950997 | 0.81689582 | +0.00738585 |
| adaptive precision | 0.94587239 | 0.95116848 | +0.00529609 |
| overfill outside adaptive band | 0.05412761 | 0.04883152 | -0.00529609 |
| low strict precision samples | 11 | 9 | -2 |
| low adaptive precision samples | 6 | 4 | -2 |
| component-aware risk score | 0.09653159 | 0.09592804 | -0.00060355 |
| review samples | 15 | 13 | -2 |
| hard fail | 0 | 0 | 0 |

M2.64 improves precision and review load, but it is a tradeoff: mean loss and jump count rise slightly.

## Text-Specific Result

Rendered text changes:

| Metric | Delta |
|---|---:|
| coverage ratio | -0.00717420 |
| strict precision | +0.04874660 |
| adaptive coverage | -0.03063040 |
| adaptive precision | +0.03495420 |
| overfill outside adaptive band | -0.03495420 |

This confirms the original diagnosis: rendered text benefits from a separate precision-aware branch, but pure precision maximization can reduce coverage or add jumps.

## Gate Design

The text rescue is deliberately conservative. A candidate must satisfy:

```text
adaptive precision >= 0.95
adaptive precision gain >= 0.02
adaptive coverage >= 0.88
coverage >= 0.95
loss increase <= 0.04
jump increase <= 4
trim increase <= 2
visible connector increase <= 0
off-mask increase <= 0.25 mm
```

This prevents the selector from choosing very clean but under-covered text skeletons.

## Artifacts

Tool:

```text
tools/apply_m2_text_precision_rescue.py
```

Selector output:

```text
results/public_benchmark_v1_ext33_m2_64_text_precision_rescue_selector/
```

Adaptive precision audit:

```text
results/public_benchmark_v1_ext33_m2_64_text_precision_rescue_adaptive_audit/
```

Component-aware risk audit:

```text
results/public_benchmark_v1_ext33_m2_64_text_precision_rescue_component_audit/
```

Probe:

```text
results/m2_64_text_precision_rescue_probe/
```

## Interpretation

M2.64 is a positive text-domain tradeoff:

- improves overall adaptive precision
- improves rendered-text adaptive precision
- reduces low precision review samples
- keeps hard fail at zero
- keeps visible connector count at zero
- slightly increases loss and jump count

It should be used when text precision matters more than the tiny increase in execution cost.

## Remaining Bottleneck

`txt_005` / `HELLO` remains the weakest text sample:

```text
coverage is high
adaptive precision is low
aggressive rescue improves precision but adds too much loss and too many jumps
```

This means the next step should not simply select from the existing candidate pool. It needs a dedicated legibility-preserving text branch.

Recommended next stage:

```text
M2.65 text HELLO legibility rescue
```

Potential components:

- preserve text counters and holes
- use letter/component-aware routing
- penalize horizontal overfill across letter gaps
- allow per-letter local fill instead of whole-word fill
- evaluate legibility and adaptive precision together
