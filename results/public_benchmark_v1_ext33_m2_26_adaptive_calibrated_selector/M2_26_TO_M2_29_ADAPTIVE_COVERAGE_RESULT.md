# M2.26-M2.29 Adaptive Coverage Selector Result

Date: 2026-06-22

M2.26-M2.29 tested the next step after the fixed coverage-floor experiments: source-aware adaptive coverage targets.

Instead of using one global `min_coverage`, the selector can now use a coverage profile based on `source_name` and `target_branch`:

```text
Openclipart          higher flat-icon coverage target
OpenMoji             moderate multi-color icon target
QuickDraw            lower line-drawing target
Oxford-IIIT Pet      moderate real-photo target
Rendered text        lower line/text target
Incoming review      moderate holdout target
```

The implementation lives in:

```text
tools/tune_m2_calibrated_selector.py
```

New fields:

```text
coverage_profile
effective_min_coverage
mean_effective_min_coverage
mean_coverage_margin
```

New optional gate:

```text
--enforce-adaptive-coverage-floor
```

## Results

| Model | Main Change | Public Full Loss | Public Full Coverage | Public Full Jump | Public Full Trim | Incoming Loss | Incoming Coverage | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| M2.14 current best | gated inset-outline selector | 0.111060 | 0.826159 | 10.1875 | 3.7500 | 0.090709 | 0.856321 | keep current best |
| M2.26 | adaptive target, no adaptive floor | 0.095392 | 0.734277 | 9.2500 | 3.0000 | 0.090585 | 0.854175 | not promoted: under-coverage |
| M2.27 | balanced/conservative adaptive targets | 0.117751 | 0.834037 | 10.7500 | 4.6875 | 0.094245 | 0.927795 | not promoted: loss regression |
| M2.28 | adaptive floor, tolerance 0.05 | 0.122887 | 0.844804 | 11.1875 | 4.8125 | not applied | not applied | not promoted: over-constrained |
| M2.29 | adaptive floor, tolerance 0.10 | 0.122887 | 0.844804 | 11.1875 | 4.8125 | not applied | not applied | not promoted: same as M2.28 |

## Interpretation

M2.26 gives the clearest signal:

```text
Public full loss improves: 0.111060 -> 0.095392
Jump improves:            10.1875  -> 9.2500
Trim improves:            3.7500   -> 3.0000
Coverage regresses:       0.826159 -> 0.734277
```

This means source-aware adaptive coverage targets are useful, but the soft target is too permissive. The selector learns to choose low-command-risk candidates that do not cover enough of the design.

M2.28 and M2.29 show the opposite failure mode:

```text
Coverage recovers: 0.844804
Loss regresses:    0.122887
Jump regresses:    11.1875
Trim regresses:    4.8125
```

So adaptive floors are still too blunt. They behave like M2.25: once coverage becomes a hard filter, the system jumps to high-coverage candidates with worse command-level behavior.

## Decision

M2.26-M2.29 are **not promoted**.

M2.14 remains the current best.

## Research Lesson

The selector needs a continuous coverage-risk tradeoff, not a hard coverage gate.

The next step should be:

```text
predict source/image-specific coverage target
rank candidates with calibrated score
add a smooth coverage-risk penalty
avoid hard filtering unless coverage is catastrophically low
```

A promising next version is:

```text
M2.30 = adaptive coverage target + smooth coverage risk curve
```

The risk curve should penalize coverage deficits more sharply below a critical lower bound, while keeping the penalty mild above that bound. This should preserve M2.26's lower loss without repeating its under-coverage failure.
