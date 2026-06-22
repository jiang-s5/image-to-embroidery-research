# M2.19-M2.21 Hard-Safe Gate Experiments

Date: 2026-06-22

These experiments follow M2.18, where component-level style-aware routing reduced fill-branch jumps but was over-selected on incoming review samples.

The goal was to keep the useful style-aware/fill candidates while preventing unsafe selections.

## New Selector Gates

The candidate selector now supports optional hard gates:

- `--style-aware-hard-gate`
- `--mask-fill-hard-gate`

The gates can reject candidates by:

- off-mask stitch length;
- jump count;
- trim count;
- stitch precision;
- mask coverage.

These gates are disabled by default, so previous M2.14-M2.18 results remain reproducible.

## Experiments

### M2.19: Style-Aware Gate

Thresholds:

```text
style-aware off-mask <= 0.05 mm
style-aware jump <= 15
style-aware precision >= 0.75
style-aware coverage >= 0.80
```

Result:

| Evaluation | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Off-Mask mm | Mean Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Public ext33 LOO | 0 | 0.095790 | 8.939394 | 3.212121 | 0.110715 | 0.856572 |
| Core-to-full | 0 | 0.111060 | 10.1875 | 3.7500 | 0.187262 | 0.826159 |
| Incoming review | 0 | 0.095860 | 8.5000 | 3.7500 | 0.123875 | 0.932133 |

M2.19 successfully blocks the worst style-aware mistakes, but the learned selector still shifts toward unsafe mask-fill/outline choices on incoming review.

### M2.20: Strict Mask-Fill Family Gate

Thresholds:

```text
mask-fill off-mask <= 0.05 mm
mask-fill jump <= 20
mask-fill trim <= 3
mask-fill precision >= 0.70
mask-fill coverage >= 0.80
```

Result:

| Evaluation | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Off-Mask mm | Mean Coverage | Mean Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Public ext33 LOO | 0 | 0.096824 | 8.909091 | 3.848485 | 0.000000 | 0.805151 | 0.861442 |
| Core-to-full | 0 | 0.111759 | 10.2500 | 4.5000 | 0.000000 | 0.746555 | 0.873835 |
| Incoming review | 0 | 0.123580 | 11.5000 | 4.7500 | 0.000000 | 0.720063 | 0.816424 |

M2.20 eliminates off-mask stitch length, but it is too conservative. Coverage drops too much, and unified loss regresses.

### M2.21: Relaxed Mask-Fill Family Gate

Thresholds:

```text
mask-fill off-mask <= 0.05 mm
mask-fill jump <= 20
mask-fill trim <= 4
mask-fill precision >= 0.69
mask-fill coverage >= 0.80
```

Result:

| Evaluation | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Off-Mask mm | Mean Coverage | Mean Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Public ext33 LOO | 0 | 0.097512 | 8.969697 | 3.878788 | 0.000000 | 0.821567 | 0.857007 |
| Core-to-full | 0 | 0.113179 | 10.3750 | 4.5625 | 0.000000 | 0.780414 | 0.864687 |
| Incoming review | 0 | 0.116902 | 10.5000 | 5.0000 | 0.000000 | 0.814743 | 0.821912 |

M2.21 recovers some coverage compared with M2.20, but it still underperforms M2.14.

## Comparison With Current Best

| Model | Public LOO Loss | Core-to-Full Loss | Incoming Loss | Off-Mask Behavior | Decision |
|---|---:|---:|---:|---|---|
| M2.14 current best | 0.096228 | 0.111060 | 0.090709 | small but acceptable | keep current best |
| M2.19 style gate | 0.095790 | 0.111060 | 0.095860 | not fully solved | not promoted |
| M2.20 strict family gate | 0.096824 | 0.111759 | 0.123580 | off-mask zero, coverage too low | not promoted |
| M2.21 relaxed family gate | 0.097512 | 0.113179 | 0.116902 | off-mask zero, still too conservative | not promoted |

## Decision

M2.19-M2.21 are **not promoted**.

M2.14 remains the current best.

## Research Interpretation

The gate experiments are still useful. They show that:

1. hard gates can remove visible/off-mask risks;
2. pure hard gating can over-prune high-coverage candidates;
3. the next selector should not use fixed thresholds only;
4. style/fill safety should become a learned calibration problem.

The next promising direction is a two-stage selector:

```text
stage 1: hard reject truly unsafe candidates
stage 2: calibrated scorer balances coverage, precision, jump, trim, off-mask
```

The scorer should be trained with incoming-like hard samples or a validation split that penalizes off-mask and under-coverage at the same time.

