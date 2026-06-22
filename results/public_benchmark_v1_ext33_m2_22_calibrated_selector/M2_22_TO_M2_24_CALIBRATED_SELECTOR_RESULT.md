# M2.22-M2.25 Two-Stage Calibrated Selector Result

Date: 2026-06-22

M2.22-M2.25 tested a two-stage calibrated selector:

```text
stage 1: reject hard-fail candidates
stage 2: rank remaining candidates with calibrated score
```

The calibrated score is:

```text
score =
  unified_loss
  + coverage_weight * coverage_deficit
  + precision_weight * precision_deficit
  + jump_weight * normalized_jump
  + trim_weight * normalized_trim
  + off_mask_weight * off_mask_mm
  + visible_weight * visible_connector_count
```

This is different from the earlier learned ridge selector. It is interpretable and explicitly balances executability, coverage, precision, jump, trim, and off-mask risk.

## New Tool

Implemented:

```text
tools/tune_m2_calibrated_selector.py
```

The tool supports:

- grid search on a train phase;
- evaluation on a test phase;
- applying a saved `calibrated_config.json` to incoming/holdout data;
- optional branch/source coverage-floor enforcement;
- optional reuse of style-aware and mask-fill hard gates.

## M2.22: Low-Loss Calibrated Selector

Best config selected on the public benchmark core split:

```text
coverage_weight = 0.20
precision_weight = 0.05
jump_weight = 0.00
trim_weight = 0.02
off_mask_weight = 0.10
min_coverage = 0.80
min_precision = 0.72
```

| Evaluation | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Off-Mask mm | Mean Coverage | Mean Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Public core train | 0 | 0.072126 | 7.117647 | 2.176471 | 0.000000 | 0.810057 | 0.824644 |
| Public full test | 0 | 0.091982 | 9.0625 | 2.8125 | 0.000000 | 0.689666 | 0.869991 |
| Incoming review | 0 | 0.090585 | 7.5000 | 4.7500 | 0.000000 | 0.854175 | 0.800173 |

M2.22 is the first post-M2.14 selector to slightly improve incoming review unified loss:

```text
M2.14 incoming loss: 0.090709
M2.22 incoming loss: 0.090585
```

However, public full coverage drops to `0.689666`, which is too low for promotion. The low-loss selector can over-prefer skeleton/low-coverage candidates.

Incoming selected candidates:

| Candidate | Count |
|---|---:|
| auto_evalrepair_r10_c20 | 2 |
| mask_fill_edgewalk_styleaware_r008_d45 | 1 |
| mask_fill_edgewalk_rows16_p40 | 1 |

The incoming gain is mainly from choosing style-aware fill on `pair_008`, where it slightly improves loss compared with normal edge-walk fill.

## M2.23: High-Coverage Calibrated Selector

M2.23 raised the coverage target:

```text
coverage_weight = 0.70
min_coverage = 0.86
```

| Evaluation | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Off-Mask mm | Mean Coverage | Mean Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Public core train | 0 | 0.093593 | 8.411765 | 4.000000 | 0.000000 | 0.941958 | 0.813420 |
| Public full test | 0 | 0.121659 | 11.1250 | 4.8125 | 0.030969 | 0.846421 | 0.838613 |

M2.23 recovers coverage, but unified loss regresses beyond M2.14.

## M2.24: Balanced Coverage Selector

M2.24 used a middle coverage target:

```text
coverage_weight = 0.35
min_coverage = 0.83
off_mask_weight = 0.05
```

| Evaluation | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Off-Mask mm | Mean Coverage | Mean Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Public core train | 0 | 0.093593 | 8.411765 | 4.000000 | 0.000000 | 0.941958 | 0.813420 |
| Public full test | 0 | 0.117751 | 10.7500 | 4.6875 | 0.030969 | 0.834037 | 0.847179 |
| Incoming review | 0 | 0.094245 | 8.5000 | 3.5000 | 0.123875 | 0.927795 | 0.777570 |

M2.24 improves coverage compared with M2.22, but no longer improves incoming loss and still does not beat M2.14 overall.

## M2.25: Coverage-Floor Calibrated Selector

M2.25 added a hard coverage-floor filter before calibrated ranking:

```text
exclude hard_fail
enforce branch coverage floor with tolerance = 0.05
rank remaining candidates with calibrated score
```

Best config:

```text
coverage_weight = 0.20
precision_weight = 0.05
jump_weight = 0.00
trim_weight = 0.00
off_mask_weight = 0.05
min_coverage = 0.80
min_precision = 0.72
```

| Evaluation | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Off-Mask mm | Mean Coverage | Mean Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Public core train | 0 | 0.093593 | 8.411765 | 4.000000 | 0.000000 | 0.941958 | 0.813420 |
| Public full test | 0 | 0.122887 | 11.1875 | 4.8125 | 0.061937 | 0.844804 | 0.841947 |
| Incoming review | 0 | 0.094245 | 8.5000 | 3.5000 | 0.123875 | 0.927795 | 0.777570 |

M2.25 confirms that the under-coverage failure can be corrected, but the corrected selector pays for that coverage with higher unified loss, more jumps, more trims, and worse incoming loss. It does not preserve the small M2.22 incoming gain.

## Comparison With Current Best

| Model | Public Full Loss | Public Full Coverage | Incoming Loss | Incoming Coverage | Decision |
|---|---:|---:|---:|---:|---|
| M2.14 current best | 0.111060 | 0.826159 | 0.090709 | 0.856321 | keep current best |
| M2.22 low-loss calibrated | 0.091982 | 0.689666 | 0.090585 | 0.854175 | not promoted: under-coverage risk |
| M2.23 high-coverage calibrated | 0.121659 | 0.846421 | not applied | not applied | not promoted: loss regression |
| M2.24 balanced calibrated | 0.117751 | 0.834037 | 0.094245 | 0.927795 | not promoted: loss regression |
| M2.25 coverage-floor calibrated | 0.122887 | 0.844804 | 0.094245 | 0.927795 | not promoted: coverage recovered but loss regressed |

## Decision

M2.22-M2.25 are **not promoted**.

M2.14 remains the current best.

## Research Interpretation

This is a stronger result than the fixed hard-gate experiments because it exposes the real tradeoff:

- low-loss calibration can reduce jump/trim/off-mask but may under-cover the design;
- high-coverage calibration protects coverage but increases command risk and loss;
- incoming review can improve slightly, but public full coverage must remain acceptable before promotion.

M2.25 tested the explicit coverage floor suggested by M2.22-M2.24. It fixed the public-full coverage problem, but it also showed that a coverage floor alone is too blunt: it forces safer/high-coverage candidates even when their command-level loss is worse.

The next selector should make the coverage floor adaptive rather than fixed. A promising next version is:

```text
stage 1: exclude hard_fail
stage 2: predict branch/source-specific coverage target from image and source features
stage 3: calibrated score among candidates that pass adaptive coverage
stage 4: reject candidates only when coverage loss is visually meaningful
```

This keeps the useful lesson from M2.22 without repeating the M2.25 over-correction.
