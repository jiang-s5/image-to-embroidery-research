# M2.36 / M2.37 Fine-Grained Learned Selector Notes

Date: 2026-06-22

This experiment tests whether the M2 learned candidate selector can approach or outperform the calibrated M2.34 selector when it can see fine-grained candidate-family features.

M2.34 remains the current best model/system. M2.36 and M2.37 are recorded as learning-system ablations.

## Motivation

Earlier learned selector experiments were much weaker than the calibrated selector. One reason was that the feature set only distinguished broad families such as skeleton, auto, mask-fill, style-aware, outline, and satin. It did not tell the model whether a mask-fill candidate was:

- raw nearest-row,
- fixed inset1,
- fixed inset2,
- adaptive inset,
- edgewalk,
- quantized/strict variants.

That is a serious blind spot because M2.34's improvement comes specifically from the nearest-row fixed inset candidates.

## Implementation

`tools/train_m2_candidate_selector.py` now adds fine-grained candidate features:

- `candidate_is_edgewalk`
- `candidate_is_nearestrow`
- `candidate_is_raw_nearestrow`
- `candidate_is_fixed_inset`
- `candidate_is_inset1`
- `candidate_is_inset2`
- `candidate_fixed_inset_px`
- `candidate_is_adaptive_inset`
- `candidate_adaptive_thin_px`
- `candidate_adaptive_mid_px`
- `candidate_row_spacing_tag`
- `candidate_mask_path_limit_tag`
- `candidate_is_satinrail`
- `candidate_is_quantvalid`
- `candidate_is_segvalid`
- `candidate_is_qdirect`
- `candidate_is_qpath`
- `candidate_is_strict`

It also adds candidate-family interactions with mask area, skeleton ratio, line score, and coverage.

The learned selector now supports two training targets:

| Target | Selection direction | Meaning |
|---|---|---|
| `oracle_score` | lower is better | Regress the oracle/candidate score. |
| `oracle_choice` | higher is better | Predict whether the candidate is the per-sample oracle choice. |

`tools/apply_m2_candidate_selector.py` now respects `select_direction` from the trained model.

## Public Benchmark Results

Benchmark: `datasets/public_benchmark_v1_ext33`, 33 samples. Candidate pool includes the M2.34 candidates plus adaptive inset candidates.

### Score-Regression Target (`oracle_score`)

| Variant | Alpha | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.34 calibrated selector | - | 0 | 0.051174 | 5.636364 | 0.424242 | 0.082527 | 0.968881 | 0.819738 |
| M2.36 learned score | 0.1 | 0 | 0.052442 | 5.666667 | 0.484848 | 0.112558 | 0.974139 | 0.821405 |
| M2.36 learned score | 1 | 0 | 0.052442 | 5.666667 | 0.484848 | 0.112558 | 0.974139 | 0.821405 |
| M2.36 learned score | 10 | 0 | 0.052028 | 5.636364 | 0.454545 | 0.112558 | 0.973941 | 0.822276 |
| M2.36 learned score | 100 | 0 | 0.055327 | 6.121212 | 0.515152 | 0.062606 | 0.983517 | 0.826600 |

Best public learned-score run: alpha 10.

This is a major improvement over earlier learned selectors, but it still does not beat M2.34. It improves coverage and precision slightly, but off-mask length is higher.

### Choice Target (`oracle_choice`)

| Variant | Alpha | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.37 learned choice | 1 | 0 | 0.074472 | 7.969697 | 1.060606 | 0.081876 | 0.983665 | 0.849245 |
| M2.37 learned choice | 10 | 0 | 0.071836 | 7.696970 | 1.000000 | 0.081876 | 0.983408 | 0.853190 |
| M2.37 learned choice | 100 | 0 | 0.071837 | 7.696970 | 1.000000 | 0.081876 | 0.982203 | 0.854755 |

The choice target mostly collapses toward fixed inset2 behavior. It is not competitive with M2.34's calibrated multi-candidate selector.

## Incoming Review Check

The best score-regression model from public alpha10 was applied to the incoming review evaluation set.

| Variant | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.34 incoming LOO | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.995549 | 0.781068 |
| M2.36 learned score alpha10 applied | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.995549 | 0.780990 |
| M2.37 learned choice alpha100 applied | 0 | 0.095803 | 10.250000 | 1.750000 | 0.000000 | 0.924272 | 0.861063 |

The score-regression learned selector matches M2.34 on incoming review, while the choice target is worse.

## Decision

Do not promote M2.36 or M2.37.

M2.34 remains the current best because it has the lowest public leave-one-out loss and the best validated balance of loss, jumps, trims, off-mask length, coverage, and precision.

## Research Takeaways

1. Fine-grained candidate-family features are important.
2. Score regression is more stable than direct oracle-choice classification on this small benchmark.
3. The learned selector has now become competitive with the calibrated selector, but it still needs a better ranking/preference formulation before replacing M2.34.
4. The next learning direction should be pairwise/listwise ranking rather than one-vs-rest oracle-choice regression.

## Next Candidate Direction

The next learned selector should train on pairwise preferences:

```text
same image:
candidate A has lower evaluation score than candidate B
=> model should rank A above B
```

This is closer to the actual planner-selection problem than either scalar score regression or binary oracle-choice prediction.
