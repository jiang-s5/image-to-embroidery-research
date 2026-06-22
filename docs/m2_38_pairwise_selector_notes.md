# M2.38 Pairwise Learned Selector Notes

Date: 2026-06-22

M2.38 tests the next learning direction after M2.36: pairwise candidate ranking. Instead of predicting an absolute candidate score or a binary oracle-choice label, the model learns same-image pairwise score differences:

```text
candidate A score - candidate B score
```

At inference, the model still assigns a scalar ranking score to each candidate and selects the lowest-scoring candidate.

## Why This Was Tried

M2.36 showed that fine-grained candidate-family features make the learned selector competitive with the calibrated M2.34 selector. However, score regression still treats each candidate independently. The real selection problem is relative:

```text
for this image, should we choose candidate A or candidate B?
```

M2.38 therefore adds a linear pairwise ranker trained from candidate feature differences within the same sample.

## Implementation

`tools/train_m2_candidate_selector.py` now supports:

```text
--target pairwise_score_delta
--pairwise-min-score-gap <float>
```

The pairwise training path:

1. Build candidate rows as in M2.36.
2. Group rows by `sample_id`.
3. For every same-sample pair, compute normalized feature difference.
4. Train a ridge model to predict `oracle_score(left) - oracle_score(right)`.
5. At inference, compute a per-candidate ranking score and choose the minimum.

The saved model has:

```json
{
  "type": "ridge_pairwise_candidate_ranker",
  "target": "pairwise_score_delta",
  "select_direction": "min"
}
```

`tools/apply_m2_candidate_selector.py` can apply this model through the same interface as earlier learned selectors.

## Public Benchmark Results

Benchmark: `datasets/public_benchmark_v1_ext33`, 33 samples.

| Variant | Alpha | Min Gap | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.34 calibrated selector | - | - | 0 | 0.051174 | 5.636364 | 0.424242 | 0.082527 | 0.968881 | 0.819738 |
| M2.36 score regression | 10 | - | 0 | 0.052028 | 5.636364 | 0.454545 | 0.112558 | 0.973941 | 0.822276 |
| M2.38 pairwise | 0.1 | 0.000 | 0 | 0.052711 | 5.606061 | 0.484848 | 0.146842 | 0.973414 | 0.823411 |
| M2.38 pairwise | 1 | 0.000 | 0 | 0.052711 | 5.606061 | 0.484848 | 0.146842 | 0.973414 | 0.823411 |
| M2.38 pairwise | 10 | 0.000 | 0 | 0.052442 | 5.666667 | 0.484848 | 0.112558 | 0.974139 | 0.821405 |
| M2.38 pairwise | 100 | 0.000 | 0 | 0.052442 | 5.666667 | 0.484848 | 0.112558 | 0.974139 | 0.821405 |
| M2.38 pairwise | 10 | 0.002 | 0 | 0.052442 | 5.666667 | 0.484848 | 0.112558 | 0.974139 | 0.821405 |
| M2.38 pairwise | 10 | 0.005 | 0 | 0.052442 | 5.666667 | 0.484848 | 0.112558 | 0.974139 | 0.821405 |
| M2.38 pairwise | 10 | 0.010 | 0 | 0.052442 | 5.666667 | 0.484848 | 0.112558 | 0.974260 | 0.821601 |
| M2.38 pairwise | 10 | 0.020 | 0 | 0.052442 | 5.666667 | 0.484848 | 0.112558 | 0.974260 | 0.821601 |

M2.38 does not beat M2.34 or M2.36 on the public benchmark.

## Incoming Review Check

The best M2.38 model tested for incoming was alpha 10, min gap 0.

| Variant | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.34 incoming LOO | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.995549 | 0.781068 |
| M2.36 score regression applied | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.995549 | 0.780990 |
| M2.38 pairwise applied | 0 | 0.077491 | 8.250000 | 1.000000 | 0.123875 | 0.994320 | 0.765355 |

M2.38 slightly improves incoming unified loss, jumps, and trims, but it introduces a small off-mask length and lowers precision. Because public validation is weaker, this is not enough for promotion.

## Decision

M2.38 is not promoted.

M2.34 remains the current best model/system. M2.38 is useful as a learning-system experiment: it confirms that relative ranking is plausible, but a linear pairwise delta model is not enough to replace the calibrated selector.

## Research Takeaway

Pairwise ranking is still the right conceptual direction, but the first implementation is too weak. It learns local score differences, yet it does not model the listwise choice context well enough.

The next selector should test:

- listwise softmax ranking over all candidates for the same image,
- calibrated-score distillation from M2.34 as a teacher,
- source-family-aware ranking heads,
- hard-example weighting for samples where learned ranking differs from the oracle,
- explicit precision/off-mask secondary objectives.

In short: M2.38 moves the learning loop closer to the real selection problem, but current best remains the calibrated M2.34 selector.
