# M2.39 Listwise Softmax Selector Notes

Date: 2026-06-22

M2.39 tests a listwise learned selector. Unlike M2.36 score regression and M2.38 pairwise ranking, this model sees all candidates for the same image as a list and learns a soft teacher distribution derived from oracle candidate scores.

M2.39 is the first learned selector that beats M2.34 on public unified loss, but it is not promoted as the current best because it trades away too much coverage. It is recorded as a low-loss experimental branch.

## Implementation

`tools/train_m2_candidate_selector.py` now supports:

```text
--target listwise_softmax
--listwise-temperature <float>
--listwise-epochs <int>
--listwise-learning-rate <float>
```

Training process:

1. Group candidate rows by `sample_id`.
2. Convert oracle scores into a teacher distribution:

```text
teacher = softmax(-oracle_score / temperature)
```

3. Predict a score for every candidate in the same image.
4. Convert predicted scores into a candidate distribution:

```text
predicted = softmax(-predicted_score / temperature)
```

5. Minimize cross-entropy from teacher to predicted with L2 regularization.

At inference, the selector chooses the candidate with the lowest predicted score.

## Public Benchmark Results

Benchmark: `datasets/public_benchmark_v1_ext33`, 33 samples.

| Variant | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.34 calibrated selector | 0 | 0.051174 | 5.636364 | 0.424242 | 0.082527 | 0.968881 | 0.819738 |
| M2.36 score regression alpha10 | 0 | 0.052028 | 5.636364 | 0.454545 | 0.112558 | 0.973941 | 0.822276 |
| M2.38 pairwise alpha10 | 0 | 0.052442 | 5.666667 | 0.484848 | 0.112558 | 0.974139 | 0.821405 |
| M2.39 listwise a0.001 t0.02 | 0 | 0.050832 | 5.575758 | 0.393939 | 0.097542 | 0.938372 | 0.823438 |
| M2.39 listwise a0.001 t0.05 | 0 | 0.056238 | 6.000000 | 0.575758 | 0.131827 | 0.948867 | 0.828373 |
| M2.39 listwise a0.01 t0.05 | 0 | 0.056309 | 6.000000 | 0.575758 | 0.134994 | 0.948957 | 0.826577 |

The best low-loss run is `alpha=0.001`, `temperature=0.02`.

It improves unified loss, jump count, and trim count compared with M2.34, but mean coverage drops from `0.968881` to `0.938372`.

The lowest-coverage selected samples reveal the problem:

| Sample | Chosen Candidate | Coverage |
|---|---|---:|
| `omj_010` | `skeleton_c30_m20` | 0.249187 |
| `txt_003` | `skeleton_c30_m4` | 0.609455 |
| `ocp_010` | `auto_evalrepair_r10_c20` | 0.688911 |
| `txt_005` | `skeleton_c30_m4` | 0.711090 |

The selector sometimes chooses very low-coverage candidates because unified loss does not directly include stitch coverage.

## Coverage-Guard Variants

To test whether M2.39 could be made balanced, several source-mode coverage floors were evaluated.

| Variant | Flat Floor | Line Floor | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.39 no floor | - | - | 0 | 0.050832 | 5.575758 | 0.393939 | 0.097542 | 0.938372 | 0.823438 |
| M2.39 floor050 | 0.50 | 0.55 | 0 | 0.052041 | 5.606061 | 0.424242 | 0.131827 | 0.960100 | 0.825030 |
| M2.39 floor070 | 0.70 | 0.55 | 0 | 0.053485 | 5.757576 | 0.454545 | 0.131827 | 0.969439 | 0.823320 |
| M2.39 floor075 | 0.75 | 0.55 | 0 | 0.053216 | 5.818182 | 0.454545 | 0.097542 | 0.970164 | 0.821314 |
| M2.39 floor084 | 0.84 | 0.58 | 0 | 0.054262 | 5.848485 | 0.454545 | 0.131827 | 0.969346 | 0.825693 |
| M2.39 floor090 | 0.90 | 0.60 | 0 | 0.053739 | 5.787879 | 0.454545 | 0.131827 | 0.969380 | 0.824695 |

Coverage guards recover coverage but lose the public loss advantage over M2.34.

## Incoming Review Check

Incoming review set: 4 private evaluation-only samples.

| Variant | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.34 incoming LOO | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.995549 | 0.781068 |
| M2.39 no floor applied | 0 | 0.073831 | 7.250000 | 2.250000 | 0.000000 | 0.920700 | 0.788282 |
| M2.39 floor050 applied | 0 | 0.079998 | 8.000000 | 2.250000 | 0.000000 | 0.918531 | 0.794791 |

The unconstrained listwise selector improves incoming unified loss and jump count, but coverage drops sharply. The coverage-floor version does not beat M2.34 on incoming.

## Decision

Do not promote M2.39 as the current best.

M2.34 remains the recommended balanced model/system because it has a better public loss/coverage tradeoff:

- M2.34 public LOO: `loss 0.051174`, `coverage 0.968881`
- M2.39 no floor: `loss 0.050832`, `coverage 0.938372`
- M2.39 floor variants: coverage recovers, but loss becomes worse than M2.34

## Research Takeaway

M2.39 is important because it proves the listwise direction can beat the calibrated selector on unified loss. The remaining issue is objective mismatch: unified loss does not sufficiently penalize low coverage, so the learned selector can choose visually incomplete candidates.

The next step should train a listwise teacher that includes coverage directly:

```text
teacher_score =
  unified_loss
  + lambda_coverage * max(0, target_coverage - coverage)
  + lambda_precision * max(0, target_precision - precision)
```

This would combine the advantage of M2.39's listwise learning with the balanced behavior of M2.34's calibrated selector.
