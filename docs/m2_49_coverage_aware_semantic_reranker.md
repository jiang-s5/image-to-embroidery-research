# M2.49 Coverage-Aware Semantic Reranker

Date: 2026-06-25

M2.49 continues from M2.48. M2.48 improved profile semantics and found a useful low-jump branch, but it also exposed a coverage reward-hacking risk: a stronger semantic rerank can reduce loss and jumps by selecting candidates with weak stitch coverage.

M2.49 adds coverage-aware decoding controls to `tools/apply_m2_candidate_selector.py`:

```text
--profile-coverage-constrained-rerank
--profile-coverage-soft-penalty
--profile-min-coverage
--profile-min-coverage-by-name
--profile-coverage-penalty-weight
--profile-coverage-fallback-margin
```

The important change is conceptual:

```text
M2.48:
  learned rank + semantic metric score

M2.49:
  learned rank + semantic metric score + profile-aware coverage risk
```

## Why This Was Needed

The previous M2.48 sweep showed that increasing semantic rerank strength can lower loss and jump count, but may lower coverage too much. A hard per-sample coverage floor prevents collapse, but it can also remove legitimate tradeoffs. For example, one incoming sample can reduce jump count sharply by choosing a lower-coverage candidate while the overall set still keeps reasonable mean coverage.

Therefore M2.49 supports two modes:

- hard constrained rerank: select the best candidate that satisfies a profile coverage floor;
- soft coverage penalty: penalize coverage deficit without rejecting the candidate outright.

The recommended setting is the soft penalty mode.

## Recommended Apply-Time Settings

```text
--profile-semantic-rerank
--profile-rerank-strength 0.90
--profile-coverage-soft-penalty
--profile-coverage-penalty-weight 0.35
--profile-min-coverage-by-name balanced=0.85,low_jump=0.85,strict_low_jump=0.85
```

This keeps the low-jump behavior of M2.48 but makes the coverage tradeoff explicit in the decoder.

## Incoming Review Result

The incoming review set contains 4 held-out paired review samples.

| Method / Profile | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Visible | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.42 default | 0 | 0.074710 | 8.250000 | 1.000000 | 0.000000 | 0.000000 | 0.996541 | 0.753519 |
| M2.48 balanced / low_jump | 0 | 0.073831 | 7.250000 | 2.250000 | 0.000000 | 0.000000 | 0.920700 | 0.788361 |
| M2.49 low_loss | 0 | 0.074710 | 8.250000 | 1.000000 | 0.000000 | 0.000000 | 0.996541 | 0.753519 |
| M2.49 balanced | 0 | 0.073831 | 7.250000 | 2.250000 | 0.000000 | 0.000000 | 0.920700 | 0.788361 |
| M2.49 precision | 0 | 0.091316 | 9.000000 | 2.750000 | 0.000000 | 0.000000 | 0.863788 | 0.858754 |
| M2.49 coverage | 0 | 0.076681 | 8.500000 | 1.000000 | 0.000000 | 0.000000 | 0.994577 | 0.772159 |
| M2.49 low_jump | 0 | 0.073831 | 7.250000 | 2.250000 | 0.000000 | 0.000000 | 0.920700 | 0.788361 |
| M2.49 strict_precision | 0 | 0.091321 | 9.000000 | 2.750000 | 0.000000 | 0.000000 | 0.852231 | 0.865129 |
| M2.49 strict_coverage | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.000000 | 0.995549 | 0.781068 |
| M2.49 strict_low_jump | 0 | 0.073831 | 7.250000 | 2.250000 | 0.000000 | 0.000000 | 0.920700 | 0.788361 |

## Sanity Check On Public ext33

This is not a generalization claim because the selector was trained from the public ext33 candidate table. It is only a sanity check that the new apply-time coverage controls do not introduce hard failures.

| Profile | Samples | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Visible | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | 33 | 0 | 0.052103 | 5.606061 | 0.757576 | 0.047591 | 0.000000 | 0.982012 | 0.806207 |
| low_jump | 33 | 0 | 0.046240 | 4.515152 | 0.727273 | 0.199076 | 0.000000 | 0.925772 | 0.799349 |

## Interpretation

M2.49 does not replace M2.42 as the safest high-coverage default. Its value is that it makes the M2.48 tradeoff explicit and controllable:

- use M2.42 when maximum coverage preservation is required;
- use M2.49 balanced / low_jump when fewer jumps and lower loss are preferred, while reporting the coverage tradeoff;
- use M2.49 coverage / strict_coverage when preserving coverage is the primary objective;
- use M2.49 precision / strict_precision only when stitch precision is more important than coverage.

## Artifacts

- `configs/best_current_model_m2_49_coverage_aware_semantic_reranker.json`
- `results/incoming_review_eval_v1_m2_49_softcov_s090_f085_w035_*_applied/`
- `results/incoming_review_eval_v1_m2_49_softcov_s090_f085_w035_summary.csv`
- `results/incoming_review_eval_v1_m2_49_softcov_s090_f085_w035_summary.json`
- `results/incoming_review_eval_v1_m2_49_covsemantic_soft_sweep.csv`
- `results/public_benchmark_v1_ext33_m2_49_softcov_sanity_summary.csv`
- `results/public_benchmark_v1_ext33_m2_49_softcov_sanity_summary.json`

