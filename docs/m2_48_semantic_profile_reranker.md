# M2.48 Semantic Profile Reranker

Date: 2026-06-25

M2.48 continues from M2.47. M2.47 solved profile collapse by storing separate selector heads, but the profile names were not semantically reliable enough: for example, a `precision` profile did not always maximize precision, and `low_jump` could still choose a high-jump branch.

M2.48 adds an optional apply-time semantic calibration layer:

```text
learned profile-head score
        +
profile-specific normalized metric score
        ↓
calibrated candidate ranking
```

The new options are:

```text
--profile-semantic-rerank
--profile-rerank-strength 0.75
```

This does not change legacy behavior unless the flag is enabled.

## Semantic Objectives

The semantic score is computed per sample after candidate generation, using normalized candidate metrics:

- `unified_loss`
- `jump_count`
- `trim_count`
- `coverage_ratio`
- `stitch_precision_ratio`
- `off_mask_stitch_length_mm`
- `visible_connector_count`

Profile-specific preferences:

- `low_loss`: prioritize low unified loss while still discouraging coverage collapse.
- `balanced`: combine low loss, low jump, coverage, precision, and off-mask safety.
- `precision` / `strict_precision`: prioritize stitch precision.
- `coverage` / `strict_coverage`: prioritize coverage.
- `low_jump` / `strict_low_jump`: prioritize fewer jumps and trims.

## Incoming Review Result

M2.48 uses the M2.47 profile-specific selector model with semantic reranking strength `0.75`.

| Method / Profile | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.42 default | 0 | 0.074710 | 8.250000 | 1.000000 | 0.000000 | 0.996541 | 0.753519 |
| M2.47 low_loss | 0 | 0.085912 | 9.250000 | 1.500000 | 0.000000 | 0.986371 | 0.809823 |
| M2.48 low_loss | 0 | 0.074710 | 8.250000 | 1.000000 | 0.000000 | 0.996541 | 0.753519 |
| M2.48 balanced | 0 | 0.073831 | 7.250000 | 2.250000 | 0.000000 | 0.920700 | 0.788361 |
| M2.48 precision | 0 | 0.095775 | 10.250000 | 1.750000 | 0.000000 | 0.938636 | 0.851462 |
| M2.48 coverage | 0 | 0.076681 | 8.500000 | 1.000000 | 0.000000 | 0.994577 | 0.772159 |
| M2.48 low_jump | 0 | 0.073831 | 7.250000 | 2.250000 | 0.000000 | 0.920700 | 0.788361 |
| M2.48 strict_precision | 0 | 0.091316 | 9.000000 | 2.750000 | 0.000000 | 0.863788 | 0.858754 |
| M2.48 strict_coverage | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.995549 | 0.781068 |
| M2.48 strict_low_jump | 0 | 0.073831 | 7.250000 | 2.250000 | 0.000000 | 0.920700 | 0.788361 |

## Interpretation

M2.48 is the first branch that improves the incoming review loss and jump count relative to M2.42:

- M2.42: loss `0.074710`, jump `8.25`
- M2.48 balanced / low_jump: loss `0.073831`, jump `7.25`

However, the improvement comes with a coverage tradeoff:

- M2.42 coverage: `0.996541`
- M2.48 balanced / low_jump coverage: `0.920700`

This is not a free win. It is a better low-jump / low-loss candidate for cases where slight coverage reduction is acceptable, not a universal replacement for M2.42.

The `precision` and `strict_precision` branches now behave more semantically:

- M2.48 precision: precision `0.851462`
- M2.48 strict_precision: precision `0.858754`

## Sweep Finding

A strength sweep showed a reward-hacking risk:

- `strength=0.90` can reduce loss to `0.067516` and jump to `6.50`, but coverage drops to `0.715100`.
- coverage guard at `0.90` prevents collapse, but also removes most low-jump gains.

Therefore the recommended profile-rerank setting is:

```text
--profile-semantic-rerank --profile-rerank-strength 0.75
```

and all reports must include coverage alongside loss and jump.

## Conclusion

M2.48 should be treated as a semantic calibration layer on top of M2.47, not as a pure learned model. It improves profile interpretability and exposes a real tradeoff surface:

- M2.42 remains the safest default when coverage preservation is critical.
- M2.48 `balanced` / `low_jump` is the best low-jump exploratory branch so far.
- M2.48 `precision` / `strict_precision` is useful when precision is preferred over coverage and jump count.

## Artifacts

- `configs/best_current_model_m2_48_semantic_profile_reranker.json`
- `results/incoming_review_eval_v1_m2_48_semantic_rerank_s075_*_applied/`
- `results/incoming_review_eval_v1_m2_48_semantic_rerank_sweep_summary.csv`
- `results/incoming_review_eval_v1_m2_48_semantic_rerank_sweep_summary.json`
