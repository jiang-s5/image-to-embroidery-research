# M2.89 Hard-Failure-Aware Selector Refresh

Date: 2026-07-07

## Summary

M2.89 is a robustness refresh after the M2.88 no-gate failure.

M2.88 showed that the learned preview selector was safe only when it remained behind the deterministic M2.86 command and preview gate. Removing that gate produced:

| Metric | M2.88 no-gate |
|---|---:|
| hard fail | 3 |
| mean unified loss | 0.09428522 |
| mean jumps | 59.42424242 |
| mean trims | 8.66666667 |
| mean off-mask length mm | 4.58460303 |
| mean visible connectors | 1.39393939 |

M2.89 turns those no-gate false positives into high-weight reject labels and adds explicit execution-risk features to the learned selector.

This is not a new professional-preview quality jump. It is a hard-failure awareness step: the learned selector is taught to reject candidates that look preview-positive but explode in execution metrics.

## Added Teacher Signal

New tool:

```text
tools/build_hard_failure_aware_teacher_refresh.py
```

Inputs:

```text
results/public_benchmark_v1_ext33_m2_87_preview_teacher_refresh/preview_teacher_seed_rows.csv
results/public_benchmark_v1_ext33_m2_88_preview_learned_deployment_policy_relaxed_nogate/preview_learned_decision_rows.csv
results/public_benchmark_v1_ext33_m2_88_preview_learned_deployment_policy_relaxed_nogate/preview_learned_selected_rows.csv
```

Output:

```text
results/public_benchmark_v1_ext33_m2_89_hard_failure_teacher_refresh/hard_failure_teacher_seed_rows.csv
```

The refresh adds `6` high-weight hard-negative rows from `4` affected samples:

| Item | Value |
|---|---:|
| final seed rows | 2250 |
| added hard-negative rows | 6 |
| positive teacher rows | 73 |
| hard-negative rows | 1155 |
| mean added teacher weight | 12.2198957 |
| mean added execution penalty | 28.55182602 |
| mean added jump explosion ratio | 56.33666667 |
| mean added trim explosion ratio | 28.40555556 |
| mean added visible connector count | 9.66666667 |

Added sample IDs:

```text
ocp_001
ocp_009
omj_010
qd_007
```

All added hard negatives are `mask_fill_serpentine` candidates. These candidates can score well on preview texture, but they are dangerous when the hard gate is removed.

## New Risk Features

M2.89 adds these numeric features to the selector:

```text
execution_penalty_score
hard_fail_probe_selected
jump_explosion_ratio
trim_explosion_ratio
offmask_explosion_mm
visible_explosion_count
```

The same features are computed both during teacher refresh and deployment.

## Selector Training

Command:

```powershell
python tools\train_stitch_family_selector.py `
  --seed-rows results\public_benchmark_v1_ext33_m2_89_hard_failure_teacher_refresh\hard_failure_teacher_seed_rows.csv `
  --output-dir results\public_benchmark_v1_ext33_m2_89_hard_failure_selector_rejectw16_margin010 `
  --model-id m2_89_hard_failure_selector_rejectw16_margin010 `
  --alpha 1.0 `
  --holdout-column source_name `
  --reject-weight 16.0 `
  --reject-margin 0.10 `
  --include-professional-gate-feature
```

Source-held-out result:

| Metric | M2.87 | M2.89 |
|---|---:|---:|
| rows | 1218 | 1224 |
| accuracy | 0.95073892 | 0.95016340 |
| positive accuracy | 0.80821918 | 0.80821918 |
| non-base positive accuracy | 0.68181818 | 0.68181818 |
| reject recall | 0.95982533 | 0.95916594 |
| false texture promotions | 0 | 0 |

The classifier metrics are almost unchanged, with a tiny accuracy and reject-recall cost. The reason to keep M2.89 is not classifier-score gain; it is that the no-gate hard-failure replay is fixed.

## Deployment Results

Strict deployment:

```powershell
python tools\apply_preview_learned_deployment_policy.py `
  --model-id m2_89_hard_failure_deployment_strict `
  --selector-model results\public_benchmark_v1_ext33_m2_89_hard_failure_selector_rejectw16_margin010\stitch_family_selector_model.json `
  --output-dir results\public_benchmark_v1_ext33_m2_89_hard_failure_deployment_strict
```

Relaxed no-gate replay:

```powershell
python tools\apply_preview_learned_deployment_policy.py `
  --model-id m2_89_hard_failure_deployment_relaxed_nogate `
  --selector-model results\public_benchmark_v1_ext33_m2_89_hard_failure_selector_rejectw16_margin010\stitch_family_selector_model.json `
  --no-require-preview-gate `
  --output-dir results\public_benchmark_v1_ext33_m2_89_hard_failure_deployment_relaxed_nogate
```

Both strict and relaxed replay now select the same safe final outputs as M2.86/M2.88 strict:

| Metric | M2.89 strict | M2.89 no-gate replay |
|---|---:|---:|
| samples | 33 | 33 |
| learned policy switches | 7 | 7 |
| hard fail | 0 | 0 |
| mean unified loss | 0.05706876 | 0.05706876 |
| mean jumps | 6.15151515 | 6.15151515 |
| mean trims | 0.60606061 | 0.60606061 |
| mean off-mask length mm | 0.10619091 | 0.10619091 |
| mean visible connectors | 0.0 | 0.0 |
| mean professional preview score | 0.73544111 | 0.73544111 |
| mean generator texture score | 0.22612121 | 0.22612121 |
| outputs different from M2.86 | 0 | 0 |

## Interpretation

M2.89 proves that execution-failure feedback can be folded back into the learned selector.

The important change from M2.88 is:

```text
M2.88 no-gate:
  preview gains, but 3 hard failures and jump explosion

M2.89 no-gate replay:
  hard_fail = 0, visible = 0, final selections return to safe M2.86-equivalent outputs
```

This means the model has become more conservative around dangerous preview-positive candidates. The cost is that M2.89 does not yet create a better final preview than M2.86/M2.88 strict.

## Current Position

M2.89 should be promoted as the current learned-control training package, but not as a new visual-output profile.

Current best output profile remains:

```text
M2.86/M2.88/M2.89 strict-equivalent final outputs
```

Current best learned selector package becomes:

```text
M2.89 hard-failure-aware selector
```

## Next Step

The next real quality gain should focus on professional stitch appearance, not another safety-only refresh.

Recommended M2.90 direction:

```text
professional texture objective + multi-stitch-family preview reranking
```

Priority targets:

- separate fill, satin, outline, running-stitch visual objectives;
- prevent texture gain from reintroducing jump/trim/off-mask failures;
- add a preview-quality metric that rewards layered stitch appearance instead of only line safety;
- keep M2.89 hard-failure features as the safety floor.
