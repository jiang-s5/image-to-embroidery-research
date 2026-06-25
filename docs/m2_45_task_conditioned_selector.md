# M2.45 Task-Conditioned Learned Selector

Date: 2026-06-25

M2.45 extends the M2.44 Pareto-teacher selector into a task-conditioned selector. Instead of training one fixed tradeoff, the same learned selector can be conditioned with a task profile:

- `low_loss`
- `balanced`
- `precision`
- `coverage`
- `low_jump`

The goal is not to replace the current default selector yet. The goal is to make the selector controllable so later evaluation can ask different questions, such as "prefer precision" or "prefer coverage", without retraining a different model for every scalar objective.

## Code Change

`tools/train_m2_candidate_selector.py` now supports:

```text
--task-profiles low_loss,balanced,precision,coverage,low_jump
--eval-task-profile balanced
```

The training script expands candidate rows across task profiles and injects task-profile features plus profile-specific teacher weights.

`tools/apply_m2_candidate_selector.py` now supports:

```text
--task-profile balanced
```

If a selector model contains `task_profiles`, the apply script activates the requested profile before ranking candidates.

## Tested Setting

```text
--target listwise_softmax
--alpha 0.001
--listwise-temperature 0.02
--listwise-epochs 1200
--listwise-learning-rate 0.05
--task-profiles low_loss,balanced,precision,coverage,low_jump
--eval-task-profile balanced
--listwise-teacher-flat-min-coverage 0.84
--listwise-teacher-line-min-coverage 0.58
--listwise-teacher-min-precision 0.78
```

Candidate pool:

- `mask_fill_edgewalk_nearestrow_inset1_rows16_p40`
- `mask_fill_edgewalk_nearestrow_inset2_rows16_p40`
- `mask_fill_edgewalk_nearestrow_rows16_p40`
- `mask_fill_edgewalk_rows16_p40`
- `auto_evalrepair_r10_c20`
- `skeleton_c30_m20`
- `skeleton_c30_m4`
- `mask_fill_edgewalk_nearestrow_safeadt_t2m5_r055_a64_rows16_p40`

## Result

Public ext33 leave-one-out, evaluated with `balanced`:

| Selector | Samples | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.42 calibrated default | 33 | 0 | 0.049160 | 5.484848 | 0.424242 | 0.049952 | 0.978737 | 0.801506 |
| M2.44 Pareto teacher | 33 | 0 | 0.050318 | 5.484848 | 0.454545 | 0.093536 | 0.956618 | 0.816809 |
| M2.45 task-conditioned balanced | 33 | 0 | 0.050318 | 5.484848 | 0.454545 | 0.093536 | 0.956618 | 0.816809 |

Incoming review application:

| Profile | Samples | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | 4 | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.995549 | 0.781068 |
| coverage | 4 | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.995549 | 0.781068 |
| low_jump | 4 | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.995549 | 0.781068 |
| precision | 4 | 0 | 0.080262 | 8.750000 | 1.250000 | 0.000000 | 0.993585 | 0.799709 |
| low_loss | 4 | 0 | 0.080262 | 8.750000 | 1.250000 | 0.000000 | 0.993585 | 0.799709 |

## Interpretation

M2.45 confirms that the learned selector can carry task-conditioning features without breaking the M2.44 Pareto-teacher behavior. On the current incoming set, `precision` shifts the selected candidates toward higher stitch precision, but it pays for that with slightly worse unified loss and jump count.

Therefore:

- M2.42 remains the best default for lowest unified loss.
- M2.44 remains the fixed Pareto-teacher learned branch.
- M2.45 is the first controllable selector branch and should be used for profile-based ablations.

## Next Step

The task-conditioned profiles are still shallow because each profile shares the same candidate pool and only changes the selector preferences. The next meaningful improvement is to add profile-diverse candidates, especially candidates that explicitly trade off fill coverage, precision, and jump count. Without candidate diversity, some profiles collapse to the same selected outputs.

## Artifacts

- `configs/best_current_model_m2_45_task_conditioned_selector.json`
- `results/public_benchmark_v1_ext33_m2_45_task_conditioned_balanced/`
- `results/incoming_review_eval_v1_m2_45_task_conditioned_balanced_applied/`
- `results/incoming_review_eval_v1_m2_45_task_conditioned_precision_applied/`
- `results/incoming_review_eval_v1_m2_45_task_conditioned_low_loss_applied/`
- `results/incoming_review_eval_v1_m2_45_task_conditioned_coverage_applied/`
- `results/incoming_review_eval_v1_m2_45_task_conditioned_low_jump_applied/`
