# M2.44 Pareto-Teacher Learned Selector

Date: 2026-06-25

M2.44 connects the M2.43 Pareto audit back into the learned selector. It extends the listwise teacher used by `tools/train_m2_candidate_selector.py` so the teacher score can include:

- coverage deficit
- precision deficit
- off-mask stitch length
- visible connector count
- normalized jump count
- normalized trim count

This changes the learned selector from a single-objective listwise learner into a multi-objective preference learner.

## Code Change

`tools/train_m2_candidate_selector.py` now supports:

```text
--listwise-teacher-off-mask-weight
--listwise-teacher-visible-weight
--listwise-teacher-jump-weight
--listwise-teacher-trim-weight
--listwise-teacher-jump-scale
--listwise-teacher-trim-scale
```

The default values are zero, so older listwise runs are backward-compatible.

## Best Tested Setting

```text
--target listwise_softmax
--alpha 0.001
--listwise-temperature 0.02
--listwise-teacher-coverage-weight 0.10
--listwise-teacher-precision-weight 0.05
--listwise-teacher-off-mask-weight 0.02
--listwise-teacher-visible-weight 0.005
--listwise-teacher-jump-weight 0.003
--listwise-teacher-trim-weight 0.003
--listwise-teacher-min-precision 0.78
```

## Result

Public ext33 leave-one-out:

| Selector | Samples | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.42 calibrated default | 33 | 0 | 0.049160 | 5.484848 | 0.424242 | 0.049952 | 0.978737 | 0.801506 |
| M2.44 Pareto teacher learned | 33 | 0 | 0.050318 | 5.484848 | 0.454545 | 0.093536 | 0.956618 | 0.816809 |

Incoming review application:

| Selector | Samples | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.42 calibrated default | 4 | 0 | 0.074710 | 8.250000 | 1.000000 | 0.000000 | 0.996541 | 0.753519 |
| M2.44 Pareto teacher learned | 4 | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.995549 | 0.781068 |

## Interpretation

M2.44 succeeds at the learning objective: it shifts the learned selector toward higher precision and stays hard-fail free. However, it does not beat M2.42 on unified loss, jump, trim, or coverage.

Therefore:

- M2.42 remains the default current selector.
- M2.43 remains the Pareto audit / optional public-balanced selector.
- M2.44 is a research branch proving that Pareto-style supervision can be injected into the learned selector.

The next learned-selector step should avoid choosing one fixed teacher scalar. A better version would train from Pareto-front preferences or allow a user/task-conditioned tradeoff vector, for example `prefer_precision`, `prefer_coverage`, or `prefer_low_loss`.

## Artifacts

- `configs/best_current_model_m2_44_pareto_teacher_selector.json`
- `results/public_benchmark_v1_ext33_m2_44_pareto_teacher_weak_a001_t002/`
- `results/incoming_review_eval_v1_m2_44_pareto_teacher_weak_a001_t002_applied/`
