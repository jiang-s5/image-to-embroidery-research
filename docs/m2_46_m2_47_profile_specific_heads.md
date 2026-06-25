# M2.46 / M2.47 Profile-Conditioned Selector Follow-up

Date: 2026-06-25

This follow-up continues from M2.45. M2.45 introduced task-conditioned selector features, but the incoming review results showed that several profiles collapsed to the same selected candidates. M2.46 and M2.47 test whether this collapse is caused by weak candidate diversity or by the shared selector head itself.

## M2.46: Profile-Diverse Shared Head

M2.46 expanded the candidate pool from 8 to 14 candidates and added stronger task profiles:

- `strict_precision`
- `strict_coverage`
- `strict_low_jump`

The added candidates include `adapt_t2/t3/t4`, extra skeleton candidates, and an auto-hybrid skeleton candidate.

Public ext33 leave-one-out, evaluated with `balanced`:

| Model | Candidates | Profiles | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.45 shared task head | 8 | 5 | 0 | 0.050318 | 5.484848 | 0.454545 | 0.093536 | 0.956618 | 0.816809 |
| M2.46 shared task head | 14 | 8 | 0 | 0.059078 | 6.454545 | 0.727273 | 0.049952 | 0.969613 | 0.821866 |

Incoming review application:

All M2.46 task profiles selected the same candidate set on the 4-sample incoming set:

| Profile Group | Loss | Jump | Trim | Coverage | Precision |
|---|---:|---:|---:|---:|---:|
| all M2.46 profiles | 0.080262 | 8.750000 | 1.250000 | 0.993585 | 0.799709 |

Interpretation:

M2.46 improves coverage and precision on public LOO, but the single shared selector head still collapses to the same incoming decisions. Candidate diversity alone is not enough.

## M2.47: Profile-Specific Heads

M2.47 adds `--profile-specific-models`. Instead of forcing one learned selector head to represent all profile objectives, the trainer now stores one selector head per task profile under `profile_models`.

`tools/apply_m2_candidate_selector.py` now uses the selected profile head automatically when a model contains `profile_models`.

Public ext33 leave-one-out, evaluated with `balanced`:

| Model | Candidates | Profiles | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.47 profile-specific heads | 14 | 8 | 0 | 0.060065 | 6.666667 | 0.727273 | 0.015015 | 0.981206 | 0.830467 |

Incoming review application:

| Profile | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision | Main Behavior |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| low_loss | 0 | 0.085912 | 9.250000 | 1.500000 | 0.000000 | 0.986371 | 0.809823 | mixed adapt candidates |
| balanced | 0 | 0.093669 | 10.000000 | 1.750000 | 0.000000 | 0.951701 | 0.836955 | all adapt_t4 |
| precision | 0 | 0.093669 | 10.000000 | 1.750000 | 0.000000 | 0.950943 | 0.840096 | mostly adapt_t3 |
| coverage | 0 | 0.082284 | 9.000000 | 1.250000 | 0.000000 | 0.994104 | 0.789821 | high-coverage nearest/adapt mix |
| low_jump | 0 | 0.095803 | 10.250000 | 1.750000 | 0.000000 | 0.924272 | 0.861063 | high-precision inset2 |
| strict_precision | 0 | 0.074710 | 8.250000 | 1.000000 | 0.000000 | 0.996541 | 0.753519 | recovers M2.42-like low-loss selector |
| strict_coverage | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.995549 | 0.781068 | M2.44-like coverage/precision balance |
| strict_low_jump | 0 | 0.080262 | 8.750000 | 1.250000 | 0.000000 | 0.993585 | 0.799709 | M2.45 precision-like branch |

## Conclusion

M2.47 is not the new default best selector. M2.42 still remains the best default for low unified loss on incoming review:

- M2.42 incoming loss: `0.074710`
- M2.47 best incoming loss: `0.074710`

However, M2.47 fixes the main M2.45/M2.46 limitation: profile outputs no longer collapse to a single candidate set. This is useful research progress because it shows that task conditioning needs separate heads or a stronger conditional architecture.

The next optimization should focus on semantic alignment of profiles. In the current M2.47 run, some labels are not semantically perfect: for example, `strict_precision` recovers the low-loss M2.42-like behavior instead of maximizing precision, while `low_jump` actually selects a higher-jump high-precision branch. This suggests that the learned profile heads need either:

- profile-specific calibration on an external validation split, or
- an apply-time semantic reranker that combines learned score with normalized profile utility.

## Artifacts

- `configs/best_current_model_m2_47_profile_specific_heads.json`
- `results/public_benchmark_v1_ext33_m2_46_profile_diverse_task_conditioned_e180/`
- `results/public_benchmark_v1_ext33_m2_47_profile_specific_heads_e180/`
- `results/incoming_review_eval_v1_m2_46_profile_diverse_*_applied/`
- `results/incoming_review_eval_v1_m2_47_profile_heads_*_applied/`
